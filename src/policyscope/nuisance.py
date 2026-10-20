"""Nuisance-модели: пропенсити ``pi_A(a|x)`` и модель исхода ``mu(x, a)``.

Обе модели **не зависят от оцениваемой политики B**, поэтому обучаются один раз и
переиспользуются для любого числа кандидатов.

Обе всегда считаются **out-of-fold** (cross-fitting): прогноз для строки делает
модель, которая эту строку не видела. In-sample прогнозы систематически подстроены
под шум данных: гибкая модель пропенсити «запоминает» залогированные действия и
сжимает веса, а модель исхода занижает остатки, на которых держится DR.

Оценённые пропенсити дополнительно ограничиваются снизу **полом для каждого
действия** ``floor_a = propensity_floor * доля действия a в логах``. Без пола
модель, не воспроизводящая ступенчатую логирующую политику (ε-greedy), выдаёт
пропенсити порядка 1e-4 там, где на деле 0.05, и единичные веса в тысячи определяют
всю оценку.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from policyscope.data import LogData

__all__ = ["NuisanceConfig", "Nuisance", "make_folds", "fit_nuisance", "outcome_design"]


@dataclass(frozen=True)
class NuisanceConfig:
    """Настройки nuisance-моделей.

    Attributes:
        propensity_model: sklearn-классификатор с ``predict_proba`` для ``pi_A(a|x)``.
            ``None`` — мультиномиальная логистическая регрессия.
        outcome_model: sklearn-модель для ``mu(x, a)``; получает признаки, one-hot
            действия и их попарные произведения. ``None`` — логистическая регрессия для
            бинарной награды и Ridge для непрерывной.
        n_folds: Число фолдов cross-fitting.
        propensity_floor: Доля от частоты действия, ниже которой оценённое пропенсити
            не опускается. Применяется только к оценённым пропенсити.
    """

    propensity_model: Any = None
    outcome_model: Any = None
    n_folds: int = 5
    propensity_floor: float = 0.1

    def __post_init__(self) -> None:
        if self.n_folds < 2:
            raise ValueError("n_folds должен быть не меньше 2")
        if not 0.0 < self.propensity_floor < 1.0:
            raise ValueError("propensity_floor должен лежать в (0, 1)")


@dataclass(frozen=True)
class Nuisance:
    """Out-of-fold прогнозы nuisance-моделей, выровненные по строкам выборки.

    Attributes:
        p_taken: Пропенсити залогированного действия, используемые в весах
            (для оценённых — с полом).
        source: ``"logged"`` или ``"estimated"``.
        p_raw: Оценённое пропенсити без пола (только для оценённых).
        p_max: Максимальная оценённая вероятность по действиям — признак
            детерминированности A (только для оценённых).
        floor: Пол для каждого действия ``(k,)`` (только для оценённых).
        unseen_share: Доля строк, чьё действие не встретилось в обучающем фолде.
        mu: Метрика -> прогнозы ``(n, k)``.
        fit_seconds: Время обучения (для прогноза длительности bootstrap).
    """

    p_taken: np.ndarray
    source: str
    p_raw: Optional[np.ndarray]
    p_max: Optional[np.ndarray]
    floor: Optional[np.ndarray]
    unseen_share: float
    mu: dict[str, np.ndarray]
    fit_seconds: float

    def floored(self, a: np.ndarray, factor: float) -> np.ndarray:
        """Пропенсити с полом, умноженным на ``factor`` (анализ чувствительности)."""
        if self.source != "estimated":
            return self.p_taken
        assert self.p_raw is not None and self.floor is not None
        return np.maximum(self.p_raw, factor * self.floor[a])


def _default_propensity_model():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))


def _default_outcome_model(binary: bool):
    if binary:
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    return make_pipeline(StandardScaler(), Ridge(alpha=1.0))


def make_folds(groups: np.ndarray, n_folds: int, rng: np.random.Generator) -> np.ndarray:
    """Номер фолда для каждой строки; все строки одной группы — в одном фолде.

    Группа — кластер (клиент) или, в bootstrap-реплике, исходная строка: копии одной
    строки не должны попадать одновременно в обучение и в прогноз.
    """
    unique_groups, group_of_row = np.unique(groups, return_inverse=True)
    n_groups = len(unique_groups)
    if n_groups < n_folds:
        raise ValueError(f"Групп ({n_groups}) меньше, чем фолдов ({n_folds})")
    fold_of_group = rng.permutation(n_groups) % n_folds
    return fold_of_group[group_of_row]


def outcome_design(X: np.ndarray, a: np.ndarray, k: int) -> np.ndarray:
    """Дизайн-матрица модели исхода: ``[x, onehot(a), x * onehot(a)]``.

    Блок произведений даёт каждому действию собственные наклоны: без него модель
    отличает действия только сдвигом и не может выразить, что оффер работает по-разному
    для разных клиентов.
    """
    onehot = np.zeros((len(a), k))
    onehot[np.arange(len(a)), a] = 1.0
    inter = (X[:, :, None] * onehot[:, None, :]).reshape(len(a), -1)
    return np.hstack([X, onehot, inter])


def _propensity_oof(
    X: np.ndarray, a: np.ndarray, k: int, folds: np.ndarray, n_folds: int, model: Any
) -> tuple[np.ndarray, int]:
    probs = np.zeros((len(a), k))
    unseen = 0
    for f in range(n_folds):
        train, hold = folds != f, folds == f
        classes = np.unique(a[train])
        if len(classes) == 1:
            probs[np.ix_(hold, classes)] = 1.0
        else:
            fitted = clone(model).fit(X[train], a[train])
            probs[np.ix_(hold, np.asarray(fitted.classes_, dtype=int))] = fitted.predict_proba(X[hold])
        unseen += int(np.sum(~np.isin(a[hold], classes)))
    return probs, unseen


def _outcome_oof(
    X: np.ndarray, a: np.ndarray, r: np.ndarray, k: int, binary: bool,
    folds: np.ndarray, n_folds: int, model: Any,
) -> np.ndarray:
    mu = np.zeros((len(a), k))
    base = model if model is not None else _default_outcome_model(binary)
    for f in range(n_folds):
        train, hold = folds != f, folds == f
        n_hold = int(hold.sum())
        r_train = r[train]
        if binary and np.unique(r_train).size < 2:
            mu[hold] = float(r_train.mean())
            continue
        fitted = clone(base).fit(outcome_design(X[train], a[train], k), r_train)
        stacked = np.vstack([
            outcome_design(X[hold], np.full(n_hold, j), k) for j in range(k)
        ])
        if binary and hasattr(fitted, "predict_proba"):
            pred = fitted.predict_proba(stacked)[:, 1]
        else:
            pred = fitted.predict(stacked)
        mu[hold] = np.clip(pred, r_train.min(), r_train.max()).reshape(k, n_hold).T
    return mu


def fit_nuisance(
    data: LogData,
    config: NuisanceConfig,
    *,
    seed: int,
    positions: Optional[np.ndarray] = None,
) -> Nuisance:
    """Обучает nuisance-модели out-of-fold на всей выборке или на bootstrap-реплике.

    Args:
        data: Подготовленные логи.
        config: Настройки моделей.
        seed: Зерно разбиения на фолды.
        positions: Позиции строк реплики; ``None`` — вся выборка.
    """
    # Многопоточный BLAS на небольших матрицах в разы медленнее однопоточного,
    # а параллелизм у bootstrap и так есть на уровне реплик.
    with threadpool_limits(limits=1, user_api="blas"):
        return _fit_nuisance(data, config, seed=seed, positions=positions)


def _fit_nuisance(
    data: LogData, config: NuisanceConfig, *, seed: int, positions: Optional[np.ndarray]
) -> Nuisance:
    start = time.perf_counter()
    idx = np.arange(data.n) if positions is None else np.asarray(positions)
    X = data.X[idx]
    if X.shape[1] == 0:  # оценка без контекста
        X = np.zeros((len(idx), 1))
    a = data.a[idx]
    k = data.k
    groups = data.clusters[idx] if data.is_clustered else idx
    folds = make_folds(groups, config.n_folds, np.random.default_rng(seed))

    unseen = 0
    if data.propensity is not None:
        p_taken = data.propensity[idx]
        source, p_raw, p_max, floor = "logged", None, None, None
    else:
        model = config.propensity_model or _default_propensity_model()
        probs, unseen = _propensity_oof(X, a, k, folds, config.n_folds, model)
        rows = np.arange(len(a))
        p_raw = probs[rows, a]
        p_max = probs.max(axis=1)
        floor = config.propensity_floor * np.bincount(a, minlength=k) / len(a)
        p_taken = np.maximum(p_raw, floor[a])
        source = "estimated"

    mu = {
        metric: _outcome_oof(
            X, a, data.rewards[metric][idx], k, data.binary[metric],
            folds, config.n_folds, config.outcome_model,
        )
        for metric in data.reward_cols
    }
    return Nuisance(
        p_taken=p_taken,
        source=source,
        p_raw=p_raw,
        p_max=p_max,
        floor=floor,
        unseen_share=unseen / len(a),
        mu=mu,
        fit_seconds=time.perf_counter() - start,
    )
