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
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import GammaRegressor, LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from policyscope.data import LogData

__all__ = [
    "NuisanceConfig",
    "Nuisance",
    "TwoPartRegressor",
    "make_folds",
    "fit_nuisance",
    "outcome_design",
    "outcome_kind",
]


@dataclass(frozen=True)
class NuisanceConfig:
    """Настройки nuisance-моделей.

    Attributes:
        propensity_model: sklearn-классификатор с ``predict_proba`` для ``pi_A(a|x)``.
            ``None`` — мультиномиальная логистическая регрессия.
        outcome_model: sklearn-модель для ``mu(x, a)`` или словарь «метрика -> модель»;
            получает признаки, one-hot действия и их попарные произведения. ``None`` (или
            метрика, отсутствующая в словаре) — модель по умолчанию, см. :func:`outcome_kind`.
        n_folds: Число фолдов cross-fitting.
        propensity_floor: Доля от частоты действия, ниже которой оценённое пропенсити
            не опускается. Применяется только к оценённым пропенсити.
    """

    propensity_model: Any = None
    outcome_model: Any = None
    n_folds: int = 5
    propensity_floor: float = 0.1

    def model_for(self, metric: str) -> Any:
        """Пользовательская модель исхода для метрики или ``None`` (модель по умолчанию)."""
        if isinstance(self.outcome_model, Mapping):
            return self.outcome_model.get(metric)
        return self.outcome_model

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


class TwoPartRegressor(RegressorMixin, BaseEstimator):
    """Модель неотрицательной награды с нулями: ``E[r|z] = P(r > 0|z) · E[r | r > 0, z]``.

    Первая часть — логистическая регрессия, вторая — Gamma-GLM с log-связью на строках с
    ``r > 0``. Так устроены выручка, сумма покупки, LTV: чаще всего ноль, иначе
    положительная величина с длинным хвостом. Линейная модель на таких данных выдаёт
    отрицательные прогнозы, а их обрезка смещает оценку DM на величину порядка его
    стандартной ошибки — это показали симуляции (см. docs/estimators.md).

    Прогноз положительной части не превышает наблюдённый максимум ``r``. Log-связь
    экстраполирует экспоненциально, и в областях без данных (слабое перекрытие) прогноз
    без ограничения уходит в разы выше любого наблюдения: в симуляциях это смещало и DM,
    и DR. Ограничение действует только там, где модель экстраполирует.

    Args:
        C: Обратная сила L2-штрафа логистической части (как в ``LogisticRegression``).
            Штраф Gamma-части согласован с ним: ``alpha = 1 / (C · n_pos)``.
    """

    def __init__(self, C: float = 1.0) -> None:
        self.C = C

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if (y < 0).any():
            raise ValueError("TwoPartRegressor ожидает неотрицательную награду")
        self.scaler_ = StandardScaler().fit(X)
        Z = self.scaler_.transform(X)
        positive = y > 0
        n_pos = int(positive.sum())
        self.p_const_ = float(positive.mean())
        self.classifier_ = None
        if 0 < n_pos < len(y):
            self.classifier_ = LogisticRegression(C=self.C, max_iter=1000).fit(Z, positive)
        self.m_const_ = float(y[positive].mean()) if n_pos else 0.0
        self.cap_ = float(y.max())
        self.regressor_ = None
        if n_pos >= 10:
            self.regressor_ = GammaRegressor(
                alpha=1.0 / (self.C * n_pos), solver="newton-cholesky", max_iter=300,
            ).fit(Z[positive], y[positive])
        return self

    def predict(self, X):
        Z = self.scaler_.transform(np.asarray(X, dtype=float))
        p = (self.classifier_.predict_proba(Z)[:, 1] if self.classifier_ is not None
             else np.full(len(Z), self.p_const_))
        m = self.regressor_.predict(Z) if self.regressor_ is not None else np.full(len(Z), self.m_const_)
        return p * np.minimum(m, self.cap_)


def outcome_kind(r: np.ndarray, binary: bool) -> str:
    """Тип модели исхода по умолчанию для вектора наград.

    - ``binary`` — логистическая регрессия;
    - ``two_part`` — неотрицательная награда, где заметная доля нулей (выручка):
      :class:`TwoPartRegressor`;
    - ``linear`` — иначе: Ridge.
    """
    if binary:
        return "binary"
    if (r >= 0).all() and np.mean(r == 0) >= 0.05 and np.sum(r > 0) >= 20:
        return "two_part"
    return "linear"


OUTCOME_KIND_LABELS = {
    "binary": "LogisticRegression",
    "two_part": "двухчастная: LogisticRegression × GammaRegressor (log)",
    "linear": "Ridge",
}


def _default_outcome_model(kind: str):
    if kind == "binary":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    if kind == "two_part":
        return TwoPartRegressor()
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
    folds: np.ndarray, n_folds: int, model: Any, kind: str,
) -> np.ndarray:
    mu = np.zeros((len(a), k))
    base = model if model is not None else _default_outcome_model(kind)
    for f in range(n_folds):
        train, hold = folds != f, folds == f
        n_hold = int(hold.sum())
        r_train = r[train]
        if np.unique(r_train).size < 2:
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
            if binary:  # регрессор на бинарной награде: прогноз — вероятность
                pred = np.clip(pred, 0.0, 1.0)
        # Непрерывные прогнозы не обрезаются: обрезка по [min r, max r] смещает DM.
        mu[hold] = np.asarray(pred, dtype=float).reshape(k, n_hold).T
    if not np.isfinite(mu).all():
        raise ValueError(
            "Модель исхода выдала нечисловые прогнозы (inf/NaN). Проверьте признаки на выбросы "
            "или задайте более устойчивую outcome_model."
        )
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
    if isinstance(config.outcome_model, Mapping):
        extra = sorted(set(config.outcome_model) - set(data.reward_cols))
        if extra:
            raise ValueError(f"outcome_model задан для метрик, которых нет в reward: {extra}")
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
            folds, config.n_folds, config.model_for(metric),
            outcome_kind(data.rewards[metric], data.binary[metric]),
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
