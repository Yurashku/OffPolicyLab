"""
policyscope.estimators
======================

Универсальные оценщики для off-policy сравнения политик.

Ключевые идеи:
- все имена колонок можно передавать аргументами;
- набор признаков задаётся явно через ``feature_cols``;
- поддерживаются произвольные метки действий (в т.ч. строковые),
  если передать ``action_space``;
- тип модели исхода определяется по данным, а не по имени целевой колонки;
- nuisance-модели подключаемые (см. :class:`NuisanceSpec`).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)

__all__ = [
    "NuisanceSpec",
    "ess",
    "infer_feature_columns",
    "resolve_feature_cols",
    "resolve_action_space",
    "validate_propensity",
    "validate_policy_probs",
    "make_design",
    "train_pi_hat",
    "pi_hat_predict",
    "take_action_probabilities",
    "take_action_probabilities_with_fallback",
    "is_binary_target",
    "train_mu_hat",
    "mu_hat_predict",
    "value_on_policy",
    "replay_value",
    "prepare_piB_taken",
    "importance_weights",
    "estimator_contributions",
    "dm_part",
    "ips_value",
    "snips_value",
    "dm_value",
    "dr_value",
    "sndr_value",
    "switch_dr_value",
]

#: Нижняя граница клиппинга вероятностей, чтобы веса не уходили в бесконечность.
PROB_EPS = 1e-6


@dataclass(frozen=True)
class NuisanceSpec:
    """Конфигурация nuisance-моделей (behavior ``pi_A`` и outcome ``mu``).

    Attributes:
        propensity_model: sklearn-совместимый классификатор для ``pi_A(a|x)``.
            ``None`` — мультиклассовая ``LogisticRegression``. Клонируется на каждый fit,
            поэтому один и тот же объект безопасно переиспользовать между фолдами
            и bootstrap-репликами.
        outcome_model: sklearn-совместимый estimator для ``mu(x, a)``. ``None`` —
            ``LogisticRegression`` для бинарной цели и ``LinearRegression`` иначе.
        interactions: включать ли блок взаимодействий ``x ⊗ onehot(a)`` в дизайн-матрицу
            модели исхода. Без него ``mu(x,a) = f(x'b + alpha_a)``, то есть общий наклон и
            только сдвиги по действиям — такая модель структурно не способна выразить
            зависящий от действия эффект, и DM смещается по построению.
    """

    propensity_model: Any = None
    outcome_model: Any = None
    interactions: bool = True


DEFAULT_NUISANCE_SPEC = NuisanceSpec()


class _ConstantModel:
    """Запасной предсказатель для вырожденной цели (в выборке один класс/значение).

    Нужен, потому что bootstrap-реплика или узкий фолд cross-fitting вполне могут
    не содержать ни одного положительного примера, а ``LogisticRegression`` на одном
    классе обучиться не может.
    """

    def __init__(self, value: float) -> None:
        self.value = float(value)

    def fit(self, X, y):  # noqa: D102 - интерфейсная заглушка
        return self

    def predict(self, X) -> np.ndarray:  # noqa: D102
        return np.full(len(X), self.value, dtype=float)

    def predict_proba(self, X) -> np.ndarray:  # noqa: D102
        p = np.full(len(X), self.value, dtype=float)
        return np.column_stack([1.0 - p, p])


# ---------------------------------------------------------------------------
# Признаки
# ---------------------------------------------------------------------------


def infer_feature_columns(
    df: pd.DataFrame,
    *,
    exclude: Optional[Sequence[str]] = None,
) -> list[str]:
    """Автоматически подбирает числовые/булевы признаки из DataFrame.

    Args:
        df: Логи.
        exclude: Колонки, которые нельзя использовать как признаки (награда, действие,
            пропенсити, идентификатор кластера).

    Returns:
        Список имён колонок-признаков.
    """
    excluded = set(exclude or ())
    cols: list[str] = []
    for c in df.columns:
        if c in excluded:
            continue
        if pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]):
            cols.append(c)
    if not cols:
        raise ValueError(
            "Не удалось автоматически определить числовые признаки. "
            "Передайте feature_cols явно."
        )
    return cols


def resolve_feature_cols(
    df: pd.DataFrame,
    feature_cols: Optional[Sequence[str]],
    *,
    target: Optional[str] = None,
    action_col: Optional[str] = None,
    cluster_col: Optional[str] = None,
    propensity_col: Optional[str] = None,
    extra_exclude: Optional[Sequence[str]] = None,
) -> list[str]:
    """Возвращает список признаков, гарантированно не содержащий служебных колонок.

    Автовыбор исключает фактические ``target``/``action_col``/``cluster_col``/
    ``propensity_col``, а не заранее зашитый список имён. Это важно: попадание колонки
    награды в признаки nuisance-моделей делает DM/DR произвольно смещёнными.
    """
    if feature_cols is not None:
        cols = list(feature_cols)
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"В данных нет колонок признаков: {missing}")
        leaked = [c for c in (target, action_col) if c is not None and c in cols]
        if leaked:
            raise ValueError(
                f"Колонки {leaked} одновременно указаны как признаки и как "
                "target/действие. Это утечка: уберите их из feature_cols."
            )
        return cols

    exclude = {c for c in (target, action_col, cluster_col, propensity_col) if c is not None}
    exclude.update(extra_exclude or ())
    cols = infer_feature_columns(df, exclude=exclude)
    logger.debug("Признаки выбраны автоматически: %s", cols)
    return cols


def _validate_logs(df: pd.DataFrame, target: str, action_col: str) -> None:
    missing = {action_col, target} - set(df.columns)
    if missing:
        raise ValueError(f"В данных нет обязательных колонок: {', '.join(sorted(missing))}")


# ---------------------------------------------------------------------------
# Валидаторы
# ---------------------------------------------------------------------------


def validate_propensity(pA: np.ndarray, n: int, *, name: str = "pA") -> np.ndarray:
    """Проверяет вектор пропенсити: длина, отсутствие NaN, диапазон ``(0, 1]``.

    Отдельная проверка на NaN обязательна: сравнения ``nan <= 0`` и ``nan > 1``
    оба дают False, поэтому наивная проверка диапазона пропускает NaN дальше в веса.
    """
    p = np.asarray(pA, dtype=float)
    if p.shape[0] != n:
        raise ValueError(f"{name} должен иметь длину {n}, получено {p.shape[0]}")
    if np.isnan(p).any():
        raise ValueError(f"{name} содержит NaN")
    if np.any((p <= 0.0) | (p > 1.0)):
        raise ValueError(f"{name} должен лежать в интервале (0, 1]")
    return p


def validate_policy_probs(probs: np.ndarray, n: int, *, tol: float = 1e-6) -> np.ndarray:
    """Проверяет матрицу ``pi(a|x)``: форма, отсутствие NaN, неотрицательность, суммы строк.

    Проверка сумм важна для DM: если строки не суммируются в 1, оценка
    ``sum_a pi_B(a|x) mu(x,a)`` молча масштабируется.
    """
    p = np.asarray(probs, dtype=float)
    if p.ndim != 2 or p.shape[0] != n:
        raise ValueError(f"action_probs должен иметь форму ({n}, k), получено {p.shape}")
    if np.isnan(p).any():
        raise ValueError("action_probs содержит NaN")
    if (p < 0.0).any():
        raise ValueError("action_probs содержит отрицательные вероятности")
    row_sums = p.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=tol):
        worst = float(np.max(np.abs(row_sums - 1.0)))
        raise ValueError(
            f"Строки action_probs должны суммироваться в 1 (максимальное отклонение {worst:.3g}). "
            "Проверьте политику B."
        )
    return p


def resolve_action_space(probs: np.ndarray, action_space: Optional[Sequence]) -> list:
    """Возвращает список действий, соответствующий столбцам ``probs``."""
    if action_space is None:
        return list(range(probs.shape[1]))
    actions = list(action_space)
    if len(actions) != probs.shape[1]:
        raise ValueError(
            f"Длина action_space ({len(actions)}) не совпадает с числом столбцов "
            f"политики ({probs.shape[1]})"
        )
    if len(set(actions)) != len(actions):
        raise ValueError("action_space содержит повторяющиеся действия")
    return actions


def ess(weights: np.ndarray) -> float:
    """Считает effective sample size (ESS) для массива весов.

    ESS показывает, какое «эквивалентное» число наблюдений остаётся после
    применения весов (чем сильнее разброс весов, тем меньше ESS).

    Args:
        weights: Веса важности ``w_i`` длины ``n``.

    Returns:
        ``(sum(w)^2 / sum(w^2))``. Если веса вырождены или не конечны, возвращается ``0.0``.
    """
    w = np.asarray(weights, dtype=float)
    if w.size == 0 or not np.isfinite(w).all():
        return 0.0
    s1 = w.sum()
    s2 = (w**2).sum()
    if s2 <= 0.0 or not np.isfinite(s2):
        return 0.0
    return float((s1 * s1) / s2)


# ---------------------------------------------------------------------------
# Дизайн-матрица
# ---------------------------------------------------------------------------


def _build_scaler(df: pd.DataFrame, feature_cols: Sequence[str]) -> tuple[np.ndarray, StandardScaler]:
    X = df[list(feature_cols)].to_numpy(dtype=float)
    if X.shape[1] == 0:
        raise ValueError("feature_cols должен содержать хотя бы один признак")
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    return Xs, scaler


def _transform_features(
    df: pd.DataFrame, feature_cols: Sequence[str], scaler: Optional[StandardScaler]
) -> np.ndarray:
    X = df[list(feature_cols)].to_numpy(dtype=float)
    return scaler.transform(X) if scaler is not None else X


def _fit_one_hot(actions: np.ndarray, action_space: Optional[Sequence]) -> OneHotEncoder:
    """Обучает one-hot энкодер на *полном* пространстве действий.

    Если обучать только на наблюдённых действиях с ``handle_unknown="ignore"``, то
    действие, отсутствовавшее в train-выборке, даёт нулевой one-hot, и ``mu(x, a)``
    молча превращается в предсказание для несуществующего «нулевого» действия.
    Поэтому категории задаются явно, а неизвестное действие — ошибка.
    """
    values = np.asarray(actions).reshape(-1, 1)
    if action_space is None:
        categories: Any = "auto"
    else:
        categories = [np.asarray(list(action_space), dtype=object)]
        values = values.astype(object)
    try:
        oh = OneHotEncoder(categories=categories, sparse_output=False, handle_unknown="error")
    except TypeError:  # sklearn < 1.2
        oh = OneHotEncoder(categories=categories, sparse=False, handle_unknown="error")
    oh.fit(values)
    return oh


def _encode_actions(oh: OneHotEncoder, actions: np.ndarray) -> np.ndarray:
    values = np.asarray(actions).reshape(-1, 1)
    if oh.categories_[0].dtype == object:
        values = values.astype(object)
    try:
        return oh.transform(values)
    except ValueError as exc:
        raise ValueError(
            "Модель исхода получила действие, отсутствующее в action_space, на котором "
            f"она обучалась: {exc}"
        ) from exc


def _interactions(X_num: np.ndarray, A_oh: np.ndarray) -> np.ndarray:
    """Строит блок взаимодействий ``x ⊗ onehot(a)`` формы ``(n, p*k)``."""
    return (X_num[:, :, None] * A_oh[:, None, :]).reshape(len(X_num), -1)


def make_design(
    df: pd.DataFrame,
    *,
    feature_cols: Sequence[str],
    action_col: str = "a_A",
    action_space: Optional[Sequence] = None,
    interactions: bool = True,
) -> Tuple[np.ndarray, OneHotEncoder, StandardScaler, list[str]]:
    """Собирает дизайн-матрицу для обучения модели исхода ``mu(x, a)``.

    Формирует матрицу ``[scaled(x), onehot(a)]`` и, при ``interactions=True``,
    дополнительный блок ``x ⊗ onehot(a)``. Без взаимодействий модель имеет общий
    наклон по признакам и отличается между действиями только сдвигом.

    Returns:
        Кортеж ``(X, oh, scaler, feats)``.
    """
    feats = list(feature_cols)
    X_num, scaler = _build_scaler(df, feats)
    a = df[action_col].to_numpy()
    oh = _fit_one_hot(a, action_space)
    A_oh = _encode_actions(oh, a)
    blocks = [X_num, A_oh]
    if interactions:
        blocks.append(_interactions(X_num, A_oh))
    return np.hstack(blocks), oh, scaler, feats


# ---------------------------------------------------------------------------
# Behavior model: pi_A(a|x)
# ---------------------------------------------------------------------------


def train_pi_hat(
    df: pd.DataFrame,
    *,
    feature_cols: Sequence[str],
    action_col: str = "a_A",
    spec: NuisanceSpec = DEFAULT_NUISANCE_SPEC,
):
    """Обучает оценку behavior policy ``pi_A(a|x)``.

    Args:
        df: Логи с признаками и действием ``action_col``.
        feature_cols: Колонки признаков (не должны содержать награду или действие).
        action_col: Колонка с действием, выбранным политикой A.
        spec: Конфигурация nuisance-моделей.

    Returns:
        Обученная модель классификации со служебными атрибутами ``_scaler``,
        ``_feature_cols``, ``_action_col``.
    """
    feats = list(feature_cols)
    X_num, scaler = _build_scaler(df, feats)
    y = df[action_col].to_numpy()
    if len(np.unique(y)) < 2:
        raise ValueError(
            f"В колонке действий '{action_col}' только одно уникальное значение — "
            "behavior-модель необучаема. Передайте логированные пропенсити "
            "(propensity_source='logged')."
        )
    base = (
        spec.propensity_model
        if spec.propensity_model is not None
        else LogisticRegression(max_iter=1000)
    )
    model = clone(base)
    model.fit(X_num, y)
    model._scaler = scaler  # type: ignore[attr-defined]
    model._feature_cols = feats  # type: ignore[attr-defined]
    model._action_col = action_col  # type: ignore[attr-defined]
    return model


def pi_hat_predict(model, df: pd.DataFrame) -> np.ndarray:
    """Предсказывает ``pi_A(a|x)`` для каждой строки.

    Returns:
        Матрица ``(n, k)``, столбцы соответствуют ``model.classes_``. Значения
        клипуются снизу для устойчивости и перенормируются, чтобы строки
        по-прежнему суммировались в 1.
    """
    X = _transform_features(df, model._feature_cols, model._scaler)  # type: ignore[attr-defined]
    probs = np.clip(model.predict_proba(X), PROB_EPS, 1.0)
    return probs / probs.sum(axis=1, keepdims=True)


def take_action_probabilities(
    probs: np.ndarray,
    actions: Sequence,
    *,
    action_space: Sequence,
) -> np.ndarray:
    """Извлекает вероятность конкретно выбранного действия для каждой строки.

    Args:
        probs: Матрица вероятностей ``(n, k)``.
        actions: Последовательность действий длины ``n``.
        action_space: Порядок действий, соответствующий столбцам ``probs``.

    Raises:
        ValueError: если встречено действие вне ``action_space``.
    """
    idx_map = {a: i for i, a in enumerate(action_space)}
    actions_arr = np.asarray(actions)
    try:
        idx = np.array([idx_map[a] for a in actions_arr], dtype=int)
    except KeyError as exc:
        unknown = sorted({a for a in actions_arr if a not in idx_map}, key=repr)
        raise ValueError(
            f"Действия {unknown[:5]} отсутствуют в action_space {list(action_space)[:10]}"
        ) from exc
    return probs[np.arange(len(idx)), idx]


def take_action_probabilities_with_fallback(
    probs: np.ndarray,
    actions: Sequence,
    *,
    action_space: Sequence,
    fallback: float = PROB_EPS,
) -> tuple[np.ndarray, float]:
    """Как :func:`take_action_probabilities`, но неизвестное действие получает ``fallback``.

    Returns:
        ``(вероятности, доля строк с fallback)``. Доля обязательно должна попадать
        в диагностику: подстановка ``1e-6`` порождает вес порядка ``1e6``, который
        в одиночку доминирует в среднем.
    """
    idx_map = {a: i for i, a in enumerate(action_space)}
    actions_arr = np.asarray(actions)
    out = np.full(len(actions_arr), float(fallback), dtype=float)
    n_missing = 0
    for i, action in enumerate(actions_arr):
        idx = idx_map.get(action)
        if idx is None:
            n_missing += 1
        else:
            out[i] = probs[i, idx]
    share = float(n_missing) / max(len(actions_arr), 1)
    if n_missing:
        logger.warning(
            "%d из %d строк (%.2f%%) содержат действие, неизвестное behavior-модели; "
            "им присвоено pi_A=%.1e, что даёт экстремальные веса важности.",
            n_missing,
            len(actions_arr),
            100.0 * share,
            fallback,
        )
    return out, share


# ---------------------------------------------------------------------------
# Outcome model: mu(x, a)
# ---------------------------------------------------------------------------


def is_binary_target(y: np.ndarray) -> bool:
    """Определяет, является ли цель бинарной ``{0, 1}``, по данным, а не по имени колонки."""
    arr = np.asarray(y)
    if arr.dtype == bool:
        return True
    if not np.issubdtype(arr.dtype, np.number):
        return False
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return False
    uniq = np.unique(finite)
    return uniq.size <= 2 and bool(np.isin(uniq, (0.0, 1.0)).all())


def train_mu_hat(
    df: pd.DataFrame,
    target: str = "accept",
    *,
    feature_cols: Sequence[str],
    action_col: str = "a_A",
    action_space: Optional[Sequence] = None,
    spec: NuisanceSpec = DEFAULT_NUISANCE_SPEC,
):
    """Обучает модель исхода ``mu(x, a) = E[r|x, a]``.

    Тип модели выбирается **по данным**: бинарная цель ``{0,1}`` — классификатор,
    иначе регрессор. Раньше выбор шёл по строковому литералу ``target == "accept"``,
    из-за чего бинарная награда под любым другим именем молча получала OLS.

    Args:
        df: Логи с признаками, действием и целевой колонкой.
        target: Название целевой метрики.
        feature_cols: Колонки признаков.
        action_col: Колонка с действием в логах.
        action_space: Полное пространство действий (для one-hot).
        spec: Конфигурация nuisance-моделей.

    Returns:
        Обученная модель со служебными атрибутами ``_oh``, ``_scaler``,
        ``_feature_cols``, ``_action_col``, ``_is_binary``, ``_interactions``.
    """
    X, oh, scaler, feats = make_design(
        df,
        feature_cols=feature_cols,
        action_col=action_col,
        action_space=action_space,
        interactions=spec.interactions,
    )
    y = df[target].to_numpy()
    binary = is_binary_target(y)

    if binary and np.unique(y[np.isfinite(y)]).size < 2:
        # Вырожденная выборка (например, узкая bootstrap-реплика): один класс.
        model: Any = _ConstantModel(float(np.mean(y)))
    elif spec.outcome_model is not None:
        model = clone(spec.outcome_model)
        model.fit(X, y)
    else:
        model = LogisticRegression(max_iter=1000) if binary else LinearRegression()
        model.fit(X, y)

    model._oh = oh  # type: ignore[attr-defined]
    model._scaler = scaler  # type: ignore[attr-defined]
    model._feature_cols = feats  # type: ignore[attr-defined]
    model._action_col = action_col  # type: ignore[attr-defined]
    model._is_binary = binary  # type: ignore[attr-defined]
    model._interactions = spec.interactions  # type: ignore[attr-defined]
    model._reward_range = (float(np.min(y)), float(np.max(y)))  # type: ignore[attr-defined]
    return model


def mu_hat_predict(model, df: pd.DataFrame, action, target: Optional[str] = None) -> np.ndarray:
    """Предсказывает ``mu(x, a)`` для заданного действия.

    Args:
        model: Модель из :func:`train_mu_hat`.
        df: DataFrame с признаками.
        action: Скаляр или массив действий длины ``n``.
        target: Не используется; оставлен для совместимости вызовов.

    Returns:
        Вектор предсказаний длины ``n``. Для бинарной цели — вероятности класса 1,
        клипованные в ``[PROB_EPS, 1-PROB_EPS]``; для непрерывной — предсказание,
        ограниченное наблюдённым диапазоном награды (регрессия легко выходит за
        пределы носителя, а DR тогда получает поправку вне допустимых значений).
    """
    del target  # тип цели хранится на модели
    X_num = _transform_features(df, model._feature_cols, model._scaler)  # type: ignore[attr-defined]
    if np.isscalar(action) or (isinstance(action, np.ndarray) and action.ndim == 0):
        act = np.full(len(df), action, dtype=object)
    else:
        act = np.asarray(action)
    A_oh = _encode_actions(model._oh, act)  # type: ignore[attr-defined]
    blocks = [X_num, A_oh]
    if getattr(model, "_interactions", False):
        blocks.append(_interactions(X_num, A_oh))
    X = np.hstack(blocks)
    if getattr(model, "_is_binary", False):
        return np.clip(model.predict_proba(X)[:, 1], PROB_EPS, 1.0 - PROB_EPS)
    lo, hi = getattr(model, "_reward_range", (-np.inf, np.inf))
    return np.clip(model.predict(X), lo, hi)


# ---------------------------------------------------------------------------
# Общие строительные блоки оценщиков
# ---------------------------------------------------------------------------


def importance_weights(
    piB_taken: np.ndarray,
    pA: np.ndarray,
    *,
    weight_clip: Optional[float] = None,
) -> tuple[np.ndarray, float]:
    """Считает веса важности ``w = pi_B(a|x)/pi_A(a|x)`` с опциональным клиппингом сверху.

    Returns:
        ``(w, clip_share)`` — веса и доля наблюдений, где вес был обрезан.
    """
    w = np.asarray(piB_taken, dtype=float) / np.asarray(pA, dtype=float)
    if weight_clip is None:
        return w, 0.0
    if weight_clip <= 0:
        raise ValueError("weight_clip должен быть положительным")
    clip_share = float(np.mean(w > weight_clip))
    return np.minimum(w, weight_clip), clip_share


def estimator_contributions(
    method: str,
    *,
    r: np.ndarray,
    w: Optional[np.ndarray] = None,
    dm: Optional[np.ndarray] = None,
    mu_taken: Optional[np.ndarray] = None,
    tau: Optional[float] = None,
) -> np.ndarray:
    """Построчные вклады ``psi_i`` выбранного оценщика: ``mean(psi) == оценка``.

    Единственный источник истины для формул. Отсюда берут значение и точечные
    оценщики, и аналитический (influence-function) инференс, поэтому они не могут
    разойтись между собой.

    Для самонормированных оценщиков (SNIPS, SNDR) возвращается **линеаризация**
    отношения: обычная дельта-метод-форма ``V + w(r - V)/mean(w)``. Её среднее равно
    самой оценке, а разброс вокруг среднего даёт корректную асимптотическую дисперсию.

    Args:
        method: Имя оценщика.
        r: Наблюдённая награда.
        w: Веса важности (для взвешенных оценщиков).
        dm: Построчный DM-вклад ``sum_a pi_B(a|x) mu(x,a)``.
        mu_taken: ``mu(x, a_logged)``.
        tau: Порог переключения для ``switch_dr``.
    """
    r = np.asarray(r, dtype=float)

    if method == "on_policy":
        return r
    if method == "dm":
        assert dm is not None
        return dm

    assert w is not None, f"Оценщику '{method}' нужны веса важности"
    w_mean = float(np.mean(w))

    if method == "ips":
        return w * r
    if method == "snips":
        if w_mean <= 0.0:
            return np.full_like(r, np.nan)
        value = float(np.mean(w * r) / w_mean)
        return value + w * (r - value) / w_mean

    assert dm is not None and mu_taken is not None, f"Оценщику '{method}' нужна модель исхода"
    resid = r - mu_taken

    if method == "dr":
        return dm + w * resid
    if method == "sndr":
        if w_mean <= 0.0:
            return dm
        correction = float(np.mean(w * resid) / w_mean)
        return dm + correction + (w * resid - correction * w) / w_mean
    if method == "switch_dr":
        if tau is None:
            raise ValueError("switch_dr требует значение tau")
        return dm + np.where(w <= tau, w, 0.0) * resid

    raise ValueError(f"Неизвестный оценщик: {method}")


def dm_part(
    df: pd.DataFrame,
    policyB,
    mu_model,
    *,
    action_space: Optional[Sequence] = None,
) -> np.ndarray:
    """Считает построчный DM-вклад ``sum_a pi_B(a|x_i) mu(x_i, a)``.

    Общий блок для DM/DR/SNDR/Switch-DR — раньше он был скопирован в каждый из
    четырёх оценщиков.
    """
    probsB = validate_policy_probs(policyB.action_probs(df), len(df))
    actions = resolve_action_space(probsB, action_space)
    out = np.zeros(len(df), dtype=float)
    for idx, action in enumerate(actions):
        pa = probsB[:, idx]
        if not np.any(pa > 0.0):
            continue
        out += pa * mu_hat_predict(mu_model, df, np.full(len(df), action, dtype=object))
    return out


def _weighted_inputs(
    df: pd.DataFrame,
    policyB,
    mu_model,
    pA: np.ndarray,
    *,
    target: str,
    action_col: str,
    action_space: Optional[Sequence],
    weight_clip: Optional[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Собирает ``(dm, w, r, mu_taken, clip_share)`` для DR-семейства."""
    _validate_logs(df, target, action_col)
    n = len(df)
    pA = validate_propensity(pA, n)

    probsB = validate_policy_probs(policyB.action_probs(df), n)
    actions = resolve_action_space(probsB, action_space)
    aA = df[action_col].to_numpy()

    dm = np.zeros(n, dtype=float)
    for idx, action in enumerate(actions):
        pa = probsB[:, idx]
        if not np.any(pa > 0.0):
            continue
        dm += pa * mu_hat_predict(mu_model, df, np.full(n, action, dtype=object))

    piB_taken = take_action_probabilities(probsB, aA, action_space=actions)
    w, clip_share = importance_weights(piB_taken, pA, weight_clip=weight_clip)
    mu_taken = mu_hat_predict(mu_model, df, aA)
    return dm, w, df[target].to_numpy(dtype=float), mu_taken, clip_share


# ---------------------------------------------------------------------------
# Оценщики
# ---------------------------------------------------------------------------


def value_on_policy(df: pd.DataFrame, target: str = "accept") -> float:
    """Значение логирующей политики A как средняя наблюдённая награда."""
    return float(df[target].mean())


def replay_value(
    df: pd.DataFrame,
    a_B: np.ndarray,
    target: str = "accept",
    *,
    action_col: str = "a_A",
) -> float:
    """Replay-оценка: средняя награда на строках, где действие A совпало с действием B.

    .. warning::
       Replay несмещён только при **равномерной** логирующей политике. На
       ε-greedy или softmax логах он оценивает ``E[r | a_A = a_B(x)]``, а не
       ``V(pi_B)``, и не сопоставим с остальными оценщиками. Используйте его как
       грубый ориентир перекрытия, а не как оценку значения политики.

    Returns:
        Replay-оценка. Если совпадений нет — ``nan``.
    """
    mask = df[action_col].to_numpy() == np.asarray(a_B)
    n_match = int(mask.sum())
    if n_match == 0:
        logger.warning("[replay] Нет совпадающих действий: политика B вне области данных A")
        return float("nan")
    logger.debug("[replay] Совпадающих действий: %d из %d", n_match, len(df))
    return float(df.loc[mask, target].mean())


def prepare_piB_taken(
    df: pd.DataFrame,
    policyB,
    *,
    action_col: str = "a_A",
    action_space: Optional[Sequence] = None,
) -> np.ndarray:
    """Готовит вектор ``pi_B(a_logged|x)`` для IPS/DR-оценок."""
    probsB = validate_policy_probs(policyB.action_probs(df), len(df))
    actions = resolve_action_space(probsB, action_space)
    return take_action_probabilities(probsB, df[action_col].to_numpy(), action_space=actions)


def ips_value(
    df: pd.DataFrame,
    piB_taken: np.ndarray,
    pA: np.ndarray,
    target: str = "accept",
    weight_clip: Optional[float] = None,
    *,
    action_col: str = "a_A",
) -> Tuple[float, float, float]:
    """Inverse Propensity Scoring: ``mean(w_i * r_i)``, ``w_i = pi_B(a_i|x_i)/pi_A(a_i|x_i)``.

    Returns:
        ``(value, ess, clip_share)``.
    """
    _validate_logs(df, target, action_col)
    n = len(df)
    pA = validate_propensity(pA, n)
    if len(piB_taken) != n:
        raise ValueError(f"piB_taken должен иметь длину {n}, получено {len(piB_taken)}")
    w, clip_share = importance_weights(piB_taken, pA, weight_clip=weight_clip)
    r = df[target].to_numpy(dtype=float)
    value = float(np.mean(estimator_contributions("ips", r=r, w=w)))
    return value, ess(w), clip_share


def snips_value(
    df: pd.DataFrame,
    piB_taken: np.ndarray,
    pA: np.ndarray,
    target: str = "accept",
    weight_clip: Optional[float] = None,
    *,
    action_col: str = "a_A",
) -> Tuple[float, float, float]:
    """Self-Normalized IPS: ``sum(w r) / sum(w)``.

    Обычно снижает дисперсию ценой небольшого смещения. При полном отсутствии
    перекрытия (``sum(w) == 0``) возвращает ``nan``, а не ноль: ноль неотличим от
    правдоподобного значения награды.
    """
    _validate_logs(df, target, action_col)
    n = len(df)
    pA = validate_propensity(pA, n)
    if len(piB_taken) != n:
        raise ValueError(f"piB_taken должен иметь длину {n}, получено {len(piB_taken)}")
    w, clip_share = importance_weights(piB_taken, pA, weight_clip=weight_clip)
    if float(np.sum(w)) <= 0.0:
        logger.warning("[snips] Сумма весов равна нулю: перекрытие политик отсутствует")
        return float("nan"), 0.0, clip_share
    r = df[target].to_numpy(dtype=float)
    value = float(np.mean(estimator_contributions("snips", r=r, w=w)))
    return value, ess(w), clip_share


def dm_value(
    df: pd.DataFrame,
    policyB,
    mu_model,
    target: str = "accept",
    *,
    action_space: Optional[Sequence] = None,
) -> float:
    """Direct Method: ``mean_i sum_a pi_B(a|x_i) mu(x_i, a)``."""
    return float(np.mean(dm_part(df, policyB, mu_model, action_space=action_space)))


def dr_value(
    df: pd.DataFrame,
    policyB,
    mu_model,
    pA: np.ndarray,
    target: str = "accept",
    weight_clip: Optional[float] = None,
    *,
    action_col: str = "a_A",
    action_space: Optional[Sequence] = None,
) -> Tuple[float, float, float]:
    """Doubly Robust: ``mean(sum_a pi_B(a|x) mu(x,a) + w (r - mu(x, a_logged)))``.

    Консистентна, если корректна хотя бы одна из моделей — ``pi_A`` или ``mu``.
    """
    dm, w, r, mu_taken, clip_share = _weighted_inputs(
        df, policyB, mu_model, pA,
        target=target, action_col=action_col, action_space=action_space, weight_clip=weight_clip,
    )
    value = float(np.mean(estimator_contributions("dr", r=r, w=w, dm=dm, mu_taken=mu_taken)))
    return value, ess(w), clip_share


def sndr_value(
    df: pd.DataFrame,
    policyB,
    mu_model,
    pA: np.ndarray,
    target: str = "accept",
    weight_clip: Optional[float] = None,
    *,
    action_col: str = "a_A",
    action_space: Optional[Sequence] = None,
) -> Tuple[float, float, float]:
    """Self-Normalized DR: самонормируется только IPS-поправка, DM-база — нет."""
    dm, w, r, mu_taken, clip_share = _weighted_inputs(
        df, policyB, mu_model, pA,
        target=target, action_col=action_col, action_space=action_space, weight_clip=weight_clip,
    )
    if float(np.mean(w)) <= 0.0:
        logger.warning("[sndr] Средний вес равен нулю: перекрытие политик отсутствует")
        return float(np.mean(dm)), 0.0, clip_share
    value = float(np.mean(estimator_contributions("sndr", r=r, w=w, dm=dm, mu_taken=mu_taken)))
    return value, ess(w), clip_share


def switch_dr_value(
    df: pd.DataFrame,
    policyB,
    mu_model,
    pA: np.ndarray,
    tau: float,
    target: str = "accept",
    weight_clip: Optional[float] = None,
    *,
    action_col: str = "a_A",
    action_space: Optional[Sequence] = None,
) -> Tuple[float, float, float]:
    """Switch-DR: IPS-поправка отключается там, где вес превышает ``tau``.

    ``mean(sum_a pi_B mu + 1{w <= tau} * w * (r - mu_logged))`` — DM-база остаётся
    всегда, зануляется только поправка. Снижает дисперсию ценой смещения.

    Args:
        weight_clip: применяется до сравнения с ``tau``. Раньше параметр молча
            игнорировался именно для этого оценщика.

    Returns:
        ``(value, ess по эффективным весам, switched_share)``.
    """
    if weight_clip is not None and weight_clip <= tau:
        raise ValueError(
            f"weight_clip={weight_clip} не больше tau={tau}: клиппинг сделал бы "
            "переключение Switch-DR недостижимым. Задайте weight_clip > tau или None."
        )
    dm, w, r, mu_taken, clip_share = _weighted_inputs(
        df, policyB, mu_model, pA,
        target=target, action_col=action_col, action_space=action_space, weight_clip=weight_clip,
    )
    mask = w <= tau
    w_sw = np.where(mask, w, 0.0)
    value = float(
        np.mean(estimator_contributions("switch_dr", r=r, w=w, dm=dm, mu_taken=mu_taken, tau=tau))
    )
    return value, ess(w_sw), float(np.mean(~mask))
