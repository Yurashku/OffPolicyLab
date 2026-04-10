"""
policyscope.estimators
======================

Универсальные оценщики для off-policy сравнения политик.

Ключевые идеи:
- все имена колонок можно передавать аргументами;
- набор признаков задаётся явно через ``feature_cols``;
- поддерживаются произвольные метки действий (в т.ч. строковые),
  если передать ``action_space``.
"""

from __future__ import annotations

import logging
from typing import Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from policyscope.policies import BasePolicy

__all__ = [
    "ess",
    "infer_feature_columns",
    "make_design",
    "train_pi_hat",
    "pi_hat_predict",
    "take_action_probabilities",
    "train_mu_hat",
    "mu_hat_predict",
    "value_on_policy",
    "replay_value",
    "prepare_piB_taken",
    "ips_value",
    "snips_value",
    "dm_value",
    "dr_value",
    "sndr_value",
    "switch_dr_value",
    "estimator_with_bootstrap_ci",
    "ate_from_values",
]


def infer_feature_columns(
    df: pd.DataFrame,
    *,
    exclude: Optional[Sequence[str]] = None,
) -> list[str]:
    """Автоматически подбирает числовые/булевы признаки из DataFrame."""
    excluded = set(exclude or ())
    cols: list[str] = []
    for c in df.columns:
        if c in excluded:
            continue
        if pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]):
            cols.append(c)
    if not cols:
        raise ValueError("Could not infer numeric feature columns. Pass feature_cols explicitly.")
    return cols


def _default_feature_cols(df: pd.DataFrame) -> list[str]:
    preferred = ["loyal", "age", "risk", "income"]
    cols = [c for c in preferred if c in df.columns]
    if cols:
        return cols
    return infer_feature_columns(
        df,
        exclude=["user_id", "a_A", "a_B", "propensity_A", "accept", "cltv"],
    )


def _resolve_feature_cols(df: pd.DataFrame, feature_cols: Optional[Sequence[str]]) -> list[str]:
    cols = list(feature_cols) if feature_cols is not None else _default_feature_cols(df)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    return cols


def _validate_logs(df: pd.DataFrame, target: str, action_col: str) -> None:
    required = {action_col, target}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")


def ess(weights: np.ndarray) -> float:
    """Считает effective sample size (ESS) для массива весов.

    ESS показывает, какое «эквивалентное» число наблюдений остаётся после
    применения весов (чем сильнее разброс весов, тем меньше ESS).

    Args:
        weights: Веса важности ``w_i`` длины ``n``.

    Returns:
        Значение ``(sum(w)^2 / sum(w^2))``. Если все веса нулевые, возвращается ``0.0``.
    """
    s1 = weights.sum()
    s2 = (weights**2).sum()
    if s2 <= 0.0:
        return 0.0
    return float((s1 * s1) / s2)


def _build_scaler(df: pd.DataFrame, feature_cols: Sequence[str]) -> tuple[np.ndarray, Optional[StandardScaler]]:
    X = df[list(feature_cols)].to_numpy()
    if X.shape[1] == 0:
        raise ValueError("feature_cols must contain at least one feature")
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    return Xs, scaler


def _transform_features(df: pd.DataFrame, feature_cols: Sequence[str], scaler: Optional[StandardScaler]) -> np.ndarray:
    X = df[list(feature_cols)].to_numpy()
    return scaler.transform(X) if scaler is not None else X


def _fit_one_hot(values: np.ndarray) -> OneHotEncoder:
    try:
        oh = OneHotEncoder(categories="auto", sparse_output=False, handle_unknown="ignore")
    except TypeError:  # sklearn < 1.2
        oh = OneHotEncoder(categories="auto", sparse=False, handle_unknown="ignore")
    oh.fit(values)
    return oh


def make_design(
    df: pd.DataFrame,
    *,
    feature_cols: Optional[Sequence[str]] = None,
    action_col: str = "a_A",
) -> Tuple[np.ndarray, np.ndarray, OneHotEncoder, StandardScaler, list[str]]:
    """Собирает дизайн-матрицу для обучения модели исхода ``mu(x, a)``.

    Формирует матрицу признаков вида ``[scaled(feature_cols), onehot(action_col)]``.

    Args:
        df: Логи с признаками и колонкой действия.
        feature_cols: Список колонок признаков. Если ``None``, используется авто-выбор.
        action_col: Колонка с фактическим действием из логов политики A.

    Returns:
        Кортеж ``(X, A_oh, oh, scaler, feats)``:
        - ``X`` — итоговая матрица для обучения,
        - ``A_oh`` — one-hot действий,
        - ``oh`` — обученный ``OneHotEncoder`` для действий,
        - ``scaler`` — обученный ``StandardScaler`` для числовых фич,
        - ``feats`` — фактический список использованных признаков.
    """
    feats = _resolve_feature_cols(df, feature_cols)
    X_num, scaler = _build_scaler(df, feats)
    a = df[action_col].to_numpy().reshape(-1, 1)
    oh = _fit_one_hot(a)
    A_oh = oh.transform(a)
    X = np.hstack([X_num, A_oh])
    return X, A_oh, oh, scaler, feats


def train_pi_hat(
    df: pd.DataFrame,
    *,
    feature_cols: Optional[Sequence[str]] = None,
    action_col: str = "a_A",
):
    """Обучает оценку behavior policy: ``pi_A(a|x)``.

    Используется мультиклассовая ``LogisticRegression`` по признакам ``x``,
    чтобы оценить вероятность каждого действия, с которым работала текущая
    политика A в логах.

    Args:
        df: Логи с признаками и действием ``action_col``.
        feature_cols: Колонки признаков. Если ``None``, выбираются автоматически.
        action_col: Колонка с действием, выбранным политикой A.

    Returns:
        Обученная sklearn-модель классификации с сохранёнными служебными
        атрибутами: ``_scaler``, ``_feature_cols``, ``_action_col``.
    """
    feats = _resolve_feature_cols(df, feature_cols)
    X_num, scaler = _build_scaler(df, feats)
    y = df[action_col].to_numpy()
    model = LogisticRegression(max_iter=1000)
    model.fit(X_num, y)
    model._scaler = scaler  # type: ignore[attr-defined]
    model._feature_cols = feats  # type: ignore[attr-defined]
    model._action_col = action_col  # type: ignore[attr-defined]
    return model


def pi_hat_predict(model, df: pd.DataFrame) -> np.ndarray:
    """Предсказывает ``pi_A(a|x)`` для каждой строки.

    Args:
        model: Модель из ``train_pi_hat``.
        df: DataFrame с теми же признаками, что использовались при обучении.

    Returns:
        Матрица вероятностей формы ``(n, k)``, где ``k`` — число действий.
        Значения клипуются в интервал ``[1e-6, 1 - 1e-6]`` для устойчивости.
    """
    X = _transform_features(df, model._feature_cols, model._scaler)  # type: ignore[attr-defined]
    probs = model.predict_proba(X)
    return np.clip(probs, 1e-6, 1 - 1e-6)


def take_action_probabilities(
    probs: np.ndarray,
    actions: Sequence,
    *,
    action_space: Sequence,
) -> np.ndarray:
    """Извлекает вероятность конкретно выбранного действия для каждой строки.

    Часто нужно перейти от полной матрицы ``p(a|x)`` к вектору ``p(a_logged|x)``
    или ``p(a_B|x)``. Эта функция делает такое извлечение по ``action_space``.

    Args:
        probs: Матрица вероятностей формы ``(n, k)``.
        actions: Последовательность действий длины ``n`` (например, ``df[action_col]``).
        action_space: Порядок действий, соответствующий столбцам ``probs``.

    Returns:
        Вектор длины ``n`` с вероятностью указанного действия в каждой строке.
    """
    idx_map = {a: i for i, a in enumerate(action_space)}
    idx = np.array([idx_map[a] for a in actions], dtype=int)
    return probs[np.arange(len(idx)), idx]


def train_mu_hat(
    df: pd.DataFrame,
    target: Literal["accept", "cltv"] = "accept",
    *,
    feature_cols: Optional[Sequence[str]] = None,
    action_col: str = "a_A",
):
    """Обучает модель исхода ``mu(x, a)=E[r|x,a]``.

    Для бинарной цели ``accept`` используется ``LogisticRegression`` и далее
    возвращаются вероятности класса 1. Для непрерывной цели (например ``cltv``)
    используется ``LinearRegression``.

    Args:
        df: Логи с признаками, действием и целевой колонкой.
        target: Название целевой метрики (``accept`` или ``cltv``).
        feature_cols: Колонки признаков. Если ``None``, выбираются автоматически.
        action_col: Колонка с действием в логах.

    Returns:
        Обученная модель исхода с сохранёнными служебными атрибутами:
        ``_oh``, ``_scaler``, ``_feature_cols``, ``_action_col``.
    """
    X, _, oh, scaler, feats = make_design(df, feature_cols=feature_cols, action_col=action_col)
    y = df[target].to_numpy()
    model = LogisticRegression(max_iter=1000) if target == "accept" else LinearRegression()
    model.fit(X, y)
    model._oh = oh  # type: ignore[attr-defined]
    model._scaler = scaler  # type: ignore[attr-defined]
    model._feature_cols = feats  # type: ignore[attr-defined]
    model._action_col = action_col  # type: ignore[attr-defined]
    return model


def mu_hat_predict(model, df: pd.DataFrame, action: np.ndarray, target: str) -> np.ndarray:
    """Предсказывает ``mu(x, a)`` для заданного действия.

    Args:
        model: Модель из ``train_mu_hat``.
        df: DataFrame с признаками.
        action: Одно действие (скаляр) или массив действий длины ``n``.
        target: Тип цели. Для ``accept`` вернёт вероятности, иначе регрессионный прогноз.

    Returns:
        Вектор предсказаний ``mu(x, a)`` длины ``n``.
    """
    X_num = _transform_features(df, model._feature_cols, model._scaler)  # type: ignore[attr-defined]
    if np.isscalar(action):
        act = np.full(len(df), action, dtype=object).reshape(-1, 1)
    else:
        act = np.asarray(action, dtype=object).reshape(-1, 1)
    A_oh = model._oh.transform(act)
    X = np.hstack([X_num, A_oh])
    if target == "accept":
        return np.clip(model.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)
    return model.predict(X)


def value_on_policy(df: pd.DataFrame, target: str = "accept") -> float:
    """Оценивает значение текущей (логирующей) политики A как среднюю награду.

    Args:
        df: Логи политики A.
        target: Целевая метрика/награда.

    Returns:
        Среднее ``df[target]``.
    """
    return float(df[target].mean())


def replay_value(
    df: pd.DataFrame,
    a_B: np.ndarray,
    target: str = "accept",
    *,
    action_col: str = "a_A",
) -> float:
    """Replay-оценка: средняя награда на пересечении действий A и B.

    Берутся только строки, где ``a_A == a_B`` (или, в общем виде,
    ``df[action_col] == a_B[i]``), и считается среднее по ``target``.

    Args:
        df: Логи политики A.
        a_B: Действия, которые выбирает новая политика B на тех же объектах.
        target: Целевая метрика.
        action_col: Колонка действия из логов A.

    Returns:
        Replay-оценка ``V_B``. Если совпадений нет, возвращается ``nan``.
    """
    logging.info("[Replay] Начинаем оценку Replay для новой политики…")
    mask = df[action_col].to_numpy() == a_B
    if mask.sum() == 0:
        logging.info("[Replay] Нет совпадающих действий – политика B вне области данных A")
        return float("nan")
    val = float(df.loc[mask, target].mean())
    logging.info(f"[Replay] Совпадающих действий: {mask.sum()} из {len(df)}")
    logging.info(f"[Replay] Оценённое значение метрики: {val:.4f}")
    return val


def prepare_piB_taken(
    df: pd.DataFrame,
    policyB,
    *,
    action_col: str = "a_A",
    action_space: Optional[Sequence] = None,
) -> np.ndarray:
    """Готовит вектор ``pi_B(a_logged|x)`` для IPS/DR-оценок.

    Args:
        df: Логи с колонкой ``action_col``.
        policyB: Объект политики B с методом ``action_probs(df) -> (n, k)``.
        action_col: Колонка с фактическим действием из логов A.
        action_space: Порядок действий для столбцов ``policyB.action_probs``.
            Если ``None``, берётся ``[0, 1, ..., k-1]``.

    Returns:
        Вектор длины ``n``: вероятность, что B выбрала бы логированное действие.
    """
    probsB = policyB.action_probs(df)
    actions = df[action_col].to_numpy()
    if action_space is None:
        action_space = list(range(probsB.shape[1]))
    return take_action_probabilities(probsB, actions, action_space=action_space)


def ips_value(
    df: pd.DataFrame,
    piB_taken: np.ndarray,
    pA: np.ndarray,
    target: str = "accept",
    weight_clip: Optional[float] = None,
    *,
    action_col: str = "a_A",
) -> Tuple[float, float, float]:
    """Inverse Propensity Scoring (IPS).

    Оценка строится как ``mean(w_i * r_i)``, где ``w_i = pi_B(a_i|x_i)/pi_A(a_i|x_i)``.

    Args:
        df: Логи политики A.
        piB_taken: Вектор ``pi_B(a_logged|x)``.
        pA: Вектор ``pi_A(a_logged|x)`` (propensity score).
        target: Целевая метрика.
        weight_clip: Опциональный верхний порог для клиппинга весов ``w_i``.
        action_col: Колонка логированного действия (для валидации входа).

    Returns:
        Кортеж ``(value, ess_value, clip_share)``:
        - IPS-оценка,
        - ESS по использованным весам,
        - доля наблюдений, где вес был обрезан.
    """
    logging.info("[IPS] Начинаем оценку IPS для новой политики…")
    _validate_logs(df, target, action_col)
    if len(pA) != len(df):
        raise ValueError("pA must have same length as df")
    if np.any((pA <= 0) | (pA > 1)):
        raise ValueError("propensity scores must be in (0,1]")
    w = piB_taken / pA
    clip_share = 0.0
    if weight_clip is not None:
        clip_share = float((w > weight_clip).mean())
        w = np.minimum(w, weight_clip)
    r = df[target].to_numpy()
    value = float(np.mean(r * w))
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
    """Self-Normalized IPS (SNIPS).

    Вариант IPS с самонормировкой весов: ``sum(w_i r_i) / sum(w_i)``.
    Обычно снижает дисперсию, но может вносить смещение.

    Args:
        df: Логи политики A.
        piB_taken: Вектор ``pi_B(a_logged|x)``.
        pA: Вектор ``pi_A(a_logged|x)``.
        target: Целевая метрика.
        weight_clip: Опциональный клиппинг весов.
        action_col: Колонка логированного действия (для валидации входа).

    Returns:
        Кортеж ``(value, ess_value, clip_share)`` аналогично ``ips_value``.
    """
    logging.info("[SNIPS] Начинаем оценку SNIPS для новой политики…")
    _validate_logs(df, target, action_col)
    if len(pA) != len(df):
        raise ValueError("pA must have same length as df")
    if np.any((pA <= 0) | (pA > 1)):
        raise ValueError("propensity scores must be in (0,1]")
    w = piB_taken / pA
    clip_share = 0.0
    if weight_clip is not None:
        clip_share = float((w > weight_clip).mean())
        w = np.minimum(w, weight_clip)
    r = df[target].to_numpy()
    value = float(np.sum(r * w) / (np.sum(w) + 1e-12))
    return value, ess(w), clip_share


def _action_values_from_probs(probsB: np.ndarray, action_space: Optional[Sequence]) -> list:
    if action_space is None:
        return list(range(probsB.shape[1]))
    if len(action_space) != probsB.shape[1]:
        raise ValueError("action_space length must match policy output dimension")
    return list(action_space)


def dm_value(
    df: pd.DataFrame,
    policyB,
    mu_model,
    target: str = "accept",
    *,
    action_space: Optional[Sequence] = None,
) -> float:
    """Direct Method (DM): модельная оценка ``E_{x,a~pi_B}[mu(x,a)]``.

    Args:
        df: Данные с признаками.
        policyB: Политика B с ``action_probs(df)``.
        mu_model: Модель исхода из ``train_mu_hat``.
        target: Целевая метрика.
        action_space: Список действий, соответствующий столбцам ``action_probs``.

    Returns:
        DM-оценка значения политики B.
    """
    probsB = policyB.action_probs(df)
    actions = _action_values_from_probs(probsB, action_space)
    val = 0.0
    for idx, action in enumerate(actions):
        pa = probsB[:, idx]
        if pa.sum() == 0:
            continue
        mu = mu_hat_predict(mu_model, df, np.full(len(df), action, dtype=object), target)
        val += float(np.mean(pa * mu))
    return val


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
    """Doubly Robust (DR) оценка.

    Совмещает DM и IPS-коррекцию:
    ``mean( sum_a pi_B(a|x) mu(x,a) + w * (r - mu(x,a_logged)) )``.
    Консистентна, если корректна хотя бы одна из моделей: ``pi_A`` или ``mu``.

    Args:
        df: Логи политики A.
        policyB: Политика B.
        mu_model: Модель исхода ``mu``.
        pA: Вектор ``pi_A(a_logged|x)``.
        target: Целевая метрика.
        weight_clip: Опциональный клиппинг весов ``w``.
        action_col: Колонка логированного действия.
        action_space: Пространство действий для соответствия столбцов.

    Returns:
        Кортеж ``(value, ess_value, clip_share)``.
    """
    _validate_logs(df, target, action_col)
    if len(pA) != len(df):
        raise ValueError("pA must have same length as df")
    if np.any((pA <= 0) | (pA > 1)):
        raise ValueError("propensity scores must be in (0,1]")

    probsB = policyB.action_probs(df)
    actions_space = _action_values_from_probs(probsB, action_space)
    r = df[target].to_numpy()
    aA = df[action_col].to_numpy()

    dm_part = np.zeros(len(df), dtype=float)
    for idx, action in enumerate(actions_space):
        pa = probsB[:, idx]
        if pa.sum() == 0:
            continue
        mu = mu_hat_predict(mu_model, df, np.full(len(df), action, dtype=object), target)
        dm_part += pa * mu

    piB_taken = take_action_probabilities(probsB, aA, action_space=actions_space)
    w = piB_taken / pA
    clip_share = 0.0
    if weight_clip is not None:
        clip_share = float((w > weight_clip).mean())
        w = np.minimum(w, weight_clip)

    mu_taken = mu_hat_predict(mu_model, df, aA, target)
    value = float(np.mean(dm_part + w * (r - mu_taken)))
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
    """Self-Normalized Doubly Robust (SNDR).

    То же, что DR, но IPS-поправка нормируется на средний вес
    ``w / mean(w)`` для стабилизации дисперсии.

    Args:
        df: Логи политики A.
        policyB: Политика B.
        mu_model: Модель исхода ``mu``.
        pA: Вектор ``pi_A(a_logged|x)``.
        target: Целевая метрика.
        weight_clip: Опциональный клиппинг весов.
        action_col: Колонка логированного действия.
        action_space: Пространство действий для соответствия столбцов.

    Returns:
        Кортеж ``(value, ess_value, clip_share)``.
    """
    _validate_logs(df, target, action_col)
    if len(pA) != len(df):
        raise ValueError("pA must have same length as df")
    if np.any((pA <= 0) | (pA > 1)):
        raise ValueError("propensity scores must be in (0,1]")

    probsB = policyB.action_probs(df)
    actions_space = _action_values_from_probs(probsB, action_space)
    r = df[target].to_numpy()
    aA = df[action_col].to_numpy()

    dm_part = np.zeros(len(df), dtype=float)
    for idx, action in enumerate(actions_space):
        pa = probsB[:, idx]
        if pa.sum() == 0:
            continue
        mu = mu_hat_predict(mu_model, df, np.full(len(df), action, dtype=object), target)
        dm_part += pa * mu

    piB_taken = take_action_probabilities(probsB, aA, action_space=actions_space)
    w = piB_taken / pA
    clip_share = 0.0
    if weight_clip is not None:
        clip_share = float((w > weight_clip).mean())
        w = np.minimum(w, weight_clip)

    mu_taken = mu_hat_predict(mu_model, df, aA, target)
    value = float(np.mean(dm_part + (w / (w.mean() + 1e-12)) * (r - mu_taken)))
    return value, ess(w), clip_share


def switch_dr_value(
    df: pd.DataFrame,
    policyB,
    mu_model,
    pA: np.ndarray,
    tau: float,
    target: str = "accept",
    *,
    action_col: str = "a_A",
    action_space: Optional[Sequence] = None,
) -> Tuple[float, float, float]:
    """Switch-DR: DR с отключением IPS-поправки на «тяжёлых» весах.

    Для наблюдений с ``w_i > tau`` поправка зануляется, остаётся только DM-часть.
    Это снижает дисперсию ценой дополнительного смещения.

    Args:
        df: Логи политики A.
        policyB: Политика B.
        mu_model: Модель исхода ``mu``.
        pA: Вектор ``pi_A(a_logged|x)``.
        tau: Порог switch для веса важности.
        target: Целевая метрика.
        action_col: Колонка логированного действия.
        action_space: Пространство действий.

    Returns:
        Кортеж ``(value, ess_switched, switched_share)``, где ``switched_share`` —
        доля наблюдений, в которых сработало отключение IPS-поправки.
    """
    _validate_logs(df, target, action_col)
    if len(pA) != len(df):
        raise ValueError("pA must have same length as df")
    if np.any((pA <= 0) | (pA > 1)):
        raise ValueError("propensity scores must be in (0,1]")

    probsB = policyB.action_probs(df)
    actions_space = _action_values_from_probs(probsB, action_space)
    r = df[target].to_numpy()
    aA = df[action_col].to_numpy()

    dm_part = np.zeros(len(df), dtype=float)
    for idx, action in enumerate(actions_space):
        pa = probsB[:, idx]
        if pa.sum() == 0:
            continue
        mu = mu_hat_predict(mu_model, df, np.full(len(df), action, dtype=object), target)
        dm_part += pa * mu

    piB_taken = take_action_probabilities(probsB, aA, action_space=actions_space)
    w = piB_taken / pA
    mask = w <= tau
    w_sw = np.where(mask, w, 0.0)
    mu_taken = mu_hat_predict(mu_model, df, aA, target)
    value = float(np.mean(dm_part + w_sw * (r - mu_taken)))
    return value, ess(w_sw), float((~mask).mean())


def estimator_with_bootstrap_ci(
    df: pd.DataFrame,
    estimator_fn,
    *,
    cluster_col: Optional[str] = "user_id",
    n_boot: int = 300,
    alpha: float = 0.05,
):
    """Универсальная обёртка: point-estimator + bootstrap CI.

    Эта функция отделяет две сущности:
    1) ``estimator_fn(df_part) -> float`` — как считается точечная оценка;
    2) bootstrap — как считается доверительный интервал для этой оценки.

    Args:
        df: Исходные логи.
        estimator_fn: Функция, возвращающая ``float`` на переданном DataFrame.
            Пример: ``lambda part: ips_value(part, piB_taken_part, pA_part)[0]``.
        cluster_col: Колонка для кластерного bootstrap (например ``user_id``).
            Если ``None`` или колонки нет, используется bootstrap по строкам.
        n_boot: Число bootstrap-реплик.
        alpha: Уровень значимости (0.05 => 95% CI).

    Returns:
        Словарь вида:
        ``{'value': ..., 'CI': (low, high), 'n_boot': ..., 'alpha': ...}``.
    """
    from policyscope.bootstrap import cluster_bootstrap_ci

    value, low, high = cluster_bootstrap_ci(
        df,
        estimator_fn,
        cluster_col=cluster_col,
        n_boot=n_boot,
        alpha=alpha,
    )
    return {
        "value": value,
        "CI": (low, high),
        "n_boot": n_boot,
        "alpha": alpha,
    }


def ate_from_values(vB: float, vA: float) -> float:
    """Возвращает разницу значений политик ``vB - vA``.

    Args:
        vB: Оценка значения новой политики B.
        vA: Оценка значения текущей политики A.

    Returns:
        ATE/Delta между политиками.
    """
    return float(vB - vA)
