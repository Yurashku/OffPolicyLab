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
    "dr_with_bootstrap_ci",
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
    """Вычисляет эффективный размер выборки (ESS) для вектора весов.

    ESS используется как быстрая диагностика качества пере-взвешивания:
    чем меньше значение, тем выше дисперсия IPS-подобных оценок.
    Формула:

    .. math::
        ESS = (\\sum_i w_i)^2 / \\sum_i w_i^2

    Parameters
    ----------
    weights : np.ndarray
        Вектор importance-весов :math:`w_i`.

    Returns
    -------
    float
        Эффективный размер выборки. Если все веса нулевые, возвращается 0.
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
    """Создаёт дизайн-матрицу ``[scaled_features, onehot(action)]``."""
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
    """Обучает модель поведения :math:`\\hat\\pi_A(a\\mid x)`.

    Модель нужна для оценки вероятности логированного действия под
    исторической политикой A, когда истинные propensity не сохранены в логах.
    Используется многоклассовая логистическая регрессия по признакам
    ``feature_cols``.

    Parameters
    ----------
    df : pd.DataFrame
        Логи с признаками и колонкой действия поведения.
    feature_cols : Sequence[str], optional
        Признаки контекста :math:`x`.
    action_col : str, default=\"a_A\"
        Колонка с действием, выбранным политикой A.

    Returns
    -------
    sklearn estimator
        Обученная модель с сохранёнными служебными атрибутами
        ``_feature_cols`` и ``_scaler``.
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
    """Предсказывает :math:`\\hat\\pi_A(a\\mid x)` для всех действий.

    Parameters
    ----------
    model : sklearn estimator
        Обученная модель из :func:`train_pi_hat`.
    df : pd.DataFrame
        Данные контекста.

    Returns
    -------
    np.ndarray
        Матрица вероятностей формы ``(n_samples, n_actions)``.
        Значения клиппируются в ``[1e-6, 1-1e-6]`` для численной устойчивости.
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
    """Извлекает вероятность фактически выбранного действия для каждой строки.

    Эта функция устраняет хрупкость индексации по предположению, что действия
    обязательно кодируются числами ``0..K-1``. Поддерживаются и строковые
    метки (например, ``\"offer_A\"``, ``\"offer_B\"``), если они перечислены в
    ``action_space`` в том же порядке, что и столбцы матрицы ``probs``.
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
    """Обучает модель исхода :math:`\\hat\\mu(x, a)`.

    ``\\hat\\mu(x, a)`` используется в методах DM/DR/SNDR/Switch-DR для
    аппроксимации ожидаемого отклика при заданном контексте и действии.
    Для бинарной метрики обычно применяют логистическую регрессию, для
    непрерывной — линейную регрессию.

    Parameters
    ----------
    df : pd.DataFrame
        Логи с признаками, действием и целевой метрикой.
    target : str, default=\"accept\"
        Название целевой метрики (например, ``accept``, ``cltv``, ``reward``).
    feature_cols : Sequence[str], optional
        Список признаков контекста.
    action_col : str, default=\"a_A\"
        Колонка действия, которое было показано в логах.

    Returns
    -------
    sklearn estimator
        Обученная модель исхода с one-hot кодировщиком действий.
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
    """Предсказывает :math:`\\hat\\mu(x, a)` для переданного действия.

    Parameters
    ----------
    model : sklearn estimator
        Модель из :func:`train_mu_hat`.
    df : pd.DataFrame
        Контексты пользователей/объектов.
    action : np.ndarray or scalar
        Действия, для которых нужно предсказать отклик.
    target : str
        Тип таргета: для ``accept`` возвращаются вероятности класса 1,
        иначе — численный прогноз.
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
    """Считает on-policy baseline: среднее наблюдаемой метрики в логах A.

    Это оценка :math:`V(A)` без пере-взвешивания, т.е. просто эмпирическое
    ожидание таргета в собранных данных.
    """
    return float(df[target].mean())


def replay_value(
    df: pd.DataFrame,
    a_B: np.ndarray,
    target: str = "accept",
    *,
    action_col: str = "a_A",
) -> float:
    """Оценивает value новой политики B методом Replay.

    Replay учитывает только те записи, где действие B совпало с действием,
    реально показанным в логах A. Метод очень простой и интерпретируемый,
    но требует достаточного перекрытия действий между A и B.

    Returns
    -------
    float
        Средняя метрика на совпавших записях; ``NaN``, если совпадений нет.
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
    """Вычисляет :math:`\\pi_B(a_A \\mid x)` для каждой записи.

    Нужен вспомогательный вектор вероятностей того, что новая политика B
    выбрала бы именно логированное действие A. Этот вектор используется в
    IPS/SNIPS и DR-подобных оценщиках.
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
    """Оценка value(B) через IPS (Inverse Propensity Scoring).

    Формула:

    .. math::
        \\hat V_{IPS} = \\frac{1}{n}\\sum_i w_i r_i,\\;
        w_i = \\frac{\\pi_B(a_i\\mid x_i)}{\\hat\\pi_A(a_i\\mid x_i)}

    IPS несмещён при корректных propensity, но может иметь высокую дисперсию.
    Для стабилизации поддерживается клиппинг весов ``weight_clip``.

    Returns
    -------
    tuple(float, float, float)
        ``(value, ess, clip_share)`` — оценка value(B), ESS и доля клиппинга.
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
    """Оценка value(B) через SNIPS (Self-Normalized IPS).

    В отличие от IPS, SNIPS нормализует сумму весов:

    .. math::
        \\hat V_{SNIPS} = \\frac{\\sum_i w_i r_i}{\\sum_i w_i}

    Это обычно уменьшает дисперсию, но может вносить небольшой bias
    на конечной выборке.
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
    """Оценивает value(B) методом Direct Method (DM).

    DM строит модель исхода :math:`\\hat\\mu(x,a)` и затем усредняет прогноз
    по распределению действий новой политики B:

    .. math::
        \\hat V_{DM} = \\frac{1}{n}\\sum_i \\sum_a \\pi_B(a\\mid x_i)\\hat\\mu(x_i,a)

    Метод не использует propensity напрямую и хорошо работает при качественной
    модели исхода.
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
    r"""Оценивает value(B) методом Doubly Robust (DR).

    DR объединяет DM и IPS-поправку:

    .. math::
        \hat V_{DR} =
        \frac{1}{n}\sum_i\left[
            \sum_a \pi_B(a\mid x_i)\hat\mu(x_i,a) +
            w_i\big(r_i - \hat\mu(x_i,a_i)\big)
        \right]

    Свойство *double robustness*: оценка остаётся состоятельной, если
    корректна хотя бы одна из двух частей — propensity-модель или модель исхода.
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
    r"""Оценивает value(B) методом Self-Normalized Doubly Robust (SNDR).

    SNDR использует DR-структуру, но нормализует IPS-поправку средним весом.
    Это часто делает оценку стабильнее на практике при тяжёлых хвостах весов.
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
    r"""Оценивает value(B) методом Switch-DR.

    Идея: использовать IPS-поправку только на «безопасных» весах
    :math:`w_i \le \tau`, а при слишком больших весах полагаться на DM-часть.
    Это снижает дисперсию ценой возможного дополнительного смещения.
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


def dr_with_bootstrap_ci(
    df: pd.DataFrame,
    policyB,
    *,
    target: str = "accept",
    feature_cols: Optional[Sequence[str]] = None,
    action_col: str = "a_A",
    action_space: Optional[Sequence] = None,
    cluster_col: Optional[str] = "user_id",
    n_boot: int = 300,
    alpha: float = 0.05,
    weight_clip: Optional[float] = None,
):
    """Считает DR и доверительные интервалы бутстрэпом в одном вызове.

    Удобная high-level обёртка для туториала и прикладных экспериментов:
    внутри обучает ``mu_hat`` и ``pi_hat``, затем запускает парный bootstrap
    и возвращает словарь в формате ``paired_bootstrap_ci``.
    """
    from policyscope.bootstrap import paired_bootstrap_ci

    def estimator_pair(df_part: pd.DataFrame):
        mu = train_mu_hat(df_part, target=target, feature_cols=feature_cols, action_col=action_col)
        pi_model = train_pi_hat(df_part, feature_cols=feature_cols, action_col=action_col)
        pA_all = pi_hat_predict(pi_model, df_part)
        pA_taken = take_action_probabilities(pA_all, df_part[action_col].to_numpy(), action_space=pi_model.classes_)
        vA = value_on_policy(df_part, target=target)
        vB, _, _ = dr_value(
            df_part,
            policyB,
            mu,
            pA_taken,
            target=target,
            weight_clip=weight_clip,
            action_col=action_col,
            action_space=action_space,
        )
        return vA, vB, vB - vA

    return paired_bootstrap_ci(
        df,
        estimator_pair,
        cluster_col=cluster_col,
        n_boot=n_boot,
        alpha=alpha,
    )


def ate_from_values(vB: float, vA: float) -> float:
    return float(vB - vA)
