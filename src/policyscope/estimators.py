"""
policyscope.estimators
======================

Реализация оценщиков для off‑policy сравнения политик.

Содержит функции для оценки значения новой политики по логам старой
политики, включая:

* on-policy value (`value_on_policy`)
* replay value (`replay_value`)
* IPS и SNIPS (`ips_value`, `snips_value`)
* Direct Method (`dm_value`) и Doubly Robust (`dr_value`)
* Self-Normalized DR (`sndr_value`) и Switch-DR (`switch_dr_value`)

Также реализованы вспомогательные функции для обучения модели исхода,
расчёта веса и эффективного размера выборки (ESS) и подготовки
вероятностей выбранного действия под политикой B.
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Literal
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from policyscope.policies import BasePolicy

__all__ = [
    "ess",
    "make_design",
    "train_pi_hat",
    "pi_hat_predict",
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
    "ate_from_values",
]


def _validate_logs(df: pd.DataFrame, target: str) -> None:
    """Проверяет наличие необходимых колонок в логах.

    Parameters
    ----------
    df : pd.DataFrame
        Логи политики A.
    target : str
        Имя целевой метрики.
    """
    required = {"a_A", target}
    missing = required - set(df.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"Missing required columns: {missing_str}")


def ess(weights: np.ndarray) -> float:
    """Эффективный размер выборки (ESS).

    Определяется как (∑w)^2 / ∑w^2. Чем меньше ESS, тем выше дисперсия
    оценщика IPS.

    Parameters
    ----------
    weights : np.ndarray
        Веса (важности).

    Returns
    -------
    float
        Эффективный размер выборки.
    """
    s1 = weights.sum()
    s2 = (weights ** 2).sum()
    if s2 <= 0.0:
        return 0.0
    return float((s1 * s1) / s2)


def make_design(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, OneHotEncoder, StandardScaler]:
    """Создаёт дизайн‑матрицу для обучения модели исхода.

    Числовые признаки ``age``, ``risk`` и ``income`` нормализуются с помощью
    ``StandardScaler`` перед конкатенацией с ``loyal`` и one-hot представлением
    действия.

    Returns
    -------
    X : np.ndarray
        Полная матрица признаков.
    A_oh : np.ndarray
        One-hot представление действий.
    oh : OneHotEncoder
        Обученный кодировщик (полезен для предсказаний).
    scaler : StandardScaler
        Обученный нормализатор числовых признаков.
    """
    scaler = StandardScaler()
    num_scaled = scaler.fit_transform(df[["age", "risk", "income"]])
    X_base = np.hstack([df[["loyal"]].values, num_scaled])
    a = df["a_A"].values.reshape(-1, 1)
    try:
        oh = OneHotEncoder(
            categories="auto", sparse_output=False, handle_unknown="ignore"
        )
    except TypeError:  # scikit-learn < 1.2
        oh = OneHotEncoder(
            categories="auto", sparse=False, handle_unknown="ignore"
        )
    A_oh = oh.fit_transform(a)
    X = np.hstack([X_base, A_oh])
    return X, A_oh, oh, scaler


def train_pi_hat(df: pd.DataFrame):
    r"""Обучает модель пропенсити ``\hat\pi(a|x)``.

    Используется многоклассовая логистическая регрессия. Перед обучением
    признаки ``age``, ``risk`` и ``income`` нормализуются с помощью
    ``StandardScaler``.
    """
    scaler = StandardScaler()
    num_scaled = scaler.fit_transform(df[["age", "risk", "income"]])
    X = np.hstack([df[["loyal"]].values, num_scaled])
    y = df["a_A"].values
    # ``multi_class`` deprecated in scikit-learn>=1.5 and will always use
    # ``multinomial`` в будущих версиях. Оставляем значение по умолчанию,
    # чтобы сохранить совместимость и не получать FutureWarning.
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    model._scaler = scaler  # type: ignore[attr-defined]
    return model


def pi_hat_predict(model, df: pd.DataFrame) -> np.ndarray:
    r"""Предсказывает ``\hat\pi(a|x)`` для всех действий."""
    num_scaled = model._scaler.transform(df[["age", "risk", "income"]])  # type: ignore[attr-defined]
    X = np.hstack([df[["loyal"]].values, num_scaled])
    probs = model.predict_proba(X)
    return np.clip(probs, 1e-6, 1 - 1e-6)


def train_mu_hat(df: pd.DataFrame, target: Literal["accept", "cltv"] = "accept"):
    r"""Обучает модель исхода \(\hat\mu(x,a)\).

    Для бинарного таргета `accept` используется логистическая регрессия.
    Для вещественного `cltv` — линейная регрессия.

    Возвращает обученную модель с атрибутом `_oh` для кодировщика one-hot.
    """
    X, _, oh, scaler = make_design(df)
    y = df[target].values
    if target == "accept":
        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)
    else:
        model = LinearRegression()
        model.fit(X, y)
    model._oh = oh  # type: ignore[attr-defined]
    model._scaler = scaler  # type: ignore[attr-defined]
    return model


def mu_hat_predict(model, df: pd.DataFrame, action: np.ndarray, target: str) -> np.ndarray:
    r"""Предсказывает \(\hat\mu(x,a)\) для заданного действия.

    Параметр `action` может быть массивом тех же размеров, что и `df`, или
    скалярным значением (будет вещательно расширён).
    """
    num_scaled = model._scaler.transform(df[["age", "risk", "income"]])  # type: ignore[attr-defined]
    num = np.hstack([df[["loyal"]].values, num_scaled])
    if np.isscalar(action):
        act = np.full(len(df), action, dtype=int).reshape(-1, 1)
    else:
        act = action.reshape(-1, 1)
    A_oh = model._oh.transform(act)
    X = np.hstack([num, A_oh])
    if target == "accept":
        probs = model.predict_proba(X)[:, 1]
        return np.clip(probs, 1e-6, 1 - 1e-6)
    return model.predict(X)


def value_on_policy(df: pd.DataFrame, target: str = "accept") -> float:
    """Среднее значение метрики в логах (on-policy)."""
    return float(df[target].mean())


def replay_value(df: pd.DataFrame, a_B: np.ndarray, target: str = "accept") -> float:
    """Оценка значения новой политики через повтор (replay).

    Отбираем только те записи, где новое действие совпадает с логом.
    Оставляем соответствующие исходы и усредняем.
    Если совпадений нет, возвращаем NaN.
    """
    logging.info("[Replay] Начинаем оценку Replay для новой политики…")
    mask = df["a_A"].values == a_B
    if mask.sum() == 0:
        logging.info(
            "[Replay] Нет совпадающих действий – политика B вне области данных A"
        )
        return float("nan")
    logging.info(
        f"[Replay] Совпадающих действий: {mask.sum()} из {len(df)}"
    )
    val = float(df.loc[mask, target].mean())
    logging.info(f"[Replay] Оценённое значение метрики: {val:.4f}")
    return val


def prepare_piB_taken(df: pd.DataFrame, policyB) -> np.ndarray:
    """Возвращает вероятности того, что политика B выбрала фактическое действие A.

    Parameters
    ----------
    df : pd.DataFrame
        Логи политики A (с колонкой a_A).
    policyB : BasePolicy
        Политика B.

    Returns
    -------
    np.ndarray
        Вектор вероятностей `π_B(a_A|x)`.
    """
    probsB = policyB.action_probs(df)
    aA = df["a_A"].values
    return probsB[np.arange(len(df)), aA]


def ips_value(
    df: pd.DataFrame,
    piB_taken: np.ndarray,
    pA: np.ndarray,
    target: str = "accept",
    weight_clip: Optional[float] = None,
) -> Tuple[float, float, float]:
    """IPS‑оценка значения новой политики.

    Parameters
    ----------
    df : pd.DataFrame
        Логи политики A с исходом `target`.
    piB_taken : np.ndarray
        Probabilities π_B(a_A|x) — вероятности, с которыми политика B выбрала бы логированные действия.
    pA : np.ndarray
        Оценённые вероятности поведения `π_A(a_A|x)`.
    target : {"accept", "cltv"}
        Название колонки исхода.
    weight_clip : float, optional
        Обрезка весов: w = min(w, weight_clip).

    Returns
    -------
    (value, ess, clip_share)
        Оценка метрики, эффективный размер выборки и доля обрезанных весов.
    """
    logging.info("[IPS] Начинаем оценку IPS для новой политики…")
    _validate_logs(df, target)
    if len(pA) != len(df):
        raise ValueError("pA must have same length as df")
    if np.any((pA <= 0) | (pA > 1)):
        logging.warning("[IPS] Обнаружены нулевые или отрицательные пропенсити")
        raise ValueError("propensity scores must be in (0,1]")
    w = piB_taken / pA
    clip_share = 0.0
    if weight_clip is not None:
        clip_mask = w > weight_clip
        clip_share = float(clip_mask.mean())
        w = np.minimum(w, weight_clip)
    r = df[target].values
    value = float(np.mean(r * w))
    ess_w = ess(w)
    logging.info(f"[IPS] ESS = {ess_w:.1f} из {len(df)}")
    if ess_w < 0.5 * len(df):
        logging.warning("[IPS] Низкий ESS — данных может быть недостаточно")
    if weight_clip is not None:
        logging.info(f"[IPS] Обрезано весов: {clip_share:.2%}")
    logging.info(f"[IPS] Оценённое значение метрики: {value:.4f}")
    return value, ess_w, clip_share


def snips_value(
    df: pd.DataFrame,
    piB_taken: np.ndarray,
    pA: np.ndarray,
    target: str = "accept",
    weight_clip: Optional[float] = None,
) -> Tuple[float, float, float]:
    """SNIPS‑оценка значения новой политики.

    Считается как (∑r_i w_i)/(∑w_i). Применяет обрезку весов, если указано.
    """
    logging.info("[SNIPS] Начинаем оценку SNIPS для новой политики…")
    _validate_logs(df, target)
    if len(pA) != len(df):
        raise ValueError("pA must have same length as df")
    if np.any((pA <= 0) | (pA > 1)):
        logging.warning("[SNIPS] Обнаружены нулевые или отрицательные пропенсити")
        raise ValueError("propensity scores must be in (0,1]")
    w = piB_taken / pA
    clip_share = 0.0
    if weight_clip is not None:
        clip_mask = w > weight_clip
        clip_share = float(clip_mask.mean())
        w = np.minimum(w, weight_clip)
    r = df[target].values
    num = np.sum(r * w)
    den = np.sum(w) + 1e-12
    value = float(num / den)
    ess_w = ess(w)
    logging.info("[SNIPS] Веса нормализованы")
    logging.info(f"[SNIPS] ESS = {ess_w:.1f} из {len(df)}")
    if ess_w < 0.5 * len(df):
        logging.warning("[SNIPS] Низкий ESS — данных может быть недостаточно")
    if weight_clip is not None:
        logging.info(f"[SNIPS] Обрезано весов: {clip_share:.2%}")
    logging.info(f"[SNIPS] Оценённое значение метрики: {value:.4f}")
    return value, ess_w, clip_share


def dm_value(df: pd.DataFrame, policyB, mu_model, target: str = "accept") -> float:
    """Direct Method: ожидание предсказанного исхода под политикой B."""
    logging.info(
        "[DM] Начинаем оценку Direct Method для новой политики…"
    )
    probsB = policyB.action_probs(df)
    val = 0.0
    for a in BasePolicy.ACTIONS:
        pa = probsB[:, a]
        if pa.sum() == 0:
            continue
        mu = mu_hat_predict(mu_model, df, np.full(len(df), a), target)
        val += float(np.mean(pa * mu))
    logging.info(f"[DM] Оценённое значение метрики: {val:.4f}")
    return val


def dr_value(
    df: pd.DataFrame,
    policyB,
    mu_model,
    pA: np.ndarray,
    target: str = "accept",
    weight_clip: Optional[float] = None,
) -> Tuple[float, float, float]:
    r"""Doubly Robust оценка значения новой политики.

    Комбинирует Direct Method и IPS‑поправку. При корректном
    моделировании хотя бы одной составляющей даёт несмещённую оценку.

    Parameters
    ----------
    df : pd.DataFrame
        Логи политики A с исходом `target`.
    policyB : BasePolicy
        Новая политика.
    mu_model : модель исхода
        Обученная модель `\hat\mu(x,a)`.
    pA : np.ndarray
        Оценённые вероятности поведения `π_A(a_A|x)`.
    target : {"accept", "cltv"}
        Название колонки исхода.
    weight_clip : float, optional
        Обрезка весов.
    """
    logging.info("[DR] Начинаем оценку Doubly Robust для новой политики…")
    _validate_logs(df, target)
    if len(pA) != len(df):
        raise ValueError("pA must have same length as df")
    if np.any((pA <= 0) | (pA > 1)):
        logging.warning("[DR] Обнаружены нулевые или отрицательные пропенсити")
        raise ValueError("propensity scores must be in (0,1]")
    probsB = policyB.action_probs(df)
    r = df[target].values
    aA = df["a_A"].values

    dm_part = np.zeros(len(df), dtype=float)
    for a in BasePolicy.ACTIONS:
        pa = probsB[:, a]
        if pa.sum() == 0:
            continue
        mu = mu_hat_predict(mu_model, df, np.full(len(df), a), target)
        dm_part += pa * mu

    piB_taken = probsB[np.arange(len(df)), aA]
    w = piB_taken / pA
    clip_share = 0.0
    if weight_clip is not None:
        clip_mask = w > weight_clip
        clip_share = float(clip_mask.mean())
        w = np.minimum(w, weight_clip)

    logging.info("[DR] Комбинируем DM и IPS-поправку")
    mu_taken = mu_hat_predict(mu_model, df, aA, target)
    adj = w * (r - mu_taken)
    value = float(np.mean(dm_part + adj))
    ess_w = ess(w)
    logging.info(f"[DR] ESS = {ess_w:.1f} из {len(df)}")
    if ess_w < 0.5 * len(df):
        logging.warning("[DR] Низкий ESS — данных может быть недостаточно")
    if weight_clip is not None:
        logging.info(f"[DR] Обрезано весов: {clip_share:.2%}")
    logging.info(f"[DR] Оценённое значение метрики: {value:.4f}")
    return value, ess_w, clip_share


def sndr_value(
    df: pd.DataFrame,
    policyB,
    mu_model,
    pA: np.ndarray,
    target: str = "accept",
    weight_clip: Optional[float] = None,
) -> Tuple[float, float, float]:
    r"""Self-Normalized Doubly Robust оценка значения новой политики.

    Для каждого наблюдения добавляет IPS-поправку, нормализованную на
    средний вес:
    \(\frac{1}{n} \sum_i \big[q^\,(x_i, \pi_B) + \frac{w_i}{\bar{w}} (r_i - q^\,(x_i, a_i))\big]\).

    Parameters
    ----------
    df : pd.DataFrame
        Логи политики A с исходом `target`.
    policyB : BasePolicy
        Новая политика.
    mu_model : модель исхода
        Обученная модель `\hat\mu(x,a)`.
    pA : np.ndarray
        Оценённые вероятности поведения `π_A(a_A|x)`.
    target : {"accept", "cltv"}
        Название колонки исхода.
    weight_clip : float, optional
        Обрезка весов перед нормализацией.

    Returns
    -------
    (value, ess, clip_share)
        Оценка метрики, эффективный размер выборки и доля обрезанных весов.
    """
    logging.info(
        "[SNDR] Начинаем оценку Self-Normalized DR для новой политики…"
    )
    _validate_logs(df, target)
    if len(pA) != len(df):
        raise ValueError("pA must have same length as df")
    if np.any((pA <= 0) | (pA > 1)):
        logging.warning("[SNDR] Обнаружены нулевые или отрицательные пропенсити")
        raise ValueError("propensity scores must be in (0,1]")
    probsB = policyB.action_probs(df)
    r = df[target].values
    aA = df["a_A"].values

    dm_part = np.zeros(len(df), dtype=float)
    for a in BasePolicy.ACTIONS:
        pa = probsB[:, a]
        if pa.sum() == 0:
            continue
        mu = mu_hat_predict(mu_model, df, np.full(len(df), a), target)
        dm_part += pa * mu

    piB_taken = probsB[np.arange(len(df)), aA]
    w = piB_taken / pA
    clip_share = 0.0
    if weight_clip is not None:
        clip_mask = w > weight_clip
        clip_share = float(clip_mask.mean())
        w = np.minimum(w, weight_clip)

    logging.info("[SNDR] Комбинируем DM и IPS-поправку")
    mu_taken = mu_hat_predict(mu_model, df, aA, target)
    mean_w = np.mean(w) + 1e-12
    logging.info("[SNDR] Веса нормализованы")
    adj = (w / mean_w) * (r - mu_taken)
    value = float(np.mean(dm_part + adj))
    ess_w = ess(w)
    logging.info(f"[SNDR] ESS = {ess_w:.1f} из {len(df)}")
    if ess_w < 0.5 * len(df):
        logging.warning("[SNDR] Низкий ESS — данных может быть недостаточно")
    if weight_clip is not None:
        logging.info(f"[SNDR] Обрезано весов: {clip_share:.2%}")
    logging.info(f"[SNDR] Оценённое значение метрики: {value:.4f}")
    return value, ess_w, clip_share


def switch_dr_value(
    df: pd.DataFrame,
    policyB,
    mu_model,
    pA: np.ndarray,
    tau: float,
    target: str = "accept",
) -> Tuple[float, float, float]:
    r"""Switch-DR оценка с переключением по весу.

    Добавляет IPS-поправку только если вес \(w_i = \pi_B/\pi_A\) не превышает
    порог `tau`. При больших весах используется только DM‑часть, что снижает
    дисперсию оценщика.

    Parameters
    ----------
    df : pd.DataFrame
        Логи политики A с исходом `target`.
    policyB : BasePolicy
        Новая политика.
    mu_model : модель исхода
        Обученная модель `\hat\mu(x,a)`.
    pA : np.ndarray
        Оценённые вероятности поведения `π_A(a_A|x)`.
    tau : float
        Порог для применения IPS-поправки.
    target : {"accept", "cltv"}
        Название колонки исхода.

    Returns
    -------
    (value, ess, switch_share)
        Оценка метрики, эффективный размер выборки и доля записей без IPS‑поправки.
    """
    logging.info(
        "[Switch-DR] Начинаем оценку Switch-DR: IPS-поправка только при весах ≤ τ…"
    )
    _validate_logs(df, target)
    if len(pA) != len(df):
        raise ValueError("pA must have same length as df")
    if np.any((pA <= 0) | (pA > 1)):
        logging.warning("[Switch-DR] Обнаружены нулевые или отрицательные пропенсити")
        raise ValueError("propensity scores must be in (0,1]")
    probsB = policyB.action_probs(df)
    r = df[target].values
    aA = df["a_A"].values

    dm_part = np.zeros(len(df), dtype=float)
    for a in BasePolicy.ACTIONS:
        pa = probsB[:, a]
        if pa.sum() == 0:
            continue
        mu = mu_hat_predict(mu_model, df, np.full(len(df), a), target)
        dm_part += pa * mu

    piB_taken = probsB[np.arange(len(df)), aA]
    w = piB_taken / pA
    mask = w <= tau
    switch_share = float((~mask).mean())
    w_sw = np.where(mask, w, 0.0)

    mu_taken = mu_hat_predict(mu_model, df, aA, target)
    adj = w_sw * (r - mu_taken)
    value = float(np.mean(dm_part + adj))
    ess_w = ess(w_sw)
    logging.info(f"[Switch-DR] ESS = {ess_w:.1f} из {len(df)}")
    if ess_w < 0.5 * len(df):
        logging.warning("[Switch-DR] Низкий ESS — данных может быть недостаточно")
    logging.info(
        f"[Switch-DR] Без IPS-поправки: {switch_share:.2%}"
    )
    logging.info(f"[Switch-DR] Оценённое значение метрики: {value:.4f}")
    return value, ess_w, switch_share


def ate_from_values(vB: float, vA: float) -> float:
    """Разность двух значений (ATE = V(B) - V(A))."""
    return float(vB - vA)
