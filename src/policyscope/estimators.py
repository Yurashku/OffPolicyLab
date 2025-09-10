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

Также реализованы вспомогательные функции для обучения модели исхода,
расчёта веса и эффективного размера выборки (ESS) и подготовки
вероятностей выбранного действия под политикой B.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Literal
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import OneHotEncoder

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


def make_design(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, OneHotEncoder]:
    """Создаёт дизайн‑матрицу для обучения модели исхода.

    Признаки: [loyal, age_z, risk_z, income_z] + one-hot по действию.

    Returns
    -------
    X : np.ndarray
        Полная матрица признаков.
    A_oh : np.ndarray
        One-hot представление действий.
    oh : OneHotEncoder
        Обученный кодировщик (полезен для предсказаний).
    """
    X_base = df[["loyal", "age_z", "risk_z", "income_z"]].values
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
    return X, A_oh, oh


def train_pi_hat(df: pd.DataFrame):
    """Обучает модель пропенсити ``\hat\pi(a|x)``.

    Используется многоклассовая логистическая регрессия по признакам
    [loyal, age_z, risk_z, income_z].
    """
    X = df[["loyal", "age_z", "risk_z", "income_z"]].values
    y = df["a_A"].values
    model = LogisticRegression(max_iter=1000, multi_class="multinomial")
    model.fit(X, y)
    return model


def pi_hat_predict(model, df: pd.DataFrame) -> np.ndarray:
    """Предсказывает ``\hat\pi(a|x)`` для всех действий."""
    X = df[["loyal", "age_z", "risk_z", "income_z"]].values
    probs = model.predict_proba(X)
    return np.clip(probs, 1e-6, 1 - 1e-6)


def train_mu_hat(df: pd.DataFrame, target: Literal["accept", "cltv"] = "accept"):
    r"""Обучает модель исхода \(\hat\mu(x,a)\).

    Для бинарного таргета `accept` используется логистическая регрессия.
    Для вещественного `cltv` — линейная регрессия.

    Возвращает обученную модель с атрибутом `_oh` для кодировщика one-hot.
    """
    X, _, oh = make_design(df)
    y = df[target].values
    if target == "accept":
        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)
    else:
        model = LinearRegression()
        model.fit(X, y)
    model._oh = oh  # type: ignore[attr-defined]
    return model


def mu_hat_predict(model, df: pd.DataFrame, action: np.ndarray, target: str) -> np.ndarray:
    r"""Предсказывает \(\hat\mu(x,a)\) для заданного действия.

    Параметр `action` может быть массивом тех же размеров, что и `df`, или
    скалярным значением (будет вещательно расширён).
    """
    num = df[["loyal", "age_z", "risk_z", "income_z"]].values
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
    mask = (df["a_A"].values == a_B)
    if mask.sum() == 0:
        return float('nan')
    return float(df.loc[mask, target].mean())


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
    _validate_logs(df, target)
    if len(pA) != len(df):
        raise ValueError("pA must have same length as df")
    if np.any((pA <= 0) | (pA > 1)):
        raise ValueError("propensity scores must be in (0,1]")
    w = piB_taken / pA
    clip_share = 0.0
    if weight_clip is not None:
        # доля весов, которые были клипнуты
        clip_mask = w > weight_clip
        clip_share = float(clip_mask.mean())
        w = np.minimum(w, weight_clip)
    r = df[target].values
    value = float(np.mean(r * w))
    return value, ess(w), clip_share


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
    _validate_logs(df, target)
    if len(pA) != len(df):
        raise ValueError("pA must have same length as df")
    if np.any((pA <= 0) | (pA > 1)):
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
    return value, ess(w), clip_share


def dm_value(df: pd.DataFrame, policyB, mu_model, target: str = "accept") -> float:
    """Direct Method: ожидание предсказанного исхода под политикой B."""
    probsB = policyB.action_probs(df)
    val = 0.0
    for a in BasePolicy.ACTIONS:
        pa = probsB[:, a]
        if pa.sum() == 0:
            continue
        mu = mu_hat_predict(mu_model, df, np.full(len(df), a), target)
        val += float(np.mean(pa * mu))
    return val


def dr_value(
    df: pd.DataFrame,
    policyB,
    mu_model,
    pA: np.ndarray,
    target: str = "accept",
    weight_clip: Optional[float] = None,
) -> Tuple[float, float, float]:
    """Doubly Robust оценка значения новой политики.

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
    _validate_logs(df, target)
    if len(pA) != len(df):
        raise ValueError("pA must have same length as df")
    if np.any((pA <= 0) | (pA > 1)):
        raise ValueError("propensity scores must be in (0,1]")
    probsB = policyB.action_probs(df)
    r = df[target].values
    aA = df["a_A"].values

    # DM‑часть: ожидание модели
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

    mu_taken = mu_hat_predict(mu_model, df, aA, target)
    adj = w * (r - mu_taken)
    value = float(np.mean(dm_part + adj))
    return value, ess(w), clip_share


def ate_from_values(vB: float, vA: float) -> float:
    """Разность двух значений (ATE = V(B) - V(A))."""
    return float(vB - vA)
