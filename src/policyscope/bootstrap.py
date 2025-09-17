"""
policyscope.bootstrap
=====================

Функции для оценки доверительных интервалов с помощью бутстрэпа.

Используется кластеризация по пользователям (или другим единицам), чтобы
учитывать зависимость внутри кластера. Для сравнения двух политик
предусмотрена парная процедура, оценивающая интервалы одновременно для
значения A, значения B и их разности (ATE).
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Callable, Dict, Tuple

import numpy as np
import pandas as pd

__all__ = ["cluster_bootstrap_ci", "paired_bootstrap_ci"]


@contextmanager
def _suppress_logging(level: int = logging.CRITICAL):
    """Временно отключает логирование ниже указанного уровня."""

    previous_disable = logging.root.manager.disable
    logging.disable(level)
    try:
        yield
    finally:
        logging.disable(previous_disable)


def cluster_bootstrap_ci(
    df: pd.DataFrame,
    estimator: Callable[[pd.DataFrame], float],
    cluster_col: str = "user_id",
    n_boot: int = 300,
    alpha: float = 0.05,
    rng_seed: int = 1234,
) -> Tuple[float, float, float]:
    """Оценивает доверительный интервал бутстрэпом по кластерам.

    Parameters
    ----------
    df : pd.DataFrame
        Исходные данные.
    estimator : Callable[[pd.DataFrame], float]
        Функция, возвращающая оценку метрики.
    cluster_col : str
        Название колонки, по которой происходит кластеризация (обычно user_id).
    n_boot : int
        Число бутстрэп-выборок.
    alpha : float
        Уровень значимости (0.05 ⇒ 95% интервал).
    rng_seed : int
        Зерно генератора случайных чисел.

    Returns
    -------
    (theta_hat, low, high)
        Оценка метрики и нижняя/верхняя границы CI.
    """
    rng = np.random.default_rng(rng_seed)
    theta_hat = float(estimator(df))
    clusters = df[cluster_col].unique()
    B = []
    for _ in range(n_boot):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        part = df[df[cluster_col].isin(sampled)].copy()
        with _suppress_logging():
            B.append(float(estimator(part)))
    low = np.percentile(B, 100 * alpha / 2)
    high = np.percentile(B, 100 * (1 - alpha / 2))
    return theta_hat, float(low), float(high)


def paired_bootstrap_ci(
    df: pd.DataFrame,
    estimator_pair: Callable[[pd.DataFrame], Tuple[float, float, float]],
    cluster_col: str = "user_id",
    n_boot: int = 300,
    alpha: float = 0.05,
    rng_seed: int = 4321,
) -> Dict[str, Any]:
    """Парный кластерный бутстрэп для сравнения двух политик.

    Функция `estimator_pair` должна возвращать тройку (V_A, V_B, Δ),
    где Δ = V_B - V_A. На каждой бутстрэп‑итерации вычисляются значения
    и их разность на переформированной выборке, затем строится интервал.

    Returns
    -------
    dict
        Содержит оценки V_A, V_B, Δ и 95% CI для каждой величины.
    """
    rng = np.random.default_rng(rng_seed)
    clusters = df[cluster_col].unique()
    vA, vB, dlt = estimator_pair(df)
    BA: list[float] = []
    BB: list[float] = []
    BD: list[float] = []
    for _ in range(n_boot):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        part = df[df[cluster_col].isin(sampled)].copy()
        with _suppress_logging():
            a, b, d = estimator_pair(part)
        BA.append(a)
        BB.append(b)
        BD.append(d)

    def ci(arr):
        return (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))

    return {
        "V_A": vA,
        "V_A_CI": ci(BA),
        "V_B": vB,
        "V_B_CI": ci(BB),
        "Delta": dlt,
        "Delta_CI": ci(BD),
        "n_boot": n_boot,
    }
