"""
policyscope.bootstrap
=====================

Функции для оценки доверительных интервалов с помощью бутстрэпа.

Используется кластеризация по пользователям (или другим единицам), чтобы
учитывать зависимость внутри кластера. Для сравнения двух политик
предусмотрена парная процедура, оценивающая интервалы одновременно для
значения A, значения B и их разности (delta).
"""

from __future__ import annotations

import pandas as pd
from typing import Callable, Tuple, Dict, Any

from policyscope.inference import infer_policy_comparison_bootstrap, infer_scalar_bootstrap

__all__ = ["cluster_bootstrap_ci", "paired_bootstrap_ci"]


def cluster_bootstrap_ci(
    df: pd.DataFrame,
    estimator: Callable[[pd.DataFrame], float],
    cluster_col: str | None = "user_id",
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
    out = infer_scalar_bootstrap(
        df,
        estimator,
        cluster_col=cluster_col,
        n_boot=n_boot,
        alpha=alpha,
        rng_seed=rng_seed,
    )
    low, high = out["CI"]
    return float(out["value"]), float(low), float(high)


def paired_bootstrap_ci(
    df: pd.DataFrame,
    estimator_pair: Callable[[pd.DataFrame], Tuple[float, float, float]],
    cluster_col: str | None = "user_id",
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
    comp = infer_policy_comparison_bootstrap(
        df,
        estimator_pair,
        cluster_col=cluster_col,
        n_boot=n_boot,
        alpha=alpha,
        rng_seed=rng_seed,
    ).to_dict()
    return {
        "V_A": comp["V_A"],
        "V_A_CI": comp["V_A_CI"],
        "V_B": comp["V_B"],
        "V_B_CI": comp["V_B_CI"],
        "Delta": comp["Delta"],
        "Delta_CI": comp["Delta_CI"],
        "p_value": comp["p_value"],
        "is_significant": comp["is_significant"],
        "significance_rule": comp["significance_rule"],
        "alpha": comp["alpha"],
        "n_boot": comp["n_boot"],
        "inference_method": comp["inference_method"],
        "inference_warnings": comp["inference_warnings"],
    }
