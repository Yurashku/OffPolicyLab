"""Формулы оценщиков значения политики.

Каждый оценщик задаётся **построчными вкладами** ``psi_i``, среднее которых равно
оценке: ``V_hat = mean(psi)``. Из этих же вкладов строится аналитический доверительный
интервал, поэтому точечная оценка и интервал не могут разойтись.

Обозначения (для строки ``i`` с контекстом ``x_i``, действием ``a_i`` и наградой ``r_i``):

- ``w_i = pi(a_i|x_i) / p_i`` — вес важности, ``p_i`` — пропенсити политики A;
- ``dm_i = sum_a pi(a|x_i) mu(x_i, a)`` — прогноз модели исхода под политикой ``pi``;
- ``mu_i = mu(x_i, a_i)`` — прогноз для залогированного действия.

=========  ===============================================================
``ips``    ``w r``
``snips``  ``sum(w r) / sum(w)``
``dm``     ``dm``
``dr``     ``dm + w (r - mu)``
``sndr``   ``mean(dm) + sum(w (r - mu)) / sum(w)``
=========  ===============================================================

Для самонормированных оценщиков (SNIPS, SNDR) вклад — линеаризация отношения
(дельта-метод): его среднее равно оценке, а разброс даёт асимптотическую дисперсию.

Источники: Dudík, Langford, Li (2011) «Doubly Robust Policy Evaluation and Learning»;
Swaminathan, Joachims (2015) «The Self-Normalized Estimator for Counterfactual Learning»;
Saito et al. (2021) «Open Bandit Dataset and Pipeline».
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["ESTIMATORS", "PolicyTerms", "policy_terms", "contributions", "ess"]

#: Все оценщики в порядке показа в отчёте.
ESTIMATORS: tuple[str, ...] = ("dr", "sndr", "ips", "snips", "dm")
#: Самонормированные оценщики.
SELF_NORMALIZED = frozenset({"snips", "sndr"})


@dataclass(frozen=True)
class PolicyTerms:
    """Построчные величины, из которых складываются все оценщики для одной политики."""

    w: np.ndarray        # веса важности
    dm: np.ndarray       # sum_a pi(a|x) mu(x, a)
    mu_taken: np.ndarray  # mu(x, a_logged)


def policy_terms(pi: np.ndarray, a: np.ndarray, p_taken: np.ndarray, mu: np.ndarray) -> PolicyTerms:
    """Собирает веса и прогнозы модели исхода для политики ``pi``.

    Args:
        pi: Матрица политики ``(n, k)``.
        a: Коды залогированных действий ``(n,)``.
        p_taken: Пропенсити политики A для залогированных действий ``(n,)``.
        mu: Прогнозы модели исхода для каждого действия ``(n, k)``.
    """
    rows = np.arange(len(a))
    return PolicyTerms(
        w=pi[rows, a] / p_taken,
        dm=np.einsum("ij,ij->i", pi, mu),
        mu_taken=mu[rows, a],
    )


def contributions(name: str, r: np.ndarray, t: PolicyTerms) -> np.ndarray:
    """Построчные вклады ``psi`` оценщика ``name``: ``mean(psi)`` — значение оценки."""
    w, dm, resid = t.w, t.dm, r - t.mu_taken
    if name == "dm":
        return dm.copy()
    if name == "ips":
        return w * r
    if name == "dr":
        return dm + w * resid
    w_mean = float(np.mean(w))
    if w_mean <= 0.0:
        return np.full_like(r, np.nan, dtype=float)
    if name == "snips":
        value = float(np.mean(w * r)) / w_mean
        return value + w * (r - value) / w_mean
    if name == "sndr":
        correction = float(np.mean(w * resid)) / w_mean
        return dm + correction + w * (resid - correction) / w_mean
    raise ValueError(f"Неизвестный оценщик '{name}'. Доступны: {ESTIMATORS}")


def ess(w: np.ndarray) -> float:
    """Эффективный размер выборки ``(sum w)^2 / sum(w^2)`` (Kish)."""
    w = np.asarray(w, dtype=float)
    s2 = float(np.sum(w * w))
    if w.size == 0 or s2 <= 0.0 or not np.isfinite(s2):
        return 0.0
    return float(np.sum(w)) ** 2 / s2
