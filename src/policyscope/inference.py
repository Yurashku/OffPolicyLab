"""Доверительные интервалы, p-value и поправка на множественность.

Два метода:

**Аналитический** (influence function). Оценка — среднее построчных вкладов ``psi``,
поэтому её дисперсия — дисперсия среднего. Стандартная ошибка кластерно-робастная
(CR1, с поправкой ``G/(G-1)``), критические значения — распределение Стьюдента с
``G-1`` степенями свободы, где ``G`` — число кластеров (без кластеризации — строк).
Корректен, когда пропенсити известны: ошибка модели исхода в DR входит лишь вторым
порядком. Не учитывает ошибку *оценки* пропенсити — для этого случая есть bootstrap.

**Bootstrap** с переобучением nuisance-моделей на каждой реплике. Ресэмплируются
кластеры целиком, поэтому зависимость строк одного клиента сохраняется. Интервал —
percentile, p-value — центрированный (доля реплик с ``|θ* - θ̂| >= |θ̂|``).

**Romano–Wolf** — stepdown-поправка на множественность по бутстреп-распределению
максимума |t|-статистик (Romano, Wolf, 2005, 2016). Контролирует вероятность хотя бы
одного ложного открытия (FWER) и, в отличие от Holm/Bonferroni, учитывает зависимость
гипотез: метрики и кандидаты оцениваются на одних и тех же логах. Нулевое
распределение берётся из multiplier bootstrap по кластерным суммам вкладов (для
аналитического метода — без переобучения) или из тех же bootstrap-реплик.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)

__all__ = [
    "cluster_sums",
    "cluster_se",
    "t_critical",
    "analytic_summary",
    "multiplier_tstats",
    "romano_wolf",
    "bootstrap_positions",
    "cluster_members",
    "run_bootstrap",
    "percentile_interval",
    "centered_pvalue",
]


# ---------------------------------------------------------------------------
# Аналитический метод
# ---------------------------------------------------------------------------


def cluster_sums(psi: np.ndarray, clusters: np.ndarray, n_clusters: int) -> np.ndarray:
    """Суммы центрированных вкладов по кластерам: ``(G, J)``."""
    psi = np.asarray(psi, dtype=float)
    if psi.ndim == 1:
        psi = psi[:, None]
    centered = psi - psi.mean(axis=0, keepdims=True)
    return np.column_stack([
        np.bincount(clusters, weights=centered[:, j], minlength=n_clusters)
        for j in range(psi.shape[1])
    ])


def cluster_se(sums: np.ndarray, n: int) -> np.ndarray:
    """CR1-стандартная ошибка среднего по кластерным суммам."""
    g = sums.shape[0]
    if g < 2:
        return np.full(sums.shape[1], np.nan)
    return np.sqrt(g / (g - 1) * np.sum(sums**2, axis=0)) / n


def t_critical(alpha: float, n_clusters: int) -> float:
    """Двусторонний критический уровень Стьюдента с ``G-1`` степенями свободы."""
    return float(stats.t.ppf(1.0 - alpha / 2.0, df=max(n_clusters - 1, 1)))


def analytic_summary(
    psi: np.ndarray, clusters: np.ndarray, n_clusters: int, alpha: float
) -> dict[str, np.ndarray]:
    """Оценки, SE, интервалы и p-value (для H0: оценка = 0) по столбцам ``psi``.

    Returns:
        Словарь массивов длины ``J``: ``estimate, se, low, high, p_value``, а также
        ``sums`` — кластерные суммы для поправки Romano–Wolf.
    """
    psi = np.asarray(psi, dtype=float)
    if psi.ndim == 1:
        psi = psi[:, None]
    n = psi.shape[0]
    est = psi.mean(axis=0)
    sums = cluster_sums(psi, clusters, n_clusters)
    se = cluster_se(sums, n)
    tc = t_critical(alpha, n_clusters)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(se > 0, est / se, np.nan)
    p = 2.0 * stats.t.sf(np.abs(t), df=max(n_clusters - 1, 1))
    return {
        "estimate": est,
        "se": se,
        "low": est - tc * se,
        "high": est + tc * se,
        "p_value": p,
        "sums": sums,
    }


# ---------------------------------------------------------------------------
# Romano–Wolf
# ---------------------------------------------------------------------------


def multiplier_tstats(
    sums: np.ndarray, se: np.ndarray, n: int, n_draws: int, rng: np.random.Generator
) -> np.ndarray:
    """Нулевое распределение t-статистик через multiplier bootstrap: ``(n_draws, J)``.

    ``t*_j = sum_g xi_g S_gj / (n se_j)``, ``xi ~ N(0, 1)``. Совместное распределение
    сохраняет корреляцию между гипотезами — ровно то, что нужно Romano–Wolf.
    """
    g = sums.shape[0]
    scale = np.sqrt(g / max(g - 1, 1)) / (n * se)
    out = np.empty((n_draws, sums.shape[1]))
    batch = max(1, min(n_draws, 20_000_000 // max(g, 1)))
    for start in range(0, n_draws, batch):
        stop = min(start + batch, n_draws)
        xi = rng.standard_normal((stop - start, g))
        out[start:stop] = (xi @ sums) * scale
    return out


def romano_wolf(t_obs: np.ndarray, t_null: np.ndarray) -> np.ndarray:
    """Скорректированные p-value Romano–Wolf (stepdown по max |t|).

    Args:
        t_obs: Наблюдённые t-статистики ``(J,)``.
        t_null: Центрированные бутстреп-статистики ``(B, J)``.

    Returns:
        Скорректированные p-value ``(J,)``; монотонны по |t| и не меньше
        нескорректированных бутстреп-p. Для гипотез с NaN — NaN.
    """
    t_obs = np.asarray(t_obs, dtype=float)
    p_adj = np.full(t_obs.shape, np.nan)
    valid = np.flatnonzero(np.isfinite(t_obs))
    if valid.size == 0:
        return p_adj
    null_abs = np.abs(t_null[:, valid])
    null_abs = null_abs[np.isfinite(null_abs).all(axis=1)]
    n_draws = null_abs.shape[0]
    if n_draws == 0:
        return p_adj
    obs_abs = np.abs(t_obs[valid])
    order = np.argsort(-obs_abs)
    running = 0.0
    for step, j in enumerate(order):
        max_null = null_abs[:, order[step:]].max(axis=1)
        p = (1.0 + np.sum(max_null >= obs_abs[j])) / (n_draws + 1.0)
        running = max(running, p)
        p_adj[valid[j]] = running
    return p_adj


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


def bootstrap_positions(
    clusters: np.ndarray, n_clusters: int, rng: np.random.Generator,
    members: Optional[list[np.ndarray]] = None,
) -> np.ndarray:
    """Позиции строк одной реплики: кластеры выбираются с возвращением целиком."""
    if members is None:
        return rng.integers(0, len(clusters), size=len(clusters))
    picked = rng.integers(0, n_clusters, size=n_clusters)
    return np.concatenate([members[c] for c in picked])


def cluster_members(clusters: np.ndarray, n_clusters: int) -> list[np.ndarray]:
    """Позиции строк каждого кластера (считаются один раз — реплика стоит O(n))."""
    order = np.argsort(clusters, kind="stable")
    bounds = np.searchsorted(clusters[order], np.arange(n_clusters + 1))
    return [order[bounds[i]:bounds[i + 1]] for i in range(n_clusters)]


def run_bootstrap(
    replicate: Callable[[np.ndarray, int], np.ndarray],
    *,
    clusters: np.ndarray,
    n_clusters: int,
    clustered: bool,
    n_boot: int,
    seed: int,
    n_jobs: Optional[int] = None,
) -> np.ndarray:
    """Считает ``n_boot`` реплик функции ``replicate(positions, seed) -> (J,)``.

    Зёрна реплик порождаются из ``seed`` детерминированно, поэтому результат не
    зависит от ``n_jobs``. Неудавшаяся реплика (например, вырожденный фолд)
    возвращает NaN и учитывается в отчёте.
    """
    members = cluster_members(clusters, n_clusters) if clustered else None
    children = np.random.SeedSequence(seed).spawn(n_boot)

    def one(child: np.random.SeedSequence) -> Optional[np.ndarray]:
        rng = np.random.default_rng(child)
        positions = bootstrap_positions(clusters, n_clusters, rng, members)
        try:
            return np.asarray(replicate(positions, int(rng.integers(2**31 - 1))), dtype=float)
        except (ValueError, np.linalg.LinAlgError) as exc:
            logger.debug("Bootstrap-реплика отброшена: %s", exc)
            return None

    if n_jobs in (None, 1):
        results = [one(c) for c in children]
    else:
        from joblib import Parallel, delayed

        results = Parallel(n_jobs=n_jobs)(delayed(one)(c) for c in children)
    width = next((len(r) for r in results if r is not None), 0)
    out = np.full((n_boot, width), np.nan)
    for i, r in enumerate(results):
        if r is not None:
            out[i] = r
    return out


def percentile_interval(samples: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """Percentile-интервалы по столбцам, игнорируя NaN-реплики."""
    with np.errstate(all="ignore"):
        low = np.nanpercentile(samples, 100 * alpha / 2, axis=0)
        high = np.nanpercentile(samples, 100 * (1 - alpha / 2), axis=0)
    return low, high


def centered_pvalue(samples: np.ndarray, estimate: np.ndarray) -> np.ndarray:
    """Двусторонний центрированный bootstrap p-value для H0: параметр = 0."""
    finite = np.isfinite(samples)
    extreme = (np.abs(samples - estimate) >= np.abs(estimate)) & finite
    return (1.0 + extreme.sum(axis=0)) / (1.0 + finite.sum(axis=0))
