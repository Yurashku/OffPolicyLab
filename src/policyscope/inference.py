"""
policyscope.inference
=====================

Слой инференса: доверительные интервалы и значимость для ``V_A``, ``V_B`` и
``delta = V_B - V_A``.

Доступны два взаимодополняющих метода:

``bootstrap``
    Парный (кластерный) percentile bootstrap. Nuisance-модели переобучаются внутри
    каждой реплики, поэтому интервал учитывает и неопределённость оценки nuisance.
    Дороже, но консервативнее; работает для любого оценщика.

``influence``
    Аналитический интервал по построчным вкладам ``psi_i`` с кластерно-робастной
    («сэндвич») дисперсией. Один проход по данным вместо сотен переобучений.
    **Не** учитывает неопределённость nuisance: асимптотически корректен для
    DR-семейства при cross-fitting (свойство Неймановой ортогональности), но
    антиконсервативен для DM, где ошибка модели исхода входит в оценку первым порядком.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

__all__ = [
    "Interval",
    "ComparisonInference",
    "ClusterResampler",
    "cluster_robust_se",
    "infer_comparison_bootstrap",
    "infer_comparison_influence",
]

#: Ниже этого числа реплик percentile-хвосты оцениваются слишком грубо.
MIN_BOOT_FOR_CI = 500
#: Ниже этого числа независимых кластеров любой CI ненадёжен.
MIN_CLUSTERS = 20
#: Доля NaN-реплик, выше которой интервал объявляется неопределённым.
MAX_NAN_SHARE = 0.1


@dataclass(frozen=True)
class Interval:
    """Доверительный интервал."""

    low: float
    high: float

    def as_tuple(self) -> tuple[float, float]:
        return (self.low, self.high)

    def excludes(self, value: float = 0.0) -> bool:
        """Лежит ли ``value`` вне интервала."""
        if not np.isfinite(self.low) or not np.isfinite(self.high):
            return False
        return value < self.low or value > self.high


@dataclass(frozen=True)
class ComparisonInference:
    """Результат инференса для пары политик.

    Attributes:
        p_value: Двусторонний p-value для ``H0: delta = 0``.
        is_significant: Решение по p-value на уровне ``alpha``.
        ci_excludes_zero: Решение по доверительному интервалу для delta.
            Приводится отдельно, потому что это **другая процедура**: на скошенном
            распределении percentile-интервал и центрированный тест могут расходиться.
        se_delta: Стандартная ошибка delta (только для аналитического метода).
    """

    v_a: float
    v_b: float
    delta: float
    v_a_ci: Interval
    v_b_ci: Interval
    delta_ci: Interval
    p_value: float
    is_significant: bool
    ci_excludes_zero: bool
    significance_rule: str
    alpha: float
    method: str
    n_samples: int
    n_clusters: Optional[int] = None
    n_boot: Optional[int] = None
    se_delta: Optional[float] = None
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "V_A": self.v_a,
            "V_A_CI": self.v_a_ci.as_tuple(),
            "V_B": self.v_b,
            "V_B_CI": self.v_b_ci.as_tuple(),
            "Delta": self.delta,
            "Delta_CI": self.delta_ci.as_tuple(),
            "p_value": self.p_value,
            "is_significant": self.is_significant,
            "ci_excludes_zero": self.ci_excludes_zero,
            "significance_rule": self.significance_rule,
            "alpha": self.alpha,
            "inference_method": self.method,
            "n_samples": self.n_samples,
            "n_clusters": self.n_clusters,
            "n_boot": self.n_boot,
            "se_delta": self.se_delta,
            "inference_warnings": list(self.warnings),
        }


# ---------------------------------------------------------------------------
# Ресэмплинг
# ---------------------------------------------------------------------------


class ClusterResampler:
    """Кластерный (или построчный) bootstrap-ресэмплер.

    Позиции строк внутри каждого кластера вычисляются **один раз** при создании
    объекта, поэтому каждая реплика стоит ``O(n)``. Наивная реализация
    ``pd.concat([df[df[col] == c] for c in sampled])`` делает полный проход по
    таблице на каждый выбранный кластер, то есть ``O(C*n)`` на реплику, и на
    реальных объёмах не заканчивается за разумное время.
    """

    def __init__(self, df: pd.DataFrame, cluster_col: Optional[str]) -> None:
        self.n = len(df)
        self.cluster_col = cluster_col
        self._groups: Optional[list[np.ndarray]] = None

        if cluster_col is None:
            return
        if cluster_col not in df.columns:
            raise ValueError(
                f"Колонка кластеров '{cluster_col}' отсутствует в данных. "
                "Передайте существующую колонку или cluster_col=None для построчного "
                "bootstrap. Молчаливый откат недопустим: без кластеризации интервалы "
                "получаются антиконсервативными."
            )
        codes, uniques = pd.factorize(df[cluster_col], sort=False)
        order = np.argsort(codes, kind="stable")
        sorted_codes = codes[order]
        bounds = np.searchsorted(sorted_codes, np.arange(len(uniques) + 1))
        self._groups = [order[bounds[i]:bounds[i + 1]] for i in range(len(uniques))]

    @property
    def n_clusters(self) -> Optional[int]:
        return None if self._groups is None else len(self._groups)

    def positions(self, rng: np.random.Generator) -> np.ndarray:
        """Возвращает позиционные индексы одной bootstrap-реплики."""
        if self._groups is None:
            return rng.integers(0, self.n, size=self.n)
        picked = rng.integers(0, len(self._groups), size=len(self._groups))
        return np.concatenate([self._groups[i] for i in picked])

    def resample(self, df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
        """Возвращает одну bootstrap-реплику данных."""
        return df.take(self.positions(rng))


# ---------------------------------------------------------------------------
# Общие вспомогательные функции
# ---------------------------------------------------------------------------


def _percentile_interval(samples: np.ndarray, alpha: float) -> Interval:
    """Percentile-интервал, устойчивый к NaN-репликам."""
    finite = samples[np.isfinite(samples)]
    if finite.size == 0:
        return Interval(float("nan"), float("nan"))
    return Interval(
        low=float(np.percentile(finite, 100 * alpha / 2)),
        high=float(np.percentile(finite, 100 * (1 - alpha / 2))),
    )


def _centered_bootstrap_pvalue(delta_boot: np.ndarray, delta_hat: float) -> float:
    """Двусторонний центрированный bootstrap p-value для ``H0: delta = 0``.

    Считает долю реплик, где ``|b* - b_hat| >= |b_hat|``, со сглаживанием
    «плюс один» для конечного числа реплик.
    """
    finite = delta_boot[np.isfinite(delta_boot)]
    if finite.size == 0 or not np.isfinite(delta_hat):
        return float("nan")
    n_extreme = int(np.count_nonzero(np.abs(finite - delta_hat) >= abs(delta_hat)))
    return float((n_extreme + 1) / (finite.size + 1))


def cluster_robust_se(
    contributions: np.ndarray,
    *,
    clusters: Optional[Sequence] = None,
) -> float:
    """Кластерно-робастная («сэндвич») стандартная ошибка среднего от ``contributions``.

    ``SE^2 = n^-2 * sum_c (sum_{i in c} (psi_i - mean(psi)))^2``.

    При ``clusters=None`` каждая строка считается независимой, и формула сводится
    к обычной ``sd(psi)/sqrt(n)``.
    """
    psi = np.asarray(contributions, dtype=float)
    n = psi.size
    if n == 0 or not np.isfinite(psi).all():
        return float("nan")
    centered = psi - psi.mean()
    if clusters is None:
        total = float(np.sum(centered**2))
    else:
        codes, uniques = pd.factorize(pd.Series(clusters), sort=False)
        per_cluster = np.bincount(codes, weights=centered, minlength=len(uniques))
        total = float(np.sum(per_cluster**2))
    if total < 0 or not np.isfinite(total):
        return float("nan")
    return float(np.sqrt(total) / n)


def _quality_warnings(
    *,
    n_clusters: Optional[int],
    n_boot: Optional[int] = None,
    nan_share: float = 0.0,
) -> list[str]:
    warns: list[str] = []
    if n_boot is not None and n_boot < MIN_BOOT_FOR_CI:
        warns.append("low_n_boot_ci_tails_unstable")
    if n_clusters is not None and n_clusters < MIN_CLUSTERS:
        warns.append("few_clusters_inference_unreliable")
    if nan_share > 0.0:
        warns.append("bootstrap_replicates_with_nan_dropped")
    return warns


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


def infer_comparison_bootstrap(
    df: pd.DataFrame,
    estimator_pair: Callable[[pd.DataFrame], tuple[float, float]],
    *,
    cluster_col: Optional[str] = "user_id",
    n_boot: int = 500,
    alpha: float = 0.05,
    rng_seed: int = 4321,
    observed: Optional[tuple[float, float]] = None,
) -> ComparisonInference:
    """Парный кластерный percentile bootstrap для ``(V_A, V_B, delta)``.

    Пара значений считается на **одной и той же** реплике, поэтому корреляция между
    ``V_A`` и ``V_B`` учитывается, и интервал для delta уже, чем разность
    независимых интервалов.

    Args:
        df: Исходные логи.
        estimator_pair: Функция ``part -> (V_A, V_B)`` на переданной выборке.
        cluster_col: Колонка кластеров; ``None`` — построчный bootstrap.
        n_boot: Число реплик.
        alpha: Уровень значимости.
        rng_seed: Зерно генератора.
        observed: Уже посчитанная пара ``(V_A, V_B)`` на исходных данных. Позволяет
            не пересчитывать точечную оценку (полный fit nuisance) второй раз.
    """
    if n_boot < 2:
        raise ValueError("n_boot должен быть не меньше 2")
    resampler = ClusterResampler(df, cluster_col)
    rng = np.random.default_rng(rng_seed)

    v_a, v_b = observed if observed is not None else estimator_pair(df)
    delta = float(v_b) - float(v_a)

    boot = np.empty((n_boot, 3), dtype=float)
    for i in range(n_boot):
        part = resampler.resample(df, rng)
        try:
            a, b = estimator_pair(part)
        except (ValueError, np.linalg.LinAlgError) as exc:
            logger.debug("Bootstrap-реплика %d отброшена: %s", i, exc)
            boot[i] = np.nan
            continue
        boot[i] = (a, b, b - a)

    nan_share = float(np.mean(~np.isfinite(boot[:, 2])))
    warns = _quality_warnings(n_clusters=resampler.n_clusters, n_boot=n_boot, nan_share=nan_share)

    if nan_share > MAX_NAN_SHARE:
        logger.warning(
            "%.0f%% bootstrap-реплик не удалось посчитать; интервалы неопределены",
            100 * nan_share,
        )
        warns.append("too_many_failed_bootstrap_replicates")
        nan_interval = Interval(float("nan"), float("nan"))
        return ComparisonInference(
            v_a=float(v_a), v_b=float(v_b), delta=delta,
            v_a_ci=nan_interval, v_b_ci=nan_interval, delta_ci=nan_interval,
            p_value=float("nan"), is_significant=False, ci_excludes_zero=False,
            significance_rule="centered_paired_bootstrap_p_value_lt_alpha",
            alpha=alpha, method="paired_percentile_bootstrap",
            n_samples=len(df), n_clusters=resampler.n_clusters, n_boot=n_boot,
            warnings=tuple(warns),
        )

    delta_ci = _percentile_interval(boot[:, 2], alpha)
    p_value = _centered_bootstrap_pvalue(boot[:, 2], delta)

    return ComparisonInference(
        v_a=float(v_a),
        v_b=float(v_b),
        delta=delta,
        v_a_ci=_percentile_interval(boot[:, 0], alpha),
        v_b_ci=_percentile_interval(boot[:, 1], alpha),
        delta_ci=delta_ci,
        p_value=p_value,
        is_significant=bool(np.isfinite(p_value) and p_value < alpha),
        ci_excludes_zero=delta_ci.excludes(0.0),
        significance_rule="centered_paired_bootstrap_p_value_lt_alpha",
        alpha=alpha,
        method="paired_percentile_bootstrap",
        n_samples=len(df),
        n_clusters=resampler.n_clusters,
        n_boot=n_boot,
        warnings=tuple(warns),
    )


# ---------------------------------------------------------------------------
# Аналитический (influence function) инференс
# ---------------------------------------------------------------------------


def infer_comparison_influence(
    contributions_b: np.ndarray,
    rewards: np.ndarray,
    *,
    clusters: Optional[Sequence] = None,
    alpha: float = 0.05,
) -> ComparisonInference:
    """Аналитический CI по построчным вкладам с кластерно-робастной дисперсией.

    ``V_A`` — среднее наблюдённой награды (вклад ``psi^A_i = r_i``), ``V_B`` — среднее
    вкладов оценщика, ``delta`` использует парный вклад ``psi^delta_i = psi^B_i - r_i``,
    поэтому корреляция между плечами учитывается точно, без ресэмплинга.

    Args:
        contributions_b: Построчные вклады оценщика для политики B.
        rewards: Наблюдённая награда.
        clusters: Значения кластерной колонки; ``None`` — независимые строки.
        alpha: Уровень значимости.
    """
    psi_b = np.asarray(contributions_b, dtype=float)
    r = np.asarray(rewards, dtype=float)
    if psi_b.shape != r.shape:
        raise ValueError("contributions_b и rewards должны быть одной длины")

    n = psi_b.size
    psi_delta = psi_b - r
    v_a, v_b = float(np.mean(r)), float(np.mean(psi_b))
    delta = float(np.mean(psi_delta))

    se_a = cluster_robust_se(r, clusters=clusters)
    se_b = cluster_robust_se(psi_b, clusters=clusters)
    se_d = cluster_robust_se(psi_delta, clusters=clusters)

    z = float(stats.norm.ppf(1 - alpha / 2))

    def interval(center: float, se: float) -> Interval:
        if not np.isfinite(se):
            return Interval(float("nan"), float("nan"))
        return Interval(center - z * se, center + z * se)

    if np.isfinite(se_d) and se_d > 0:
        p_value = float(2 * stats.norm.sf(abs(delta) / se_d))
    else:
        p_value = float("nan")

    delta_ci = interval(delta, se_d)
    n_clusters = None if clusters is None else int(pd.Series(clusters).nunique())

    return ComparisonInference(
        v_a=v_a,
        v_b=v_b,
        delta=delta,
        v_a_ci=interval(v_a, se_a),
        v_b_ci=interval(v_b, se_b),
        delta_ci=delta_ci,
        p_value=p_value,
        is_significant=bool(np.isfinite(p_value) and p_value < alpha),
        ci_excludes_zero=delta_ci.excludes(0.0),
        significance_rule="cluster_robust_normal_p_value_lt_alpha",
        alpha=alpha,
        method="influence_function_cluster_robust",
        n_samples=n,
        n_clusters=n_clusters,
        se_delta=se_d if np.isfinite(se_d) else None,
        warnings=tuple(_quality_warnings(n_clusters=n_clusters)),
    )
