"""Unified inference layer for contextual bandit OPE."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class IntervalResult:
    low: float
    high: float


@dataclass(frozen=True)
class ComparisonInferenceResult:
    v_a_ci: IntervalResult
    v_b_ci: IntervalResult
    delta_ci: IntervalResult
    p_value: float
    is_significant: bool
    significance_rule: str
    alpha: float
    n_boot: int
    method: str = "paired_percentile_bootstrap"
    warnings: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class PolicyComparisonResult:
    v_a: float
    v_b: float
    delta: float
    inference: ComparisonInferenceResult
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "V_A": self.v_a,
            "V_A_CI": (self.inference.v_a_ci.low, self.inference.v_a_ci.high),
            "V_B": self.v_b,
            "V_B_CI": (self.inference.v_b_ci.low, self.inference.v_b_ci.high),
            "Delta": self.delta,
            "Delta_CI": (self.inference.delta_ci.low, self.inference.delta_ci.high),
            "p_value": self.inference.p_value,
            "is_significant": self.inference.is_significant,
            "significance_rule": self.inference.significance_rule,
            "alpha": self.inference.alpha,
            "n_boot": self.inference.n_boot,
            "inference_method": self.inference.method,
            "inference_warnings": list(self.inference.warnings),
            "diagnostics": self.diagnostics,
        }


def _resample_df(
    df: pd.DataFrame,
    *,
    cluster_col: Optional[str],
    rng: np.random.Generator,
) -> pd.DataFrame:
    if cluster_col is None or cluster_col not in df.columns:
        idx = rng.integers(0, len(df), size=len(df))
        return df.iloc[idx].copy()
    clusters = df[cluster_col].unique()
    sampled = rng.choice(clusters, size=len(clusters), replace=True)
    return pd.concat([df[df[cluster_col] == c] for c in sampled]).copy()


def _percentile_interval(samples: list[float], alpha: float) -> IntervalResult:
    low = float(np.percentile(samples, 100 * alpha / 2))
    high = float(np.percentile(samples, 100 * (1 - alpha / 2)))
    return IntervalResult(low=low, high=high)


def _centered_bootstrap_pvalue(delta_boot: list[float], delta_hat: float) -> float:
    """Two-sided centered paired-bootstrap p-value for H0: delta = 0."""
    arr = np.asarray(delta_boot, dtype=float)
    centered = arr - float(delta_hat)
    test_stat = abs(float(delta_hat))
    extreme = np.mean(np.abs(centered) >= test_stat)
    # Add-one smoothing for finite bootstrap samples.
    p = (extreme * len(arr) + 1.0) / (len(arr) + 1.0)
    return float(min(max(p, 0.0), 1.0))


def infer_scalar_bootstrap(
    df: pd.DataFrame,
    estimator: Callable[[pd.DataFrame], float],
    *,
    cluster_col: Optional[str] = "user_id",
    n_boot: int = 300,
    alpha: float = 0.05,
    rng_seed: int = 1234,
    method: str = "percentile_bootstrap",
) -> dict[str, Any]:
    """Official scalar inference entrypoint (point estimate + bootstrap CI)."""
    rng = np.random.default_rng(rng_seed)
    value = float(estimator(df))
    boot: list[float] = []
    for _ in range(n_boot):
        part = _resample_df(df, cluster_col=cluster_col, rng=rng)
        boot.append(float(estimator(part)))
    ci = _percentile_interval(boot, alpha=alpha)
    return {
        "value": value,
        "CI": (ci.low, ci.high),
        "alpha": alpha,
        "n_boot": n_boot,
        "cluster_col": cluster_col,
        "inference_method": method,
    }


def infer_policy_comparison_bootstrap(
    df: pd.DataFrame,
    estimator_pair: Callable[[pd.DataFrame], tuple[float, float, float]],
    *,
    cluster_col: Optional[str] = "user_id",
    n_boot: int = 300,
    alpha: float = 0.05,
    rng_seed: int = 4321,
    method: str = "paired_percentile_bootstrap",
) -> PolicyComparisonResult:
    """Official paired comparison inference entrypoint for (V_A, V_B, delta).

    Uses percentile bootstrap CIs for (V_A, V_B, delta) and a centered paired
    bootstrap two-sided test for H0: delta = 0.
    """
    rng = np.random.default_rng(rng_seed)
    v_a, v_b, delta = estimator_pair(df)
    b_a: list[float] = []
    b_b: list[float] = []
    b_d: list[float] = []
    for _ in range(n_boot):
        part = _resample_df(df, cluster_col=cluster_col, rng=rng)
        a, b, d = estimator_pair(part)
        b_a.append(float(a))
        b_b.append(float(b))
        b_d.append(float(d))

    delta_ci = _percentile_interval(b_d, alpha=alpha)
    p_value = _centered_bootstrap_pvalue(b_d, float(delta))
    warnings: list[str] = []
    if n_boot < 200:
        warnings.append("low_n_boot_p_value_may_be_unstable")

    inference = ComparisonInferenceResult(
        v_a_ci=_percentile_interval(b_a, alpha=alpha),
        v_b_ci=_percentile_interval(b_b, alpha=alpha),
        delta_ci=delta_ci,
        p_value=p_value,
        is_significant=bool(p_value < alpha),
        significance_rule="centered_paired_bootstrap_p_value_lt_alpha",
        alpha=alpha,
        n_boot=n_boot,
        method=f"{method}+centered_delta_test",
        warnings=tuple(warnings),
    )
    return PolicyComparisonResult(v_a=float(v_a), v_b=float(v_b), delta=float(delta), inference=inference)
