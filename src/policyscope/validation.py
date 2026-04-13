"""Synthetic simulation validation harness for contextual bandit OPE estimators."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from policyscope.comparison import compare_policies
from policyscope.policies import make_policy
from policyscope.synthetic import SynthConfig, SyntheticRecommenderEnv


@dataclass(frozen=True)
class ValidationMode:
    name: str
    propensity_source: str = "auto"
    propensity_col: Optional[str] = "propensity_A"
    use_crossfit: bool = False
    weight_clip: Optional[float] = None


@dataclass(frozen=True)
class ValidationRunRow:
    run_id: int
    seed: int
    estimator: str
    mode: str
    oracle_v_a: float
    oracle_v_b: float
    oracle_delta: float
    v_a_hat: float
    v_b_hat: float
    delta_hat: float
    v_b_bias: float
    delta_bias: float
    abs_delta_error: float
    delta_ci_low: Optional[float]
    delta_ci_high: Optional[float]
    delta_ci_covers_oracle: Optional[bool]
    is_significant: Optional[bool]
    p_value: Optional[float]
    propensity_source_used: Optional[str]
    propensity_column_used: Optional[str]
    ess_ratio: Optional[float]
    weight_p99: Optional[float]
    behavior_log_loss: Optional[float]
    outcome_log_loss: Optional[float]
    outcome_rmse: Optional[float]


@dataclass(frozen=True)
class ValidationExperimentResult:
    config: dict
    run_rows: list[ValidationRunRow]
    aggregate: pd.DataFrame

    def to_dict(self) -> dict:
        return {
            "config": self.config,
            "run_rows": [asdict(r) for r in self.run_rows],
            "aggregate": self.aggregate.to_dict(orient="records"),
        }


def _attach_logged_propensity(logs: pd.DataFrame, policyA, *, action_col: str, propensity_col: str) -> pd.DataFrame:
    probsA = policyA.action_probs(logs)
    actions = logs[action_col].to_numpy().astype(int)
    logged = probsA[np.arange(len(logs)), actions]
    out = logs.copy()
    out[propensity_col] = np.clip(logged, 1e-6, 1.0)
    return out


def _aggregate_rows(rows: list[ValidationRunRow]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([asdict(r) for r in rows])
    grouped = (
        df.groupby(["mode", "estimator"], dropna=False)
        .agg(
            runs=("run_id", "count"),
            v_b_bias_mean=("v_b_bias", "mean"),
            v_b_std=("v_b_hat", "std"),
            v_b_rmse=("v_b_bias", lambda x: float(np.sqrt(np.mean(np.square(x))))),
            delta_bias_mean=("delta_bias", "mean"),
            delta_std=("delta_hat", "std"),
            delta_rmse=("delta_bias", lambda x: float(np.sqrt(np.mean(np.square(x))))),
            delta_abs_error_mean=("abs_delta_error", "mean"),
            delta_ci_coverage=("delta_ci_covers_oracle", "mean"),
            significance_rate=("is_significant", "mean"),
            mean_ess_ratio=("ess_ratio", "mean"),
            mean_weight_p99=("weight_p99", "mean"),
            mean_behavior_log_loss=("behavior_log_loss", "mean"),
            mean_outcome_log_loss=("outcome_log_loss", "mean"),
            mean_outcome_rmse=("outcome_rmse", "mean"),
        )
        .reset_index()
    )
    return grouped


def run_simulation_validation(
    *,
    target: str = "accept",
    estimators: Sequence[str] = ("replay", "ips", "snips", "dm", "dr", "sndr", "switch_dr"),
    modes: Optional[Sequence[ValidationMode]] = None,
    n_runs: int = 5,
    n_users: int = 400,
    base_seed: int = 2026,
    n_boot: int = 40,
    alpha: float = 0.05,
    feature_cols: Sequence[str] = ("loyal", "age", "risk", "income"),
    action_col: str = "a_A",
    cluster_col: Optional[str] = "user_id",
    tau: float = 20.0,
    policyA_epsilon: float = 0.1,
    policyB_tau: float = 0.8,
) -> ValidationExperimentResult:
    """Run repeated synthetic OPE validation experiments against oracle values."""
    if modes is None:
        modes = (
            ValidationMode(name="estimated_default", propensity_source="estimated"),
            ValidationMode(name="logged_auto", propensity_source="auto", propensity_col="propensity_A"),
            ValidationMode(name="crossfit_estimated", propensity_source="estimated", use_crossfit=True),
            ValidationMode(name="estimated_clip20", propensity_source="estimated", weight_clip=20.0),
        )

    rows: list[ValidationRunRow] = []
    for run_id in range(n_runs):
        seed = base_seed + run_id
        env = SyntheticRecommenderEnv(SynthConfig(n_users=n_users, seed=seed, horizon_days=30))
        X = env.sample_users()
        policyA = make_policy("epsilon_greedy", seed=seed, epsilon=policyA_epsilon)
        policyB = make_policy("softmax", seed=seed + 1, tau=policyB_tau)

        logs = env.simulate_logs_A(policyA, X)
        logs = _attach_logged_propensity(logs, policyA, action_col=action_col, propensity_col="propensity_A")

        oracle_v_a = env.oracle_value(policyA, X, metric=target, n_mc=2)
        oracle_v_b = env.oracle_value(policyB, X, metric=target, n_mc=2)
        oracle_delta = float(oracle_v_b - oracle_v_a)

        for mode in modes:
            for estimator in estimators:
                summary = compare_policies(
                    logs,
                    policyB,
                    estimator=estimator,
                    target=target,
                    feature_cols=feature_cols,
                    action_col=action_col,
                    cluster_col=cluster_col,
                    n_boot=n_boot,
                    alpha=alpha,
                    tau=tau,
                    weight_clip=mode.weight_clip,
                    with_ci=True,
                    use_crossfit=mode.use_crossfit,
                    propensity_source=mode.propensity_source,  # type: ignore[arg-type]
                    propensity_col=mode.propensity_col,
                )
                delta_ci_low = summary.delta_ci[0] if summary.delta_ci is not None else None
                delta_ci_high = summary.delta_ci[1] if summary.delta_ci is not None else None
                covers = None
                if delta_ci_low is not None and delta_ci_high is not None:
                    covers = bool(delta_ci_low <= oracle_delta <= delta_ci_high)

                diag = summary.diagnostics.to_dict() if summary.diagnostics is not None else {}
                rows.append(
                    ValidationRunRow(
                        run_id=run_id,
                        seed=seed,
                        estimator=estimator,
                        mode=mode.name,
                        oracle_v_a=float(oracle_v_a),
                        oracle_v_b=float(oracle_v_b),
                        oracle_delta=float(oracle_delta),
                        v_a_hat=float(summary.v_a),
                        v_b_hat=float(summary.v_b),
                        delta_hat=float(summary.delta),
                        v_b_bias=float(summary.v_b - oracle_v_b),
                        delta_bias=float(summary.delta - oracle_delta),
                        abs_delta_error=float(abs(summary.delta - oracle_delta)),
                        delta_ci_low=delta_ci_low,
                        delta_ci_high=delta_ci_high,
                        delta_ci_covers_oracle=covers,
                        is_significant=summary.is_significant,
                        p_value=summary.p_value,
                        propensity_source_used=summary.propensity_source,
                        propensity_column_used=summary.propensity_column,
                        ess_ratio=diag.get("weight_ess_ratio"),
                        weight_p99=diag.get("weight_p99"),
                        behavior_log_loss=(
                            summary.nuisance_diagnostics.behavior.multiclass_log_loss
                            if summary.nuisance_diagnostics is not None
                            else None
                        ),
                        outcome_log_loss=(
                            summary.nuisance_diagnostics.outcome.log_loss
                            if summary.nuisance_diagnostics is not None
                            else None
                        ),
                        outcome_rmse=(
                            summary.nuisance_diagnostics.outcome.rmse
                            if summary.nuisance_diagnostics is not None
                            else None
                        ),
                    )
                )

    aggregate = _aggregate_rows(rows)
    config = {
        "target": target,
        "estimators": list(estimators),
        "modes": [asdict(m) for m in modes],
        "n_runs": n_runs,
        "n_users": n_users,
        "base_seed": base_seed,
        "n_boot": n_boot,
        "alpha": alpha,
        "action_col": action_col,
        "cluster_col": cluster_col,
    }
    return ValidationExperimentResult(config=config, run_rows=rows, aggregate=aggregate)
