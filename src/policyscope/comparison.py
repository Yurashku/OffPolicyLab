"""Official high-level comparison/orchestration API for contextual bandit OPE."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import pandas as pd

from policyscope.ci import estimate_value
from policyscope.data import LoggedBanditDataset
from policyscope.diagnostics import compute_policy_diagnostics, PolicyDiagnostics
from policyscope.inference import infer_policy_comparison_bootstrap
from policyscope.estimators import value_on_policy
from policyscope.nuisance import (
    CrossFitNuisanceBundle,
    PropensitySource,
    fit_crossfit_nuisance_bundle,
    resolve_behavior_predictions,
)


@dataclass(frozen=True)
class PolicyValueResult:
    name: str
    value: float
    ci: Optional[tuple[float, float]] = None


@dataclass(frozen=True)
class PolicyComparisonSummary:
    estimator: str
    target: str
    v_a: float
    v_b: float
    delta: float
    v_a_ci: Optional[tuple[float, float]] = None
    v_b_ci: Optional[tuple[float, float]] = None
    delta_ci: Optional[tuple[float, float]] = None
    p_value: Optional[float] = None
    is_significant: Optional[bool] = None
    significance_rule: Optional[str] = None
    alpha: Optional[float] = None
    n_boot: Optional[int] = None
    inference_method: Optional[str] = None
    inference_warnings: tuple[str, ...] = field(default_factory=tuple)
    diagnostics: PolicyDiagnostics | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)
    propensity_source: Optional[str] = None
    propensity_column: Optional[str] = None

    def to_dict(self) -> dict:
        out = {
            "estimator": self.estimator,
            "target": self.target,
            "V_A": self.v_a,
            "V_B": self.v_b,
            "Delta": self.delta,
            "diagnostics": self.diagnostics.to_dict() if self.diagnostics is not None else {},
            "notes": list(self.notes),
        }
        if self.v_a_ci is not None:
            out["V_A_CI"] = self.v_a_ci
        if self.v_b_ci is not None:
            out["V_B_CI"] = self.v_b_ci
        if self.delta_ci is not None:
            out["Delta_CI"] = self.delta_ci
        if self.p_value is not None:
            out["p_value"] = self.p_value
        if self.is_significant is not None:
            out["is_significant"] = self.is_significant
        if self.significance_rule is not None:
            out["significance_rule"] = self.significance_rule
        if self.alpha is not None:
            out["alpha"] = self.alpha
        if self.n_boot is not None:
            out["n_boot"] = self.n_boot
        if self.inference_method is not None:
            out["inference_method"] = self.inference_method
        if self.inference_warnings:
            out["inference_warnings"] = list(self.inference_warnings)
        if self.propensity_source is not None:
            out["propensity_source"] = self.propensity_source
        if self.propensity_column is not None:
            out["propensity_column"] = self.propensity_column
        return out


@dataclass(frozen=True)
class MultiMetricComparisonResult:
    estimator: str
    results: dict[str, PolicyComparisonSummary]

    def to_dict(self) -> dict:
        return {
            "estimator": self.estimator,
            "results": {k: v.to_dict() for k, v in self.results.items()},
        }



def _unwrap_dataset_input(
    df_or_dataset: pd.DataFrame | LoggedBanditDataset,
    *,
    target: str,
    action_col: str,
    cluster_col: Optional[str],
    propensity_col: Optional[str],
    feature_cols: Optional[Sequence[str]],
) -> tuple[pd.DataFrame, str, str, Optional[str], Optional[str], Optional[Sequence[str]]]:
    if isinstance(df_or_dataset, LoggedBanditDataset):
        ds = df_or_dataset
        return (
            ds.df,
            ds.schema.reward_col,
            ds.schema.action_col,
            ds.schema.cluster_col if cluster_col == "user_id" else cluster_col,
            ds.schema.propensity_col if propensity_col is None else propensity_col,
            ds.schema.feature_cols if feature_cols is None else feature_cols,
        )
    return df_or_dataset, target, action_col, cluster_col, propensity_col, feature_cols


def compare_policies(
    df: pd.DataFrame | LoggedBanditDataset,
    policyB,
    *,
    estimator: str = "dr",
    target: str = "accept",
    feature_cols: Optional[Sequence[str]] = None,
    action_col: str = "a_A",
    action_space: Optional[Sequence] = None,
    cluster_col: Optional[str] = "user_id",
    n_boot: int = 300,
    alpha: float = 0.05,
    weight_clip: Optional[float] = None,
    tau: float = 20.0,
    with_ci: bool = True,
    nuisance_bundle: Optional[CrossFitNuisanceBundle] = None,
    use_crossfit: bool = False,
    crossfit_n_splits: int = 5,
    crossfit_random_state: int = 123,
    propensity_source: PropensitySource = "auto",
    propensity_col: Optional[str] = None,
) -> PolicyComparisonSummary:
    df, target, action_col, cluster_col, propensity_col, feature_cols = _unwrap_dataset_input(
        df,
        target=target,
        action_col=action_col,
        cluster_col=cluster_col,
        propensity_col=propensity_col,
        feature_cols=feature_cols,
    )

    if use_crossfit and nuisance_bundle is None:
        nuisance_bundle = fit_crossfit_nuisance_bundle(
            df,
            policyB,
            target=target,
            n_splits=crossfit_n_splits,
            random_state=crossfit_random_state,
            feature_cols=feature_cols,
            action_col=action_col,
            action_space=action_space,
        )

    resolved_behavior = None
    resolved_source: Optional[str] = None
    resolved_propensity_col: Optional[str] = None
    propensity_notes: tuple[str, ...] = tuple()
    if estimator in {"ips", "snips", "dr", "sndr", "switch_dr"} and (
        nuisance_bundle is None or nuisance_bundle.behavior is None
    ):
        resolved_behavior, resolved_source, resolved_propensity_col, propensity_notes = resolve_behavior_predictions(
            df,
            policyB,
            propensity_source=propensity_source,
            propensity_col=propensity_col,
            feature_cols=feature_cols,
            action_col=action_col,
            action_space=action_space,
        )

    def point_on(part: pd.DataFrame) -> float:
        behavior_preds = None
        if nuisance_bundle is not None and nuisance_bundle.behavior is not None and len(part) == len(df):
            behavior_preds = nuisance_bundle.behavior
        elif resolved_behavior is not None and len(part) == len(df):
            behavior_preds = resolved_behavior
        return estimate_value(
            part,
            policyB,
            method=estimator,  # type: ignore[arg-type]
            target=target,
            feature_cols=feature_cols,
            action_col=action_col,
            action_space=action_space,
            weight_clip=weight_clip,
            tau=tau,
            nuisance_behavior=behavior_preds,
            nuisance_outcome=(
                nuisance_bundle.outcome
                if nuisance_bundle is not None and nuisance_bundle.outcome is not None and len(part) == len(df)
                else None
            ),
            propensity_source=propensity_source,
            propensity_col=propensity_col,
        )

    v_a = value_on_policy(df, target=target)
    v_b = point_on(df)
    diag = compute_policy_diagnostics(
        df,
        policyB,
        method=estimator,
        target=target,
        feature_cols=feature_cols,
        action_col=action_col,
        action_space=action_space,
        weight_clip=weight_clip,
        tau=tau,
        behavior_predictions=(
            nuisance_bundle.behavior if nuisance_bundle is not None and nuisance_bundle.behavior is not None else resolved_behavior
        ),
        propensity_source=propensity_source,
        propensity_col=propensity_col,
    )

    if not with_ci:
        return PolicyComparisonSummary(
            estimator=estimator,
            target=target,
            v_a=float(v_a),
            v_b=float(v_b),
            delta=float(v_b - v_a),
            diagnostics=diag,
            notes=propensity_notes + tuple(diag.warnings),
            propensity_source=diag.propensity_source or resolved_source,
            propensity_column=diag.propensity_column or resolved_propensity_col,
        )

    def estimator_pair(part: pd.DataFrame):
        a = value_on_policy(part, target=target)
        b = point_on(part)
        return a, b, b - a

    inf = infer_policy_comparison_bootstrap(
        df,
        estimator_pair,
        cluster_col=cluster_col,
        n_boot=n_boot,
        alpha=alpha,
    ).to_dict()
    notes = propensity_notes + tuple(diag.warnings) + tuple(inf.get("inference_warnings", []))
    return PolicyComparisonSummary(
        estimator=estimator,
        target=target,
        v_a=float(inf["V_A"]),
        v_b=float(inf["V_B"]),
        delta=float(inf["Delta"]),
        v_a_ci=inf["V_A_CI"],
        v_b_ci=inf["V_B_CI"],
        delta_ci=inf["Delta_CI"],
        p_value=inf.get("p_value"),
        is_significant=inf.get("is_significant"),
        significance_rule=inf.get("significance_rule"),
        alpha=inf.get("alpha"),
        n_boot=inf.get("n_boot"),
        inference_method=inf.get("inference_method"),
        inference_warnings=tuple(inf.get("inference_warnings", [])),
        diagnostics=diag,
        notes=notes,
        propensity_source=diag.propensity_source or resolved_source,
        propensity_column=diag.propensity_column or resolved_propensity_col,
    )


def compare_policies_multi_target(
    df: pd.DataFrame | LoggedBanditDataset,
    policyB,
    *,
    estimator: str = "dr",
    targets: Sequence[str],
    feature_cols: Optional[Sequence[str]] = None,
    action_col: str = "a_A",
    action_space: Optional[Sequence] = None,
    cluster_col: Optional[str] = "user_id",
    n_boot: int = 300,
    alpha: float = 0.05,
    weight_clip: Optional[float] = None,
    tau: float = 20.0,
    with_ci: bool = True,
    nuisance_bundle: Optional[CrossFitNuisanceBundle] = None,
    use_crossfit: bool = False,
    crossfit_n_splits: int = 5,
    crossfit_random_state: int = 123,
    propensity_source: PropensitySource = "auto",
    propensity_col: Optional[str] = None,
) -> MultiMetricComparisonResult:
    if isinstance(df, LoggedBanditDataset):
        base_df = df.df
        base_action_col = df.schema.action_col if action_col == "a_A" else action_col
        base_cluster_col = df.schema.cluster_col if cluster_col == "user_id" else cluster_col
        base_propensity_col = df.schema.propensity_col if propensity_col is None else propensity_col
        base_feature_cols = df.schema.feature_cols if feature_cols is None else feature_cols
    else:
        base_df = df
        base_action_col = action_col
        base_cluster_col = cluster_col
        base_propensity_col = propensity_col
        base_feature_cols = feature_cols

    results: dict[str, PolicyComparisonSummary] = {}
    for target in targets:
        results[target] = compare_policies(
            base_df,
            policyB,
            estimator=estimator,
            target=target,
            feature_cols=base_feature_cols,
            action_col=base_action_col,
            action_space=action_space,
            cluster_col=base_cluster_col,
            n_boot=n_boot,
            alpha=alpha,
            weight_clip=weight_clip,
            tau=tau,
            with_ci=with_ci,
            nuisance_bundle=nuisance_bundle,
            use_crossfit=use_crossfit,
            crossfit_n_splits=crossfit_n_splits,
            crossfit_random_state=crossfit_random_state,
            propensity_source=propensity_source,
            propensity_col=base_propensity_col,
        )
    return MultiMetricComparisonResult(estimator=estimator, results=results)
