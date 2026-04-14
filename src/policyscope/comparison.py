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
from policyscope.nuisance_diagnostics import NuisanceDiagnostics, compute_nuisance_diagnostics
from policyscope.nuisance import (
    CrossFitNuisanceBundle,
    PropensitySource,
    fit_crossfit_nuisance_bundle,
    resolve_behavior_predictions,
)

RECOMMENDED_ESTIMATOR = "dr"
RECOMMENDED_PROPENSITY_SOURCE_WITH_LOGGED = "auto"
RECOMMENDED_PROPENSITY_SOURCE_FALLBACK = "estimated"
RECOMMENDED_CROSSFIT_ESTIMATORS = frozenset({"dm", "dr", "sndr", "switch_dr"})


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
    info_notes: tuple[str, ...] = field(default_factory=tuple)
    diagnostic_warnings: tuple[str, ...] = field(default_factory=tuple)
    trust_notes: tuple[str, ...] = field(default_factory=tuple)
    trust_level: str = "ok"
    recommendation: Optional[str] = None
    recommended_defaults: dict[str, object] = field(default_factory=dict)
    propensity_source: Optional[str] = None
    propensity_column: Optional[str] = None
    nuisance_diagnostics: Optional[NuisanceDiagnostics] = None

    def to_dict(self) -> dict:
        out = {
            "estimator": self.estimator,
            "target": self.target,
            "V_A": self.v_a,
            "V_B": self.v_b,
            "Delta": self.delta,
            "diagnostics": self.diagnostics.to_dict() if self.diagnostics is not None else {},
            "notes": list(self.notes),
            "info_notes": list(self.info_notes),
            "diagnostic_warnings": list(self.diagnostic_warnings),
            "trust_notes": list(self.trust_notes),
            "trust_level": self.trust_level,
        }
        if self.recommendation is not None:
            out["recommendation"] = self.recommendation
        if self.recommended_defaults:
            out["recommended_defaults"] = self.recommended_defaults
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
        if self.nuisance_diagnostics is not None:
            out["nuisance_diagnostics"] = self.nuisance_diagnostics.to_dict()
        return out


def _recommended_defaults(estimator: str) -> dict[str, object]:
    return {
        "preferred_estimator_general_use": RECOMMENDED_ESTIMATOR,
        "preferred_propensity_mode_when_logged_available": RECOMMENDED_PROPENSITY_SOURCE_WITH_LOGGED,
        "preferred_propensity_fallback_when_logged_unavailable": RECOMMENDED_PROPENSITY_SOURCE_FALLBACK,
        "crossfit_recommended_for_estimator": estimator in RECOMMENDED_CROSSFIT_ESTIMATORS,
    }


def _build_trust_metadata(
    *,
    estimator: str,
    use_crossfit: bool,
    propensity_notes: tuple[str, ...],
    diagnostic_warnings: tuple[str, ...],
    inference_warnings: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[str, ...], str, Optional[str]]:
    info_notes = list(dict.fromkeys(propensity_notes))
    trust_notes: list[str] = []
    risk_score = 0
    if diagnostic_warnings:
        risk_score += len(diagnostic_warnings)
        trust_notes.append("diagnostics_warnings_present_review_weight_overlap_metrics")
    if inference_warnings:
        risk_score += len(inference_warnings)
        trust_notes.append("inference_warnings_present_ci_and_p_value_less_stable")
    if estimator in RECOMMENDED_CROSSFIT_ESTIMATORS and not use_crossfit:
        info_notes.append("crossfit_optional_recommendation_for_bias_hardening")
    if any(w in {"low_ess_ratio", "heavy_weight_tail", "extreme_max_weight"} for w in diagnostic_warnings):
        risk_score += 1
        trust_notes.append("trust_elevated_concern_unstable_importance_weights")

    trust_level = "ok"
    recommendation = None
    if risk_score >= 3:
        trust_level = "elevated_concern"
        recommendation = "Treat comparison as directional; improve overlap/weights or collect more representative logs."
    elif risk_score > 0:
        trust_level = "caution"
        recommendation = "Review diagnostics and inference warnings before making product decisions."
    return tuple(info_notes), tuple(trust_notes), trust_level, recommendation


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

    external_nuisance_bootstrap_warning = (
        "external_nuisance_not_reused_in_bootstrap_resamples_fallback_to_internal_nuisance_fit"
    )
    external_nuisance_observed = bool(
        (nuisance_bundle is not None and (nuisance_bundle.behavior is not None or nuisance_bundle.outcome is not None))
        or resolved_behavior is not None
    )
    fallback_triggered = {"value": False}
    base_index = df.index.copy()

    def _can_reuse_external_nuisance(part: pd.DataFrame) -> bool:
        # Safe reuse policy: only for the exact original row identity and order.
        # Bootstrap resamples may reorder, duplicate, or omit rows and must not
        # reuse externally supplied nuisance predictions by position.
        return part.index.equals(base_index)

    def point_on(part: pd.DataFrame) -> float:
        can_reuse = _can_reuse_external_nuisance(part)
        behavior_preds = None
        outcome_preds = None
        if can_reuse:
            if nuisance_bundle is not None and nuisance_bundle.behavior is not None:
                behavior_preds = nuisance_bundle.behavior
            elif resolved_behavior is not None:
                behavior_preds = resolved_behavior
            if nuisance_bundle is not None and nuisance_bundle.outcome is not None:
                outcome_preds = nuisance_bundle.outcome
        elif external_nuisance_observed:
            fallback_triggered["value"] = True
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
            nuisance_outcome=outcome_preds,
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

    nuisance_diag = compute_nuisance_diagnostics(
        df,
        target=target,
        estimator=estimator,
        feature_cols=feature_cols,
        action_col=action_col,
        propensity_source=diag.propensity_source or resolved_source,
        behavior_predictions=(
            nuisance_bundle.behavior if nuisance_bundle is not None and nuisance_bundle.behavior is not None else resolved_behavior
        ),
        nuisance_bundle=nuisance_bundle,
    )

    if not with_ci:
        diag_warnings = tuple(diag.warnings)
        info_notes, trust_notes, trust_level, recommendation = _build_trust_metadata(
            estimator=estimator,
            use_crossfit=use_crossfit,
            propensity_notes=propensity_notes,
            diagnostic_warnings=diag_warnings,
            inference_warnings=tuple(),
        )
        notes = tuple(dict.fromkeys(info_notes + diag_warnings + trust_notes))
        return PolicyComparisonSummary(
            estimator=estimator,
            target=target,
            v_a=float(v_a),
            v_b=float(v_b),
            delta=float(v_b - v_a),
            diagnostics=diag,
            notes=notes,
            info_notes=info_notes,
            diagnostic_warnings=diag_warnings,
            trust_notes=trust_notes,
            trust_level=trust_level,
            recommendation=recommendation,
            recommended_defaults=_recommended_defaults(estimator),
            propensity_source=diag.propensity_source or resolved_source,
            propensity_column=diag.propensity_column or resolved_propensity_col,
            nuisance_diagnostics=nuisance_diag,
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
    inference_warnings = tuple(inf.get("inference_warnings", []))
    if fallback_triggered["value"]:
        inference_warnings = inference_warnings + (external_nuisance_bootstrap_warning,)
    diag_warnings = tuple(diag.warnings)
    info_notes, trust_notes, trust_level, recommendation = _build_trust_metadata(
        estimator=estimator,
        use_crossfit=use_crossfit,
        propensity_notes=propensity_notes,
        diagnostic_warnings=diag_warnings,
        inference_warnings=inference_warnings,
    )
    notes = tuple(dict.fromkeys(info_notes + diag_warnings + inference_warnings + trust_notes))
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
        inference_warnings=inference_warnings,
        diagnostics=diag,
        notes=notes,
        info_notes=info_notes,
        diagnostic_warnings=diag_warnings,
        trust_notes=trust_notes,
        trust_level=trust_level,
        recommendation=recommendation,
        recommended_defaults=_recommended_defaults(estimator),
        propensity_source=diag.propensity_source or resolved_source,
        propensity_column=diag.propensity_column or resolved_propensity_col,
        nuisance_diagnostics=nuisance_diag,
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
