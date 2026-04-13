"""Diagnostics for trust/stability in contextual bandit OPE."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from policyscope.estimators import ess
from policyscope.nuisance import (
    BehaviorPredictions,
    PropensitySource,
    resolve_behavior_predictions,
    validate_behavior_predictions,
)


@dataclass(frozen=True)
class WeightDiagnostics:
    ess: Optional[float]
    ess_ratio: Optional[float]
    max_weight: Optional[float]
    p95_weight: Optional[float]
    p99_weight: Optional[float]
    clip_share: Optional[float] = None
    switch_share: Optional[float] = None


@dataclass(frozen=True)
class OverlapDiagnostics:
    replay_overlap: float
    replay_coverage: float


@dataclass(frozen=True)
class PolicyDiagnostics:
    method: str
    n_samples: int
    overlap: OverlapDiagnostics
    weights: WeightDiagnostics
    warnings: tuple[str, ...] = field(default_factory=tuple)
    propensity_source: Optional[str] = None
    propensity_column: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "n_samples": self.n_samples,
            "replay_overlap": self.overlap.replay_overlap,
            "replay_coverage": self.overlap.replay_coverage,
            "weight_ess": self.weights.ess,
            "weight_ess_ratio": self.weights.ess_ratio,
            "weight_max": self.weights.max_weight,
            "weight_p95": self.weights.p95_weight,
            "weight_p99": self.weights.p99_weight,
            "clip_share": self.weights.clip_share,
            "switch_share": self.weights.switch_share,
            "warnings": list(self.warnings),
            "propensity_source": self.propensity_source,
            "propensity_column": self.propensity_column,
        }


def _weight_diagnostics(
    weights: Optional[np.ndarray],
    n: int,
    *,
    clip_share: Optional[float] = None,
    switch_share: Optional[float] = None,
) -> WeightDiagnostics:
    if weights is None or len(weights) == 0:
        return WeightDiagnostics(None, None, None, None, None, clip_share=clip_share, switch_share=switch_share)
    w = np.asarray(weights, dtype=float)
    return WeightDiagnostics(
        ess=float(ess(w)),
        ess_ratio=float(ess(w) / max(n, 1)),
        max_weight=float(np.max(w)),
        p95_weight=float(np.percentile(w, 95)),
        p99_weight=float(np.percentile(w, 99)),
        clip_share=clip_share,
        switch_share=switch_share,
    )


def compute_policy_diagnostics(
    df: pd.DataFrame,
    policyB,
    *,
    method: str,
    target: str = "accept",
    feature_cols: Optional[Sequence[str]] = None,
    action_col: str = "a_A",
    action_space: Optional[Sequence] = None,
    weight_clip: Optional[float] = None,
    tau: float = 20.0,
    ess_ratio_warn_threshold: float = 0.1,
    replay_overlap_warn_threshold: float = 0.05,
    max_weight_warn_threshold: float = 50.0,
    p99_weight_warn_threshold: float = 20.0,
    clip_share_warn_threshold: float = 0.2,
    switch_share_warn_threshold: float = 0.2,
    behavior_predictions: Optional[BehaviorPredictions] = None,
    propensity_source: PropensitySource = "auto",
    propensity_col: Optional[str] = None,
) -> PolicyDiagnostics:
    n = int(len(df))
    probsB = policyB.action_probs(df)
    actions = np.asarray(action_space if action_space is not None else list(range(probsB.shape[1])))
    a_B = actions[np.argmax(probsB, axis=1)]
    overlap = float(np.mean(df[action_col].to_numpy() == a_B))
    overlap_diag = OverlapDiagnostics(replay_overlap=overlap, replay_coverage=overlap)

    weights = None
    clip_share = None
    switch_share = None
    used_propensity_source: Optional[str] = None
    used_propensity_col: Optional[str] = None
    if method in {"ips", "snips", "dr", "sndr", "switch_dr"}:
        if behavior_predictions is None:
            behavior_predictions, used_propensity_source, used_propensity_col, source_notes = resolve_behavior_predictions(
                df,
                policyB,
                propensity_source=propensity_source,
                propensity_col=propensity_col,
                feature_cols=feature_cols,
                action_col=action_col,
                action_space=action_space,
            )
        else:
            source_notes = ()
            used_propensity_source = behavior_predictions.propensity_source
            used_propensity_col = behavior_predictions.propensity_col
        validate_behavior_predictions(behavior_predictions, n)
        with np.errstate(divide="ignore", invalid="ignore"):
            weights = behavior_predictions.piB_taken / behavior_predictions.pA_taken
        weights = np.nan_to_num(weights, nan=0.0, posinf=np.finfo(float).max, neginf=0.0)
        if weight_clip is not None and method in {"ips", "snips", "dr", "sndr"}:
            clip_share = float(np.mean(weights > weight_clip))
        if method == "switch_dr":
            switch_share = float(np.mean(weights > tau))

    w_diag = _weight_diagnostics(weights, n=n, clip_share=clip_share, switch_share=switch_share)
    warnings: list[str] = list(source_notes) if method in {"ips", "snips", "dr", "sndr", "switch_dr"} else []
    if overlap_diag.replay_overlap < replay_overlap_warn_threshold:
        warnings.append("low_replay_overlap")
    if w_diag.ess_ratio is not None and w_diag.ess_ratio < ess_ratio_warn_threshold:
        warnings.append("low_ess_ratio")
    if w_diag.max_weight is not None and w_diag.max_weight > max_weight_warn_threshold:
        warnings.append("extreme_max_weight")
    if w_diag.p99_weight is not None and w_diag.p99_weight > p99_weight_warn_threshold:
        warnings.append("heavy_weight_tail")
    if w_diag.clip_share is not None and w_diag.clip_share > clip_share_warn_threshold:
        warnings.append("large_clip_share")
    if w_diag.switch_share is not None and w_diag.switch_share > switch_share_warn_threshold:
        warnings.append("large_switch_share")

    return PolicyDiagnostics(
        method=method,
        n_samples=n,
        overlap=overlap_diag,
        weights=w_diag,
        warnings=tuple(warnings),
        propensity_source=used_propensity_source,
        propensity_column=used_propensity_col,
    )
