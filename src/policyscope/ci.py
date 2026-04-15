"""
policyscope.ci
==============

Единый слой расчёта доверительных интервалов (CI) для OPE-оценщиков.

Дизайн:
- point-estimator и inference разделены явно;
- для всех поддержанных оценщиков используется единый практический baseline:
  percentile bootstrap (по строкам или кластерам).
"""

from __future__ import annotations

from typing import Optional, Sequence, Literal

import numpy as np
import pandas as pd

from policyscope.inference import infer_scalar_bootstrap
from policyscope.estimators import (
    value_on_policy,
    replay_value,
    prepare_piB_taken,
    ips_value,
    snips_value,
    dm_value,
    dr_value,
    sndr_value,
    switch_dr_value,
)
from policyscope.nuisance import (
    BehaviorPredictions,
    OutcomePredictions,
    PropensitySource,
    fit_outcome_nuisance_bundle,
    validate_behavior_predictions,
    validate_outcome_predictions,
    resolve_behavior_predictions,
)

EstimatorName = Literal["on_policy", "replay", "ips", "snips", "dm", "dr", "sndr", "switch_dr"]


__all__ = ["estimate_value", "estimate_value_with_ci"]


def _resolve_action_values(probsB: np.ndarray, action_space: Optional[Sequence]) -> list:
    if action_space is None:
        return list(range(probsB.shape[1]))
    return list(action_space)


def _dm_dr_family_from_predictions(
    df: pd.DataFrame,
    policyB,
    *,
    outcome_predictions: OutcomePredictions,
    target: str,
    action_col: str,
    action_space: Optional[Sequence],
    pA_taken: Optional[np.ndarray],
    method: EstimatorName,
    weight_clip: Optional[float],
    tau: float,
) -> float:
    probsB = policyB.action_probs(df)
    actions = _resolve_action_values(probsB, action_space)
    validate_outcome_predictions(outcome_predictions, len(df), required_actions=actions)

    assert outcome_predictions.mu_by_action is not None
    dm_part = np.zeros(len(df), dtype=float)
    for idx, action in enumerate(actions):
        dm_part += probsB[:, idx] * outcome_predictions.mu_by_action[action]

    if method == "dm":
        return float(np.mean(dm_part))

    assert pA_taken is not None
    piB_taken = prepare_piB_taken(df, policyB, action_col=action_col, action_space=action_space)
    w = piB_taken / pA_taken
    if weight_clip is not None and method in {"dr", "sndr"}:
        w = np.minimum(w, weight_clip)

    r = df[target].to_numpy()
    mu_logged = outcome_predictions.mu_logged_action

    if method == "dr":
        return float(np.mean(dm_part + w * (r - mu_logged)))
    if method == "sndr":
        return float(np.mean(dm_part + (w / (w.mean() + 1e-12)) * (r - mu_logged)))

    mask = w <= tau
    w_sw = np.where(mask, w, 0.0)
    return float(np.mean(dm_part + w_sw * (r - mu_logged)))


def _estimate_point(
    df: pd.DataFrame,
    *,
    method: EstimatorName,
    policyB,
    target: str,
    feature_cols: Optional[Sequence[str]],
    action_col: str,
    action_space: Optional[Sequence],
    weight_clip: Optional[float],
    tau: float,
    nuisance_behavior: Optional[BehaviorPredictions] = None,
    nuisance_outcome: Optional[OutcomePredictions] = None,
    propensity_source: PropensitySource = "auto",
    propensity_col: Optional[str] = None,
) -> float:
    if method == "on_policy":
        return value_on_policy(df, target=target)

    if method == "replay":
        probsB = policyB.action_probs(df)
        actions = np.asarray(action_space if action_space is not None else list(range(probsB.shape[1])))
        a_B = actions[np.argmax(probsB, axis=1)]
        return replay_value(df, a_B, target=target, action_col=action_col)

    behavior_preds: Optional[BehaviorPredictions] = nuisance_behavior
    pA_taken: Optional[np.ndarray] = None
    if method in {"ips", "snips", "dr", "sndr", "switch_dr"}:
        if behavior_preds is None:
            behavior_preds, _, _, _ = resolve_behavior_predictions(
                df,
                policyB,
                propensity_source=propensity_source,
                propensity_col=propensity_col,
                feature_cols=feature_cols,
                action_col=action_col,
                action_space=action_space,
                target_col=target,
            )
        validate_behavior_predictions(behavior_preds, len(df))
        pA_taken = behavior_preds.pA_taken

    if method == "ips":
        assert behavior_preds is not None and pA_taken is not None
        v, _, _ = ips_value(
            df,
            behavior_preds.piB_taken,
            pA_taken,
            target=target,
            weight_clip=weight_clip,
            action_col=action_col,
        )
        return v

    if method == "snips":
        assert behavior_preds is not None and pA_taken is not None
        v, _, _ = snips_value(
            df,
            behavior_preds.piB_taken,
            pA_taken,
            target=target,
            weight_clip=weight_clip,
            action_col=action_col,
        )
        return v

    if method == "dm":
        if nuisance_outcome is not None:
            return _dm_dr_family_from_predictions(
                df,
                policyB,
                outcome_predictions=nuisance_outcome,
                target=target,
                action_col=action_col,
                action_space=action_space,
                pA_taken=None,
                method=method,
                weight_clip=weight_clip,
                tau=tau,
            )
        outcome_bundle = fit_outcome_nuisance_bundle(
            df,
            target=target,
            feature_cols=feature_cols,
            action_col=action_col,
        )
        return dm_value(df, policyB, outcome_bundle.mu_model, target=target, action_space=action_space)

    if method == "dr":
        if nuisance_outcome is not None:
            return _dm_dr_family_from_predictions(
                df,
                policyB,
                outcome_predictions=nuisance_outcome,
                target=target,
                action_col=action_col,
                action_space=action_space,
                pA_taken=pA_taken,
                method=method,
                weight_clip=weight_clip,
                tau=tau,
            )
        outcome_bundle = fit_outcome_nuisance_bundle(
            df,
            target=target,
            feature_cols=feature_cols,
            action_col=action_col,
        )
        assert pA_taken is not None
        v, _, _ = dr_value(
            df,
            policyB,
            outcome_bundle.mu_model,
            pA_taken,
            target=target,
            weight_clip=weight_clip,
            action_col=action_col,
            action_space=action_space,
        )
        return v

    if method == "sndr":
        if nuisance_outcome is not None:
            return _dm_dr_family_from_predictions(
                df,
                policyB,
                outcome_predictions=nuisance_outcome,
                target=target,
                action_col=action_col,
                action_space=action_space,
                pA_taken=pA_taken,
                method=method,
                weight_clip=weight_clip,
                tau=tau,
            )
        outcome_bundle = fit_outcome_nuisance_bundle(
            df,
            target=target,
            feature_cols=feature_cols,
            action_col=action_col,
        )
        assert pA_taken is not None
        v, _, _ = sndr_value(
            df,
            policyB,
            outcome_bundle.mu_model,
            pA_taken,
            target=target,
            weight_clip=weight_clip,
            action_col=action_col,
            action_space=action_space,
        )
        return v

    if method == "switch_dr":
        if nuisance_outcome is not None:
            return _dm_dr_family_from_predictions(
                df,
                policyB,
                outcome_predictions=nuisance_outcome,
                target=target,
                action_col=action_col,
                action_space=action_space,
                pA_taken=pA_taken,
                method=method,
                weight_clip=weight_clip,
                tau=tau,
            )
        outcome_bundle = fit_outcome_nuisance_bundle(
            df,
            target=target,
            feature_cols=feature_cols,
            action_col=action_col,
        )
        assert pA_taken is not None
        v, _, _ = switch_dr_value(
            df,
            policyB,
            outcome_bundle.mu_model,
            pA_taken,
            tau=tau,
            target=target,
            action_col=action_col,
            action_space=action_space,
        )
        return v

    raise ValueError(f"Unsupported method: {method}")


def estimate_value(
    df: pd.DataFrame,
    policyB,
    *,
    method: EstimatorName,
    target: str = "accept",
    feature_cols: Optional[Sequence[str]] = None,
    action_col: str = "a_A",
    action_space: Optional[Sequence] = None,
    weight_clip: Optional[float] = None,
    tau: float = 20.0,
    nuisance_behavior: Optional[BehaviorPredictions] = None,
    nuisance_outcome: Optional[OutcomePredictions] = None,
    propensity_source: PropensitySource = "auto",
    propensity_col: Optional[str] = None,
) -> float:
    """Считает только point-estimate для выбранного OPE-оценщика."""
    return _estimate_point(
        df,
        method=method,
        policyB=policyB,
        target=target,
        feature_cols=feature_cols,
        action_col=action_col,
        action_space=action_space,
        weight_clip=weight_clip,
        tau=tau,
        nuisance_behavior=nuisance_behavior,
        nuisance_outcome=nuisance_outcome,
        propensity_source=propensity_source,
        propensity_col=propensity_col,
    )


def estimate_value_with_ci(
    df: pd.DataFrame,
    policyB,
    *,
    method: EstimatorName,
    target: str = "accept",
    feature_cols: Optional[Sequence[str]] = None,
    action_col: str = "a_A",
    action_space: Optional[Sequence] = None,
    weight_clip: Optional[float] = None,
    tau: float = 20.0,
    cluster_col: Optional[str] = "user_id",
    n_boot: int = 300,
    alpha: float = 0.05,
    rng_seed: int = 12345,
    propensity_source: PropensitySource = "auto",
    propensity_col: Optional[str] = None,
):
    """Считает point-estimate и bootstrap CI для выбранного OPE-оценщика.

    По умолчанию используется percentile bootstrap (row/cluster).
    """
    out = infer_scalar_bootstrap(
        df,
        lambda part: _estimate_point(
            part,
            method=method,
            policyB=policyB,
            target=target,
            feature_cols=feature_cols,
            action_col=action_col,
            action_space=action_space,
            weight_clip=weight_clip,
            tau=tau,
            propensity_source=propensity_source,
            propensity_col=propensity_col,
        ),
        cluster_col=cluster_col,
        n_boot=n_boot,
        alpha=alpha,
        rng_seed=rng_seed,
    )
    low, high = out["CI"]
    return {
        "method": method,
        "value": float(out["value"]),
        "CI": (low, high),
        "n_boot": n_boot,
        "alpha": alpha,
        "cluster_col": cluster_col,
        "inference_method": out["inference_method"],
    }
