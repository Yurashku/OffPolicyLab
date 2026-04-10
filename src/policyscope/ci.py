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

from policyscope.estimators import (
    train_pi_hat,
    pi_hat_predict,
    take_action_probabilities,
    prepare_piB_taken,
    train_mu_hat,
    value_on_policy,
    replay_value,
    ips_value,
    snips_value,
    dm_value,
    dr_value,
    sndr_value,
    switch_dr_value,
)

EstimatorName = Literal["on_policy", "replay", "ips", "snips", "dm", "dr", "sndr", "switch_dr"]


__all__ = ["estimate_value", "estimate_value_with_ci"]


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
    return pd.concat([df[df[cluster_col] == c] for c in sampled], ignore_index=True)


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
) -> float:
    if method == "on_policy":
        return value_on_policy(df, target=target)

    if method == "replay":
        probsB = policyB.action_probs(df)
        actions = np.asarray(action_space if action_space is not None else list(range(probsB.shape[1])))
        a_B = actions[np.argmax(probsB, axis=1)]
        return replay_value(df, a_B, target=target, action_col=action_col)

    if method in {"ips", "snips", "dr", "sndr", "switch_dr"}:
        pi_model = train_pi_hat(df, feature_cols=feature_cols, action_col=action_col)
        pA_all = pi_hat_predict(pi_model, df)
        pA_taken = take_action_probabilities(pA_all, df[action_col].to_numpy(), action_space=pi_model.classes_)

    if method == "ips":
        piB_taken = prepare_piB_taken(df, policyB, action_col=action_col, action_space=action_space)
        v, _, _ = ips_value(df, piB_taken, pA_taken, target=target, weight_clip=weight_clip, action_col=action_col)
        return v

    if method == "snips":
        piB_taken = prepare_piB_taken(df, policyB, action_col=action_col, action_space=action_space)
        v, _, _ = snips_value(df, piB_taken, pA_taken, target=target, weight_clip=weight_clip, action_col=action_col)
        return v

    if method == "dm":
        mu_model = train_mu_hat(df, target=target, feature_cols=feature_cols, action_col=action_col)
        return dm_value(df, policyB, mu_model, target=target, action_space=action_space)

    if method == "dr":
        mu_model = train_mu_hat(df, target=target, feature_cols=feature_cols, action_col=action_col)
        v, _, _ = dr_value(
            df,
            policyB,
            mu_model,
            pA_taken,
            target=target,
            weight_clip=weight_clip,
            action_col=action_col,
            action_space=action_space,
        )
        return v

    if method == "sndr":
        mu_model = train_mu_hat(df, target=target, feature_cols=feature_cols, action_col=action_col)
        v, _, _ = sndr_value(
            df,
            policyB,
            mu_model,
            pA_taken,
            target=target,
            weight_clip=weight_clip,
            action_col=action_col,
            action_space=action_space,
        )
        return v

    if method == "switch_dr":
        mu_model = train_mu_hat(df, target=target, feature_cols=feature_cols, action_col=action_col)
        v, _, _ = switch_dr_value(
            df,
            policyB,
            mu_model,
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
):
    """Считает point-estimate и bootstrap CI для выбранного OPE-оценщика.

    По умолчанию используется percentile bootstrap (row/cluster).
    """
    rng = np.random.default_rng(rng_seed)
    point = _estimate_point(
        df,
        method=method,
        policyB=policyB,
        target=target,
        feature_cols=feature_cols,
        action_col=action_col,
        action_space=action_space,
        weight_clip=weight_clip,
        tau=tau,
    )

    boots: list[float] = []
    for _ in range(n_boot):
        part = _resample_df(df, cluster_col=cluster_col, rng=rng)
        v = _estimate_point(
            part,
            method=method,
            policyB=policyB,
            target=target,
            feature_cols=feature_cols,
            action_col=action_col,
            action_space=action_space,
            weight_clip=weight_clip,
            tau=tau,
        )
        boots.append(float(v))

    low = float(np.percentile(boots, 100 * alpha / 2))
    high = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return {
        "method": method,
        "value": float(point),
        "CI": (low, high),
        "n_boot": n_boot,
        "alpha": alpha,
        "cluster_col": cluster_col,
    }
