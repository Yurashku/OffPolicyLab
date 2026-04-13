"""Internal nuisance-model helpers for OPE estimators and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from policyscope.estimators import (
    pi_hat_predict,
    prepare_piB_taken,
    take_action_probabilities,
    train_mu_hat,
    train_pi_hat,
)


@dataclass(frozen=True)
class BehaviorNuisanceBundle:
    """Behavior-side nuisance objects reused across estimators/diagnostics."""

    pi_model: object
    pA_all: np.ndarray
    pA_taken: np.ndarray
    piB_taken: np.ndarray


@dataclass(frozen=True)
class OutcomeNuisanceBundle:
    """Outcome-model nuisance object reused by DM/DR-family estimators."""

    mu_model: object


def fit_behavior_nuisance_bundle(
    df: pd.DataFrame,
    policyB,
    *,
    feature_cols: Optional[Sequence[str]] = None,
    action_col: str = "a_A",
    action_space: Optional[Sequence] = None,
) -> BehaviorNuisanceBundle:
    """Fit behavior model and derive reusable propensity vectors."""
    pi_model = train_pi_hat(df, feature_cols=feature_cols, action_col=action_col)
    pA_all = pi_hat_predict(pi_model, df)
    pA_taken = take_action_probabilities(pA_all, df[action_col].to_numpy(), action_space=pi_model.classes_)
    piB_taken = prepare_piB_taken(df, policyB, action_col=action_col, action_space=action_space)
    return BehaviorNuisanceBundle(
        pi_model=pi_model,
        pA_all=pA_all,
        pA_taken=pA_taken,
        piB_taken=piB_taken,
    )


def fit_outcome_nuisance_bundle(
    df: pd.DataFrame,
    *,
    target: str,
    feature_cols: Optional[Sequence[str]] = None,
    action_col: str = "a_A",
) -> OutcomeNuisanceBundle:
    """Fit reusable outcome model for DM/DR-family estimators."""
    mu_model = train_mu_hat(df, target=target, feature_cols=feature_cols, action_col=action_col)
    return OutcomeNuisanceBundle(mu_model=mu_model)
