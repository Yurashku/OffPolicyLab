"""Internal nuisance-model helpers for OPE estimators and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from policyscope.estimators import (
    mu_hat_predict,
    pi_hat_predict,
    prepare_piB_taken,
    train_mu_hat,
    train_pi_hat,
)


@dataclass(frozen=True)
class BehaviorPredictions:
    """Precomputed behavior-side nuisance predictions aligned with ``df`` rows."""

    pA_taken: np.ndarray
    piB_taken: np.ndarray
    pA_all: Optional[np.ndarray] = None
    is_out_of_fold: bool = False
    fold_index: Optional[np.ndarray] = None


@dataclass(frozen=True)
class OutcomePredictions:
    """Precomputed outcome predictions aligned with ``df`` rows."""

    mu_logged_action: np.ndarray
    mu_by_action: Optional[dict[object, np.ndarray]] = None
    target: Optional[str] = None
    is_out_of_fold: bool = False
    fold_index: Optional[np.ndarray] = None


@dataclass(frozen=True)
class CrossFitNuisanceBundle:
    """Container for fold-aware nuisance predictions for future cross-fitting."""

    behavior: Optional[BehaviorPredictions] = None
    outcome: Optional[OutcomePredictions] = None
    n_splits: Optional[int] = None
    notes: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class BehaviorNuisanceBundle:
    """Behavior-side fitted objects reused across estimators/diagnostics."""

    pi_model: object
    predictions: BehaviorPredictions


@dataclass(frozen=True)
class OutcomeNuisanceBundle:
    """Outcome-model fitted object reused by DM/DR-family estimators."""

    mu_model: object




def _take_logged_probabilities_with_safe_default(
    probs: np.ndarray,
    actions: np.ndarray,
    *,
    action_space: Sequence,
    eps: float = 1e-6,
) -> np.ndarray:
    idx_map = {a: i for i, a in enumerate(action_space)}
    out = np.full(len(actions), eps, dtype=float)
    for i, action in enumerate(actions):
        idx = idx_map.get(action)
        if idx is not None:
            out[i] = probs[i, idx]
    return out

def _validate_length(name: str, values: np.ndarray, n: int) -> None:
    if len(values) != n:
        raise ValueError(f"{name} must have length {n}, got {len(values)}")


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
    pA_taken = _take_logged_probabilities_with_safe_default(
        pA_all,
        df[action_col].to_numpy(),
        action_space=pi_model.classes_,
    )
    piB_taken = prepare_piB_taken(df, policyB, action_col=action_col, action_space=action_space)
    preds = BehaviorPredictions(pA_taken=pA_taken, piB_taken=piB_taken, pA_all=pA_all)
    return BehaviorNuisanceBundle(pi_model=pi_model, predictions=preds)


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


def make_kfold_indices(
    n_samples: int,
    *,
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int = 123,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create train/holdout indices for fold-based nuisance prediction."""
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    idx = np.arange(n_samples)
    return [(train_idx, holdout_idx) for train_idx, holdout_idx in kf.split(idx)]


def generate_oof_behavior_predictions(
    df: pd.DataFrame,
    policyB,
    *,
    n_splits: int = 5,
    random_state: int = 123,
    feature_cols: Optional[Sequence[str]] = None,
    action_col: str = "a_A",
    action_space: Optional[Sequence] = None,
) -> BehaviorPredictions:
    """Generate out-of-fold behavior predictions aligned to the original rows."""
    n = len(df)
    pA_taken = np.zeros(n, dtype=float)
    piB_taken = np.zeros(n, dtype=float)
    fold_index = np.full(n, -1, dtype=int)

    for fold_id, (train_idx, holdout_idx) in enumerate(
        make_kfold_indices(n, n_splits=n_splits, random_state=random_state)
    ):
        train_df = df.iloc[train_idx]
        holdout_df = df.iloc[holdout_idx]

        pi_model = train_pi_hat(train_df, feature_cols=feature_cols, action_col=action_col)
        pA_all_holdout = pi_hat_predict(pi_model, holdout_df)
        pA_taken_holdout = _take_logged_probabilities_with_safe_default(
            pA_all_holdout,
            holdout_df[action_col].to_numpy(),
            action_space=pi_model.classes_,
        )
        piB_taken_holdout = prepare_piB_taken(
            holdout_df,
            policyB,
            action_col=action_col,
            action_space=action_space,
        )

        pA_taken[holdout_idx] = pA_taken_holdout
        piB_taken[holdout_idx] = piB_taken_holdout
        fold_index[holdout_idx] = fold_id

    return BehaviorPredictions(
        pA_taken=pA_taken,
        piB_taken=piB_taken,
        pA_all=None,
        is_out_of_fold=True,
        fold_index=fold_index,
    )


def generate_oof_outcome_predictions(
    df: pd.DataFrame,
    *,
    target: str,
    n_splits: int = 5,
    random_state: int = 123,
    feature_cols: Optional[Sequence[str]] = None,
    action_col: str = "a_A",
    requested_actions: Optional[Sequence] = None,
) -> OutcomePredictions:
    """Generate out-of-fold outcome predictions for logged and optional requested actions."""
    n = len(df)
    mu_logged = np.zeros(n, dtype=float)
    fold_index = np.full(n, -1, dtype=int)
    mu_by_action: dict[object, np.ndarray] = {}
    if requested_actions is not None:
        mu_by_action = {a: np.zeros(n, dtype=float) for a in requested_actions}

    for fold_id, (train_idx, holdout_idx) in enumerate(
        make_kfold_indices(n, n_splits=n_splits, random_state=random_state)
    ):
        train_df = df.iloc[train_idx]
        holdout_df = df.iloc[holdout_idx]
        mu_model = train_mu_hat(train_df, target=target, feature_cols=feature_cols, action_col=action_col)

        logged_actions = holdout_df[action_col].to_numpy()
        mu_logged[holdout_idx] = mu_hat_predict(mu_model, holdout_df, logged_actions, target)
        for action, values in mu_by_action.items():
            values[holdout_idx] = mu_hat_predict(mu_model, holdout_df, np.full(len(holdout_df), action, dtype=object), target)
        fold_index[holdout_idx] = fold_id

    return OutcomePredictions(
        mu_logged_action=mu_logged,
        mu_by_action=mu_by_action or None,
        target=target,
        is_out_of_fold=True,
        fold_index=fold_index,
    )


def validate_behavior_predictions(predictions: BehaviorPredictions, n: int) -> None:
    """Validate that precomputed behavior predictions align with current data length."""
    _validate_length("pA_taken", predictions.pA_taken, n)
    _validate_length("piB_taken", predictions.piB_taken, n)
