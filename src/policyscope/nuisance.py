"""Internal nuisance-model helpers for OPE estimators and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

PropensitySource = Literal["auto", "logged", "estimated"]

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
    propensity_source: Optional[str] = None
    propensity_col: Optional[str] = None
    action_space: Optional[tuple[object, ...]] = None


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
    preds = BehaviorPredictions(
        pA_taken=pA_taken,
        piB_taken=piB_taken,
        pA_all=pA_all,
        propensity_source="estimated",
        action_space=tuple(pi_model.classes_),
    )
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
        propensity_source="estimated",
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




def _validate_logged_propensity_values(p: pd.Series, col: str) -> None:
    if p.isna().any():
        raise ValueError(f"Propensity column '{col}' contains NaN values")
    invalid = (~((p > 0.0) & (p <= 1.0))).any()
    if invalid:
        raise ValueError(f"Propensity column '{col}' must be in (0, 1]")


def build_logged_behavior_predictions(
    df: pd.DataFrame,
    policyB,
    *,
    propensity_col: str,
    action_col: str = "a_A",
    action_space: Optional[Sequence] = None,
) -> BehaviorPredictions:
    """Build behavior predictions from logged propensity column."""
    if propensity_col not in df.columns:
        raise ValueError(f"Missing propensity column: {propensity_col}")
    _validate_logged_propensity_values(df[propensity_col], propensity_col)
    piB_taken = prepare_piB_taken(df, policyB, action_col=action_col, action_space=action_space)
    return BehaviorPredictions(
        pA_taken=df[propensity_col].to_numpy(dtype=float),
        piB_taken=piB_taken,
        propensity_source="logged",
        propensity_col=propensity_col,
    )


def resolve_behavior_predictions(
    df: pd.DataFrame,
    policyB,
    *,
    propensity_source: PropensitySource = "auto",
    propensity_col: Optional[str] = None,
    feature_cols: Optional[Sequence[str]] = None,
    action_col: str = "a_A",
    action_space: Optional[Sequence] = None,
) -> tuple[BehaviorPredictions, str, Optional[str], tuple[str, ...]]:
    """Resolve behavior predictions from logged or estimated propensity path."""
    notes: list[str] = []
    if propensity_source == "logged":
        if propensity_col is None:
            raise ValueError("propensity_source='logged' requires propensity_col")
        preds = build_logged_behavior_predictions(
            df,
            policyB,
            propensity_col=propensity_col,
            action_col=action_col,
            action_space=action_space,
        )
        return preds, "logged", propensity_col, tuple(notes)

    if propensity_source == "estimated":
        bundle = fit_behavior_nuisance_bundle(
            df,
            policyB,
            feature_cols=feature_cols,
            action_col=action_col,
            action_space=action_space,
        )
        notes.append("estimated_propensity_from_behavior_model")
        return bundle.predictions, "estimated", None, tuple(notes)

    # auto
    if propensity_col is not None and propensity_col in df.columns:
        try:
            preds = build_logged_behavior_predictions(
                df,
                policyB,
                propensity_col=propensity_col,
                action_col=action_col,
                action_space=action_space,
            )
            notes.append("auto_selected_logged_propensity")
            return preds, "logged", propensity_col, tuple(notes)
        except ValueError:
            notes.append("logged_propensity_invalid_fallback_to_estimated")

    if propensity_col is None or propensity_col not in df.columns:
        notes.append("logged_propensity_missing_fallback_to_estimated")
    bundle = fit_behavior_nuisance_bundle(
        df,
        policyB,
        feature_cols=feature_cols,
        action_col=action_col,
        action_space=action_space,
    )
    return bundle.predictions, "estimated", None, tuple(notes)

def validate_behavior_predictions(predictions: BehaviorPredictions, n: int) -> None:
    """Validate that precomputed behavior predictions align with current data length."""
    _validate_length("pA_taken", predictions.pA_taken, n)
    _validate_length("piB_taken", predictions.piB_taken, n)


def validate_outcome_predictions(
    predictions: OutcomePredictions,
    n: int,
    *,
    required_actions: Optional[Sequence] = None,
) -> None:
    """Validate that precomputed outcome predictions align with current data length."""
    _validate_length("mu_logged_action", predictions.mu_logged_action, n)
    if required_actions is None:
        return
    if predictions.mu_by_action is None:
        raise ValueError("Outcome predictions must include mu_by_action for DM/DR-family cross-fitting path")
    for action in required_actions:
        if action not in predictions.mu_by_action:
            raise ValueError(f"Outcome predictions missing action {action}")
        _validate_length(f"mu_by_action[{action}]", predictions.mu_by_action[action], n)


def fit_crossfit_nuisance_bundle(
    df: pd.DataFrame,
    policyB,
    *,
    target: str,
    n_splits: int = 5,
    random_state: int = 123,
    feature_cols: Optional[Sequence[str]] = None,
    action_col: str = "a_A",
    action_space: Optional[Sequence] = None,
) -> CrossFitNuisanceBundle:
    """Fit OOF nuisance predictions and return a single structured bundle."""
    probsB = policyB.action_probs(df)
    actions = list(action_space) if action_space is not None else list(range(probsB.shape[1]))
    behavior = generate_oof_behavior_predictions(
        df,
        policyB,
        n_splits=n_splits,
        random_state=random_state,
        feature_cols=feature_cols,
        action_col=action_col,
        action_space=action_space,
    )
    outcome = generate_oof_outcome_predictions(
        df,
        target=target,
        n_splits=n_splits,
        random_state=random_state,
        feature_cols=feature_cols,
        action_col=action_col,
        requested_actions=actions,
    )
    return CrossFitNuisanceBundle(
        behavior=behavior,
        outcome=outcome,
        n_splits=n_splits,
        notes=("oof_crossfit_bundle",),
    )
