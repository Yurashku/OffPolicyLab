"""Nuisance-model quality diagnostics for behavior and outcome models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score

from policyscope.estimators import mu_hat_predict
from policyscope.nuisance import (
    BehaviorPredictions,
    CrossFitNuisanceBundle,
    OutcomePredictions,
    fit_outcome_nuisance_bundle,
)


@dataclass(frozen=True)
class BehaviorModelDiagnostics:
    applicable: bool
    propensity_source: str
    is_out_of_fold: bool
    multiclass_log_loss: Optional[float] = None
    top1_accuracy: Optional[float] = None
    mean_logged_action_prob: Optional[float] = None
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        return {
            "applicable": self.applicable,
            "propensity_source": self.propensity_source,
            "is_out_of_fold": self.is_out_of_fold,
            "multiclass_log_loss": self.multiclass_log_loss,
            "top1_accuracy": self.top1_accuracy,
            "mean_logged_action_prob": self.mean_logged_action_prob,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class OutcomeModelDiagnostics:
    applicable: bool
    target: str
    is_binary_target: bool
    is_out_of_fold: bool
    log_loss: Optional[float] = None
    brier_score: Optional[float] = None
    roc_auc: Optional[float] = None
    rmse: Optional[float] = None
    mae: Optional[float] = None
    r2: Optional[float] = None
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        return {
            "applicable": self.applicable,
            "target": self.target,
            "is_binary_target": self.is_binary_target,
            "is_out_of_fold": self.is_out_of_fold,
            "log_loss": self.log_loss,
            "brier_score": self.brier_score,
            "roc_auc": self.roc_auc,
            "rmse": self.rmse,
            "mae": self.mae,
            "r2": self.r2,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class NuisanceDiagnostics:
    behavior: BehaviorModelDiagnostics
    outcome: OutcomeModelDiagnostics
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        return {
            "behavior": self.behavior.to_dict(),
            "outcome": self.outcome.to_dict(),
            "warnings": list(self.warnings),
        }


def _compute_behavior_diagnostics(
    df: pd.DataFrame,
    *,
    action_col: str,
    behavior_predictions: Optional[BehaviorPredictions],
    propensity_source: Optional[str],
) -> BehaviorModelDiagnostics:
    if behavior_predictions is None or propensity_source not in {"estimated", "auto"}:
        return BehaviorModelDiagnostics(
            applicable=False,
            propensity_source=propensity_source or "unknown",
            is_out_of_fold=False,
            warnings=("behavior_model_not_applicable_for_logged_propensity",),
        )

    y = df[action_col].to_numpy()
    p_taken = np.clip(behavior_predictions.pA_taken, 1e-12, 1.0)
    ll = float(-np.mean(np.log(p_taken)))
    top1 = None
    if behavior_predictions.pA_all is not None:
        top1 = float(np.mean(np.argmax(behavior_predictions.pA_all, axis=1) == y))
    warnings: list[str] = []
    if ll > 1.2:
        warnings.append("weak_behavior_log_loss")
    if top1 is not None and top1 < 0.4:
        warnings.append("weak_behavior_top1_accuracy")

    return BehaviorModelDiagnostics(
        applicable=True,
        propensity_source=propensity_source or behavior_predictions.propensity_source or "estimated",
        is_out_of_fold=bool(behavior_predictions.is_out_of_fold),
        multiclass_log_loss=ll,
        top1_accuracy=top1,
        mean_logged_action_prob=float(np.mean(p_taken)),
        warnings=tuple(warnings),
    )


def _compute_outcome_diagnostics(
    df: pd.DataFrame,
    *,
    target: str,
    feature_cols: Optional[Sequence[str]],
    action_col: str,
    estimator: str,
    outcome_predictions: Optional[OutcomePredictions],
) -> OutcomeModelDiagnostics:
    if estimator not in {"dm", "dr", "sndr", "switch_dr"}:
        return OutcomeModelDiagnostics(
            applicable=False,
            target=target,
            is_binary_target=False,
            is_out_of_fold=False,
            warnings=("outcome_model_not_used_for_estimator",),
        )

    y = df[target].to_numpy()
    is_binary = np.array_equal(np.unique(y), np.array([0, 1])) or np.array_equal(np.unique(y), np.array([0.0, 1.0]))

    if outcome_predictions is None:
        mu_bundle = fit_outcome_nuisance_bundle(df, target=target, feature_cols=feature_cols, action_col=action_col)
        pred = mu_hat_predict(mu_bundle.mu_model, df, df[action_col].to_numpy(), target)
        is_oof = False
    else:
        pred = outcome_predictions.mu_logged_action
        is_oof = bool(outcome_predictions.is_out_of_fold)

    warnings: list[str] = []
    if is_binary:
        p = np.clip(pred, 1e-12, 1 - 1e-12)
        ll = float(log_loss(y, p, labels=[0, 1]))
        br = float(brier_score_loss(y, p))
        try:
            auc = float(roc_auc_score(y, p))
        except ValueError:
            auc = None
        if ll > 0.69:
            warnings.append("weak_outcome_log_loss")
        if br > 0.25:
            warnings.append("weak_outcome_brier")
        if auc is not None and auc < 0.6:
            warnings.append("weak_outcome_auc")
        return OutcomeModelDiagnostics(
            applicable=True,
            target=target,
            is_binary_target=True,
            is_out_of_fold=is_oof,
            log_loss=ll,
            brier_score=br,
            roc_auc=auc,
            warnings=tuple(warnings),
        )

    rmse = float(np.sqrt(mean_squared_error(y, pred)))
    mae = float(mean_absolute_error(y, pred))
    r2 = float(r2_score(y, pred))
    if r2 < 0.0:
        warnings.append("weak_outcome_r2")
    return OutcomeModelDiagnostics(
        applicable=True,
        target=target,
        is_binary_target=False,
        is_out_of_fold=is_oof,
        rmse=rmse,
        mae=mae,
        r2=r2,
        warnings=tuple(warnings),
    )


def compute_nuisance_diagnostics(
    df: pd.DataFrame,
    *,
    target: str,
    estimator: str,
    feature_cols: Optional[Sequence[str]],
    action_col: str,
    propensity_source: Optional[str],
    behavior_predictions: Optional[BehaviorPredictions] = None,
    nuisance_bundle: Optional[CrossFitNuisanceBundle] = None,
) -> NuisanceDiagnostics:
    """Compute structured nuisance quality diagnostics for official outputs."""
    if behavior_predictions is None and nuisance_bundle is not None:
        behavior_predictions = nuisance_bundle.behavior
    outcome_predictions = nuisance_bundle.outcome if nuisance_bundle is not None else None

    behavior = _compute_behavior_diagnostics(
        df,
        action_col=action_col,
        behavior_predictions=behavior_predictions,
        propensity_source=propensity_source,
    )
    outcome = _compute_outcome_diagnostics(
        df,
        target=target,
        feature_cols=feature_cols,
        action_col=action_col,
        estimator=estimator,
        outcome_predictions=outcome_predictions,
    )
    warnings = tuple(list(behavior.warnings) + list(outcome.warnings))
    return NuisanceDiagnostics(behavior=behavior, outcome=outcome, warnings=warnings)
