"""Диагностика качества nuisance-моделей (behavior ``pi_A`` и outcome ``mu``).

Дополняет диагностику весов: та показывает, устойчива ли оценка численно, а эта —
насколько правдоподобны модели, на которых оценка держится.

Все метрики считаются **out-of-fold**. In-sample метрики оптимистичны по построению
и ровно поэтому бесполезны для своей задачи — обнаружить слабую или переобученную
nuisance-модель.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

from policyscope.estimate import OUTCOME_ESTIMATORS, WEIGHTED_ESTIMATORS
from policyscope.estimators import DEFAULT_NUISANCE_SPEC, NuisanceSpec, is_binary_target
from policyscope.nuisance import (
    BehaviorPredictions,
    OutcomePredictions,
    generate_oof_outcome_predictions,
)

logger = logging.getLogger(__name__)

__all__ = [
    "NuisanceThresholds",
    "BehaviorModelDiagnostics",
    "OutcomeModelDiagnostics",
    "NuisanceDiagnostics",
    "compute_nuisance_diagnostics",
]


@dataclass(frozen=True)
class NuisanceThresholds:
    """Пороги качества nuisance-моделей."""

    #: Отношение log-loss модели к log-loss равномерного предсказания (log k).
    #: Значение около 1 означает, что модель почти не лучше подбрасывания монетки.
    behavior_relative_log_loss: float = 0.95
    outcome_relative_log_loss: float = 0.95
    outcome_brier: float = 0.25
    outcome_roc_auc: float = 0.6
    outcome_r2: float = 0.0


@dataclass(frozen=True)
class BehaviorModelDiagnostics:
    """Качество модели поведения ``pi_A(a|x)``."""

    applicable: bool
    propensity_source: str
    is_out_of_fold: bool
    multiclass_log_loss: Optional[float] = None
    uniform_log_loss: Optional[float] = None
    relative_log_loss: Optional[float] = None
    top1_accuracy: Optional[float] = None
    mean_logged_action_prob: Optional[float] = None
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        return {
            "applicable": self.applicable,
            "propensity_source": self.propensity_source,
            "is_out_of_fold": self.is_out_of_fold,
            "multiclass_log_loss": self.multiclass_log_loss,
            "uniform_log_loss": self.uniform_log_loss,
            "relative_log_loss": self.relative_log_loss,
            "top1_accuracy": self.top1_accuracy,
            "mean_logged_action_prob": self.mean_logged_action_prob,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class OutcomeModelDiagnostics:
    """Качество модели исхода ``mu(x, a)``."""

    applicable: bool
    target: str
    is_binary_target: bool
    is_out_of_fold: bool
    log_loss: Optional[float] = None
    relative_log_loss: Optional[float] = None
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
            "relative_log_loss": self.relative_log_loss,
            "brier_score": self.brier_score,
            "roc_auc": self.roc_auc,
            "rmse": self.rmse,
            "mae": self.mae,
            "r2": self.r2,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class NuisanceDiagnostics:
    """Сводная диагностика обеих nuisance-сторон."""

    behavior: BehaviorModelDiagnostics
    outcome: OutcomeModelDiagnostics
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        return {
            "behavior": self.behavior.to_dict(),
            "outcome": self.outcome.to_dict(),
            "warnings": list(self.warnings),
        }


def _not_applicable_behavior(source: Optional[str], reason: str) -> BehaviorModelDiagnostics:
    return BehaviorModelDiagnostics(
        applicable=False,
        propensity_source=source or "unknown",
        is_out_of_fold=False,
        warnings=(reason,),
    )


def _compute_behavior_diagnostics(
    df: pd.DataFrame,
    *,
    action_col: str,
    behavior: Optional[BehaviorPredictions],
    estimator: str,
    thresholds: NuisanceThresholds,
) -> BehaviorModelDiagnostics:
    if estimator not in WEIGHTED_ESTIMATORS or behavior is None:
        return _not_applicable_behavior(None, "behavior_model_not_used_for_estimator")
    if behavior.propensity_source != "estimated":
        return _not_applicable_behavior(
            behavior.propensity_source, "behavior_model_not_applicable_for_logged_propensity"
        )

    y = df[action_col].to_numpy()
    n_actions = max(int(pd.Series(y).nunique()), 2)
    p_taken = np.clip(behavior.pA_taken, 1e-12, 1.0)
    ll = float(-np.mean(np.log(p_taken)))
    uniform_ll = float(np.log(n_actions))
    relative = ll / uniform_ll if uniform_ll > 0 else None

    top1 = None
    if behavior.pA_all is not None and behavior.pA_all_actions is not None:
        # argmax даёт индекс столбца, а y хранит метку действия. Сопоставление идёт
        # через pA_all_actions; прямое сравнение индекса с меткой верно только для
        # меток 0..k-1 в отсортированном порядке.
        labels = np.asarray(behavior.pA_all_actions, dtype=object)
        predicted = labels[np.argmax(behavior.pA_all, axis=1)]
        top1 = float(np.mean(predicted == y))

    warnings: list[str] = []
    if relative is not None and relative > thresholds.behavior_relative_log_loss:
        warnings.append("weak_behavior_log_loss")
    if not behavior.is_out_of_fold:
        warnings.append("behavior_metrics_in_sample_optimistic")

    return BehaviorModelDiagnostics(
        applicable=True,
        propensity_source="estimated",
        is_out_of_fold=bool(behavior.is_out_of_fold),
        multiclass_log_loss=ll,
        uniform_log_loss=uniform_ll,
        relative_log_loss=relative,
        top1_accuracy=top1,
        mean_logged_action_prob=float(np.mean(p_taken)),
        warnings=tuple(warnings),
    )


def _compute_outcome_diagnostics(
    df: pd.DataFrame,
    *,
    target: str,
    estimator: str,
    outcome: Optional[OutcomePredictions],
    thresholds: NuisanceThresholds,
) -> OutcomeModelDiagnostics:
    if estimator not in OUTCOME_ESTIMATORS or outcome is None:
        return OutcomeModelDiagnostics(
            applicable=False,
            target=target,
            is_binary_target=False,
            is_out_of_fold=False,
            warnings=("outcome_model_not_used_for_estimator",),
        )

    y = df[target].to_numpy()
    binary = is_binary_target(y)
    pred = np.asarray(outcome.mu_logged_action, dtype=float)
    warnings: list[str] = []
    if not outcome.is_out_of_fold:
        warnings.append("outcome_metrics_in_sample_optimistic")

    if binary:
        p = np.clip(pred, 1e-12, 1 - 1e-12)
        base_rate = float(np.mean(y))
        try:
            ll = float(log_loss(y, p, labels=[0, 1]))
        except ValueError:
            ll = float("nan")
        # Опорная точка — константное предсказание базовой ставки.
        if 0.0 < base_rate < 1.0:
            baseline_ll = float(
                -(base_rate * np.log(base_rate) + (1 - base_rate) * np.log(1 - base_rate))
            )
        else:
            baseline_ll = float("nan")
        relative = ll / baseline_ll if np.isfinite(baseline_ll) and baseline_ll > 0 else None
        br = float(brier_score_loss(y, p))
        # ROC AUC не определён при единственном классе: проверяем заранее, чтобы
        # не ловить исключение после предупреждения sklearn.
        auc: Optional[float] = None
        if 0.0 < base_rate < 1.0:
            try:
                auc = float(roc_auc_score(y, p))
            except ValueError:
                auc = None

        if relative is not None and relative > thresholds.outcome_relative_log_loss:
            warnings.append("weak_outcome_log_loss")
        if br > thresholds.outcome_brier:
            warnings.append("weak_outcome_brier")
        if auc is not None and auc < thresholds.outcome_roc_auc:
            warnings.append("weak_outcome_auc")

        return OutcomeModelDiagnostics(
            applicable=True,
            target=target,
            is_binary_target=True,
            is_out_of_fold=bool(outcome.is_out_of_fold),
            log_loss=ll,
            relative_log_loss=relative,
            brier_score=br,
            roc_auc=auc,
            warnings=tuple(warnings),
        )

    rmse = float(np.sqrt(mean_squared_error(y, pred)))
    mae = float(mean_absolute_error(y, pred))
    r2 = float(r2_score(y, pred))
    if r2 < thresholds.outcome_r2:
        warnings.append("weak_outcome_r2")
    return OutcomeModelDiagnostics(
        applicable=True,
        target=target,
        is_binary_target=False,
        is_out_of_fold=bool(outcome.is_out_of_fold),
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
    action_col: str,
    feature_cols: Sequence[str],
    behavior: Optional[BehaviorPredictions] = None,
    outcome: Optional[OutcomePredictions] = None,
    cluster_col: Optional[str] = None,
    action_space: Optional[Sequence] = None,
    spec: NuisanceSpec = DEFAULT_NUISANCE_SPEC,
    honest_outcome_metrics: bool = True,
    n_splits: int = 5,
    random_state: int = 123,
) -> NuisanceDiagnostics:
    """Считает диагностику качества nuisance-моделей.

    Args:
        honest_outcome_metrics: если предсказания исхода получены in-sample,
            посчитать отдельные out-of-fold предсказания только ради метрик.
            Это один дополнительный k-fold fit на полных данных (не внутри bootstrap),
            зато метрики перестают быть оптимистичными.
    """
    if (
        honest_outcome_metrics
        and estimator in OUTCOME_ESTIMATORS
        and outcome is not None
        and not outcome.is_out_of_fold
    ):
        try:
            outcome = generate_oof_outcome_predictions(
                df,
                target=target,
                feature_cols=feature_cols,
                n_splits=n_splits,
                random_state=random_state,
                action_col=action_col,
                action_space=action_space,
                cluster_col=cluster_col,
                spec=spec,
            )
        except ValueError as exc:
            logger.debug("Не удалось посчитать OOF-метрики исхода: %s", exc)

    behavior_diag = _compute_behavior_diagnostics(
        df,
        action_col=action_col,
        behavior=behavior,
        estimator=estimator,
        thresholds=NuisanceThresholds(),
    )
    outcome_diag = _compute_outcome_diagnostics(
        df,
        target=target,
        estimator=estimator,
        outcome=outcome,
        thresholds=NuisanceThresholds(),
    )
    return NuisanceDiagnostics(
        behavior=behavior_diag,
        outcome=outcome_diag,
        warnings=tuple(list(behavior_diag.warnings) + list(outcome_diag.warnings)),
    )
