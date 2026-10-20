"""
policyscope.estimate
====================

Единый диспетчер точечных оценок.

Любой оценщик считается по одной схеме:

1. разрешить признаки, пространство действий и источник пропенсити;
2. получить nuisance-предсказания (внутренний fit либо готовые, в т.ч. OOF);
3. собрать построчные вклады ``psi_i`` через
   :func:`policyscope.estimators.estimator_contributions`;
4. значение оценки — ``mean(psi)``.

Благодаря шагу 3 точечная оценка и аналитический инференс используют одну и ту же
формулу, а внутренний и cross-fit пути не могут разойтись численно.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal, Optional, Sequence

import numpy as np
import pandas as pd

from policyscope.estimators import (
    DEFAULT_NUISANCE_SPEC,
    NuisanceSpec,
    estimator_contributions,
    importance_weights,
    mu_hat_predict,
    replay_value,
    resolve_action_space,
    resolve_feature_cols,
    validate_policy_probs,
    validate_propensity,
)
from policyscope.nuisance import (
    BehaviorPredictions,
    OutcomePredictions,
    PropensitySource,
    fit_outcome_nuisance_bundle,
    resolve_behavior_predictions,
    validate_behavior_predictions,
    validate_outcome_predictions,
)

logger = logging.getLogger(__name__)

EstimatorName = Literal["on_policy", "replay", "ips", "snips", "dm", "dr", "sndr", "switch_dr"]

#: Оценщики, которым нужны веса важности (а значит — пропенсити).
WEIGHTED_ESTIMATORS = frozenset({"ips", "snips", "dr", "sndr", "switch_dr"})
#: Оценщики, которым нужна модель исхода.
OUTCOME_ESTIMATORS = frozenset({"dm", "dr", "sndr", "switch_dr"})
#: Оценщики, для которых вклад на наблюдение линеен (доступен аналитический CI).
LINEARIZABLE_ESTIMATORS = frozenset(
    {"on_policy", "ips", "snips", "dm", "dr", "sndr", "switch_dr"}
)
ALL_ESTIMATORS: tuple[str, ...] = (
    "on_policy", "replay", "ips", "snips", "dm", "dr", "sndr", "switch_dr",
)

__all__ = [
    "EstimatorName",
    "PointEstimate",
    "estimate_value",
    "estimate_point",
    "ALL_ESTIMATORS",
    "WEIGHTED_ESTIMATORS",
    "OUTCOME_ESTIMATORS",
    "LINEARIZABLE_ESTIMATORS",
]


@dataclass(frozen=True)
class PointEstimate:
    """Точечная оценка вместе со всем, что нужно для инференса и диагностики.

    Attributes:
        value: Значение оценки, равное ``mean(contributions)`` для линеаризуемых оценщиков.
        contributions: Построчные вклады ``psi_i``; ``None`` для ``replay``, который
            не является средним по всем строкам.
        weights: Эффективные веса важности (после клиппинга), если применимо.
        clip_share: Доля наблюдений, где вес был обрезан.
        switch_share: Доля наблюдений, где Switch-DR отключил IPS-поправку.
        propensity_source: Фактически использованный источник пропенсити.
        unsupported_action_share: Доля строк с действием, неизвестным behavior-модели.
    """

    value: float
    contributions: Optional[np.ndarray] = None
    weights: Optional[np.ndarray] = None
    clip_share: float = 0.0
    switch_share: Optional[float] = None
    behavior: Optional[BehaviorPredictions] = None
    outcome: Optional[OutcomePredictions] = None
    propensity_source: Optional[str] = None
    propensity_col: Optional[str] = None
    unsupported_action_share: float = 0.0
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def supports_influence_ci(self) -> bool:
        """Доступен ли аналитический CI (нужны построчные вклады)."""
        return self.contributions is not None


def _build_outcome_predictions(
    df: pd.DataFrame,
    *,
    actions: Sequence,
    target: str,
    feature_cols: Sequence[str],
    action_col: str,
    spec: NuisanceSpec,
) -> OutcomePredictions:
    """Обучает модель исхода на этих же данных и разворачивает её в предсказания.

    In-sample путь: для честной оценки качества nuisance используйте cross-fitting.
    """
    bundle = fit_outcome_nuisance_bundle(
        df,
        target=target,
        feature_cols=feature_cols,
        action_col=action_col,
        action_space=actions,
        spec=spec,
    )
    mu_model = bundle.mu_model
    n = len(df)
    mu_by_action = {
        a: mu_hat_predict(mu_model, df, np.full(n, a, dtype=object)) for a in actions
    }
    mu_logged = mu_hat_predict(mu_model, df, df[action_col].to_numpy())
    return OutcomePredictions(
        mu_logged_action=mu_logged,
        mu_by_action=mu_by_action,
        target=target,
        is_out_of_fold=False,
    )


def estimate_point(
    df: pd.DataFrame,
    policyB,
    *,
    method: EstimatorName,
    target: str = "accept",
    feature_cols: Optional[Sequence[str]] = None,
    action_col: str = "a_A",
    action_space: Optional[Sequence] = None,
    cluster_col: Optional[str] = None,
    weight_clip: Optional[float] = None,
    tau: float = 20.0,
    nuisance_behavior: Optional[BehaviorPredictions] = None,
    nuisance_outcome: Optional[OutcomePredictions] = None,
    propensity_source: PropensitySource = "auto",
    propensity_col: Optional[str] = None,
    spec: NuisanceSpec = DEFAULT_NUISANCE_SPEC,
) -> PointEstimate:
    """Считает точечную оценку и все побочные величины, нужные инференсу.

    Args:
        df: Логи политики A.
        policyB: Политика-кандидат с методом ``action_probs(df) -> (n, k)``.
        method: Имя оценщика (см. :data:`ALL_ESTIMATORS`).
        target: Колонка награды.
        feature_cols: Признаки; ``None`` — автовыбор с исключением служебных колонок.
        action_col: Колонка логированного действия.
        action_space: Порядок действий, соответствующий столбцам политики.
        cluster_col: Колонка кластера (исключается из автовыбора признаков).
        weight_clip: Верхний порог клиппинга весов важности.
        tau: Порог переключения Switch-DR.
        nuisance_behavior: Готовые behavior-предсказания (например, OOF).
        nuisance_outcome: Готовые предсказания исхода (например, OOF).
        propensity_source: ``auto`` / ``logged`` / ``estimated``.
        propensity_col: Колонка логированных пропенсити.
        spec: Конфигурация nuisance-моделей.

    Returns:
        :class:`PointEstimate`.
    """
    if method not in ALL_ESTIMATORS:
        raise ValueError(f"Неизвестный оценщик: {method}. Доступны: {list(ALL_ESTIMATORS)}")

    n = len(df)
    if n == 0:
        raise ValueError("Пустой DataFrame")

    if method == "on_policy":
        r = df[target].to_numpy(dtype=float)
        return PointEstimate(value=float(np.mean(r)), contributions=r)

    probsB = validate_policy_probs(policyB.action_probs(df), n)
    actions = resolve_action_space(probsB, action_space)

    if method == "replay":
        a_B = np.asarray(actions, dtype=object)[np.argmax(probsB, axis=1)]
        return PointEstimate(
            value=replay_value(df, a_B, target=target, action_col=action_col),
            contributions=None,  # не среднее по всем строкам -> без аналитического CI
            notes=("replay_valid_only_under_uniform_logging",),
        )

    feats = resolve_feature_cols(
        df,
        feature_cols,
        target=target,
        action_col=action_col,
        cluster_col=cluster_col,
        propensity_col=propensity_col,
    )

    # --- behavior-сторона ---
    behavior = nuisance_behavior
    source: Optional[str] = None
    used_col: Optional[str] = None
    notes: tuple[str, ...] = ()
    w: Optional[np.ndarray] = None
    clip_share = 0.0

    if method in WEIGHTED_ESTIMATORS:
        if behavior is None:
            behavior, source, used_col, notes = resolve_behavior_predictions(
                df,
                policyB,
                feature_cols=feats,
                propensity_source=propensity_source,
                propensity_col=propensity_col,
                action_col=action_col,
                action_space=actions,
                spec=spec,
            )
        else:
            source = behavior.propensity_source
            used_col = behavior.propensity_col
        validate_behavior_predictions(behavior, n)
        pA = validate_propensity(behavior.pA_taken, n, name="pA_taken")
        w, clip_share = importance_weights(behavior.piB_taken, pA, weight_clip=weight_clip)

    # --- outcome-сторона ---
    outcome = nuisance_outcome
    dm_vec: Optional[np.ndarray] = None
    mu_taken: Optional[np.ndarray] = None

    if method in OUTCOME_ESTIMATORS:
        if outcome is None:
            outcome = _build_outcome_predictions(
                df,
                actions=actions,
                target=target,
                feature_cols=feats,
                action_col=action_col,
                spec=spec,
            )
        validate_outcome_predictions(outcome, n, required_actions=actions)
        assert outcome.mu_by_action is not None
        dm_vec = np.zeros(n, dtype=float)
        for idx, action in enumerate(actions):
            dm_vec += probsB[:, idx] * outcome.mu_by_action[action]
        mu_taken = np.asarray(outcome.mu_logged_action, dtype=float)

    contributions = estimator_contributions(
        method, r=df[target].to_numpy(dtype=float), w=w, dm=dm_vec, mu_taken=mu_taken, tau=tau
    )

    switch_share = None
    effective_w = w
    if method == "switch_dr" and w is not None:
        mask = w <= tau
        switch_share = float(np.mean(~mask))
        effective_w = np.where(mask, w, 0.0)

    return PointEstimate(
        value=float(np.mean(contributions)),
        contributions=contributions,
        weights=effective_w,
        clip_share=clip_share,
        switch_share=switch_share,
        behavior=behavior,
        outcome=outcome,
        propensity_source=source,
        propensity_col=used_col,
        unsupported_action_share=(
            behavior.unsupported_action_share if behavior is not None else 0.0
        ),
        notes=notes,
    )


def estimate_value(
    df: pd.DataFrame,
    policyB,
    *,
    method: EstimatorName,
    target: str = "accept",
    feature_cols: Optional[Sequence[str]] = None,
    action_col: str = "a_A",
    action_space: Optional[Sequence] = None,
    cluster_col: Optional[str] = None,
    weight_clip: Optional[float] = None,
    tau: float = 20.0,
    nuisance_behavior: Optional[BehaviorPredictions] = None,
    nuisance_outcome: Optional[OutcomePredictions] = None,
    propensity_source: PropensitySource = "auto",
    propensity_col: Optional[str] = None,
    spec: NuisanceSpec = DEFAULT_NUISANCE_SPEC,
) -> float:
    """Возвращает только числовое значение выбранного оценщика.

    Тонкая обёртка над :func:`estimate_point` для случаев, когда диагностика не нужна.
    """
    return estimate_point(
        df,
        policyB,
        method=method,
        target=target,
        feature_cols=feature_cols,
        action_col=action_col,
        action_space=action_space,
        cluster_col=cluster_col,
        weight_clip=weight_clip,
        tau=tau,
        nuisance_behavior=nuisance_behavior,
        nuisance_outcome=nuisance_outcome,
        propensity_source=propensity_source,
        propensity_col=propensity_col,
        spec=spec,
    ).value
