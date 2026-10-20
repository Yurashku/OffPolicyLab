"""Диагностика доверия и устойчивости OPE-оценки.

Диагностика **не заменяет** доверительный интервал: узкий CI на данных без
перекрытия политик остаётся бессмысленным. Задача этого слоя — показать, при каких
условиях полученное число вообще имеет право на существование.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from policyscope.estimate import PointEstimate
from policyscope.estimators import ess, resolve_action_space, validate_policy_probs

logger = logging.getLogger(__name__)

__all__ = [
    "DiagnosticThresholds",
    "WeightDiagnostics",
    "OverlapDiagnostics",
    "PolicyDiagnostics",
    "compute_policy_diagnostics",
]


@dataclass(frozen=True)
class DiagnosticThresholds:
    """Пороги срабатывания предупреждений.

    Значения по умолчанию — практические ориентиры, а не универсальные константы;
    их можно переопределить при вызове.
    """

    #: Доля эффективной выборки, ниже которой стоит насторожиться.
    ess_ratio: float = 0.1
    #: Абсолютный эффективный размер выборки, ниже которого оценивать нечего.
    #: Именно абсолютное значение определяет, есть ли вообще информация: ESS/N = 0.1
    #: при n = 100 000 — это 10 000 наблюдений, а при n = 200 — двадцать.
    min_effective_sample: float = 50.0
    replay_overlap: float = 0.05
    max_weight: float = 50.0
    p99_weight: float = 20.0
    clip_share: float = 0.2
    switch_share: float = 0.2
    #: Допустимое отклонение среднего веса от 1.0. Для корректных пропенсити
    #: E[w] = 1 по построению, поэтому заметное отклонение — резкий признак
    #: несоответствия модели пропенсити или нарушенного перекрытия.
    mean_weight_deviation: float = 0.1
    #: Доля массы pi_B на действиях, почти не встречающихся в логах.
    unsupported_action_mass: float = 0.01
    #: Сколько раз действие должно встретиться в логах, чтобы считаться поддержанным.
    min_action_count: int = 30
    #: Минимальное отношение частоты действия в логах к массе, которую ему отдаёт pi_B.
    #: Именно это отношение задаёт масштаб веса важности: если B отдаёт действию 97%
    #: массы, а в логах оно встречается в 0.3% случаев, средний вес там порядка 300.
    min_support_ratio: float = 0.1


@dataclass(frozen=True)
class WeightDiagnostics:
    """Статистики весов важности, **фактически использованных** оценщиком."""

    ess: Optional[float] = None
    ess_ratio: Optional[float] = None
    mean_weight: Optional[float] = None
    max_weight: Optional[float] = None
    p95_weight: Optional[float] = None
    p99_weight: Optional[float] = None
    clip_share: Optional[float] = None
    switch_share: Optional[float] = None


@dataclass(frozen=True)
class OverlapDiagnostics:
    """Перекрытие носителей политик A и B."""

    replay_overlap: float
    unsupported_action_mass: float
    unsupported_action_share: float
    action_coverage: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class PolicyDiagnostics:
    """Полная диагностика одного прогона оценки."""

    method: str
    n_samples: int
    n_clusters: Optional[int]
    overlap: OverlapDiagnostics
    weights: WeightDiagnostics
    warnings: tuple[str, ...] = field(default_factory=tuple)
    propensity_source: Optional[str] = None
    propensity_column: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "n_samples": self.n_samples,
            "n_clusters": self.n_clusters,
            "replay_overlap": self.overlap.replay_overlap,
            "unsupported_action_mass": self.overlap.unsupported_action_mass,
            "unsupported_action_share": self.overlap.unsupported_action_share,
            "action_coverage": dict(self.overlap.action_coverage),
            "weight_ess": self.weights.ess,
            "weight_ess_ratio": self.weights.ess_ratio,
            "weight_mean": self.weights.mean_weight,
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
    clip_share: Optional[float],
    switch_share: Optional[float],
) -> WeightDiagnostics:
    if weights is None or len(weights) == 0:
        return WeightDiagnostics(clip_share=clip_share, switch_share=switch_share)
    w = np.asarray(weights, dtype=float)
    if not np.isfinite(w).all():
        logger.warning("Веса важности содержат не конечные значения; статистики неполны")
        w = w[np.isfinite(w)]
        if w.size == 0:
            return WeightDiagnostics(clip_share=clip_share, switch_share=switch_share)
    ess_value = ess(w)
    return WeightDiagnostics(
        ess=float(ess_value),
        ess_ratio=float(ess_value / max(n, 1)),
        mean_weight=float(np.mean(w)),
        max_weight=float(np.max(w)),
        p95_weight=float(np.percentile(w, 95)),
        p99_weight=float(np.percentile(w, 99)),
        clip_share=clip_share,
        switch_share=switch_share,
    )


def _action_support(
    df: pd.DataFrame,
    probsB: np.ndarray,
    actions: Sequence,
    *,
    action_col: str,
    min_count: int,
    min_support_ratio: float,
) -> tuple[float, dict[str, float]]:
    """Считает массу ``pi_B`` на действиях, которые почти не встречаются в логах.

    Это ключевая проверка применимости OPE: если политика B рекомендует то, чего
    политика A практически никогда не делала, ни один оценщик не восстановит
    значение из данных — сколь угодно узкий CI будет вводить в заблуждение.

    Действие считается неподдержанным, если оно либо встречается в логах слишком
    редко в абсолютном выражении, либо его доля в логах мала **относительно** массы,
    которую ему отдаёт ``pi_B``. Одного абсолютного счётчика недостаточно: сотня
    наблюдений — это много для действия, которому B отдаёт 1% массы, и катастрофически
    мало для действия с массой 97%.
    """
    n = len(df)
    logged_counts = df[action_col].value_counts()
    mean_mass = probsB.mean(axis=0)
    coverage: dict[str, float] = {}
    unsupported = 0.0
    for idx, action in enumerate(actions):
        count = int(logged_counts.get(action, 0))
        mass = float(mean_mass[idx])
        coverage[str(action)] = mass
        if mass <= 0.0:
            continue
        logged_share = count / max(n, 1)
        if count < min_count or logged_share < min_support_ratio * mass:
            unsupported += mass
    return unsupported, coverage


def compute_policy_diagnostics(
    df: pd.DataFrame,
    policyB,
    point: PointEstimate,
    *,
    method: str,
    action_col: str = "a_A",
    action_space: Optional[Sequence] = None,
    cluster_col: Optional[str] = None,
    thresholds: DiagnosticThresholds = DiagnosticThresholds(),
) -> PolicyDiagnostics:
    """Строит диагностику по уже посчитанной точечной оценке.

    Принимает готовый :class:`~policyscope.estimate.PointEstimate`, а не пересчитывает
    nuisance заново: раньше диагностика обучала behavior-модель второй раз и считала
    статистики по **неклипованным** весам, то есть описывала веса, которых оценщик
    не использовал.
    """
    n = int(len(df))
    probsB = validate_policy_probs(policyB.action_probs(df), n)
    actions = resolve_action_space(probsB, action_space)

    a_B = np.asarray(actions, dtype=object)[np.argmax(probsB, axis=1)]
    replay_overlap = float(np.mean(df[action_col].to_numpy() == a_B))
    unsupported_mass, coverage = _action_support(
        df, probsB, actions,
        action_col=action_col,
        min_count=thresholds.min_action_count,
        min_support_ratio=thresholds.min_support_ratio,
    )

    overlap = OverlapDiagnostics(
        replay_overlap=replay_overlap,
        unsupported_action_mass=unsupported_mass,
        unsupported_action_share=point.unsupported_action_share,
        action_coverage=coverage,
    )
    w_diag = _weight_diagnostics(
        point.weights,
        n=n,
        clip_share=point.clip_share if point.weights is not None else None,
        switch_share=point.switch_share,
    )

    n_clusters = (
        int(df[cluster_col].nunique())
        if cluster_col is not None and cluster_col in df.columns
        else None
    )

    warnings: list[str] = list(point.notes)
    if replay_overlap < thresholds.replay_overlap:
        warnings.append("low_replay_overlap")
    if unsupported_mass > thresholds.unsupported_action_mass:
        warnings.append("policy_b_recommends_unsupported_actions")
    if point.unsupported_action_share > 0.0:
        warnings.append("logged_actions_unknown_to_behavior_model")
    if w_diag.ess_ratio is not None and w_diag.ess_ratio < thresholds.ess_ratio:
        warnings.append("low_ess_ratio")
    if w_diag.ess is not None and w_diag.ess < thresholds.min_effective_sample:
        warnings.append("critically_low_ess")
    if w_diag.max_weight is not None and w_diag.max_weight > thresholds.max_weight:
        warnings.append("extreme_max_weight")
    if w_diag.p99_weight is not None and w_diag.p99_weight > thresholds.p99_weight:
        warnings.append("heavy_weight_tail")
    if (
        w_diag.mean_weight is not None
        and method in {"ips", "dr"}
        and abs(w_diag.mean_weight - 1.0) > thresholds.mean_weight_deviation
    ):
        warnings.append("mean_weight_far_from_one")
    if w_diag.clip_share is not None and w_diag.clip_share > thresholds.clip_share:
        warnings.append("large_clip_share")
    if w_diag.switch_share is not None and w_diag.switch_share > thresholds.switch_share:
        warnings.append("large_switch_share")

    return PolicyDiagnostics(
        method=method,
        n_samples=n,
        n_clusters=n_clusters,
        overlap=overlap,
        weights=w_diag,
        warnings=tuple(dict.fromkeys(warnings)),
        propensity_source=point.propensity_source,
        propensity_column=point.propensity_col,
    )
