"""Объектная обёртка над :func:`~policyscope.comparison.compare_policies`.

Удобна, когда одни и те же данные и настройки прогоняются несколькими оценщиками:
конфигурация задаётся один раз, а меняется только имя метода.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import pandas as pd

from policyscope.comparison import (
    InferenceMethod,
    MultiTargetComparison,
    PolicyComparison,
    compare_policies,
    compare_policies_multi_estimator,
    compare_policies_multi_target,
)
from policyscope.data import LoggedBanditDataset
from policyscope.diagnostics import DiagnosticThresholds
from policyscope.estimate import EstimatorName
from policyscope.estimators import DEFAULT_NUISANCE_SPEC, NuisanceSpec
from policyscope.nuisance import PropensitySource

__all__ = ["OPEEvaluator"]


@dataclass
class OPEEvaluator:
    """Единый объект оценки политик.

    Поля повторяют аргументы :func:`~policyscope.comparison.compare_policies`, так что
    объекту доступны все возможности библиотеки, включая логированные пропенсити,
    cross-fitting, выбор метода инференса и подключаемые nuisance-модели.

    Example:
        >>> ev = OPEEvaluator(df, policy_b, target="reward", cluster_col="user_id")
        >>> print(ev.compare("dr").to_markdown())
        >>> ev.compare_all()          # таблица по всем оценщикам
    """

    data: pd.DataFrame | LoggedBanditDataset
    policyB: object
    target: Any = None
    feature_cols: Optional[Sequence[str]] = None
    action_col: Any = None
    action_space: Optional[Sequence] = None
    cluster_col: Any = None
    propensity_col: Any = None
    propensity_source: PropensitySource = "auto"
    inference: InferenceMethod = "bootstrap"
    n_boot: int = 500
    alpha: float = 0.05
    rng_seed: int = 4321
    weight_clip: Optional[float] = None
    tau: float = 20.0
    use_crossfit: bool = False
    crossfit_n_splits: int = 5
    crossfit_random_state: int = 123
    nuisance_spec: NuisanceSpec = DEFAULT_NUISANCE_SPEC
    honest_outcome_metrics: bool = True
    thresholds: DiagnosticThresholds = field(default_factory=DiagnosticThresholds)

    def _kwargs(self) -> dict[str, Any]:
        from policyscope.comparison import UNSET

        out: dict[str, Any] = {
            "action_space": self.action_space,
            "propensity_source": self.propensity_source,
            "inference": self.inference,
            "n_boot": self.n_boot,
            "alpha": self.alpha,
            "rng_seed": self.rng_seed,
            "weight_clip": self.weight_clip,
            "tau": self.tau,
            "use_crossfit": self.use_crossfit,
            "crossfit_n_splits": self.crossfit_n_splits,
            "crossfit_random_state": self.crossfit_random_state,
            "nuisance_spec": self.nuisance_spec,
            "honest_outcome_metrics": self.honest_outcome_metrics,
            "thresholds": self.thresholds,
        }
        # None означает «не задано» и должно уступать схеме датасета,
        # поэтому такие поля передаются только когда указаны явно.
        for name in ("target", "action_col", "cluster_col", "propensity_col", "feature_cols"):
            value = getattr(self, name)
            out[name] = UNSET if value is None else value
        return out

    def compare(self, estimator: EstimatorName = "dr") -> PolicyComparison:
        """Считает сравнение выбранным оценщиком."""
        return compare_policies(self.data, self.policyB, estimator=estimator, **self._kwargs())

    def compare_all(
        self,
        estimators: Sequence[EstimatorName] = ("ips", "snips", "dm", "dr", "sndr", "switch_dr"),
    ) -> pd.DataFrame:
        """Считает несколько оценщиков и сводит их в таблицу.

        Расхождение оценщиков между собой — практичный признак того, что оценке
        доверять нельзя, даже если каждый отдельный CI выглядит узким.
        """
        return compare_policies_multi_estimator(
            self.data, self.policyB, estimators=estimators, **self._kwargs()
        )

    def compare_targets(
        self,
        targets: Sequence[str],
        estimator: EstimatorName = "dr",
    ) -> MultiTargetComparison:
        """Повторяет оценку для нескольких целевых метрик."""
        kwargs = self._kwargs()
        kwargs.pop("target", None)
        return compare_policies_multi_target(
            self.data, self.policyB, targets=targets, estimator=estimator, **kwargs
        )
