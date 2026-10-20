"""
policyscope
===========

Оффлайн-оценка политик рекомендаций (off-policy evaluation) по логам.

Задача: по логам работающей политики **A** оценить, как повела бы себя политика-кандидат
**B**, без онлайн A/B-теста.

Быстрый старт::

    import policyscope as ps

    policy_b = ps.SoftmaxPolicy(["score_0", "score_1", "score_2"], temperature=1.0)
    result = ps.compare_policies(
        df, policy_b,
        target="reward", action_col="action", cluster_col="user_id",
    )
    print(result.to_markdown())

Библиотека не настраивает логирование: чтобы видеть предупреждения, вызовите
``logging.basicConfig(level=logging.INFO)`` в своём приложении.
"""

from __future__ import annotations

import logging

from policyscope.comparison import (
    MultiTargetComparison,
    PolicyComparison,
    compare_policies,
    compare_policies_multi_estimator,
    compare_policies_multi_target,
)
from policyscope.data import BanditSchema, LoggedBanditDataset
from policyscope.diagnostics import DiagnosticThresholds, PolicyDiagnostics
from policyscope.estimate import ALL_ESTIMATORS, PointEstimate, estimate_point, estimate_value
from policyscope.estimators import NuisanceSpec, ess
from policyscope.evaluator import OPEEvaluator
from policyscope.inference import cluster_robust_se
from policyscope.nuisance import CrossFitNuisanceBundle, fit_crossfit_nuisance_bundle
from policyscope.policies import (
    BasePolicy,
    DeterministicPolicy,
    EpsilonGreedyPolicy,
    SoftmaxPolicy,
    TabularPolicy,
    make_policy,
)
from policyscope.report import dump_json, render_summary, trust_verdict

logging.getLogger(__name__).addHandler(logging.NullHandler())

__version__ = "0.2.0"

__all__ = [
    "__version__",
    # Основная точка входа
    "compare_policies",
    "compare_policies_multi_target",
    "compare_policies_multi_estimator",
    "OPEEvaluator",
    "PolicyComparison",
    "MultiTargetComparison",
    # Контракт данных
    "BanditSchema",
    "LoggedBanditDataset",
    # Политики
    "BasePolicy",
    "TabularPolicy",
    "DeterministicPolicy",
    "SoftmaxPolicy",
    "EpsilonGreedyPolicy",
    "make_policy",
    # Отчёты
    "render_summary",
    "trust_verdict",
    "dump_json",
    # Низкоуровневый слой
    "estimate_point",
    "estimate_value",
    "PointEstimate",
    "ALL_ESTIMATORS",
    "NuisanceSpec",
    "CrossFitNuisanceBundle",
    "fit_crossfit_nuisance_bundle",
    "DiagnosticThresholds",
    "PolicyDiagnostics",
    "cluster_robust_se",
    "ess",
]
