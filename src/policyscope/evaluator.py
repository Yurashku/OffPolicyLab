"""High-level unified comparator for policy evaluation estimators.

Слой для унифицированного вызова разных OPE-эстиматоров через имя метода,
с CI по умолчанию.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import pandas as pd

from policyscope.ci import estimate_value
from policyscope.inference import infer_policy_comparison_bootstrap
from policyscope.estimators import value_on_policy


@dataclass
class OPEEvaluator:
    """Унифицированный объект оценки политик.

    Args:
        df: Логи политики A.
        policyB: Кандидатная политика B.
        target: Целевая метрика.
        feature_cols: Колонки признаков.
        action_col: Колонка логированного действия.
        action_space: Пространство действий для policyB.
        cluster_col: Колонка кластеров для bootstrap.
        n_boot: Число bootstrap-реплик (для CI по умолчанию).
        alpha: Уровень значимости.
        weight_clip: Клиппинг весов для IPS/SNIPS/DR/SNDR.
        tau: Порог switch для switch_dr.
    """

    df: pd.DataFrame
    policyB: object
    target: str = "accept"
    feature_cols: Optional[Sequence[str]] = None
    action_col: str = "a_A"
    action_space: Optional[Sequence] = None
    cluster_col: Optional[str] = "user_id"
    n_boot: int = 300
    alpha: float = 0.05
    weight_clip: Optional[float] = None
    tau: float = 20.0

    def evaluate(self, estimator: str = "dr", *, with_ci: bool = True):
        """Оценивает политику B выбранным эстиматором.

        По умолчанию возвращает CI (bootstrap).
        """

        def point_on(part: pd.DataFrame) -> float:
            return estimate_value(
                part,
                self.policyB,
                method=estimator,  # type: ignore[arg-type]
                target=self.target,
                feature_cols=self.feature_cols,
                action_col=self.action_col,
                action_space=self.action_space,
                weight_clip=self.weight_clip,
                tau=self.tau,
            )

        vA = value_on_policy(self.df, target=self.target)
        vB = point_on(self.df)
        base = {
            "estimator": estimator,
            "V_A": vA,
            "V_B": vB,
            "Delta": float(vB - vA),
        }
        if not with_ci:
            return base

        def estimator_pair(part: pd.DataFrame):
            a = value_on_policy(part, target=self.target)
            b = point_on(part)
            return a, b, b - a

        comp = infer_policy_comparison_bootstrap(
            self.df,
            estimator_pair,
            cluster_col=self.cluster_col,
            n_boot=self.n_boot,
            alpha=self.alpha,
        ).to_dict()
        base.update({
            "V_A_CI": comp["V_A_CI"],
            "V_B_CI": comp["V_B_CI"],
            "Delta_CI": comp["Delta_CI"],
            "p_value": comp["p_value"],
            "is_significant": comp["is_significant"],
            "significance_rule": comp["significance_rule"],
            "n_boot": comp["n_boot"],
            "alpha": comp["alpha"],
            "inference_method": comp["inference_method"],
            "inference_warnings": comp["inference_warnings"],
        })
        return base
