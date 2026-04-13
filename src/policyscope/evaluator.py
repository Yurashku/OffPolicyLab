"""High-level unified comparator for policy evaluation estimators.

Слой для унифицированного вызова разных OPE-эстиматоров через имя метода,
с CI по умолчанию.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import pandas as pd

from policyscope.comparison import PolicyComparisonSummary, compare_policies


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
        return self.evaluate_summary(estimator=estimator, with_ci=with_ci).to_dict()

    def evaluate_summary(self, estimator: str = "dr", *, with_ci: bool = True) -> PolicyComparisonSummary:
        """Официальный структурированный результат сравнения A vs B."""
        return compare_policies(
            self.df,
            self.policyB,
            estimator=estimator,
            target=self.target,
            feature_cols=self.feature_cols,
            action_col=self.action_col,
            action_space=self.action_space,
            cluster_col=self.cluster_col,
            n_boot=self.n_boot,
            alpha=self.alpha,
            weight_clip=self.weight_clip,
            tau=self.tau,
            with_ci=with_ci,
        )
