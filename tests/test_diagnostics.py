"""Тесты диагностики доверия: веса, перекрытие, качество nuisance-моделей."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from policyscope.comparison import compare_policies
from policyscope.diagnostics import DiagnosticThresholds, compute_policy_diagnostics
from policyscope.estimate import estimate_point
from policyscope.nuisance_diagnostics import compute_nuisance_diagnostics
from policyscope.policies import TabularPolicy

from conftest import make_controlled_bandit


@pytest.fixture
def setup():
    bandit = make_controlled_bandit(n=800, seed=31)
    return bandit, bandit.policy_b()


def _diag(bandit, policy, **kwargs):
    point = estimate_point(
        bandit.df, policy, method=kwargs.pop("method", "ips"), target="reward",
        action_col="action", action_space=bandit.action_space,
        feature_cols=["x1", "x2"], propensity_source="logged",
        propensity_col="propensity", **kwargs,
    )
    return point, compute_policy_diagnostics(
        bandit.df, policy, point, method="ips", action_col="action",
        action_space=bandit.action_space,
    )


class TestWeightDiagnostics:
    def test_weight_stats_present_for_weighted_estimators(self, setup):
        bandit, policy = setup
        _, diag = _diag(bandit, policy)
        d = diag.to_dict()
        for key in ("weight_ess", "weight_ess_ratio", "weight_mean", "weight_p99"):
            assert d[key] is not None and np.isfinite(d[key])

    def test_mean_weight_is_near_one_with_true_propensities(self, setup):
        bandit, policy = setup
        _, diag = _diag(bandit, policy)
        assert diag.weights.mean_weight == pytest.approx(1.0, abs=0.1)

    def test_diagnostics_use_clipped_weights(self, setup):
        """Статистики должны описывать веса, которые оценщик реально использовал."""
        bandit, policy = setup
        _, unclipped = _diag(bandit, policy)
        _, clipped = _diag(bandit, policy, weight_clip=1.5)
        assert clipped.weights.max_weight <= 1.5
        assert clipped.weights.max_weight < unclipped.weights.max_weight
        assert clipped.weights.clip_share > 0

    def test_switch_share_reported_for_switch_dr(self, setup):
        bandit, policy = setup
        point = estimate_point(
            bandit.df, policy, method="switch_dr", target="reward", action_col="action",
            action_space=bandit.action_space, feature_cols=["x1", "x2"],
            propensity_source="logged", propensity_col="propensity", tau=1.2,
        )
        diag = compute_policy_diagnostics(
            bandit.df, policy, point, method="switch_dr",
            action_col="action", action_space=bandit.action_space,
        )
        assert diag.weights.switch_share is not None and diag.weights.switch_share > 0

    def test_no_weight_stats_for_dm(self, setup):
        bandit, policy = setup
        point = estimate_point(
            bandit.df, policy, method="dm", target="reward", action_col="action",
            action_space=bandit.action_space, feature_cols=["x1", "x2"],
        )
        diag = compute_policy_diagnostics(
            bandit.df, policy, point, method="dm",
            action_col="action", action_space=bandit.action_space,
        )
        assert diag.weights.ess is None


class TestOverlapDiagnostics:
    def test_unsupported_actions_are_detected(self):
        """Политика B кладёт массу на действие, которого A почти не совершала."""
        n = 400
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "x1": rng.normal(size=n),
            "action": np.zeros(n, dtype=int),   # логировалось только действие 0
            "propensity": np.full(n, 1.0),
            "reward": rng.integers(0, 2, n),
        })
        df.loc[:2, "action"] = 1  # три наблюдения с действием 1
        probs = np.column_stack([np.full(n, 0.3), np.full(n, 0.7)])
        policy = TabularPolicy(probs, action_space=[0, 1], index=df.index)
        point = estimate_point(
            df, policy, method="ips", target="reward", action_col="action",
            action_space=[0, 1], feature_cols=["x1"],
            propensity_source="logged", propensity_col="propensity",
        )
        diag = compute_policy_diagnostics(
            df, policy, point, method="ips", action_col="action", action_space=[0, 1]
        )
        assert diag.overlap.unsupported_action_mass == pytest.approx(0.7)
        assert "policy_b_recommends_unsupported_actions" in diag.warnings

    def test_action_coverage_sums_to_one(self, setup):
        bandit, policy = setup
        _, diag = _diag(bandit, policy)
        assert sum(diag.overlap.action_coverage.values()) == pytest.approx(1.0)

    def test_n_clusters_reported(self):
        bandit = make_controlled_bandit(n=200, seed=32, rows_per_user=4)
        policy = bandit.policy_b()
        point = estimate_point(
            bandit.df, policy, method="ips", target="reward", action_col="action",
            action_space=bandit.action_space, feature_cols=["x1", "x2"],
            propensity_source="logged", propensity_col="propensity",
        )
        diag = compute_policy_diagnostics(
            bandit.df, policy, point, method="ips", action_col="action",
            action_space=bandit.action_space, cluster_col="user_id",
        )
        assert diag.n_clusters == 50


class TestWarningRules:
    def test_extreme_weights_trigger_warnings(self):
        n = 300
        rng = np.random.default_rng(1)
        df = pd.DataFrame({
            "x1": rng.normal(size=n),
            "action": rng.integers(0, 2, n),
            "propensity": np.full(n, 0.001),   # крошечные пропенсити -> огромные веса
            "reward": rng.integers(0, 2, n),
        })
        probs = np.column_stack([np.full(n, 0.5), np.full(n, 0.5)])
        policy = TabularPolicy(probs, action_space=[0, 1], index=df.index)
        point = estimate_point(
            df, policy, method="ips", target="reward", action_col="action",
            action_space=[0, 1], feature_cols=["x1"],
            propensity_source="logged", propensity_col="propensity",
        )
        diag = compute_policy_diagnostics(
            df, policy, point, method="ips", action_col="action", action_space=[0, 1]
        )
        assert "extreme_max_weight" in diag.warnings
        assert "heavy_weight_tail" in diag.warnings
        assert "mean_weight_far_from_one" in diag.warnings

    def test_thresholds_are_configurable(self, setup):
        bandit, policy = setup
        point, _ = _diag(bandit, policy)
        strict = compute_policy_diagnostics(
            bandit.df, policy, point, method="ips", action_col="action",
            action_space=bandit.action_space,
            thresholds=DiagnosticThresholds(ess_ratio=0.99, replay_overlap=0.99),
        )
        assert "low_ess_ratio" in strict.warnings
        assert "low_replay_overlap" in strict.warnings

    def test_absolute_and_relative_ess_are_separate_signals(self, setup):
        """Отношение ESS/N и абсолютный ESS отвечают на разные вопросы."""
        bandit, policy = setup
        point, diag = _diag(bandit, policy)
        assert diag.weights.ess > 50
        assert "critically_low_ess" not in diag.warnings

        strict = compute_policy_diagnostics(
            bandit.df, policy, point, method="ips", action_col="action",
            action_space=bandit.action_space,
            thresholds=DiagnosticThresholds(min_effective_sample=1e9),
        )
        assert "critically_low_ess" in strict.warnings


class TestNuisanceDiagnostics:
    def test_behavior_diagnostics_not_applicable_for_logged_propensity(self, setup):
        bandit, policy = setup
        point, _ = _diag(bandit, policy)
        nd = compute_nuisance_diagnostics(
            bandit.df, target="reward", estimator="ips", action_col="action",
            feature_cols=["x1", "x2"], behavior=point.behavior, outcome=point.outcome,
        )
        assert nd.behavior.applicable is False

    def test_behavior_diagnostics_for_estimated_propensity(self, setup):
        bandit, policy = setup
        point = estimate_point(
            bandit.df, policy, method="ips", target="reward", action_col="action",
            action_space=bandit.action_space, feature_cols=["x1", "x2"],
            propensity_source="estimated",
        )
        nd = compute_nuisance_diagnostics(
            bandit.df, target="reward", estimator="ips", action_col="action",
            feature_cols=["x1", "x2"], behavior=point.behavior, outcome=point.outcome,
        )
        assert nd.behavior.applicable
        assert nd.behavior.multiclass_log_loss > 0
        # Опорная точка — равномерное предсказание log(k); модель должна быть лучше.
        assert nd.behavior.relative_log_loss < 1.0

    def test_top1_accuracy_uses_action_labels_not_column_indices(self):
        """Метки действий не обязаны совпадать с индексами столбцов."""
        n = 400
        rng = np.random.default_rng(2)
        df = pd.DataFrame({
            "x1": rng.normal(size=n),
            "action": rng.choice(["red", "green", "blue"], size=n),
            "reward": rng.integers(0, 2, n),
        })
        probs = np.full((n, 3), 1 / 3)
        policy = TabularPolicy(probs, action_space=["red", "green", "blue"], index=df.index)
        point = estimate_point(
            df, policy, method="ips", target="reward", action_col="action",
            action_space=["red", "green", "blue"], feature_cols=["x1"],
            propensity_source="estimated",
        )
        nd = compute_nuisance_diagnostics(
            df, target="reward", estimator="ips", action_col="action",
            feature_cols=["x1"], behavior=point.behavior, outcome=point.outcome,
        )
        # При сравнении индекса с меткой точность была бы ровно 0.
        assert nd.behavior.top1_accuracy is not None
        assert nd.behavior.top1_accuracy > 0.0

    def test_outcome_metrics_are_out_of_fold(self, setup):
        """In-sample метрики оптимистичны и потому бесполезны для своей задачи."""
        bandit, policy = setup
        result = compare_policies(
            bandit.df, policy, estimator="dr", target="reward", action_col="action",
            action_space=bandit.action_space, feature_cols=["x1", "x2"],
            propensity_col="propensity", cluster_col=None, inference="none",
        )
        assert result.nuisance_diagnostics is not None
        assert result.nuisance_diagnostics.outcome.is_out_of_fold is True
        assert "outcome_metrics_in_sample_optimistic" not in result.warnings


class TestRelativeActionSupport:
    """Абсолютного счётчика наблюдений недостаточно для оценки поддержки действия."""

    @staticmethod
    def _mass(logged_share: float, policy_mass: float, n: int = 4000) -> float:
        n_rare = max(int(round(logged_share * n)), 1)
        rng = np.random.default_rng(0)
        actions = np.zeros(n, dtype=int)
        actions[:n_rare] = 1
        df = pd.DataFrame({
            "x1": rng.normal(size=n),
            "action": actions,
            "propensity": np.where(actions == 1, logged_share, 1 - logged_share),
            "reward": rng.integers(0, 2, n),
        })
        probs = np.column_stack([
            np.full(n, 1 - policy_mass), np.full(n, policy_mass)
        ])
        policy = TabularPolicy(probs, action_space=[0, 1], index=df.index)
        point = estimate_point(
            df, policy, method="ips", target="reward", action_col="action",
            action_space=[0, 1], feature_cols=["x1"],
            propensity_source="logged", propensity_col="propensity",
        )
        diag = compute_policy_diagnostics(
            df, policy, point, method="ips", action_col="action", action_space=[0, 1]
        )
        return diag.overlap.unsupported_action_mass

    def test_rare_action_with_large_policy_mass_is_flagged(self):
        """100 наблюдений — много для массы 1% и катастрофически мало для 97%."""
        assert self._mass(logged_share=0.025, policy_mass=0.97) == pytest.approx(0.97)

    def test_same_count_is_fine_when_policy_mass_is_small(self):
        assert self._mass(logged_share=0.025, policy_mass=0.01) == pytest.approx(0.0)


class TestHonestOutcomeMetricsFlag:
    """OOF-метрики честны, но стоят дополнительного k-fold фита."""

    @staticmethod
    def _run(bandit, flag):
        return compare_policies(
            bandit.df, bandit.policy_b(), estimator="dr", target="reward",
            action_col="action", action_space=bandit.action_space,
            feature_cols=["x1", "x2"], propensity_col="propensity",
            cluster_col=None, inference="none", honest_outcome_metrics=flag,
        )

    def test_flag_controls_out_of_fold_metrics(self):
        bandit = make_controlled_bandit(n=600, seed=71)
        assert self._run(bandit, True).nuisance_diagnostics.outcome.is_out_of_fold is True
        off = self._run(bandit, False)
        assert off.nuisance_diagnostics.outcome.is_out_of_fold is False
        assert "outcome_metrics_in_sample_optimistic" in off.warnings

    def test_flag_does_not_change_the_estimate(self):
        """Диагностика обязана быть наблюдателем, а не участником расчёта."""
        bandit = make_controlled_bandit(n=600, seed=72)
        assert self._run(bandit, True).v_b == pytest.approx(self._run(bandit, False).v_b)
