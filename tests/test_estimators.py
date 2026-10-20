"""Тесты низкоуровневого слоя оценщиков: валидация входов и защита от утечек."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from policyscope.estimate import estimate_point, estimate_value
from policyscope.estimators import (
    NuisanceSpec,
    dm_value,
    dr_value,
    ess,
    importance_weights,
    ips_value,
    is_binary_target,
    make_design,
    prepare_piB_taken,
    resolve_feature_cols,
    snips_value,
    switch_dr_value,
    train_mu_hat,
    validate_policy_probs,
    validate_propensity,
)

from conftest import make_controlled_bandit


class TestESS:
    def test_uniform_weights_give_full_ess(self):
        assert ess(np.ones(100)) == pytest.approx(100.0)

    def test_single_dominant_weight_collapses_ess(self):
        w = np.zeros(100)
        w[0] = 1.0
        assert ess(w) == pytest.approx(1.0)

    def test_non_finite_weights_return_zero(self):
        assert ess(np.array([1.0, np.inf])) == 0.0


class TestFeatureResolution:
    """Утечка колонки награды в признаки делает DM/DR произвольно смещёнными."""

    def test_autoselect_excludes_target_and_action(self):
        df = pd.DataFrame({
            "user_id": [1, 2], "x1": [0.1, 0.2],
            "action": [0, 1], "reward": [1, 0], "propensity": [0.5, 0.5],
        })
        feats = resolve_feature_cols(
            df, None, target="reward", action_col="action",
            cluster_col="user_id", propensity_col="propensity",
        )
        assert feats == ["x1"]

    def test_autoselect_respects_custom_target_name(self):
        """Раньше список исключений был зашит и игнорировал фактический target."""
        df = pd.DataFrame({"x1": [0.1, 0.2], "a": [0, 1], "my_metric": [1, 0]})
        feats = resolve_feature_cols(df, None, target="my_metric", action_col="a")
        assert "my_metric" not in feats and feats == ["x1"]

    def test_explicit_leak_is_rejected(self):
        df = pd.DataFrame({"x1": [0.1, 0.2], "a": [0, 1], "reward": [1, 0]})
        with pytest.raises(ValueError, match="утечка"):
            resolve_feature_cols(df, ["x1", "reward"], target="reward", action_col="a")


class TestValidators:
    def test_nan_propensity_is_rejected(self):
        """NaN проходит наивную проверку диапазона: nan<=0 и nan>1 оба False."""
        with pytest.raises(ValueError, match="NaN"):
            validate_propensity(np.array([0.5, np.nan]), 2)

    @pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
    def test_out_of_range_propensity_is_rejected(self, bad):
        with pytest.raises(ValueError, match="интервале"):
            validate_propensity(np.array([0.5, bad]), 2)

    def test_policy_rows_must_sum_to_one(self):
        with pytest.raises(ValueError, match="суммироваться"):
            validate_policy_probs(np.array([[0.3, 0.3], [0.5, 0.5]]), 2)

    def test_negative_policy_probabilities_rejected(self):
        with pytest.raises(ValueError, match="отрицательные"):
            validate_policy_probs(np.array([[-0.1, 1.1]]), 1)


class TestTargetTypeDetection:
    """Тип модели исхода определяется по данным, а не по имени колонки."""

    def test_binary_detected_regardless_of_column_name(self):
        assert is_binary_target(np.array([0, 1, 1, 0]))
        assert is_binary_target(np.array([0.0, 1.0]))
        assert not is_binary_target(np.array([0.0, 0.5, 1.0]))
        assert not is_binary_target(np.array([10.0, 20.0]))

    def test_binary_reward_under_custom_name_gets_classifier(self):
        bandit = make_controlled_bandit(n=300, seed=1)
        model = train_mu_hat(
            bandit.df, target="reward", feature_cols=["x1", "x2"],
            action_col="action", action_space=bandit.action_space,
        )
        assert model._is_binary is True
        assert hasattr(model, "predict_proba")

    def test_continuous_target_gets_regressor(self):
        df = make_controlled_bandit(n=300, seed=2).df.copy()
        df["revenue"] = np.linspace(10.0, 500.0, len(df))
        model = train_mu_hat(
            df, target="revenue", feature_cols=["x1", "x2"],
            action_col="action", action_space=[0, 1, 2],
        )
        assert model._is_binary is False


class TestDesignMatrix:
    def test_interactions_expand_the_design(self):
        df = make_controlled_bandit(n=100, seed=3).df
        without, *_ = make_design(
            df, feature_cols=["x1", "x2"], action_col="action",
            action_space=[0, 1, 2], interactions=False,
        )
        with_inter, *_ = make_design(
            df, feature_cols=["x1", "x2"], action_col="action",
            action_space=[0, 1, 2], interactions=True,
        )
        assert without.shape[1] == 2 + 3          # признаки + one-hot
        assert with_inter.shape[1] == 2 + 3 + 6   # + взаимодействия x*a

    def test_unknown_action_raises_instead_of_zero_onehot(self):
        """handle_unknown='ignore' молча давал нулевой one-hot и предсказание
        для несуществующего «нулевого» действия."""
        df = make_controlled_bandit(n=100, seed=4).df
        with pytest.raises(ValueError):
            make_design(
                df, feature_cols=["x1"], action_col="action",
                action_space=[0, 1],  # в данных есть действие 2
            )


class TestWeightsAndClipping:
    def test_clip_share_counts_only_clipped_rows(self):
        w, share = importance_weights(
            np.array([1.0, 1.0, 1.0]), np.array([0.5, 0.1, 1.0]), weight_clip=5.0
        )
        assert w.tolist() == [2.0, 5.0, 1.0]
        assert share == pytest.approx(1 / 3)

    def test_no_clipping_by_default(self):
        w, share = importance_weights(np.ones(3), np.array([0.1, 0.5, 1.0]))
        assert share == 0.0
        assert w.max() == pytest.approx(10.0)

    def test_switch_dr_rejects_clip_below_tau(self):
        """Клиппинг ниже tau сделал бы переключение недостижимым."""
        bandit = make_controlled_bandit(n=200, seed=5)
        policy = bandit.policy_b()
        model = train_mu_hat(
            bandit.df, target="reward", feature_cols=["x1", "x2"],
            action_col="action", action_space=bandit.action_space,
        )
        with pytest.raises(ValueError, match="не больше tau"):
            switch_dr_value(
                bandit.df, policy, model, bandit.df["propensity"].to_numpy(), tau=20.0,
                target="reward", weight_clip=5.0, action_col="action",
                action_space=bandit.action_space,
            )


class TestDegenerateCases:
    def test_snips_returns_nan_without_overlap(self):
        """Ноль неотличим от правдоподобного значения награды, поэтому нужен NaN."""
        df = pd.DataFrame({"a": [0, 0], "reward": [1.0, 1.0]})
        value, _, _ = snips_value(
            df, np.zeros(2), np.array([0.5, 0.5]), target="reward", action_col="a"
        )
        assert np.isnan(value)

    def test_ips_length_mismatch_raises(self):
        df = pd.DataFrame({"a": [0, 0], "reward": [1.0, 1.0]})
        with pytest.raises(ValueError, match="piB_taken"):
            ips_value(df, np.ones(3), np.array([0.5, 0.5]), target="reward", action_col="a")


class TestLowLevelMatchesDispatcher:
    """Низкоуровневые функции и диспетчер обязаны давать одно и то же число."""

    @pytest.fixture
    def setup(self):
        bandit = make_controlled_bandit(n=600, seed=6)
        policy = bandit.policy_b()
        model = train_mu_hat(
            bandit.df, target="reward", feature_cols=["x1", "x2"],
            action_col="action", action_space=bandit.action_space,
        )
        return bandit, policy, model

    def test_ips_matches(self, setup):
        bandit, policy, _ = setup
        pA = bandit.df["propensity"].to_numpy()
        piB = prepare_piB_taken(
            bandit.df, policy, action_col="action", action_space=bandit.action_space
        )
        direct, _, _ = ips_value(bandit.df, piB, pA, target="reward", action_col="action")
        via = estimate_value(
            bandit.df, policy, method="ips", target="reward", action_col="action",
            action_space=bandit.action_space, feature_cols=["x1", "x2"],
            propensity_source="logged", propensity_col="propensity",
        )
        assert direct == pytest.approx(via)

    def test_dm_and_dr_match(self, setup):
        bandit, policy, model = setup
        pA = bandit.df["propensity"].to_numpy()
        direct_dm = dm_value(
            bandit.df, policy, model, target="reward", action_space=bandit.action_space
        )
        direct_dr, _, _ = dr_value(
            bandit.df, policy, model, pA, target="reward",
            action_col="action", action_space=bandit.action_space,
        )
        common = dict(
            target="reward", action_col="action", action_space=bandit.action_space,
            feature_cols=["x1", "x2"], propensity_source="logged",
            propensity_col="propensity",
        )
        assert direct_dm == pytest.approx(
            estimate_value(bandit.df, policy, method="dm", **common)
        )
        assert direct_dr == pytest.approx(
            estimate_value(bandit.df, policy, method="dr", **common)
        )


class TestPluggableModels:
    def test_custom_outcome_model_is_used(self):
        bandit = make_controlled_bandit(n=400, seed=8)
        policy = bandit.policy_b()
        spec = NuisanceSpec(outcome_model=RandomForestClassifier(n_estimators=5, random_state=0))
        point = estimate_point(
            bandit.df, policy, method="dm", target="reward", action_col="action",
            action_space=bandit.action_space, feature_cols=["x1", "x2"], spec=spec,
        )
        assert np.isfinite(point.value)

    def test_spec_object_is_not_mutated_between_fits(self):
        """Модель клонируется на каждый fit, иначе состояние протекает между фолдами."""
        base = RandomForestClassifier(n_estimators=5, random_state=0)
        spec = NuisanceSpec(outcome_model=base)
        bandit = make_controlled_bandit(n=300, seed=9)
        train_mu_hat(
            bandit.df, target="reward", feature_cols=["x1", "x2"],
            action_col="action", action_space=bandit.action_space, spec=spec,
        )
        assert not hasattr(base, "classes_"), "переданная модель была обучена на месте"
