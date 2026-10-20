"""Тесты слоя nuisance: фолды cross-fitting, источник пропенсити, валидация."""

from __future__ import annotations

import numpy as np
import pytest

from policyscope.nuisance import (
    BehaviorPredictions,
    OutcomePredictions,
    build_logged_behavior_predictions,
    fit_behavior_nuisance_bundle,
    fit_crossfit_nuisance_bundle,
    generate_oof_behavior_predictions,
    generate_oof_outcome_predictions,
    make_kfold_indices,
    resolve_behavior_predictions,
    validate_behavior_predictions,
    validate_outcome_predictions,
)

from conftest import make_controlled_bandit


@pytest.fixture
def setup():
    bandit = make_controlled_bandit(n=500, seed=51, rows_per_user=5)
    return bandit, bandit.policy_b()


class TestFolds:
    def test_holdouts_partition_all_rows_exactly_once(self):
        folds = make_kfold_indices(100, n_splits=5, random_state=0)
        seen = np.concatenate([h for _, h in folds])
        assert sorted(seen) == list(range(100))

    def test_train_and_holdout_are_disjoint(self):
        for train, holdout in make_kfold_indices(60, n_splits=4, random_state=1):
            assert not set(train) & set(holdout)

    def test_grouped_folds_keep_clusters_together(self):
        """Иначе строки одного пользователя попадут и в train, и в holdout."""
        groups = np.repeat(np.arange(40), 5)
        folds = make_kfold_indices(len(groups), n_splits=5, groups=groups, random_state=0)
        for train, holdout in folds:
            assert not set(groups[train]) & set(groups[holdout])

    def test_grouped_folds_still_cover_all_rows(self):
        groups = np.repeat(np.arange(40), 5)
        folds = make_kfold_indices(len(groups), n_splits=5, groups=groups, random_state=0)
        seen = np.concatenate([h for _, h in folds])
        assert sorted(seen) == list(range(len(groups)))

    def test_too_few_groups_raises(self):
        with pytest.raises(ValueError, match="Групп"):
            make_kfold_indices(10, n_splits=5, groups=np.repeat([0, 1], 5))

    def test_n_splits_below_two_rejected(self):
        with pytest.raises(ValueError, match="n_splits"):
            make_kfold_indices(10, n_splits=1)

    def test_folds_are_reproducible(self):
        a = make_kfold_indices(50, n_splits=5, random_state=7)
        b = make_kfold_indices(50, n_splits=5, random_state=7)
        for (t1, h1), (t2, h2) in zip(a, b):
            assert np.array_equal(t1, t2) and np.array_equal(h1, h2)


class TestBehaviorNuisance:
    def test_estimated_propensities_are_valid_probabilities(self, setup):
        bandit, policy = setup
        bundle = fit_behavior_nuisance_bundle(
            bandit.df, policy, feature_cols=["x1", "x2"],
            action_col="action", action_space=bandit.action_space,
        )
        preds = bundle.predictions
        assert ((preds.pA_taken > 0) & (preds.pA_taken <= 1)).all()
        assert np.allclose(preds.pA_all.sum(axis=1), 1.0)
        assert preds.pA_all_actions == tuple(bundle.pi_model.classes_)

    def test_oof_predictions_cover_all_rows(self, setup):
        bandit, policy = setup
        preds = generate_oof_behavior_predictions(
            bandit.df, policy, feature_cols=["x1", "x2"], n_splits=4,
            action_col="action", action_space=bandit.action_space,
        )
        assert preds.is_out_of_fold
        assert not np.isnan(preds.pA_taken).any()
        assert set(np.unique(preds.fold_index)) == {0, 1, 2, 3}

    def test_oof_folds_respect_cluster_column(self, setup):
        bandit, policy = setup
        preds = generate_oof_behavior_predictions(
            bandit.df, policy, feature_cols=["x1", "x2"], n_splits=4,
            action_col="action", action_space=bandit.action_space, cluster_col="user_id",
        )
        folds_per_user = bandit.df.assign(fold=preds.fold_index).groupby("user_id")["fold"].nunique()
        assert (folds_per_user == 1).all(), "пользователь распределён по нескольким фолдам"

    def test_unsupported_action_share_is_tracked(self):
        """Подстановка eps вместо неизвестного пропенсити должна быть видна."""
        preds = BehaviorPredictions(
            pA_taken=np.array([0.5, 1e-6]),
            piB_taken=np.array([0.5, 0.5]),
            unsupported_action_share=0.5,
        )
        assert preds.unsupported_action_share == 0.5


class TestPropensitySource:
    def test_logged_predictions_use_column_values(self, setup):
        bandit, policy = setup
        preds = build_logged_behavior_predictions(
            bandit.df, policy, propensity_col="propensity",
            action_col="action", action_space=bandit.action_space,
        )
        assert np.allclose(preds.pA_taken, bandit.df["propensity"].to_numpy())
        assert preds.propensity_source == "logged"

    def test_invalid_logged_propensity_rejected(self, setup):
        bandit, policy = setup
        df = bandit.df.copy()
        df.loc[df.index[0], "propensity"] = 0.0
        with pytest.raises(ValueError, match="интервале"):
            build_logged_behavior_predictions(
                df, policy, propensity_col="propensity",
                action_col="action", action_space=bandit.action_space,
            )

    def test_nan_logged_propensity_rejected(self, setup):
        bandit, policy = setup
        df = bandit.df.copy()
        df.loc[df.index[0], "propensity"] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            build_logged_behavior_predictions(
                df, policy, propensity_col="propensity",
                action_col="action", action_space=bandit.action_space,
            )

    def test_auto_prefers_logged_when_available(self, setup):
        bandit, policy = setup
        _, source, col, notes = resolve_behavior_predictions(
            bandit.df, policy, feature_cols=["x1", "x2"], propensity_col="propensity",
            action_col="action", action_space=bandit.action_space,
        )
        assert source == "logged" and col == "propensity"
        assert "auto_selected_logged_propensity" in notes

    def test_auto_falls_back_when_column_invalid(self, setup):
        bandit, policy = setup
        df = bandit.df.copy()
        df["propensity"] = 0.0
        _, source, _, notes = resolve_behavior_predictions(
            df, policy, feature_cols=["x1", "x2"], propensity_col="propensity",
            action_col="action", action_space=bandit.action_space,
        )
        assert source == "estimated"
        assert "logged_propensity_invalid_fallback_to_estimated" in notes


class TestCrossfitBundle:
    def test_bundle_contains_both_sides_and_target(self, setup):
        bandit, policy = setup
        bundle = fit_crossfit_nuisance_bundle(
            bandit.df, policy, target="reward", feature_cols=["x1", "x2"],
            action_col="action", action_space=bandit.action_space, n_splits=3,
        )
        assert bundle.behavior is not None and bundle.outcome is not None
        assert bundle.target == "reward"
        assert set(bundle.outcome.mu_by_action) == set(bandit.action_space)

    def test_logged_propensity_is_not_replaced_by_a_model(self, setup):
        """Логированные пропенсити точны и в cross-fitting не нуждаются."""
        bandit, policy = setup
        bundle = fit_crossfit_nuisance_bundle(
            bandit.df, policy, target="reward", feature_cols=["x1", "x2"],
            action_col="action", action_space=bandit.action_space, n_splits=3,
            propensity_source="logged", propensity_col="propensity",
        )
        assert bundle.behavior.propensity_source == "logged"
        assert np.allclose(bundle.behavior.pA_taken, bandit.df["propensity"].to_numpy())
        assert "crossfit_skipped_for_logged_propensity" in bundle.notes

    def test_outcome_predictions_are_out_of_fold(self, setup):
        bandit, policy = setup
        preds = generate_oof_outcome_predictions(
            bandit.df, target="reward", feature_cols=["x1", "x2"], n_splits=3,
            action_col="action", action_space=bandit.action_space,
            requested_actions=bandit.action_space,
        )
        assert preds.is_out_of_fold
        assert not np.isnan(preds.mu_logged_action).any()


class TestValidation:
    def test_behavior_length_mismatch_rejected(self):
        preds = BehaviorPredictions(pA_taken=np.ones(3), piB_taken=np.ones(3))
        with pytest.raises(ValueError, match="длину 5"):
            validate_behavior_predictions(preds, 5)

    def test_outcome_missing_action_rejected(self):
        preds = OutcomePredictions(
            mu_logged_action=np.ones(3), mu_by_action={0: np.ones(3)}
        )
        with pytest.raises(ValueError, match="нет действия"):
            validate_outcome_predictions(preds, 3, required_actions=[0, 1])

    def test_outcome_without_per_action_rejected_for_dr_family(self):
        preds = OutcomePredictions(mu_logged_action=np.ones(3))
        with pytest.raises(ValueError, match="mu_by_action"):
            validate_outcome_predictions(preds, 3, required_actions=[0])
