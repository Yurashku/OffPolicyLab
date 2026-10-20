"""Nuisance-модели: out-of-fold, группировка фолдов, пол пропенсити."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.dummy import DummyRegressor

from policyscope.data import prepare_logs
from policyscope.nuisance import NuisanceConfig, fit_nuisance, make_folds

from conftest import make_bandit


def prep(logs, **kw):
    kw.setdefault("features", ["x1", "x2"])
    return prepare_logs(logs, action="action", reward="reward", **kw)


class TestFolds:
    def test_groups_stay_together(self):
        groups = np.repeat(np.arange(40), 3)
        folds = make_folds(groups, 5, np.random.default_rng(0))
        for g in range(40):
            assert len(set(folds[groups == g])) == 1

    def test_bootstrap_duplicates_share_a_fold(self):
        """Регрессия: копии одной строки реплики не должны попадать в train и holdout."""
        positions = np.random.default_rng(1).integers(0, 100, 100)
        folds = make_folds(positions, 5, np.random.default_rng(0))
        for p in np.unique(positions):
            assert len(set(folds[positions == p])) == 1

    def test_too_few_groups(self):
        with pytest.raises(ValueError, match="меньше"):
            make_folds(np.array([0, 0, 1]), 5, np.random.default_rng(0))

    def test_invalid_config(self):
        with pytest.raises(ValueError):
            NuisanceConfig(n_folds=1)
        with pytest.raises(ValueError):
            NuisanceConfig(propensity_floor=0.0)


class TestOutOfFold:
    def test_prediction_does_not_depend_on_own_reward(self):
        """Определение OOF: прогноз строки не меняется, если поменять её награду."""
        b = make_bandit(600, seed=3)
        data = prep(b.logs, propensity="propensity")
        nuis = fit_nuisance(data, NuisanceConfig(), seed=0)
        folds = make_folds(np.arange(data.n), 5, np.random.default_rng(0))
        flipped = b.logs.copy()
        hold = folds == 0
        flipped.loc[hold, "reward"] = 1 - flipped.loc[hold, "reward"]
        nuis2 = fit_nuisance(prep(flipped, propensity="propensity"), NuisanceConfig(), seed=0)
        np.testing.assert_allclose(nuis.mu["reward"][hold], nuis2.mu["reward"][hold])
        assert not np.allclose(nuis.mu["reward"][~hold], nuis2.mu["reward"][~hold])

    def test_logged_propensities_used_as_is(self):
        b = make_bandit(300, seed=1)
        nuis = fit_nuisance(prep(b.logs, propensity="propensity"), NuisanceConfig(), seed=0)
        assert nuis.source == "logged"
        np.testing.assert_allclose(nuis.p_taken, b.logs["propensity"])

    def test_outcome_predictions_in_reward_range(self):
        b = make_bandit(500, seed=2)
        mu = fit_nuisance(prep(b.logs), NuisanceConfig(), seed=0).mu["reward"]
        assert mu.shape == (500, 3) and (mu >= 0).all() and (mu <= 1).all()


class TestEstimatedPropensity:
    def test_floor_per_action(self):
        b = make_bandit(800, seed=4)
        data = prep(b.logs)
        nuis = fit_nuisance(data, NuisanceConfig(propensity_floor=0.5), seed=0)
        shares = np.bincount(data.a) / data.n
        np.testing.assert_allclose(nuis.floor, 0.5 * shares)
        assert (nuis.p_taken >= nuis.floor[data.a] - 1e-12).all()
        assert (nuis.floored(data.a, 2.0) >= nuis.p_taken).all()

    def test_estimates_close_to_truth_on_softmax_logging(self):
        b = make_bandit(4000, seed=5)
        nuis = fit_nuisance(prep(b.logs), NuisanceConfig(), seed=0)
        assert np.mean(np.abs(nuis.p_taken - b.logs["propensity"])) < 0.03

    def test_single_action_logs(self):
        b = make_bandit(200, seed=6)
        logs = b.logs.assign(action="red")
        nuis = fit_nuisance(prep(logs), NuisanceConfig(), seed=0)
        np.testing.assert_allclose(nuis.p_taken, 1.0)


def test_custom_outcome_model_is_used():
    b = make_bandit(300, seed=8)
    data = prep(b.logs.assign(reward=b.logs["reward"] * 10.0))  # непрерывная награда
    nuis = fit_nuisance(data, NuisanceConfig(outcome_model=DummyRegressor()), seed=0)
    mu = nuis.mu["reward"]
    assert np.allclose(mu, mu[:, :1])  # константа по действиям внутри фолда


def test_context_free_evaluation():
    b = make_bandit(300, seed=9)
    nuis = fit_nuisance(prep(b.logs, features=[]), NuisanceConfig(), seed=0)
    assert np.isfinite(nuis.mu["reward"]).all()
