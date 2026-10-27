"""Nuisance-модели: out-of-fold, группировка фолдов, пол пропенсити."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.dummy import DummyRegressor

from policyscope.data import prepare_logs
from policyscope.nuisance import (
    NuisanceConfig,
    TwoPartRegressor,
    fit_nuisance,
    make_folds,
    outcome_kind,
)

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


class TestOutcomeModels:
    def test_kind_by_data(self):
        rng = np.random.default_rng(0)
        assert outcome_kind(np.array([0.0, 1.0, 1.0]), True) == "binary"
        revenue = np.where(rng.random(500) < 0.7, 0.0, rng.gamma(2.0, 50.0, 500))
        assert outcome_kind(revenue, False) == "two_part"
        assert outcome_kind(rng.normal(size=500), False) == "linear"

    def test_two_part_regressor_recovers_mean(self):
        """E[r|x] = P(r>0|x) · E[r|r>0, x]; прогноз неотрицателен и несмещён в среднем."""
        rng = np.random.default_rng(1)
        x = rng.normal(size=(20000, 1))
        p = 1 / (1 + np.exp(-(x[:, 0] - 0.5)))
        m = np.exp(3.0 + 0.4 * x[:, 0])
        y = (rng.random(20000) < p) * rng.gamma(4.0, m / 4.0)
        pred = TwoPartRegressor().fit(x, y).predict(x)
        assert (pred >= 0).all()
        assert np.mean(pred) == pytest.approx(np.mean(p * m), rel=0.03)
        assert np.corrcoef(pred, p * m)[0, 1] > 0.99

    def test_revenue_predictions_are_not_clipped(self):
        """Регрессия: обрезка прогнозов по [min r, max r] смещала DM."""
        from policyscope.synthetic import OfferSimulator

        sim = OfferSimulator(seed=3)
        logs = sim.simulate(sim.sample_clients(3000), sim.policy("epsilon_greedy", epsilon=0.2, seed=1))
        data = prepare_logs(logs, action="offer", reward="revenue", propensity="propensity",
                            features=["age", "income", "loyal", "risk", "region"])
        mu = fit_nuisance(data, NuisanceConfig(), seed=0).mu["revenue"]
        assert (mu > 0).all()  # двухчастная модель: не отрицательна и без обрезки до 0

    def test_outcome_model_per_metric(self):
        b = make_bandit(400, seed=10)
        logs = b.logs.assign(spend=b.logs["reward"] * 10.0 + b.logs["x1"])
        data = prepare_logs(logs, action="action", reward=["reward", "spend"], features=["x1", "x2"])
        config = NuisanceConfig(outcome_model={"spend": DummyRegressor()})
        mu = fit_nuisance(data, config, seed=0).mu
        assert config.model_for("reward") is None
        assert np.allclose(mu["spend"], mu["spend"][:, :1])  # Dummy: константа по действиям
        assert not np.allclose(mu["reward"], mu["reward"][:, :1])


def test_two_part_does_not_explode_on_outlying_features():
    """Регрессия: log-связь на сыром признаке экстраполирует экспоненциально."""
    rng = np.random.default_rng(2)
    x = rng.normal(size=(3000, 1))
    y = (rng.random(3000) < 0.4) * np.exp(3.0 + 0.8 * x[:, 0])
    model = TwoPartRegressor().fit(x, y)
    pred = model.predict(np.array([[50.0], [-50.0]]))
    assert np.isfinite(pred).all() and pred.max() <= y.max()


def test_non_finite_outcome_predictions_are_refused():
    class Exploding(DummyRegressor):
        def predict(self, X):
            return np.full(len(X), np.inf)

    b = make_bandit(300, seed=11)
    data = prep(b.logs.assign(reward=b.logs["reward"] * 3.0 - 1.0))
    with pytest.raises(ValueError, match="нечисловые"):
        fit_nuisance(data, NuisanceConfig(outcome_model=Exploding()), seed=0)
