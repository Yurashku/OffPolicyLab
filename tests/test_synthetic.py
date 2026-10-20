"""Симулятор: оракул совпадает с тем, что реально порождают логи."""

from __future__ import annotations

import numpy as np
import pytest

from policyscope.synthetic import OfferSimulator


@pytest.mark.parametrize("client_effect", [0.0, 1.0])
def test_oracle_matches_simulated_outcomes(client_effect):
    """Средний отклик в логах политики сходится к аналитическому значению."""
    sim = OfferSimulator(seed=0, client_effect=client_effect)
    clients = sim.sample_clients(60_000)
    policy = sim.policy("softmax", temperature=1.0, seed=3)
    logs = sim.simulate(clients, policy)
    for metric, tol in (("accepted", 0.006), ("revenue", 1.5)):
        assert logs[metric].mean() == pytest.approx(sim.true_value(clients, policy, metric), abs=tol)


def test_policies_are_distributions_over_string_actions():
    sim = OfferSimulator(seed=1)
    clients = sim.sample_clients(500)
    for kind in ("greedy", "epsilon_greedy", "softmax", "uniform"):
        probs = sim.policy(kind, seed=2).action_probs(clients)
        assert list(probs.columns) == list(sim.actions)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0)


def test_logged_propensity_is_probability_of_logged_action():
    sim = OfferSimulator(seed=2)
    clients = sim.sample_clients(300)
    policy = sim.policy("epsilon_greedy", epsilon=0.3, seed=4)
    logs = sim.simulate(clients, policy)
    probs = policy.action_probs(clients)
    expected = [probs.loc[i, a] for i, a in zip(logs.index, logs["offer"])]
    np.testing.assert_allclose(logs["propensity"], expected)


def test_visits_repeat_clients_and_share_effect():
    sim = OfferSimulator(seed=3, client_effect=2.0)
    clients = sim.sample_clients(100, visits=4)
    assert len(clients) == 400 and clients["client_id"].nunique() == 100


def test_revenue_is_zero_without_acceptance():
    sim = OfferSimulator(seed=4)
    logs = sim.simulate(sim.sample_clients(2000), sim.policy("uniform"))
    assert (logs.loc[logs["accepted"] == 0, "revenue"] == 0).all()
    assert (logs.loc[logs["accepted"] == 1, "revenue"] > 0).all()
