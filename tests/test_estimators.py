import numpy as np
import pandas as pd
import pytest
from policyscope.synthetic import SynthConfig, SyntheticRecommenderEnv
from policyscope.policies import make_policy
from policyscope.estimators import (
    train_mu_hat,
    prepare_piB_taken,
    ips_value,
    snips_value,
    dr_value,
)


def test_estimators_run_on_synthetic():
    cfg = SynthConfig(n_users=100, horizon_days=30, seed=0)
    env = SyntheticRecommenderEnv(cfg)
    X = env.sample_users()
    policyA = make_policy("epsilon_greedy", epsilon=0.1, seed=0)
    logsA = env.simulate_logs_A(policyA, X)
    policyB = make_policy("softmax", tau=0.7, seed=1)
    piB_taken = prepare_piB_taken(logsA, policyB)
    mu_accept = train_mu_hat(logsA, target="accept")
    v_ips, ess, _ = ips_value(logsA, piB_taken, target="accept", weight_clip=20)
    v_snips, _, _ = snips_value(logsA, piB_taken, target="accept", weight_clip=20)
    v_dr, _, _ = dr_value(logsA, policyB, mu_accept, target="accept", weight_clip=20)

    assert np.isfinite(v_ips)
    assert np.isfinite(v_snips)
    assert np.isfinite(v_dr)
    assert ess > 0


def test_invalid_inputs_raise_errors():
    df = pd.DataFrame({"a_A": [0], "propensity_A": [1.2], "accept": [1]})
    piB = np.array([1.0])
    with pytest.raises(ValueError):
        ips_value(df, piB, target="accept")

    df_missing = pd.DataFrame({"propensity_A": [0.5], "accept": [1]})
    dummy_policy = type("P", (), {"action_probs": lambda self, d: np.ones((len(d), 1))})()
    dummy_model = object()
    with pytest.raises(ValueError):
        dr_value(df_missing, dummy_policy, dummy_model, target="accept")
