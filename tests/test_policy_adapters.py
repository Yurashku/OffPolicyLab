import numpy as np
import pandas as pd
import pytest

from policyscope.comparison import compare_policies
from policyscope.policy_adapters import DeterministicActionColumnPolicy


def test_deterministic_adapter_integer_actions_one_hot_and_row_sums():
    df = pd.DataFrame({"action_B": [1, 0, 2, 1]})
    policy = DeterministicActionColumnPolicy(action_col="action_B", action_space=[0, 1, 2])
    probs = policy.action_probs(df)

    assert probs.shape == (4, 3)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert np.array_equal(policy.action_argmax(df), df["action_B"].to_numpy())


def test_deterministic_adapter_string_actions_supported():
    df = pd.DataFrame({"action_B": ["email", "sms", "push"]})
    policy = DeterministicActionColumnPolicy(
        action_col="action_B",
        action_space=["email", "sms", "push"],
    )
    probs = policy.action_probs(df)
    assert probs.shape == (3, 3)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert np.array_equal(policy.action_argmax(df), df["action_B"].to_numpy())


def test_deterministic_adapter_unknown_action_raises():
    df = pd.DataFrame({"action_B": [0, 1, 99]})
    policy = DeterministicActionColumnPolicy(action_col="action_B", action_space=[0, 1, 2])
    with pytest.raises(ValueError, match="Unknown actions"):
        policy.action_probs(df)


def test_deterministic_adapter_integrates_with_compare_policies():
    n = 80
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "user_id": np.arange(n),
            "loyal": rng.binomial(1, 0.4, size=n),
            "age": rng.integers(18, 70, size=n),
            "risk": rng.uniform(0.0, 1.0, size=n),
            "income": rng.lognormal(mean=10.3, sigma=0.4, size=n),
            "action_A": rng.choice([0, 1, 2], size=n),
            "action_B": rng.choice([0, 1, 2], size=n),
        }
    )
    df["reward"] = rng.binomial(1, 1.0 / (1.0 + np.exp(-(-0.5 + 0.8 * df["loyal"] - 0.6 * df["risk"]))))
    df["propensity_A"] = rng.uniform(0.1, 0.9, size=n)

    policy_b = DeterministicActionColumnPolicy(action_col="action_B", action_space=[0, 1, 2])
    summary = compare_policies(
        df,
        policy_b,
        estimator="dr",
        target="reward",
        action_col="action_A",
        feature_cols=["loyal", "age", "risk", "income"],
        cluster_col="user_id",
        propensity_source="logged",
        propensity_col="propensity_A",
        n_boot=10,
        alpha=0.1,
    )
    out = summary.to_dict()
    assert "Delta" in out
    assert np.isfinite(out["V_B"])
