import numpy as np

from policyscope.nuisance import fit_behavior_nuisance_bundle, fit_outcome_nuisance_bundle
from policyscope.policies import make_policy
from policyscope.synthetic import SynthConfig, SyntheticRecommenderEnv


def _logs_and_policy(seed: int = 7):
    cfg = SynthConfig(n_users=100, horizon_days=20, seed=seed)
    env = SyntheticRecommenderEnv(cfg)
    X = env.sample_users()
    policyA = make_policy("epsilon_greedy", epsilon=0.1, seed=seed)
    logs = env.simulate_logs_A(policyA, X)
    policyB = make_policy("softmax", tau=0.8, seed=seed + 1)
    return logs, policyB


def test_behavior_bundle_shapes_and_bounds():
    logs, policyB = _logs_and_policy(70)
    bundle = fit_behavior_nuisance_bundle(
        logs,
        policyB,
        feature_cols=["loyal", "age", "risk", "income"],
        action_col="a_A",
    )

    n = len(logs)
    assert bundle.pA_all.shape[0] == n
    assert bundle.pA_taken.shape == (n,)
    assert bundle.piB_taken.shape == (n,)
    assert np.all(bundle.pA_taken > 0)
    assert np.all(bundle.piB_taken >= 0)
    assert np.all(bundle.piB_taken <= 1)


def test_outcome_bundle_predicts_for_binary_target():
    logs, _ = _logs_and_policy(71)
    bundle = fit_outcome_nuisance_bundle(
        logs,
        target="accept",
        feature_cols=["loyal", "age", "risk", "income"],
        action_col="a_A",
    )
    design = bundle.mu_model.predict_proba
    assert callable(design)
