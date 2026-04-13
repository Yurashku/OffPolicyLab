import numpy as np
import pandas as pd
import pytest
from policyscope.synthetic import SynthConfig, SyntheticRecommenderEnv
from policyscope.policies import make_policy
from policyscope.estimators import (
    train_pi_hat,
    pi_hat_predict,
    train_mu_hat,
    prepare_piB_taken,
    ips_value,
    snips_value,
    dr_value,
    sndr_value,
    switch_dr_value,
    estimator_with_bootstrap_ci,
    take_action_probabilities,
)
from policyscope.ci import estimate_value_with_ci
from policyscope.evaluator import OPEEvaluator
from policyscope.inference import (
    ComparisonInferenceResult,
    IntervalResult,
    PolicyComparisonResult,
    infer_policy_comparison_bootstrap,
)
from policyscope.ci import estimate_value
from policyscope.estimators import value_on_policy


def test_estimators_run_on_synthetic():
    cfg = SynthConfig(n_users=100, horizon_days=30, seed=0)
    env = SyntheticRecommenderEnv(cfg)
    X = env.sample_users()
    policyA = make_policy("epsilon_greedy", epsilon=0.1, seed=0)
    logsA = env.simulate_logs_A(policyA, X)
    policyB = make_policy("softmax", tau=0.7, seed=1)
    piB_taken = prepare_piB_taken(logsA, policyB)
    pi_model = train_pi_hat(logsA)
    pA_all = pi_hat_predict(pi_model, logsA)
    pA_taken = take_action_probabilities(pA_all, logsA["a_A"].values, action_space=pi_model.classes_)
    mu_accept = train_mu_hat(logsA, target="accept")
    v_ips, ess, _ = ips_value(logsA, piB_taken, pA_taken, target="accept", weight_clip=20)
    v_snips, _, _ = snips_value(logsA, piB_taken, pA_taken, target="accept", weight_clip=20)
    v_dr, _, _ = dr_value(logsA, policyB, mu_accept, pA_taken, target="accept", weight_clip=20)
    v_sndr, _, _ = sndr_value(logsA, policyB, mu_accept, pA_taken, target="accept", weight_clip=20)
    v_switch, _, _ = switch_dr_value(logsA, policyB, mu_accept, pA_taken, tau=20, target="accept")

    assert np.isfinite(v_ips)
    assert np.isfinite(v_snips)
    assert np.isfinite(v_dr)
    assert np.isfinite(v_sndr)
    assert np.isfinite(v_switch)
    assert ess > 0


def test_invalid_inputs_raise_errors():
    df = pd.DataFrame({"a_A": [0], "accept": [1]})
    piB = np.array([1.0])
    pA = np.array([1.2])
    with pytest.raises(ValueError):
        ips_value(df, piB, pA, target="accept")

    df_missing = pd.DataFrame({"accept": [1]})
    pA2 = np.array([0.5])
    dummy_policy = type("P", (), {"action_probs": lambda self, d: np.ones((len(d), 1))})()
    dummy_model = object()
    with pytest.raises(ValueError):
        dr_value(df_missing, dummy_policy, dummy_model, pA2, target="accept")
    with pytest.raises(ValueError):
        sndr_value(df_missing, dummy_policy, dummy_model, pA2, target="accept")
    with pytest.raises(ValueError):
        switch_dr_value(df_missing, dummy_policy, dummy_model, pA2, tau=1.0, target="accept")


def test_estimators_support_custom_columns():
    cfg = SynthConfig(n_users=80, horizon_days=30, seed=7)
    env = SyntheticRecommenderEnv(cfg)
    X = env.sample_users()
    policyA = make_policy("epsilon_greedy", epsilon=0.2, seed=7)
    logs = env.simulate_logs_A(policyA, X).rename(columns={"a_A": "action_logged", "accept": "reward"})
    policyB = make_policy("softmax", tau=0.9, seed=8)
    piB_taken = prepare_piB_taken(logs, policyB, action_col="action_logged")
    pi_model = train_pi_hat(logs, feature_cols=["loyal", "age", "risk", "income"], action_col="action_logged")
    pA_all = pi_hat_predict(pi_model, logs)
    pA_taken = take_action_probabilities(pA_all, logs["action_logged"].values, action_space=pi_model.classes_)
    mu = train_mu_hat(logs, target="reward", feature_cols=["loyal", "age", "risk", "income"], action_col="action_logged")

    v_dr, _, _ = dr_value(
        logs,
        policyB,
        mu,
        pA_taken,
        target="reward",
        weight_clip=10.0,
        action_col="action_logged",
    )
    assert np.isfinite(v_dr)


def test_generic_estimator_with_bootstrap_ci_runs():
    cfg = SynthConfig(n_users=120, horizon_days=20, seed=11)
    env = SyntheticRecommenderEnv(cfg)
    X = env.sample_users()
    policyA = make_policy("epsilon_greedy", epsilon=0.1, seed=11)
    logs = env.simulate_logs_A(policyA, X)
    res = estimator_with_bootstrap_ci(
        logs,
        lambda part: float(part["accept"].mean()),
        cluster_col="user_id",
        n_boot=30,
        alpha=0.1,
    )
    assert np.isfinite(res["value"])
    low, high = res["CI"]
    assert np.isfinite(low) and np.isfinite(high)
    assert low <= high


def test_estimate_value_with_ci_for_multiple_estimators():
    cfg = SynthConfig(n_users=110, horizon_days=25, seed=21)
    env = SyntheticRecommenderEnv(cfg)
    X = env.sample_users()
    policyA = make_policy("epsilon_greedy", epsilon=0.1, seed=21)
    logs = env.simulate_logs_A(policyA, X)
    policyB = make_policy("softmax", tau=0.9, seed=22)

    for method in ("ips", "snips", "dm", "dr", "sndr", "switch_dr"):
        out = estimate_value_with_ci(
            logs,
            policyB,
            method=method,
            target="accept",
            cluster_col="user_id",
            n_boot=20,
            alpha=0.1,
        )
        assert out["method"] == method
        assert np.isfinite(out["value"])
        lo, hi = out["CI"]
        assert np.isfinite(lo) and np.isfinite(hi)


def test_unified_evaluator_object_with_default_ci():
    cfg = SynthConfig(n_users=100, horizon_days=20, seed=31)
    env = SyntheticRecommenderEnv(cfg)
    X = env.sample_users()
    policyA = make_policy("epsilon_greedy", epsilon=0.1, seed=31)
    logs = env.simulate_logs_A(policyA, X)
    policyB = make_policy("softmax", tau=0.8, seed=32)

    evaluator = OPEEvaluator(
        logs,
        policyB,
        target="accept",
        feature_cols=["loyal", "age", "risk", "income"],
        action_col="a_A",
        cluster_col="user_id",
        n_boot=20,
        alpha=0.1,
        weight_clip=20.0,
        tau=20.0,
    )
    out = evaluator.evaluate("dr")
    assert out["estimator"] == "dr"
    assert np.isfinite(out["V_A"])
    assert np.isfinite(out["V_B"])
    assert np.isfinite(out["Delta"])
    lo, hi = out["V_B_CI"]
    assert np.isfinite(lo) and np.isfinite(hi)
    assert 0.0 <= out["p_value"] <= 1.0
    assert isinstance(out["is_significant"], bool)
    assert out["significance_rule"] == "centered_paired_bootstrap_p_value_lt_alpha"
    assert "paired_percentile_bootstrap" in out["inference_method"]
    assert "diagnostics" in out
    assert "replay_overlap" in out["diagnostics"]
    assert "weight_ess_ratio" in out["diagnostics"]


def test_structured_inference_result_shape():
    cfg = SynthConfig(n_users=90, horizon_days=20, seed=41)
    env = SyntheticRecommenderEnv(cfg)
    X = env.sample_users()
    policyA = make_policy("epsilon_greedy", epsilon=0.1, seed=41)
    logs = env.simulate_logs_A(policyA, X)
    policyB = make_policy("softmax", tau=0.8, seed=42)
    evaluator = OPEEvaluator(
        logs,
        policyB,
        target="accept",
        feature_cols=["loyal", "age", "risk", "income"],
        action_col="a_A",
        cluster_col="user_id",
        n_boot=20,
        alpha=0.1,
    )

    def estimator_pair(part):
        v_a = value_on_policy(part, target="accept")
        v_b = estimate_value(
            part,
            policyB,
            method="dr",
            target="accept",
            feature_cols=["loyal", "age", "risk", "income"],
            action_col="a_A",
        )
        return v_a, v_b, v_b - v_a

    res = infer_policy_comparison_bootstrap(logs, estimator_pair, cluster_col="user_id", n_boot=20, alpha=0.1)
    assert isinstance(res, PolicyComparisonResult)
    assert isinstance(res.inference, ComparisonInferenceResult)
    assert isinstance(res.inference.delta_ci, IntervalResult)
    assert 0.0 <= res.inference.p_value <= 1.0
    assert isinstance(res.inference.is_significant, bool)
    assert res.inference.significance_rule == "centered_paired_bootstrap_p_value_lt_alpha"
    out = res.to_dict()
    assert "Delta_CI" in out and "is_significant" in out
