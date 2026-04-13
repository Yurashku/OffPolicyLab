import numpy as np

from policyscope.comparison import (
    MultiMetricComparisonResult,
    PolicyComparisonSummary,
    compare_policies,
    compare_policies_multi_target,
)
from policyscope.evaluator import OPEEvaluator
from policyscope.policies import make_policy
from policyscope.report import decision_summary
from policyscope.synthetic import SynthConfig, SyntheticRecommenderEnv


def _prepare_env(seed: int = 100):
    cfg = SynthConfig(n_users=120, horizon_days=20, seed=seed)
    env = SyntheticRecommenderEnv(cfg)
    X = env.sample_users()
    policyA = make_policy("epsilon_greedy", epsilon=0.1, seed=seed)
    logs = env.simulate_logs_A(policyA, X)
    policyB = make_policy("softmax", tau=0.8, seed=seed + 1)
    return logs, policyB


def test_official_comparison_entrypoint_shape():
    logs, policyB = _prepare_env(101)
    out = compare_policies(
        logs,
        policyB,
        estimator="dr",
        target="accept",
        feature_cols=["loyal", "age", "risk", "income"],
        action_col="a_A",
        n_boot=20,
        alpha=0.1,
        weight_clip=20.0,
    )
    assert isinstance(out, PolicyComparisonSummary)
    d = out.to_dict()
    assert d["estimator"] == "dr"
    assert d["target"] == "accept"
    assert "V_A_CI" in d and "Delta_CI" in d
    assert "diagnostics" in d and "weight_ess_ratio" in d["diagnostics"]
    assert 0.0 <= d["p_value"] <= 1.0


def test_multi_target_repeated_scalar_evaluation():
    logs, policyB = _prepare_env(102)
    out = compare_policies_multi_target(
        logs,
        policyB,
        estimator="dr",
        targets=["accept", "cltv"],
        feature_cols=["loyal", "age", "risk", "income"],
        action_col="a_A",
        n_boot=20,
        alpha=0.1,
    )
    assert isinstance(out, MultiMetricComparisonResult)
    assert set(out.results.keys()) == {"accept", "cltv"}
    assert out.results["accept"].target == "accept"
    assert out.results["cltv"].target == "cltv"


def test_evaluator_delegates_to_official_comparison_path():
    logs, policyB = _prepare_env(103)
    evaluator = OPEEvaluator(
        logs,
        policyB,
        target="accept",
        feature_cols=["loyal", "age", "risk", "income"],
        action_col="a_A",
        n_boot=20,
        alpha=0.1,
    )
    summary = evaluator.evaluate_summary("dr")
    out = evaluator.evaluate("dr")
    assert isinstance(summary, PolicyComparisonSummary)
    assert np.isclose(summary.delta, out["Delta"])
    assert summary.to_dict()["diagnostics"]["replay_overlap"] == out["diagnostics"]["replay_overlap"]


def test_report_accepts_structured_summary_object():
    logs, policyB = _prepare_env(104)
    summary = compare_policies(
        logs,
        policyB,
        estimator="dr",
        target="accept",
        feature_cols=["loyal", "age", "risk", "income"],
        action_col="a_A",
        n_boot=20,
        alpha=0.1,
    )
    text = decision_summary(summary, metric_name="accept", business_threshold=0.0)
    assert "Delta (B−A)" in text
