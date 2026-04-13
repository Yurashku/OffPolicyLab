import numpy as np
import pandas as pd

from policyscope.bootstrap import cluster_bootstrap_ci, paired_bootstrap_ci
from policyscope.inference import infer_policy_comparison_bootstrap
from policyscope.report import analyze_logs, decision_summary


def test_cluster_bootstrap_ci_basic():
    df = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2],
            "reward": [1.0, 2.0, 3.0, 4.0],
        }
    )

    def estimator(d):
        return d["reward"].mean()

    theta, low, high = cluster_bootstrap_ci(
        df, estimator, cluster_col="user_id", n_boot=200, rng_seed=0
    )
    assert isinstance(theta, float)
    assert isinstance(low, float) and isinstance(high, float)
    assert low <= theta <= high
    assert np.isclose(theta, df["reward"].mean())
    assert low <= df["reward"].mean() <= high


def test_paired_bootstrap_ci_basic():
    df = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2],
            "val_a": [1.0, 1.0, 2.0, 2.0],
            "val_b": [2.0, 2.0, 3.0, 3.0],
        }
    )

    def estimator_pair(d):
        va = d["val_a"].mean()
        vb = d["val_b"].mean()
        return va, vb, vb - va

    res = paired_bootstrap_ci(
        df, estimator_pair, cluster_col="user_id", n_boot=200, alpha=0.1, rng_seed=1
    )
    mean_a = df["val_a"].mean()
    mean_b = df["val_b"].mean()
    delta = mean_b - mean_a

    assert np.isclose(res["V_A"], mean_a)
    assert np.isclose(res["V_B"], mean_b)
    assert np.isclose(res["Delta"], delta)

    for key in ["V_A_CI", "V_B_CI", "Delta_CI"]:
        lo, hi = res[key]
        assert lo <= hi
        assert isinstance(lo, float) and isinstance(hi, float)
    lo, hi = res["Delta_CI"]
    assert lo <= delta <= hi
    assert res["n_boot"] == 200
    assert res["alpha"] == 0.1
    assert 0.0 <= res["p_value"] <= 1.0
    assert isinstance(res["is_significant"], bool)
    assert res["significance_rule"] == "centered_paired_bootstrap_p_value_lt_alpha"
    assert "paired_percentile_bootstrap" in res["inference_method"]


def test_paired_bootstrap_ci_respects_alpha_width():
    df = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2, 3, 3, 4, 4],
            "val_a": [0.0, 0.1, 0.2, 0.3, 0.0, 0.1, 0.2, 0.3],
            "val_b": [0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4],
        }
    )

    def estimator_pair(d):
        va = d["val_a"].mean()
        vb = d["val_b"].mean()
        return va, vb, vb - va

    res_95 = paired_bootstrap_ci(df, estimator_pair, cluster_col="user_id", n_boot=400, alpha=0.05, rng_seed=2)
    res_80 = paired_bootstrap_ci(df, estimator_pair, cluster_col="user_id", n_boot=400, alpha=0.20, rng_seed=2)
    w95 = res_95["Delta_CI"][1] - res_95["Delta_CI"][0]
    w80 = res_80["Delta_CI"][1] - res_80["Delta_CI"][0]
    assert w80 <= w95


def test_paired_bootstrap_ci_matches_official_inference_wrapper():
    df = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2],
            "val_a": [1.0, 1.0, 2.0, 2.0],
            "val_b": [2.0, 2.0, 3.0, 3.0],
        }
    )

    def estimator_pair(d):
        va = d["val_a"].mean()
        vb = d["val_b"].mean()
        return va, vb, vb - va

    legacy = paired_bootstrap_ci(df, estimator_pair, cluster_col="user_id", n_boot=100, alpha=0.1, rng_seed=9)
    official = infer_policy_comparison_bootstrap(
        df, estimator_pair, cluster_col="user_id", n_boot=100, alpha=0.1, rng_seed=9
    ).to_dict()
    assert legacy["Delta_CI"] == official["Delta_CI"]
    assert legacy["is_significant"] == official["is_significant"]
    assert legacy["p_value"] == official["p_value"]


def test_centered_bootstrap_p_value_obvious_cases():
    df_equal = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2, 3, 3],
            "val_a": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "val_b": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )
    df_large_effect = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2, 3, 3],
            "val_a": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "val_b": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )

    def estimator_pair(d):
        va = d["val_a"].mean()
        vb = d["val_b"].mean()
        return va, vb, vb - va

    eq = paired_bootstrap_ci(df_equal, estimator_pair, cluster_col="user_id", n_boot=400, alpha=0.05, rng_seed=5)
    big = paired_bootstrap_ci(
        df_large_effect, estimator_pair, cluster_col="user_id", n_boot=400, alpha=0.05, rng_seed=5
    )
    assert eq["p_value"] > 0.3
    assert big["p_value"] < 0.05
    assert eq["is_significant"] is False
    assert big["is_significant"] is True


def test_decision_summary_outcomes():
    base = {
        "V_A": 0.1,
        "V_B": 0.15,
        "V_A_CI": (0.09, 0.11),
        "V_B_CI": (0.14, 0.16),
    }

    res_pos = {**base, "Delta": 0.05, "Delta_CI": (0.02, 0.08)}
    txt_pos = decision_summary(res_pos, "metric", business_threshold=0.01)
    assert "модель B лучше A" in txt_pos

    res_neg = {**base, "Delta": -0.05, "Delta_CI": (-0.08, -0.02)}
    txt_neg = decision_summary(res_neg, "metric", business_threshold=0.01)
    assert "модель A лучше B" in txt_neg

    res_neu = {**base, "Delta": 0.0, "Delta_CI": (-0.03, 0.04)}
    txt_neu = decision_summary(res_neu, "metric", business_threshold=0.01)
    assert "статистически значимого отличия" in txt_neu


def test_decision_summary_uses_alpha_from_result():
    res = {
        "V_A": 0.2,
        "V_B": 0.25,
        "Delta": 0.05,
        "V_A_CI": (0.18, 0.22),
        "V_B_CI": (0.20, 0.30),
        "Delta_CI": (0.01, 0.09),
        "alpha": 0.1,
        "trust_level": "caution",
        "recommendation": "check diagnostics",
    }
    txt = decision_summary(res, "metric", business_threshold=0.0)
    assert "90% CI" in txt
    assert "Уровень доверия к оценке: caution." in txt


def test_decision_summary_legacy_output_without_ci_is_supported():
    res = {"V_A": 0.2, "V_B": 0.25, "Delta": 0.05, "is_significant": False}
    txt = decision_summary(res, "metric", business_threshold=0.0)
    assert "CI недоступен" in txt
    assert "итог следует трактовать как предварительный" in txt


def test_analyze_logs_mentions_official_workflow_and_propensity_modes():
    df = pd.DataFrame(
        {
            "user_id": [1, 2, 3],
            "a_A": [0, 1, 0],
            "accept": [1.0, 0.0, 1.0],
            "age": [20, 30, 40],
        }
    )
    txt = analyze_logs(df, policyB=None, propensity_col="propensity_A")
    assert "compare_policies" in txt
    assert "режим auto перейдёт в estimated propensity path" in txt
    assert "strict logged path требует валидную propensity_col" in txt
    assert "DR/SNDR/Switch-DR" in txt
