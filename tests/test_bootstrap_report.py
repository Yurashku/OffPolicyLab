import numpy as np
import pandas as pd

from policyscope.bootstrap import cluster_bootstrap_ci, paired_bootstrap_ci
from policyscope.report import decision_summary


def test_cluster_bootstrap_ci_basic():
    df = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2],
            "reward": [1.0, 2.0, 3.0, 4.0],
        }
    )
    estimator = lambda d: d["reward"].mean()
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
        df, estimator_pair, cluster_col="user_id", n_boot=200, rng_seed=1
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
