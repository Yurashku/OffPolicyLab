import numpy as np
import pandas as pd

from policyscope.diagnostics import compute_policy_diagnostics
from policyscope.policies import make_policy
from policyscope.synthetic import SynthConfig, SyntheticRecommenderEnv


def _make_logs(seed: int = 0):
    cfg = SynthConfig(n_users=120, horizon_days=20, seed=seed)
    env = SyntheticRecommenderEnv(cfg)
    X = env.sample_users()
    policyA = make_policy("epsilon_greedy", epsilon=0.1, seed=seed)
    logs = env.simulate_logs_A(policyA, X)
    policyB = make_policy("softmax", tau=0.8, seed=seed + 1)
    return logs, policyB


def test_ess_ratio_and_weight_stats_present_for_weighted_estimators():
    logs, policyB = _make_logs(10)
    d = compute_policy_diagnostics(
        logs,
        policyB,
        method="dr",
        target="accept",
        feature_cols=["loyal", "age", "risk", "income"],
        action_col="a_A",
    )
    out = d.to_dict()
    assert out["weight_ess"] is not None
    assert out["weight_ess_ratio"] is not None
    assert 0.0 <= out["weight_ess_ratio"] <= 1.0
    assert out["weight_max"] is not None
    assert out["weight_p95"] is not None
    assert out["weight_p99"] is not None


def test_overlap_diagnostics_for_replay_like_methods():
    logs, policyB = _make_logs(20)
    d = compute_policy_diagnostics(logs, policyB, method="replay", action_col="a_A")
    out = d.to_dict()
    assert 0.0 <= out["replay_overlap"] <= 1.0
    assert 0.0 <= out["replay_coverage"] <= 1.0
    assert out["weight_ess"] is None


def test_warning_rules_trigger_on_obvious_extreme_weights():
    n = 200
    df = pd.DataFrame(
        {
            "user_id": np.arange(n),
            "a_A": np.arange(n) % 2,
            "accept": np.ones(n, dtype=float),
            "f1": np.linspace(0.0, 1.0, n),
        }
    )

    class ExtremePolicy:
        def action_probs(self, d):
            out = np.full((len(d), 2), 1e-6)
            out[:, 0] = 1e-6
            out[:, 1] = 1.0 - 1e-6
            return out

    d = compute_policy_diagnostics(
        df,
        ExtremePolicy(),
        method="ips",
        target="accept",
        feature_cols=["f1"],
        action_col="a_A",
        weight_clip=0.5,
        max_weight_warn_threshold=1.1,
        p99_weight_warn_threshold=1.1,
        clip_share_warn_threshold=0.01,
    ).to_dict()
    assert "extreme_max_weight" in d["warnings"] or "heavy_weight_tail" in d["warnings"]
    assert "large_clip_share" in d["warnings"]
