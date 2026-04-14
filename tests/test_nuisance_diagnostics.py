import numpy as np
import pandas as pd

from policyscope.comparison import compare_policies
from policyscope.nuisance import BehaviorPredictions, fit_crossfit_nuisance_bundle, resolve_behavior_predictions
from policyscope.nuisance_diagnostics import _compute_behavior_diagnostics
from policyscope.policies import make_policy
from policyscope.synthetic import SynthConfig, SyntheticRecommenderEnv


def _logs_and_policy(seed: int = 12):
    cfg = SynthConfig(n_users=120, horizon_days=20, seed=seed)
    env = SyntheticRecommenderEnv(cfg)
    X = env.sample_users()
    policyA = make_policy("epsilon_greedy", epsilon=0.1, seed=seed)
    logs = env.simulate_logs_A(policyA, X)
    policyB = make_policy("softmax", tau=0.8, seed=seed + 1)
    est, _, _, _ = resolve_behavior_predictions(
        logs,
        policyB,
        propensity_source="estimated",
        feature_cols=["loyal", "age", "risk", "income"],
        action_col="a_A",
    )
    logs = logs.copy()
    logs["p_logged"] = est.pA_taken
    return logs, policyB


def test_behavior_diagnostics_available_for_estimated_propensity():
    logs, policyB = _logs_and_policy(40)
    summary = compare_policies(
        logs,
        policyB,
        estimator="ips",
        target="accept",
        feature_cols=["loyal", "age", "risk", "income"],
        action_col="a_A",
        with_ci=False,
        propensity_source="estimated",
    )
    nd = summary.nuisance_diagnostics
    assert nd is not None
    assert nd.behavior.applicable
    assert nd.behavior.multiclass_log_loss is not None


def test_logged_propensity_marks_behavior_diagnostics_not_applicable():
    logs, policyB = _logs_and_policy(41)
    summary = compare_policies(
        logs,
        policyB,
        estimator="ips",
        target="accept",
        feature_cols=["loyal", "age", "risk", "income"],
        action_col="a_A",
        with_ci=False,
        propensity_source="logged",
        propensity_col="p_logged",
    )
    nd = summary.nuisance_diagnostics
    assert nd is not None
    assert not nd.behavior.applicable


def test_outcome_diagnostics_binary_and_crossfit_oof_flag():
    logs, policyB = _logs_and_policy(42)
    bundle = fit_crossfit_nuisance_bundle(
        logs,
        policyB,
        target="accept",
        n_splits=3,
        random_state=9,
        feature_cols=["loyal", "age", "risk", "income"],
        action_col="a_A",
    )
    summary = compare_policies(
        logs,
        policyB,
        estimator="dr",
        target="accept",
        feature_cols=["loyal", "age", "risk", "income"],
        action_col="a_A",
        with_ci=False,
        nuisance_bundle=bundle,
        propensity_source="estimated",
    )
    nd = summary.nuisance_diagnostics
    assert nd is not None
    assert nd.outcome.applicable
    assert nd.outcome.is_binary_target
    assert nd.outcome.is_out_of_fold
    assert nd.outcome.log_loss is not None


def test_behavior_top1_diagnostics_support_string_action_labels():
    df = pd.DataFrame({"action_logged": ["email", "sms", "email", "push"]})
    behavior = BehaviorPredictions(
        pA_taken=np.array([0.9, 0.8, 0.85, 0.7]),
        piB_taken=np.array([0.2, 0.2, 0.2, 0.2]),
        pA_all=np.array(
            [
                [0.9, 0.1, 0.0],
                [0.1, 0.8, 0.1],
                [0.7, 0.2, 0.1],
                [0.1, 0.2, 0.7],
            ]
        ),
        propensity_source="estimated",
        action_space=("email", "sms", "push"),
    )
    diag = _compute_behavior_diagnostics(
        df,
        action_col="action_logged",
        behavior_predictions=behavior,
        propensity_source="estimated",
    )
    assert diag.top1_accuracy == 1.0


def test_behavior_top1_diagnostics_int_labels_without_action_space_still_work():
    df = pd.DataFrame({"a_A": [0, 1, 2, 1]})
    behavior = BehaviorPredictions(
        pA_taken=np.array([0.9, 0.8, 0.85, 0.7]),
        piB_taken=np.array([0.2, 0.2, 0.2, 0.2]),
        pA_all=np.array(
            [
                [0.9, 0.1, 0.0],
                [0.1, 0.8, 0.1],
                [0.1, 0.2, 0.7],
                [0.2, 0.6, 0.2],
            ]
        ),
        propensity_source="estimated",
    )
    diag = _compute_behavior_diagnostics(
        df,
        action_col="a_A",
        behavior_predictions=behavior,
        propensity_source="estimated",
    )
    assert diag.top1_accuracy == 1.0
