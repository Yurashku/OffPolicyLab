import numpy as np
import pandas as pd

from policyscope.nuisance import (
    CrossFitNuisanceBundle,
    fit_behavior_nuisance_bundle,
    fit_outcome_nuisance_bundle,
    generate_oof_behavior_predictions,
    generate_oof_outcome_predictions,
    make_kfold_indices,
    fit_crossfit_nuisance_bundle,
)
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

    preds = bundle.predictions
    n = len(logs)
    assert preds.pA_all is not None
    assert preds.pA_all.shape[0] == n
    assert preds.pA_taken.shape == (n,)
    assert preds.piB_taken.shape == (n,)
    assert np.all(preds.pA_taken > 0)
    assert np.all(preds.piB_taken >= 0)
    assert np.all(preds.piB_taken <= 1)


def test_outcome_bundle_predicts_for_binary_target():
    logs, _ = _logs_and_policy(71)
    bundle = fit_outcome_nuisance_bundle(
        logs,
        target="accept",
        feature_cols=["loyal", "age", "risk", "income"],
        action_col="a_A",
    )
    assert callable(bundle.mu_model.predict_proba)


def test_oof_behavior_predictions_have_fold_alignment():
    logs, policyB = _logs_and_policy(72)
    preds = generate_oof_behavior_predictions(
        logs,
        policyB,
        n_splits=4,
        random_state=9,
        feature_cols=["loyal", "age", "risk", "income"],
        action_col="a_A",
    )
    assert preds.is_out_of_fold
    assert preds.fold_index is not None
    assert preds.pA_taken.shape == (len(logs),)
    assert preds.piB_taken.shape == (len(logs),)
    assert set(np.unique(preds.fold_index)) == {0, 1, 2, 3}


def test_oof_predictions_support_custom_action_column():
    logs, policyB = _logs_and_policy(73)
    logs = logs.rename(columns={"a_A": "action_logged"})
    bpreds = generate_oof_behavior_predictions(
        logs,
        policyB,
        n_splits=3,
        random_state=7,
        feature_cols=["loyal", "age", "risk", "income"],
        action_col="action_logged",
    )
    opreds = generate_oof_outcome_predictions(
        logs,
        target="accept",
        n_splits=3,
        random_state=7,
        feature_cols=["loyal", "age", "risk", "income"],
        action_col="action_logged",
        requested_actions=[0, 1],
    )
    assert bpreds.fold_index is not None and opreds.fold_index is not None
    assert bpreds.pA_taken.shape == (len(logs),)
    assert opreds.mu_logged_action.shape == (len(logs),)
    assert opreds.mu_by_action is not None
    assert set(opreds.mu_by_action.keys()) == {0, 1}


def test_kfold_indices_cover_all_rows_once():
    n = 41
    folds = make_kfold_indices(n, n_splits=5, random_state=123)
    seen = np.concatenate([holdout for _, holdout in folds])
    assert len(seen) == n
    assert sorted(seen.tolist()) == list(range(n))


def test_crossfit_bundle_container_works():
    logs, policyB = _logs_and_policy(74)
    behavior = generate_oof_behavior_predictions(
        logs,
        policyB,
        n_splits=3,
        random_state=2,
        feature_cols=["loyal", "age", "risk", "income"],
    )
    outcome = generate_oof_outcome_predictions(
        logs,
        target="accept",
        n_splits=3,
        random_state=2,
        feature_cols=["loyal", "age", "risk", "income"],
    )
    bundle = CrossFitNuisanceBundle(behavior=behavior, outcome=outcome, n_splits=3)
    assert bundle.n_splits == 3
    assert bundle.behavior is not None and bundle.outcome is not None


def test_fit_crossfit_nuisance_bundle_contains_behavior_and_outcome():
    logs, policyB = _logs_and_policy(75)
    bundle = fit_crossfit_nuisance_bundle(
        logs,
        policyB,
        target="accept",
        n_splits=3,
        random_state=13,
        feature_cols=["loyal", "age", "risk", "income"],
        action_col="a_A",
    )
    assert bundle.behavior is not None
    assert bundle.outcome is not None
    assert bundle.behavior.is_out_of_fold
    assert bundle.outcome.is_out_of_fold
    assert bundle.outcome.mu_by_action is not None


def test_behavior_auto_feature_inference_excludes_target_and_meta_columns():
    logs, policyB = _logs_and_policy(76)
    logs = logs.rename(columns={"accept": "reward"})
    logs["propensity_custom"] = 0.25
    logs["candidate_action"] = pd.Series(logs["a_A"]).shift(1).fillna(0).astype(int)

    bundle = fit_behavior_nuisance_bundle(
        logs,
        policyB,
        feature_cols=None,
        action_col="a_A",
        target_col="reward",
        propensity_col="propensity_custom",
        cluster_col="user_id",
    )
    inferred = list(bundle.pi_model._feature_cols)  # type: ignore[attr-defined]
    assert "reward" not in inferred
    assert "a_A" not in inferred
    assert "propensity_custom" not in inferred
    assert "user_id" not in inferred
