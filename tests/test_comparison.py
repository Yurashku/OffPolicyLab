import numpy as np
import pytest

from policyscope.comparison import (
    MultiMetricComparisonResult,
    PolicyComparisonSummary,
    compare_policies,
    compare_policies_multi_target,
)
from policyscope.evaluator import OPEEvaluator
from policyscope.data import BanditSchema, LoggedBanditDataset
from policyscope.nuisance import CrossFitNuisanceBundle, generate_oof_behavior_predictions, resolve_behavior_predictions
from policyscope.policies import make_policy
from policyscope.report import decision_summary
from policyscope.synthetic import SynthConfig, SyntheticRecommenderEnv
import policyscope.comparison as comparison_mod


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
    assert d["recommended_defaults"]["preferred_estimator_general_use"] == "dr"
    assert "info_notes" in d and "diagnostic_warnings" in d and "trust_notes" in d
    assert d["trust_level"] in {"ok", "caution", "elevated_concern"}


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


def test_comparison_accepts_external_nuisance_predictions():
    logs, policyB = _prepare_env(105)
    behavior = generate_oof_behavior_predictions(
        logs,
        policyB,
        n_splits=3,
        random_state=11,
        feature_cols=["loyal", "age", "risk", "income"],
        action_col="a_A",
    )
    summary = compare_policies(
        logs,
        policyB,
        estimator="ips",
        target="accept",
        feature_cols=["loyal", "age", "risk", "income"],
        action_col="a_A",
        with_ci=False,
        nuisance_bundle=CrossFitNuisanceBundle(behavior=behavior, n_splits=3),
    )
    out = summary.to_dict()
    assert out["estimator"] == "ips"
    assert "diagnostics" in out and "weight_ess_ratio" in out["diagnostics"]


def test_crossfit_mode_dm_and_dr_family_runs():
    logs, policyB = _prepare_env(106)
    dm_summary = compare_policies(
        logs,
        policyB,
        estimator="dm",
        target="accept",
        feature_cols=["loyal", "age", "risk", "income"],
        action_col="a_A",
        with_ci=False,
        use_crossfit=True,
        crossfit_n_splits=3,
        crossfit_random_state=19,
    )
    dr_summary = compare_policies(
        logs,
        policyB,
        estimator="dr",
        target="accept",
        feature_cols=["loyal", "age", "risk", "income"],
        action_col="a_A",
        with_ci=False,
        use_crossfit=True,
        crossfit_n_splits=3,
        crossfit_random_state=19,
    )
    assert np.isfinite(dm_summary.v_b)
    assert np.isfinite(dr_summary.v_b)


def test_crossfit_mode_multi_target_still_works():
    logs, policyB = _prepare_env(107)
    out = compare_policies_multi_target(
        logs,
        policyB,
        estimator="dr",
        targets=["accept", "cltv"],
        feature_cols=["loyal", "age", "risk", "income"],
        action_col="a_A",
        with_ci=False,
        use_crossfit=True,
        crossfit_n_splits=3,
    )
    assert set(out.results.keys()) == {"accept", "cltv"}


def test_propensity_source_logged_and_estimated_modes():
    logs, policyB = _prepare_env(108)
    estimated_preds, _, _, _ = resolve_behavior_predictions(
        logs,
        policyB,
        propensity_source="estimated",
        feature_cols=["loyal", "age", "risk", "income"],
        action_col="a_A",
    )
    logs = logs.copy()
    logs["p_logged"] = estimated_preds.pA_taken

    logged_summary = compare_policies(
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
    estimated_summary = compare_policies(
        logs,
        policyB,
        estimator="ips",
        target="accept",
        feature_cols=["loyal", "age", "risk", "income"],
        action_col="a_A",
        with_ci=False,
        propensity_source="estimated",
    )
    assert logged_summary.propensity_source == "logged"
    assert logged_summary.propensity_column == "p_logged"
    assert estimated_summary.propensity_source == "estimated"


def test_propensity_source_auto_fallback_and_metadata():
    logs, policyB = _prepare_env(109)
    summary = compare_policies(
        logs,
        policyB,
        estimator="ips",
        target="accept",
        feature_cols=["loyal", "age", "risk", "income"],
        action_col="a_A",
        with_ci=False,
        propensity_source="auto",
        propensity_col="missing_propensity",
    )
    assert summary.propensity_source == "estimated"
    assert any("fallback" in n for n in summary.info_notes)
    assert summary.to_dict()["diagnostics"]["propensity_source"] == "estimated"


def test_notes_are_structured_and_legacy_notes_remain_compatible():
    logs, policyB = _prepare_env(114)
    summary = compare_policies(
        logs,
        policyB,
        estimator="dr",
        target="accept",
        feature_cols=["loyal", "age", "risk", "income"],
        action_col="a_A",
        with_ci=True,
        n_boot=10,
    )
    assert isinstance(summary.info_notes, tuple)
    assert isinstance(summary.diagnostic_warnings, tuple)
    assert isinstance(summary.inference_warnings, tuple)
    assert isinstance(summary.trust_notes, tuple)
    # Legacy combined notes stays available for backward-compatible consumers.
    assert set(summary.info_notes).issubset(set(summary.notes))


def test_propensity_source_logged_requires_valid_column():
    logs, policyB = _prepare_env(110)
    logs = logs.copy()
    logs["bad_p"] = 1.5
    try:
        compare_policies(
            logs,
            policyB,
            estimator="ips",
            target="accept",
            feature_cols=["loyal", "age", "risk", "income"],
            action_col="a_A",
            with_ci=False,
            propensity_source="logged",
            propensity_col="bad_p",
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for invalid logged propensity column")


def test_logged_bandit_dataset_input_with_propensity_column():
    logs, policyB = _prepare_env(111)
    estimated_preds, _, _, _ = resolve_behavior_predictions(
        logs,
        policyB,
        propensity_source="estimated",
        feature_cols=["loyal", "age", "risk", "income"],
        action_col="a_A",
    )
    logs = logs.copy()
    logs["ps"] = estimated_preds.pA_taken
    dataset = LoggedBanditDataset(
        df=logs,
        schema=BanditSchema(
            action_col="a_A",
            reward_col="accept",
            feature_cols=["loyal", "age", "risk", "income"],
            propensity_col="ps",
            cluster_col="user_id",
        ),
    )
    summary = compare_policies(
        dataset,
        policyB,
        estimator="ips",
        with_ci=False,
        propensity_source="auto",
    )
    assert summary.propensity_source == "logged"
    assert summary.propensity_column == "ps"


@pytest.mark.parametrize(
    "index_builder",
    [
        lambda n: np.arange(n - 1, -1, -1),  # reordered
        lambda n: np.concatenate([np.arange(n - 1), [n - 2]]),  # duplicated row
        lambda n: np.concatenate([np.arange(1, n), [1]]),  # omitted + duplicated row
    ],
)
def test_bootstrap_does_not_reuse_external_nuisance_on_non_identical_rows(monkeypatch, index_builder):
    logs, policyB = _prepare_env(112)
    behavior = generate_oof_behavior_predictions(
        logs,
        policyB,
        n_splits=3,
        random_state=21,
        feature_cols=["loyal", "age", "risk", "income"],
        action_col="a_A",
    )
    bundle = CrossFitNuisanceBundle(behavior=behavior, n_splits=3)

    original_estimate = comparison_mod.estimate_value
    captured: list[tuple[bool, bool]] = []

    def wrapped_estimate(*args, **kwargs):
        captured.append((kwargs.get("nuisance_behavior") is not None, kwargs.get("nuisance_outcome") is not None))
        return original_estimate(*args, **kwargs)

    class _FakeInferenceResult:
        def to_dict(self):
            return {
                "V_A": 0.0,
                "V_B": 0.0,
                "Delta": 0.0,
                "V_A_CI": (0.0, 0.0),
                "V_B_CI": (0.0, 0.0),
                "Delta_CI": (0.0, 0.0),
                "p_value": 1.0,
                "is_significant": False,
                "significance_rule": "centered_paired_bootstrap_p_value_lt_alpha",
                "alpha": 0.05,
                "n_boot": 3,
                "inference_method": "paired_percentile_bootstrap+centered_delta_test",
                "inference_warnings": [],
            }

    def fake_infer(df, estimator_pair, **kwargs):
        idx = index_builder(len(df))
        part = df.iloc[idx].copy()
        estimator_pair(part)
        return _FakeInferenceResult()

    monkeypatch.setattr(comparison_mod, "estimate_value", wrapped_estimate)
    monkeypatch.setattr(comparison_mod, "infer_policy_comparison_bootstrap", fake_infer)

    summary = compare_policies(
        logs,
        policyB,
        estimator="ips",
        target="accept",
        feature_cols=["loyal", "age", "risk", "income"],
        action_col="a_A",
        with_ci=True,
        nuisance_bundle=bundle,
        n_boot=3,
    )

    assert captured[0] == (True, False)
    assert captured[-1] == (False, False)
    assert "external_nuisance_not_reused_in_bootstrap_resamples_fallback_to_internal_nuisance_fit" in summary.inference_warnings
    assert "external_nuisance_not_reused_in_bootstrap_resamples_fallback_to_internal_nuisance_fit" in summary.notes


def test_with_ci_external_nuisance_fallback_warning_present():
    logs, policyB = _prepare_env(113)
    behavior = generate_oof_behavior_predictions(
        logs,
        policyB,
        n_splits=3,
        random_state=25,
        feature_cols=["loyal", "age", "risk", "income"],
        action_col="a_A",
    )
    summary = compare_policies(
        logs,
        policyB,
        estimator="ips",
        target="accept",
        feature_cols=["loyal", "age", "risk", "income"],
        action_col="a_A",
        with_ci=True,
        nuisance_bundle=CrossFitNuisanceBundle(behavior=behavior, n_splits=3),
        n_boot=10,
    )
    assert "external_nuisance_not_reused_in_bootstrap_resamples_fallback_to_internal_nuisance_fit" in summary.inference_warnings
