"""Тесты рендеринга отчёта и вердикта доверия."""

from __future__ import annotations

import json

import pytest

from policyscope.comparison import compare_policies
from policyscope.report import (
    TRUST_CAUTION,
    TRUST_OK,
    TRUST_UNRELIABLE,
    WARNING_EXPLANATIONS,
    dump_json,
    render_summary,
    trust_verdict,
)

from conftest import make_controlled_bandit


@pytest.fixture
def result():
    bandit = make_controlled_bandit(n=600, seed=41)
    return compare_policies(
        bandit.df, bandit.policy_b(), estimator="ips", target="reward",
        action_col="action", action_space=bandit.action_space,
        feature_cols=["x1", "x2"], propensity_col="propensity",
        propensity_source="logged", cluster_col=None, inference="influence",
    )


class TestTrustVerdict:
    def test_clean_run_is_trustworthy(self):
        assert trust_verdict([]) == (TRUST_OK, [])

    def test_blocking_warning_makes_result_unreliable(self):
        verdict, reasons = trust_verdict(["critically_low_ess"])
        assert verdict == TRUST_UNRELIABLE
        assert "critically_low_ess" in reasons

    def test_low_ess_ratio_alone_is_only_caution(self):
        """Малая доля ESS при большой выборке — повод присмотреться, а не приговор.

        Блокирующим является абсолютный ESS: при n = 100 000 отношение 0.1 даёт
        10 000 эффективных наблюдений, чего вполне достаточно.
        """
        verdict, _ = trust_verdict(["low_ess_ratio"])
        assert verdict == TRUST_CAUTION

    def test_soft_warning_yields_caution(self):
        verdict, reasons = trust_verdict(["heavy_weight_tail"])
        assert verdict == TRUST_CAUTION
        assert reasons == ["heavy_weight_tail"]

    def test_blocking_dominates_caution(self):
        verdict, _ = trust_verdict(["heavy_weight_tail", "critically_low_ess"])
        assert verdict == TRUST_UNRELIABLE

    def test_duplicate_codes_reported_once(self):
        _, reasons = trust_verdict(["heavy_weight_tail", "heavy_weight_tail"])
        assert reasons == ["heavy_weight_tail"]


class TestRendering:
    def test_report_contains_key_numbers(self, result):
        text = render_summary(result)
        assert "V(A)" in text and "V(B)" in text and "Delta" in text
        assert "Доверие к оценке" in text
        assert "Решение" in text

    def test_confidence_level_follows_alpha(self):
        """Уровень доверия берётся из alpha, а не зашит как «95%»."""
        bandit = make_controlled_bandit(n=400, seed=42)
        r = compare_policies(
            bandit.df, bandit.policy_b(), estimator="ips", target="reward",
            action_col="action", action_space=bandit.action_space,
            feature_cols=["x1", "x2"], propensity_col="propensity",
            cluster_col=None, inference="influence", alpha=0.10,
        )
        assert "90% CI" in render_summary(r)

    def test_report_without_ci_does_not_crash(self):
        """Раньше отчёт падал с KeyError при отсутствии CI."""
        bandit = make_controlled_bandit(n=300, seed=43)
        r = compare_policies(
            bandit.df, bandit.policy_b(), estimator="ips", target="reward",
            action_col="action", action_space=bandit.action_space,
            feature_cols=["x1", "x2"], propensity_col="propensity",
            cluster_col=None, inference="none",
        )
        text = render_summary(r)
        assert "CI не считался" in text

    def test_warning_codes_are_explained_in_words(self, result):
        text = render_summary(result)
        for code in result.warnings:
            if code in WARNING_EXPLANATIONS:
                assert WARNING_EXPLANATIONS[code][:30] in text

    def test_business_threshold_changes_decision(self):
        bandit = make_controlled_bandit(n=800, seed=44)
        r = compare_policies(
            bandit.df, bandit.policy_b(), estimator="ips", target="reward",
            action_col="action", action_space=bandit.action_space,
            feature_cols=["x1", "x2"], propensity_col="propensity",
            cluster_col=None, inference="influence",
        )
        strict = render_summary(r, business_threshold=10.0)
        assert "Значимого отличия не обнаружено" in strict

    def test_to_markdown_delegates_to_renderer(self, result):
        assert result.to_markdown() == render_summary(result)

    def test_metric_name_override(self, result):
        assert "Отклик" in render_summary(result, metric_name="Отклик")


class TestDumpJson:
    def test_writes_valid_json_from_result_object(self, result, tmp_path):
        path = tmp_path / "r.json"
        dump_json(str(path), result)
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["estimator"] == "ips"
        assert "diagnostics" in payload

    def test_nan_becomes_null_not_invalid_json(self, tmp_path):
        """Голый NaN — невалидный JSON и ломает потребителей."""
        path = tmp_path / "nan.json"
        dump_json(str(path), {"value": float("nan"), "nested": {"x": float("inf")}})
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["value"] is None
        assert payload["nested"]["x"] is None
