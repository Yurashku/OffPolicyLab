"""Отчёт: содержание, форматирование, вывод не противоречит доверию."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import policyscope as ps
from policyscope.report import _plural

from conftest import make_bandit


@pytest.fixture(scope="module")
def report():
    b = make_bandit(3000, seed=21)
    logs = b.logs.assign(revenue=b.logs["reward"] * 120.0 + 5.0)
    return ps.evaluate(logs, b.policy_b, action="action", reward=["reward", "revenue"],
                       propensity="propensity", features=["x1", "x2"])


def test_markdown_sections(report):
    md = report.to_markdown()
    for section in ("## Off-policy оценка: B против A", "### Можно ли верить оценке",
                    "### Согласие оценщиков", "### Перекрытие политик", "### Настройки и предпосылки"):
        assert section in md
    assert "Romano–Wolf" in md
    assert report._repr_markdown_() == md


def test_main_table_shows_values_with_intervals_and_agreement(report):
    md = report.to_markdown()
    assert "| V(A) [95% ДИ] | V(B) [95% ДИ] |" in md and "| Оценщики |" in md
    m = report["reward"]
    assert f"{100 * m.v_b.value:.2f}% [{100 * m.v_b.low:.2f}%; {100 * m.v_b.high:.2f}%]" in md


def test_estimator_choice_and_agreement_are_explained(report):
    md = report.to_markdown()
    assert "выбран автоматически" in md
    assert "Основной оценщик — DR, выбран автоматически: пропенсити известны точно" in md
    assert f"**reward: {report['reward'].agreement_label}.** Δ по 5 оценщикам" in md


def test_trust_reason_is_named_in_the_headline():
    b = make_bandit(1500, seed=22)
    rep = ps.evaluate(b.logs.drop(columns="propensity"), b.policy_b, action="action",
                      reward="reward", features=["x1", "x2"], inference="none")
    assert rep["reward"].trust == "caution"
    assert "с осторожностью (пропенсити оценены моделью" in rep.to_markdown()


def test_trust_reasons_keep_abbreviations():
    from policyscope.diagnostics import Finding
    from policyscope.report import MetricReport

    m = MetricReport(**{**report_fields(), "trust": "caution", "findings": (
        Finding(code="x", severity="caution", title="IPS и DR расходятся", detail=""),
        Finding(code="y", severity="caution", title="Пропенсити оценены моделью", detail=""),
    )})
    assert m.trust_reasons == ["IPS и DR расходятся", "пропенсити оценены моделью"]


def report_fields():
    from policyscope.report import Estimate

    return dict(metric="m", estimator="dr", binary=True, lower_is_better=False, v_a=Estimate(0.1),
                v_b=Estimate(0.2), delta=Estimate(0.1), p_value=None, p_value_adj=None, lift=None,
                mde=None, decision="not_computed", trust="reliable", findings=(), checks=(), rows=(),
                outcome_quality={})


def test_no_double_period_after_percentage_points():
    b = make_bandit(1500, seed=23)
    rep = ps.evaluate(b.logs.drop(columns="propensity"), b.policy_b, action="action",
                      reward="reward", features=["x1", "x2"], inference="none")
    md = rep.to_markdown()
    assert "чувствительность Δ к полу пропенсити" in md and "п.п.." not in md


def test_binary_metric_in_percentage_points(report):
    md = report.to_markdown()
    assert "п.п." in md and "%" in md


def test_settings_line_has_proper_russian(report):
    assert "3 000 строк, 3 действия" in report.to_markdown()


def test_plural():
    assert _plural(1, "действие", "действия", "действий") == "1 действие"
    assert _plural(3, "действие", "действия", "действий") == "3 действия"
    assert _plural(11, "действие", "действия", "действий") == "11 действий"
    assert _plural(22, "строка", "строки", "строк") == "22 строки"


def test_repr_is_one_line_per_metric(report):
    text = repr(report)
    assert text.count("\n") == 2 and "доверие" in text


def test_json_roundtrip(report, tmp_path):
    path = tmp_path / "report.json"
    text = report.to_json(path)
    assert json.loads(path.read_text(encoding="utf-8")) == json.loads(text)


def test_unknown_metric_message(report):
    with pytest.raises(KeyError, match="есть"):
        report["clicks"]


def test_unreliable_estimate_gets_no_decision():
    """Даже интервал, не накрывающий ноль, не превращается в «B лучше» при ненадёжной оценке."""
    b = make_bandit(3000, seed=22)
    logs = b.logs.copy()
    logs["propensity"] = np.where(logs["action"] == "blue", 0.002, 0.5)
    blue = pd.DataFrame({"red": 0.0, "green": 0.0, "blue": 1.0}, index=logs.index)
    rep = ps.evaluate(logs, blue, action="action", reward="reward",
                      propensity="propensity", features=["x1", "x2"])
    m = rep["reward"]
    assert m.trust == "unreliable" and m.decision == "inconclusive"
    assert "вывод сделать нельзя" in rep.to_markdown()


def test_no_difference_mentions_detectable_effect():
    b = make_bandit(2000, seed=23)
    rep = ps.evaluate(b.logs, b.policy_a, action="action", reward="reward",
                      propensity="propensity", features=["x1", "x2"])
    m = rep["reward"]
    assert m.decision == "no_difference"
    assert "был бы обнаружен" in m.sentence()
