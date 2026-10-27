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
