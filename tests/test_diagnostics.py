"""Диагностика: каждая находка срабатывает, когда должна, и молчит на чистых данных."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier

import policyscope as ps

from conftest import ACTIONS, make_bandit

KW = dict(action="action", reward="reward", features=["x1", "x2"])


def codes(metric_report):
    return {f.code: f.severity for f in metric_report.findings}


def test_clean_data_is_reliable():
    b = make_bandit(4000, seed=1)
    m = ps.evaluate(b.logs, b.policy_b, propensity="propensity", **KW)["reward"]
    assert m.trust == "reliable"
    assert all(s == "info" for s in codes(m).values())
    assert {c.name: c.status for c in m.checks}["propensity"] != "fail"


def test_response_probability_passed_as_propensity_blocks():
    """Частая ошибка: в колонке пропенсити — вероятность отклика, а не выбора действия."""
    b = make_bandit(3000, seed=2)
    logs = b.logs.assign(p_accept=np.clip(b.mu[np.arange(3000), 0], 0.05, 1.0) * 0.3)
    m = ps.evaluate(logs, b.policy_b, propensity="p_accept", **KW)["reward"]
    assert codes(m)["propensity_not_probability"] == "block"
    assert m.decision == "inconclusive"


def test_deterministic_logging_blocks_unless_candidate_matches():
    b = make_bandit(3000, seed=3)
    greedy_a = np.digitize(b.logs["x1"], [-0.5, 0.5])  # действие однозначно задаётся x1
    logs = b.logs.assign(action=np.asarray(ACTIONS, dtype=object)[greedy_a], propensity=1.0)
    m = ps.evaluate(logs, b.policy_b, propensity="propensity", **KW)["reward"]
    assert codes(m)["deterministic_logging"] == "block"
    same = ps.evaluate(logs, "action", propensity="propensity", **KW)["reward"]
    assert "deterministic_logging" not in codes(same)
    assert same.delta.value == 0.0


def test_support_deficit():
    """B выбирает действие там, где A его никогда не выбирает."""
    b = make_bandit(4000, seed=4)
    pa = b.policy_a.to_numpy().copy()
    pa[b.logs["x1"] > 0, 2] = 0.0  # A не выбирает blue при x1 > 0
    pa /= pa.sum(axis=1, keepdims=True)
    rng = np.random.default_rng(0)
    a = (rng.random(4000)[:, None] > pa.cumsum(axis=1)).sum(axis=1).clip(0, 2)
    logs = b.logs.assign(action=np.asarray(ACTIONS, dtype=object)[a], propensity=pa[np.arange(4000), a])
    blue = pd.DataFrame({"red": 0.1, "green": 0.1, "blue": 0.8}, index=logs.index)
    m = ps.evaluate(logs, blue, propensity="propensity", **KW)["reward"]
    assert "support_deficit" in codes(m)


def test_wrong_propensities_detected_by_triangulation():
    b = make_bandit(8000, seed=5)
    distorted = b.logs["propensity"] ** 3
    logs = b.logs.assign(propensity=distorted / distorted.max())
    ev = ps.Evaluator(logs, propensity="propensity", **KW)
    dr = ev.evaluate(b.policy_b)["reward"]
    ips = ev.evaluate(b.policy_b, estimator="ips")["reward"]
    assert {c.name: c.status for c in dr.checks}["propensity"] == "fail"
    assert codes(dr)["propensity_check_failed"] == "caution"
    assert codes(ips)["propensity_check_failed"] == "block"
    assert dr.agreement == "disagree"


def test_wrong_outcome_model_detected_but_harmless_with_logged_propensity():
    b = make_bandit(8000, seed=6)
    ev = ps.Evaluator(b.logs, propensity="propensity",
                      outcome_model=DummyClassifier(strategy="prior"), **KW)
    m = ev.evaluate(b.policy_b)["reward"]
    assert {c.name: c.status for c in m.checks}["outcome"] == "fail"
    assert codes(m)["outcome_check_failed"] == "info"
    assert m.delta.low <= b.delta <= m.delta.high
    assert m.agreement == "disagree" and m.agreement_spread > 0


def test_critically_low_ess_blocks():
    b = make_bandit(3000, seed=7)
    logs = b.logs.copy()
    rare = logs.index[logs["action"] == "blue"][5:]
    logs.loc[rare, "action"] = "red"  # blue почти не встречается
    logs["propensity"] = np.where(logs["action"] == "blue", 0.002, 0.5)
    blue = pd.DataFrame({"red": 0.0, "green": 0.0, "blue": 1.0}, index=logs.index)
    m = ps.evaluate(logs, blue, propensity="propensity", **KW)["reward"]
    assert codes(m)["critically_low_ess"] == "block"
    assert m.decision == "inconclusive"


def test_cluster_hints_and_few_clusters():
    b = make_bandit(1000, seed=8)
    logs = b.logs.assign(user_id=np.arange(1000) // 50)
    hinted = ps.evaluate(logs, b.policy_b, propensity="propensity", **KW)["reward"]
    assert codes(hinted)["cluster_not_set"] == "caution"
    few = ps.evaluate(logs, b.policy_b, propensity="propensity", cluster="user_id", **KW)["reward"]
    assert codes(few)["few_clusters"] == "caution" and "cluster_not_set" not in codes(few)


def test_auto_features_are_listed():
    b = make_bandit(1000, seed=9)
    m = ps.evaluate(b.logs, b.policy_b, action="action", reward="reward",
                    propensity="propensity")["reward"]
    assert "auto_features" in codes(m)
