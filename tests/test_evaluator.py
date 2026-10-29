"""Evaluator: сквозной путь, схема результата, кеширование, baseline, рейтинг."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import policyscope as ps
import policyscope.evaluator as evaluator_module

from conftest import ACTIONS, make_bandit


class TestEndToEnd:
    def test_report_covers_truth_and_is_reliable(self, bandit, evaluator):
        m = evaluator.evaluate(bandit.policy_b)["reward"]
        assert m.delta.low <= bandit.delta <= m.delta.high
        assert m.trust == "reliable"
        assert m.delta.value == pytest.approx(m.v_b.value - m.v_a.value)

    def test_all_estimators_reported_dm_without_interval(self, bandit, evaluator):
        table = evaluator.evaluate(bandit.policy_b)["reward"].estimators
        assert list(table["estimator"]) == ["dr", "sndr", "ips", "snips", "dm"]
        dm = table.set_index("estimator").loc["dm"]
        assert pd.isna(dm["delta_low"]) and np.isfinite(dm["delta"])

    def test_logging_policy_as_candidate_gives_zero_ips_delta(self, bandit, evaluator):
        """При B = A все веса равны 1: IPS-оценка V(B) — это ровно средняя награда."""
        rep = evaluator.evaluate(bandit.policy_a, estimator="ips")
        assert rep["reward"].delta.value == pytest.approx(0.0, abs=1e-12)

    def test_baseline_equal_to_candidate_gives_zero(self, bandit, evaluator):
        rep = evaluator.evaluate(bandit.policy_b, baseline=bandit.policy_b)
        assert rep["reward"].delta.value == pytest.approx(0.0, abs=1e-12)
        assert rep.baseline == "база"

    def test_baseline_value_is_estimated(self, bandit, evaluator):
        rep = evaluator.evaluate(bandit.policy_b, baseline=bandit.policy_a, baseline_name="A*")
        assert rep["reward"].v_a.value == pytest.approx(bandit.logs["reward"].mean(), abs=0.03)
        assert rep.baseline == "A*"

    def test_multiple_metrics(self, bandit):
        logs = bandit.logs.assign(revenue=bandit.logs["reward"] * 100 + 1.0)
        rep = ps.evaluate(logs, bandit.policy_b, action="action", reward=["reward", "revenue"],
                          propensity="propensity", features=["x1", "x2"])
        assert list(rep.metrics) == ["reward", "revenue"]
        for m in rep.metrics.values():
            assert m.p_value_adj is not None and m.p_value_adj >= m.p_value - 0.02
        assert not rep["reward"].binary is rep["revenue"].binary

    def test_lower_is_better_flips_decision(self, bandit):
        kw = dict(action="action", reward="reward", propensity="propensity", features=["x1", "x2"])
        up = ps.evaluate(bandit.logs, bandit.policy_b, **kw)["reward"].decision
        down = ps.evaluate(bandit.logs, bandit.policy_b, lower_is_better=["reward"], **kw)["reward"].decision
        flip = {"better": "worse", "worse": "better", "no_difference": "no_difference"}
        assert down == flip[up]


class TestSchema:
    def test_to_dict_is_json_and_stable(self, bandit, evaluator):
        d = evaluator.evaluate(bandit.policy_b).to_dict()
        json.dumps(d, allow_nan=False)
        assert set(d) == {"policy", "baseline", "trust", "metrics", "overlap", "findings", "settings"}
        m = d["metrics"]["reward"]
        for key in ("V_A", "V_B", "delta", "p_value", "p_value_adj", "lift", "mde", "decision",
                    "trust", "findings", "checks", "estimators", "outcome_quality"):
            assert key in m

    def test_summary_has_value_intervals_and_agreement(self, bandit, evaluator):
        s = evaluator.evaluate(bandit.policy_b).summary()
        for col in ("V_A_low", "V_A_high", "V_B_low", "V_B_high", "agreement"):
            assert col in s.columns
        assert s.loc["reward", "V_B_low"] < s.loc["reward", "V_B"] < s.loc["reward", "V_B_high"]

    def test_summary_frame(self, bandit, evaluator):
        s = evaluator.evaluate(bandit.policy_b).summary()
        assert list(s.index) == ["reward"]
        assert {"V_A", "V_B", "delta", "delta_low", "delta_high", "p_value", "decision", "trust"} <= set(s.columns)

    def test_no_inference(self, bandit, evaluator):
        m = evaluator.evaluate(bandit.policy_b, inference="none")["reward"]
        assert m.decision == "not_computed" and m.delta.low is None and m.p_value is None
        json.dumps(evaluator.evaluate(bandit.policy_b, inference="none").to_dict())


class TestCaching:
    def test_nuisance_fitted_once(self, bandit, monkeypatch):
        calls = []
        original = evaluator_module.fit_nuisance

        def counting(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)

        monkeypatch.setattr(evaluator_module, "fit_nuisance", counting)
        ev = ps.Evaluator(bandit.logs, action="action", reward="reward",
                          propensity="propensity", features=["x1", "x2"])
        ev.evaluate(bandit.policy_b)
        ev.evaluate(bandit.policy_a)
        ev.compare({"b": bandit.policy_b, "a": bandit.policy_a})
        assert len(calls) == 1

    def test_logs_mutation_after_init_has_no_effect(self, bandit):
        logs = bandit.logs.copy()
        ev = ps.Evaluator(logs, action="action", reward="reward", propensity="propensity",
                          features=["x1", "x2"])
        before = ev.evaluate(bandit.policy_b)["reward"].delta.value
        logs["reward"] = 0
        assert ev.evaluate(bandit.policy_b)["reward"].delta.value == before


class TestCompare:
    def test_leaderboard(self, bandit, evaluator):
        board = evaluator.compare({"b": bandit.policy_b, "a": bandit.policy_a})
        assert set(board.table["candidate"]) == {"a", "b"}
        deltas = board.table["delta"].tolist()
        assert deltas == sorted(deltas, reverse=True)  # рейтинг: лучший первым
        assert list(board.reports) == list(board.table["candidate"])
        assert board["b"]["reward"].p_value_adj >= board["b"]["reward"].p_value - 0.02
        md = board.to_markdown()
        assert "Romano–Wolf" in md and "| V(A) [95% ДИ] | V(кандидата) [95% ДИ] |" in md
        assert {"V_B_low", "V_B_high", "agreement"} <= set(board.table.columns)
        json.dumps(board.to_dict(), allow_nan=False)

    def test_empty_candidates(self, evaluator):
        with pytest.raises(ValueError):
            evaluator.compare({})


class TestValidation:
    def test_dm_with_custom_outcome_model_is_refused(self, bandit):
        from sklearn.ensemble import HistGradientBoostingClassifier

        ev = ps.Evaluator(bandit.logs, propensity="propensity", action="action", reward="reward",
                          features=["x1", "x2"], outcome_model=HistGradientBoostingClassifier())
        with pytest.raises(ValueError, match="встроенной моделью"):
            ev.evaluate(bandit.policy_b, estimator="dm")

    def test_dm_has_no_analytic_interval(self, bandit, evaluator):
        with pytest.raises(ValueError, match="bootstrap"):
            evaluator.evaluate(bandit.policy_b, estimator="dm", inference="influence")

    def test_ips_needs_logged_propensity(self, bandit):
        ev = ps.Evaluator(bandit.logs, action="action", reward="reward", features=["x1", "x2"])
        with pytest.raises(ValueError, match="оценённых пропенсити"):
            ev.evaluate(bandit.policy_b, estimator="ips")

    def test_bad_inference(self, bandit, evaluator):
        with pytest.raises(ValueError):
            evaluator.evaluate(bandit.policy_b, inference="jackknife")

    def test_outcome_model_for_unknown_metric(self, bandit):
        from sklearn.dummy import DummyRegressor

        with pytest.raises(ValueError, match="outcome_model"):
            ps.Evaluator(bandit.logs, action="action", reward="reward",
                         outcome_model={"rewrd": DummyRegressor()})

    def test_unknown_lower_is_better_metric(self, bandit):
        with pytest.raises(ValueError, match="lower_is_better"):
            ps.Evaluator(bandit.logs, action="action", reward="reward", lower_is_better=["cost"])


class TestDirectMethod:
    """DM — явный выбор: точнее DR при верной модели исхода, но смещение не проверяется."""

    def test_opt_in_uses_bootstrap_and_reports_interval(self):
        b = make_bandit(800, seed=12)
        ev = ps.Evaluator(b.logs, action="action", reward="reward", propensity="propensity",
                          features=["x1", "x2"], n_jobs=1)
        rep = ev.evaluate(b.policy_b, estimator="dm", n_boot=60)
        m = rep["reward"]
        assert rep.settings["inference"] == "bootstrap" and m.estimator == "dm"
        assert m.delta.low < m.delta.value < m.delta.high
        assert m.trust != "reliable"
        assert "model_based_estimate" in {f.code for f in m.findings}
        dm_row = m.estimators.set_index("estimator").loc["dm"]
        assert np.isfinite(dm_row["delta_low"])

    def test_dm_delta_is_a_contrast_of_model_predictions(self, bandit, evaluator):
        """Для DM база — прогноз той же модели на залогированных действиях, а не mean(r)."""
        rep = evaluator.evaluate(bandit.policy_b, inference="none")
        nuis = evaluator.nuisance
        data = evaluator.data
        pi = ps.policy.policy_matrix(bandit.policy_b, data, evaluator._frame)
        mu = nuis.mu["reward"]
        expected = np.mean(np.sum(pi * mu, axis=1)) - np.mean(mu[np.arange(data.n), data.a])
        dm = rep["reward"].estimators.set_index("estimator").loc["dm", "delta"]
        assert dm == pytest.approx(expected)

    def test_dm_row_without_interval_when_not_primary(self, bandit, evaluator):
        row = evaluator.evaluate(bandit.policy_b)["reward"].estimators.set_index("estimator").loc["dm"]
        assert pd.isna(row["delta_low"])


class TestAutoEstimator:
    def test_auto_is_default_and_explained(self, bandit, evaluator):
        s = evaluator.evaluate(bandit.policy_b).settings
        assert s["estimator"] == "dr" and s["estimator_auto"]
        assert "известны точно" in s["estimator_reason"]

    def test_auto_reason_for_estimated_propensities(self):
        b = make_bandit(1500, seed=12)
        ev = ps.Evaluator(b.logs.drop(columns="propensity"), action="action", reward="reward",
                          features=["x1", "x2"])
        s = ev.evaluate(b.policy_b, inference="none").settings
        assert s["estimator"] == "dr" and "оценены моделью" in s["estimator_reason"]

    def test_explicit_estimator_is_not_auto(self, bandit, evaluator):
        s = evaluator.evaluate(bandit.policy_b, estimator="sndr").settings
        assert s["estimator"] == "sndr" and not s["estimator_auto"] and s["estimator_reason"] is None

    def test_unknown_estimator_lists_auto(self, evaluator, bandit):
        with pytest.raises(ValueError, match="'auto'"):
            evaluator.evaluate(bandit.policy_b, estimator="magic")


class TestAgreement:
    def test_estimators_agree_on_well_specified_logs(self, bandit, evaluator):
        m = evaluator.evaluate(bandit.policy_b)["reward"]
        assert m.agreement == "agree"
        deltas = [row.delta.value for row in m.rows]
        assert m.agreement_spread == pytest.approx(max(deltas) - min(deltas))

    def test_absolute_values_have_intervals(self, bandit, evaluator):
        m = evaluator.evaluate(bandit.policy_b, baseline=bandit.policy_a)["reward"]
        for est in (m.v_a, m.v_b):
            assert est.low < est.value < est.high
        assert m.v_b.low <= bandit.value_b <= m.v_b.high


def test_too_wide_outcome_design_is_reported(bandit, monkeypatch):
    import policyscope.nuisance as nuisance_module

    monkeypatch.setattr(nuisance_module, "MAX_DESIGN_CELLS", 100)
    ev = ps.Evaluator(bandit.logs, action="action", reward="reward", propensity="propensity",
                      features=["x1", "x2"])
    rep = ev.evaluate(bandit.policy_b)
    assert "без x×a" in rep.settings["outcome_model"]["reward"]
    finding = next(f for f in rep["reward"].findings if f.code == "outcome_without_interactions")
    assert finding.severity == "info" and "слишком велика" in finding.detail


class TestInferenceModes:
    def test_auto_is_influence_with_logged_propensity(self, bandit, evaluator):
        assert evaluator.evaluate(bandit.policy_b).settings["inference"] == "influence"

    def test_auto_is_bootstrap_with_estimated_propensity(self):
        b = make_bandit(600, seed=3)
        ev = ps.Evaluator(b.logs, action="action", reward="reward", features=["x1", "x2"], n_jobs=1)
        rep = ev.evaluate(b.policy_b, n_boot=50)
        assert rep.settings["inference"] == "bootstrap"
        m = rep["reward"]
        assert m.delta.low < m.delta.value < m.delta.high
        assert "propensity_estimated" in {f.code for f in m.findings}

    def test_bootstrap_reproducible(self):
        b = make_bandit(400, seed=4)
        kw = dict(action="action", reward="reward", propensity="propensity",
                  features=["x1", "x2"], n_jobs=1, random_state=3)
        r1 = ps.Evaluator(b.logs, **kw).evaluate(b.policy_b, inference="bootstrap", n_boot=50)
        r2 = ps.Evaluator(b.logs, **kw).evaluate(b.policy_b, inference="bootstrap", n_boot=50)
        assert r1["reward"].delta.low == r2["reward"].delta.low

    def test_bootstrap_and_influence_agree_with_logged_propensity(self):
        b = make_bandit(2000, seed=5)
        ev = ps.Evaluator(b.logs, action="action", reward="reward", propensity="propensity",
                          features=["x1", "x2"], n_jobs=1)
        se_if = ev.evaluate(b.policy_b)["reward"].delta.se
        se_bs = ev.evaluate(b.policy_b, inference="bootstrap", n_boot=100)["reward"].delta.se
        assert se_bs == pytest.approx(se_if, rel=0.35)


def test_string_actions_and_categorical_features_end_to_end():
    b = make_bandit(1500, seed=11)
    rep = ps.evaluate(b.logs, b.policy_b, action="action", reward="reward",
                      propensity="propensity", features=["x1", "x2", "segment"])
    assert rep.settings["actions"] == sorted(ACTIONS)
    assert "segment" in rep.settings["features"]


class TestOutcomeCheckTargetsReportedDelta:
    """Проверка DR − DM сравнивает именно те Δ, что показаны в таблице оценщиков."""

    @pytest.mark.parametrize("with_baseline", [False, True])
    def test_discrepancy_equals_difference_of_rows(self, bandit, evaluator, with_baseline):
        baseline = bandit.policy_a if with_baseline else None
        m = evaluator.evaluate(bandit.policy_b, baseline=baseline, inference="none")["reward"]
        rows = m.estimators.set_index("estimator")["delta"]
        check = {c.name: c for c in m.checks}["outcome"]
        assert check.discrepancy == pytest.approx(rows["dr"] - rows["dm"], abs=1e-10)

    def test_candidate_equal_to_logging_policy_gives_zero(self, bandit, evaluator):
        m = evaluator.evaluate(bandit.policy_a, estimator="dm", inference="none")["reward"]
        check = {c.name: c for c in m.checks}["outcome"]
        assert check.discrepancy == pytest.approx(0.0, abs=1e-12)
        assert "outcome_check_failed" not in {f.code for f in m.findings}


def test_outlying_feature_does_not_break_revenue_estimates():
    """Регрессия: выброс в доходе раздувал прогноз двухчастной модели до 1e6 и выше."""
    from policyscope.synthetic import OfferSimulator

    sim = OfferSimulator(seed=5)
    clients = sim.sample_clients(3000)
    clients.loc[0, "income"] = clients["income"].max() * 5
    a, b = sim.policy("epsilon_greedy", epsilon=0.2, seed=1), sim.policy("softmax", seed=2)
    logs = sim.simulate(clients, a)
    truth = sim.true_value(logs, b, "revenue") - sim.true_value(logs, a, "revenue")
    m = ps.evaluate(logs, b, action="offer", reward="revenue", propensity="propensity",
                    features=["age", "income", "loyal", "risk", "region"])["revenue"]
    rows = m.estimators.set_index("estimator")["delta"]
    assert abs(rows["dr"] - truth) < 4 * m.delta.se
    assert abs(rows["dm"] - truth) < 10


class TestInputGuards:
    def test_constant_metric_blocks_the_decision(self, bandit):
        logs = bandit.logs.assign(visits=1)
        rep = ps.evaluate(logs, bandit.policy_b, action="action", reward=["reward", "visits"],
                          propensity="propensity", features=["x1", "x2"])
        m = rep["visits"]
        assert m.decision == "inconclusive"
        assert "constant_metric" in {f.code for f in m.findings}
        assert rep["reward"].p_value_adj is not None and np.isfinite(rep["reward"].p_value_adj)

    def test_lower_is_better_accepts_a_string(self, bandit):
        ev = ps.Evaluator(bandit.logs, action="action", reward="reward", lower_is_better="reward",
                          features=["x1", "x2"], propensity="propensity")
        assert ev.lower_is_better == frozenset({"reward"})

    def test_random_state_must_be_integer(self, bandit):
        with pytest.raises(ValueError, match="random_state"):
            ps.Evaluator(bandit.logs, action="action", reward="reward", random_state=None)

    def test_too_few_clusters_for_folds(self, bandit):
        logs = bandit.logs.assign(user_id=np.arange(len(bandit.logs)) % 3)
        with pytest.raises(ValueError, match="фолдов"):
            ps.Evaluator(logs, action="action", reward="reward", cluster="user_id")

    def test_too_few_rows_is_explained(self):
        b = make_bandit(20, seed=1)
        m = ps.evaluate(b.logs, b.policy_b, action="action", reward="reward", propensity="propensity",
                        features=["x1", "x2"])["reward"]
        assert "too_few_rows" in {f.code for f in m.findings} and m.decision == "inconclusive"

    def test_one_shot_evaluate_accepts_baseline_name(self, bandit):
        rep = ps.evaluate(bandit.logs, bandit.policy_b, action="action", reward="reward",
                          propensity="propensity", features=["x1", "x2"],
                          baseline=bandit.policy_a, baseline_name="прод")
        assert rep.baseline == "прод"


def test_leaderboard_repr_and_json(bandit, evaluator):
    board = evaluator.compare({"b": bandit.policy_b, "a": bandit.policy_a}, inference="none")
    assert "кандидат" in repr(board).lower()
    assert "Romano–Wolf" not in board.to_markdown()  # p-value не считались
    json.loads(board.to_json())
