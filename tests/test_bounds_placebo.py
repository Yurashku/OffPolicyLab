"""Границы для кандидата вне покрытия логов и плацебо-проверка скрытых факторов выбора."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import policyscope as ps
from policyscope.inference import imbens_manski_critical

F = ["age", "income", "loyal", "risk", "region"]


def version_logs(seed=4, n=6000, k=4):
    sim = ps.synthetic.OfferSimulator(seed=seed)
    versions = sim.versions(k, drift=0.4, seed=1)
    clients = sim.sample_clients(n)
    rows = np.array_split(np.arange(n), k)
    logs = pd.concat([sim.simulate(clients.iloc[r], versions[m]).assign(month=m) for m, r in zip(versions, rows)],
                     ignore_index=True).drop(columns="propensity")
    return sim, versions, logs


def wrapper_logs(seed, n, informed):
    """Скоринг рекомендует оффер, «обвязка» иногда меняет его; плацебо — отклик прошлого визита."""
    sim = ps.synthetic.OfferSimulator(seed=seed, client_effect=1.5)
    both = sim.sample_clients(n, visits=2)
    first, second = both.iloc[::2].reset_index(drop=True), both.iloc[1::2].reset_index(drop=True)
    prev = sim.simulate(first, sim.policy("softmax", temperature=2.0, seed=1))
    rec = sim.policy("greedy", seed=1).action_probs(second).to_numpy()
    u = sim.hidden_trait(second)
    d = 1 / (1 + np.exp(-(1.5 * u - 0.5))) if informed else np.full(n, 0.4)
    probs = (1 - d)[:, None] * rec + d[:, None] * (1 - rec) / 3
    logs = sim.simulate(second, pd.DataFrame(probs, columns=sim.actions, index=second.index))
    logs["prev_accepted"] = prev["accepted"].to_numpy()
    return sim, logs


class TestImbensManski:
    def test_between_one_and_two_sided(self):
        assert imbens_manski_critical(0.0, 1.0, 1.0, 0.05) == pytest.approx(1.96, abs=1e-3)
        assert imbens_manski_critical(50.0, 1.0, 1.0, 0.05) == pytest.approx(1.645, abs=1e-3)
        assert 1.645 < imbens_manski_critical(1.0, 1.0, 1.0, 0.05) < 1.96


class TestBounds:
    @pytest.fixture(scope="class")
    @staticmethod
    def setup():
        sim, versions, logs = version_logs()
        ev = ps.Evaluator(logs, action="offer", reward=["accepted", "revenue"], features=F,
                          logging_policy=versions, logger="month")
        return sim, versions, logs, ev

    def test_random_policy_bounds_contain_truth(self, setup):
        sim, versions, logs, ev = setup
        b = ev.bounds(ps.uniform(), name="случайная", high={"revenue": 1000})
        assert 0 < b.uncovered < 1
        truth_a = np.mean([sim.true_value(logs[logs.month == m], v, "accepted") for m, v in versions.items()])
        truth = sim.true_value(logs, pd.DataFrame(0.25, index=logs.index, columns=sim.actions), "accepted") - truth_a
        m = b["accepted"]
        assert m.delta_low <= truth <= m.delta_high
        assert m.ci_low <= m.at_least <= m.delta_low <= m.delta_high <= m.at_most <= m.ci_high
        assert b["revenue"].reward_range == (0.0, 1000.0) and "high задан" in b["revenue"].range_note
        assert b["revenue"].range_assumed and not b["accepted"].range_assumed  # low взят из данных
        explicit = ev.bounds(ps.uniform(), low={"revenue": 0}, high={"revenue": 1000})["revenue"]
        assert explicit.range_note == "задан" and not explicit.range_assumed
        assert "Δ не больше" in b.to_markdown() and "Чувствительность" in b.to_markdown()
        json.dumps(b.to_dict(), allow_nan=False)

    def test_covered_policy_collapses_to_point(self, setup):
        _, versions, _, ev = setup
        b = ev.bounds(versions["v2"], name="v2")
        m = b["accepted"]
        assert b.uncovered == 0 and m.delta_low == pytest.approx(m.delta_high)
        point = ev.evaluate(versions["v2"])["accepted"].delta.value
        assert m.delta_low == pytest.approx(point, rel=1e-9, abs=1e-12)
        assert m.ci_low == pytest.approx(point - 1.96 * (m.at_most - point) / 1.645, rel=1e-3)

    def test_sensitivity_is_monotone_and_breakdown_is_a_root(self, setup):
        _, _, _, ev = setup
        b = ev.bounds(ps.uniform(), name="случайная")["accepted"]
        values = list(b.sensitivity.values())
        assert all(x <= y + 1e-12 for x, y in zip(values, values[1:]))
        if b.breakdown is not None:
            assert b.delta_low <= 0 <= b.delta_high

    def test_requires_known_logging_policy(self):
        sim, _, logs = version_logs(n=1200)
        ev = ps.Evaluator(logs, action="offer", reward="accepted", features=F)
        with pytest.raises(ValueError, match="logging_policy"):
            ev.bounds(ps.uniform())


class TestPlacebo:
    def test_placebo_cannot_be_a_feature(self):
        _, logs = wrapper_logs(1, 1500, informed=False)
        with pytest.raises(ValueError, match="служебные"):
            ps.Evaluator(logs, action="offer", reward="accepted", features=[*F, "prev_accepted"],
                         placebo="prev_accepted")

    def test_auto_features_skip_placebo(self):
        _, logs = wrapper_logs(1, 1500, informed=False)
        ev = ps.Evaluator(logs.drop(columns="propensity"), action="offer", reward="accepted",
                          placebo="prev_accepted")
        assert "prev_accepted" not in ev.data.features

    def test_informed_wrapper_fails_placebo(self):
        _, logs = wrapper_logs(7, 8000, informed=True)
        rep = ps.evaluate(logs.drop(columns="propensity"), ps.uniform(), action="offer", reward="accepted",
                          features=F, placebo="prev_accepted", inference="influence")
        assert rep.placebo[0].status == "fail"
        assert "placebo_failed" in {f.code for f in rep.findings} and rep["accepted"].trust != "reliable"

    def test_random_wrapper_passes_placebo(self):
        _, logs = wrapper_logs(7, 8000, informed=False)
        rep = ps.evaluate(logs.drop(columns="propensity"), ps.uniform(), action="offer", reward="accepted",
                          features=F, placebo="prev_accepted", inference="influence")
        assert rep.placebo[0].status == "ok"
        assert "placebo_ok" in {f.code for f in rep.findings}
        json.dumps(rep.to_dict(), allow_nan=False)


class TestSharpBounds:
    def test_identical_policies_have_zero_width(self):
        sim, versions, logs = version_logs(n=3000)
        ev = ps.Evaluator(logs, action="offer", reward="accepted", features=F, logging_policy=versions, logger="month")
        b = ev.bounds(ps.uniform(), baseline=ps.uniform(), name="u1", baseline_name="u2")["accepted"]
        assert b.delta_low == pytest.approx(0, abs=1e-12) and b.delta_high == pytest.approx(0, abs=1e-12)

    def test_policy_baseline_width_uses_mass_difference(self):
        sim, versions, logs = version_logs(n=3000)
        ev = ps.Evaluator(logs, action="offer", reward="accepted", features=F, logging_policy=versions, logger="month")
        mixed = ps.mixture([ps.uniform(), versions["v4"]], weights=[0.8, 0.2])
        report = ev.bounds(mixed, baseline=ps.uniform(), name="m", baseline_name="u")
        support = ev.support().to_numpy()
        from policyscope.policy import policy_matrix

        diff = (policy_matrix(mixed, ev.data, ev._frame) - 0.25) * ~support
        width = np.abs(diff).sum(axis=1).mean()
        b = report["accepted"]
        assert b.delta_high - b.delta_low == pytest.approx(width, rel=1e-9)
        assert report.baseline_uncovered > 0 and "У «u» таких решений" in report.to_markdown()

    def test_sensitivity_clips_each_action_and_breakdown_is_exhaustive(self):
        sim, versions, logs = version_logs(n=3000)
        ev = ps.Evaluator(logs, action="offer", reward="accepted", features=F, logging_policy=versions, logger="month")
        b = ev.bounds(ps.uniform(), sensitivity=(0.0, 1e6))["accepted"]
        assert b.sensitivity[0.0] == pytest.approx(b.delta_low)      # λ = 0: отклик 0 = нижняя граница
        assert b.sensitivity[1e6] == pytest.approx(b.delta_high)     # λ → ∞: всё упёрлось в 1
        if b.delta_low < 0 < b.delta_high:
            assert b.breakdown is not None

    def test_alpha_is_validated(self):
        sim, versions, logs = version_logs(n=1200)
        ev = ps.Evaluator(logs, action="offer", reward="accepted", features=F, logging_policy=versions, logger="month")
        with pytest.raises(ValueError, match="alpha"):
            ev.bounds(ps.uniform(), alpha=1.0)

    def test_mixture_findings_are_attached(self):
        sim, versions, logs = version_logs(n=3000)
        ev = ps.Evaluator(logs, action="offer", reward="accepted", features=F, logging_policy=versions, logger="month")
        report = ev.bounds(ps.uniform())
        assert "logging_mixture" in {f.code for f in report.findings}
        assert "Предпосылки покрытой части" in report.to_markdown()


class TestPlaceboRules:
    def test_not_applicable_when_versions_fix_the_choice(self):
        sim, versions, logs = version_logs(n=3000)
        logs["prev"] = np.random.default_rng(0).normal(size=len(logs))
        rep = ps.evaluate(logs, ps.uniform(), action="offer", reward="accepted", features=F,
                          logging_policy=versions, logger="month", placebo="prev", inference="influence")
        assert rep.placebo[0].status == "not_applicable"
        assert "placebo_not_applicable" in {f.code for f in rep.findings}

    def test_values_must_be_finite_and_vary(self):
        _, logs = wrapper_logs(1, 1500, informed=False)
        with pytest.raises(ValueError, match="постоянна"):
            ps.Evaluator(logs.assign(prev_accepted=1.0), action="offer", reward="accepted", features=F,
                         placebo="prev_accepted")
        with pytest.raises(ValueError, match="бесконечност"):
            ps.Evaluator(logs.assign(prev_accepted=np.inf), action="offer", reward="accepted", features=F,
                         placebo="prev_accepted")

    def test_listed_among_skipped_features(self):
        _, logs = wrapper_logs(1, 1500, informed=False)
        ev = ps.Evaluator(logs.drop(columns="propensity"), action="offer", reward="accepted",
                          placebo="prev_accepted")
        assert "плацебо" in ev.data.skipped_features["prev_accepted"]

    def test_interval_and_rule_use_the_check_level(self):
        _, logs = wrapper_logs(7, 8000, informed=True)
        rep = ps.evaluate(logs.drop(columns="propensity"), ps.uniform(), action="offer", reward="accepted",
                          features=F, placebo="prev_accepted", inference="influence")
        c = rep.placebo[0]
        assert c.estimate - c.low == pytest.approx(c.high - c.estimate)
        assert np.isfinite(c.material) and c.material > 0
        assert (c.status == "fail") == ((c.low > 0 or c.high < 0) and abs(c.estimate) > c.material)

    def test_weights_are_centred_exactly_for_a_known_logger(self):
        """Кандидат частично вне носителя: E[w_B | x] < 1, но центрированный вес в среднем ноль."""
        sim, versions, logs = version_logs(n=3000)
        logs["prev"] = np.random.default_rng(1).normal(size=len(logs))
        ev = ps.Evaluator(logs, action="offer", reward="accepted", features=F,
                          logging_policy=versions["v1"].action_probs(logs).pipe(lambda p: 0.8 * p + 0.05),
                          placebo="prev")
        from policyscope.policy import policy_matrix

        pi = policy_matrix(ps.uniform(["card", "loan"]), ev.data, ev._frame)
        omega, determined = ev._placebo_weights(pi, None)
        assert not determined
        # E[ω | x] = 0 для каждой строки: сумма по действиям с вероятностями логирующей политики.
        rows = np.arange(ev.data.n)
        p = ev._logging_matrix
        w_all = pi / p - 1.0
        centre = (p * w_all).sum(axis=1)
        np.testing.assert_allclose(omega, (pi[rows, ev.data.a] / p[rows, ev.data.a] - 1.0) - centre)
        np.testing.assert_allclose((p * (w_all - centre[:, None])).sum(axis=1), 0.0, atol=1e-12)


class TestPlaceboAfterReview:
    def test_rounding_noise_in_deterministic_versions_is_still_not_applicable(self):
        sim, versions, logs = version_logs(n=3000)
        logs["prev"] = np.random.default_rng(2).normal(size=len(logs))
        noisy = {m: v.action_probs(logs) * (1 + 5e-7) for m, v in versions.items()}
        rep = ps.evaluate(logs, ps.uniform(), action="offer", reward="accepted", features=F,
                          logging_policy=noisy, logger="month", placebo="prev", inference="influence")
        assert rep.placebo[0].status == "not_applicable"

    def test_pass_text_does_not_claim_factors_are_weak(self):
        _, logs = wrapper_logs(7, 8000, informed=False)
        rep = ps.evaluate(logs.drop(columns="propensity"), ps.uniform(), action="offer", reward="accepted",
                          features=F, placebo="prev_accepted", inference="influence")
        c = rep.placebo[0]
        detail = next(f.detail for f in rep.findings if f.code == "placebo_ok")
        assert c.status == "ok" and "они слабые" not in detail
        if max(abs(c.low), abs(c.high)) > c.material:
            assert "прохождение не доказывает" in detail
