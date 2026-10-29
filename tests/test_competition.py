"""Случайные политики, известная логирующая политика и честный выбор лучшего кандидата."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import policyscope as ps

from conftest import ACTIONS, make_bandit

KW = dict(action="action", reward="reward", features=["x1", "x2"])


@pytest.fixture(scope="module")
def bandit():
    return make_bandit(3000, seed=51)


class TestRandomPolicies:
    def test_uniform(self, bandit):
        ev = ps.Evaluator(bandit.logs, propensity="propensity", **KW)
        from policyscope.policy import policy_matrix

        probs = policy_matrix(ps.uniform(), ev.data, ev._frame)
        np.testing.assert_allclose(probs, 1 / 3)
        subset = policy_matrix(ps.uniform(["red", "blue"]), ev.data, ev._frame)
        np.testing.assert_allclose(subset[:, ev.data.actions.index("green")], 0)
        with pytest.raises(ValueError, match="нет в логах"):
            policy_matrix(ps.uniform(["purple"]), ev.data, ev._frame)

    def test_random_like_keeps_the_balance(self, bandit):
        ev = ps.Evaluator(bandit.logs, propensity="propensity", **KW)
        from policyscope.policy import policy_matrix

        logged = bandit.logs["action"].value_counts(normalize=True)
        probs = policy_matrix(ps.random_like(), ev.data, ev._frame)
        expected = [logged[a] for a in ev.data.actions]
        np.testing.assert_allclose(probs, np.tile(expected, (ev.data.n, 1)))
        # Баланс другой политики — её средние доли; клиент на выбор не влияет.
        like_b = policy_matrix(ps.random_like(bandit.policy_b), ev.data, ev._frame)
        np.testing.assert_allclose(like_b[0], bandit.policy_b[ev.data.actions].mean().to_numpy())
        assert np.ptp(like_b, axis=0).max() == 0
        with pytest.raises(ValueError, match="в сумме давать 1"):
            policy_matrix(ps.random_like({"red": 0.5, "blue": 0.2}), ev.data, ev._frame)

    def test_random_like_value_covers_truth(self, bandit):
        report = ps.evaluate(bandit.logs, ps.random_like(), propensity="propensity", **KW)
        shares = bandit.logs["action"].value_counts(normalize=True).reindex(ACTIONS).to_numpy()
        truth = float(np.mean(bandit.mu @ shares))
        m = report["reward"]
        assert m.v_b.low <= truth <= m.v_b.high


class TestLoggingPolicy:
    def test_known_policy_gives_the_logged_propensities(self, bandit):
        ev = ps.Evaluator(bandit.logs.drop(columns="propensity"), logging_policy=bandit.policy_a, **KW)
        np.testing.assert_allclose(ev.nuisance.p_taken, bandit.logs["propensity"])
        rep = ev.evaluate(bandit.policy_b)
        assert rep.settings["propensity_origin"] == "policy" and rep.settings["inference"] == "influence"
        assert "по заданной политике A" in rep.to_markdown()

    def test_deterministic_known_policy_is_blocked(self, bandit):
        x = bandit.logs
        scores = pd.DataFrame({"red": x["x1"], "green": -x["x1"], "blue": x["x2"]}, index=x.index)
        greedy_a = ps.greedy(scores)
        logs = bandit.logs.assign(action=greedy_a.idxmax(axis=1))
        rep = ps.evaluate(logs.drop(columns="propensity"), ps.uniform(), logging_policy=greedy_a, **KW)
        m = rep["reward"]
        assert m.decision == "inconclusive"
        assert "deterministic_logging" in {f.code for f in m.findings}

    def test_policy_that_could_not_log_these_actions_is_an_error(self, bandit):
        always_red = pd.Series("red", index=bandit.logs.index)
        with pytest.raises(ValueError, match="не могла выбрать залогированное действие"):
            ps.Evaluator(bandit.logs.drop(columns="propensity"), logging_policy=always_red, **KW)

    def test_argument_errors(self, bandit):
        logs = bandit.logs.assign(period=np.where(np.arange(len(bandit.logs)) % 2, "v1", "v2"))
        with pytest.raises(ValueError, match="что-то одно"):
            ps.Evaluator(logs, propensity="propensity", logging_policy=bandit.policy_a, **KW)
        with pytest.raises(ValueError, match="только вместе"):
            ps.Evaluator(logs, logger="period", **KW)
        with pytest.raises(ValueError, match="колонку logger"):
            ps.Evaluator(logs.drop(columns="propensity"), logging_policy={"v1": bandit.policy_a}, **KW)
        with pytest.raises(ValueError, match="Нет политики"):
            ps.Evaluator(logs.drop(columns="propensity"), logging_policy={"v1": bandit.policy_a},
                         logger="period", **KW)

    def test_mixture_of_loggers(self, bandit):
        """Две политики по периодам: пропенсити — смесь с долями периодов, logger не признак."""
        n = len(bandit.logs)
        rng = np.random.default_rng(3)
        period = np.where(np.arange(n) < n // 4, "v1", "v2")
        p1 = pd.DataFrame(rng.dirichlet(np.ones(3), n), columns=ACTIONS)
        p2 = bandit.policy_a
        draws = rng.random(n)[:, None]
        chosen = np.where(period[:, None] == "v1", p1.to_numpy(), p2.to_numpy())
        a = (draws > chosen.cumsum(axis=1)).sum(axis=1).clip(0, 2)
        logs = bandit.logs.drop(columns="propensity").assign(
            action=np.asarray(ACTIONS, dtype=object)[a], period=period,
        )
        ev = ps.Evaluator(logs, action="action", reward="reward", logging_policy={"v1": p1, "v2": p2},
                          logger="period")
        expected = 0.25 * p1.to_numpy()[np.arange(n), a] + 0.75 * p2.to_numpy()[np.arange(n), a]
        np.testing.assert_allclose(ev.nuisance.p_taken, expected)
        assert "period" not in ev.data.features
        rep = ev.evaluate(bandit.policy_b)
        codes = {f.code for f in rep.findings}
        assert "logging_mixture" in codes and "logger_shift" not in codes  # периоды назначены не по клиентам

    def test_shifted_periods_are_flagged(self, bandit):
        logs = bandit.logs.drop(columns="propensity").assign(period=np.where(bandit.logs["x1"] > 0, "v1", "v2"))
        ev = ps.Evaluator(logs, action="action", reward="reward", features=["x1", "x2"],
                          logging_policy={"v1": bandit.policy_a, "v2": bandit.policy_a}, logger="period")
        codes = {f.code for f in ev.evaluate(bandit.policy_b).findings}
        assert "logger_shift" in codes


class TestCompetition:
    @pytest.fixture(scope="class")
    @staticmethod
    def board(bandit):
        ev = ps.Evaluator(bandit.logs, propensity="propensity", **KW)
        return ev.compare({
            "b": bandit.policy_b,
            "b_soft": 0.7 * bandit.policy_b + 0.3 / 3,
            "a": bandit.policy_a,
            "random": ps.uniform(),
        })

    def test_simultaneous_intervals_are_wider(self, board):
        t = board.table
        assert (t["delta_low_sim"] <= t["delta_low"] + 1e-12).all()
        assert (t["delta_high_sim"] >= t["delta_high"] - 1e-12).all()

    def test_pairwise_covers_all_pairs(self, board):
        pw = board.pairwise
        assert len(pw) == 6 and set(pw.columns) >= {"diff", "low", "high", "p_value_adj"}
        row = pw[(pw["candidate"] == "b") & (pw["other"] == "a")]
        if row.empty:
            row = pw[(pw["candidate"] == "a") & (pw["other"] == "b")]
        diff = board["b"]["reward"].v_b.value - board["a"]["reward"].v_b.value
        assert abs(abs(row["diff"].iloc[0]) - abs(diff)) < 1e-12

    def test_best(self, board):
        sel = board.best()
        deltas = {name: rep["reward"].delta.value for name, rep in board.reports.items()}
        assert sel.winner == max(deltas, key=deltas.get)
        assert sel.winner in sel.best_set
        assert sel.winner_delta.low <= sel.winner_delta.value <= sel.winner_delta.high
        assert "Кто лучший" in board.to_markdown()
        json.dumps(sel.to_dict(), allow_nan=False)
        json.dumps(board.to_dict(), allow_nan=False)

    def test_guardrails_exclude(self, board):
        sel = board.best(guardrails={"reward": 0.0})
        for name in sel.excluded:
            assert board.table.set_index("candidate").loc[name, "delta_low_sim"] < 0
        with pytest.raises(KeyError):
            board.best(guardrails={"revenue": 0.0})
        with pytest.raises(ValueError):
            board.best(guardrails={"reward": -1.0})

    def test_bootstrap_path(self, bandit):
        small = make_bandit(800, seed=52)
        ev = ps.Evaluator(small.logs.drop(columns="propensity"), action="action", reward="reward",
                          features=["x1", "x2"], n_jobs=1)
        board = ev.compare({"b": small.policy_b, "random": ps.uniform()}, n_boot=50)
        assert len(board.pairwise) == 1 and board.best().winner in {"b", "random"}

    def test_best_needs_intervals(self, bandit):
        ev = ps.Evaluator(bandit.logs, propensity="propensity", **KW)
        board = ev.compare({"b": bandit.policy_b, "random": ps.uniform()}, inference="none")
        with pytest.raises(ValueError, match="интервалы"):
            board.best()


class TestMonthlyVersions:
    @pytest.fixture(scope="class")
    @staticmethod
    def history():
        sim = ps.synthetic.OfferSimulator(seed=4)
        months = ["m1", "m2", "m3", "m4"]
        versions = sim.versions(4, drift=0.5, seed=1, names=months)
        clients = sim.sample_clients(6000)
        rows = np.array_split(np.arange(len(clients)), 4)
        logs = pd.concat([sim.simulate(clients.iloc[r], versions[m]).assign(month=m) for m, r in zip(months, rows)],
                         ignore_index=True).drop(columns="propensity")
        return sim, versions, clients, rows, logs

    def test_versions_are_similar_but_not_equal(self, history):
        sim, versions, clients, _, _ = history
        choices = pd.DataFrame({m: v.action_probs(clients).idxmax(axis=1) for m, v in versions.items()})
        agree = (choices.nunique(axis=1) == 1).mean()
        assert 0.2 < agree < 0.95
        with pytest.raises(ValueError):
            sim.versions(2, names=["only one"])

    def test_random_like_by_month_keeps_monthly_balance(self, history):
        _, versions, _, _, logs = history
        from policyscope.policy import policy_matrix

        ev = ps.Evaluator(logs, action="offer", reward="accepted", logging_policy=versions, logger="month")
        probs = pd.DataFrame(policy_matrix(ps.random_like(by="month"), ev.data, ev._frame), columns=ev.data.actions)
        for month, rows in logs.groupby("month").groups.items():
            observed = logs.loc[rows, "offer"].value_counts(normalize=True).reindex(ev.data.actions, fill_value=0)
            np.testing.assert_allclose(probs.loc[rows].mean(), observed.to_numpy())
            assert np.ptp(probs.loc[rows].to_numpy(), axis=0).max() < 1e-12
        with pytest.raises(ValueError, match="by="):
            ps.random_like({"card": 1.0}, by="month")
        # Баланс политики (таблицы) внутри месяцев — её средние доли по строкам месяца.
        v1 = pd.DataFrame(policy_matrix(versions["m1"], ev.data, ev._frame), columns=ev.data.actions)
        by_month = policy_matrix(ps.random_like(v1, by="month"), ev.data, ev._frame)
        for month, rows in logs.groupby("month").groups.items():
            np.testing.assert_allclose(by_month[rows[0]], v1.loc[rows].mean().to_numpy())

    def test_mixture_policy(self, history):
        _, versions, _, _, logs = history
        from policyscope.policy import policy_matrix

        ev = ps.Evaluator(logs, action="offer", reward="accepted", logging_policy=versions, logger="month")
        mixed = policy_matrix(ps.mixture(versions), ev.data, ev._frame)
        parts = [policy_matrix(v, ev.data, ev._frame) for v in versions.values()]
        np.testing.assert_allclose(mixed, np.mean(parts, axis=0))
        rep = ev.evaluate(ps.mixture(versions))
        assert rep["accepted"].decision in ("no_difference", "better", "worse")
        with pytest.raises(ValueError):
            ps.mixture(versions, weights=[0.5, 0.5])

    def test_random_policy_needs_exploration_in_logs(self, history):
        _, versions, _, _, logs = history
        rep = ps.evaluate(logs, ps.random_like(), action="offer", reward="accepted",
                          logging_policy=versions, logger="month")
        assert rep["accepted"].decision == "inconclusive"

    def test_seasonal_response_is_flagged(self, history):
        sim, versions, clients, rows, logs = history
        kw = dict(action="offer", reward="accepted", logging_policy=versions, logger="month")
        calm = {f.code for f in ps.evaluate(logs, ps.mixture(versions), **kw).findings}
        assert "logger_drift" not in calm
        seasonal = pd.concat(
            [sim.simulate(clients.iloc[r], versions[m], shift=1.0 if m == "m4" else 0.0).assign(month=m)
             for m, r in zip(versions, rows)], ignore_index=True,
        ).drop(columns="propensity")
        codes = {f.code for f in ps.evaluate(seasonal, ps.mixture(versions), **kw).findings}
        assert "logger_drift" in codes

    def test_leaderboard_shows_leader_versus_others(self, history):
        _, versions, _, _, logs = history
        board = ps.Evaluator(logs, action="offer", reward="accepted", logging_policy=versions,
                             logger="month").compare(versions)
        md = board.to_markdown()
        assert "Лидер против остальных" in md
        assert md.count("| m") >= len(versions) - 1


class TestNewModelOnVersionLogs:
    def test_allowed_restricts_every_helper(self):
        scores = pd.DataFrame({"red": [3.0, 1.0], "green": [2.0, 5.0], "blue": [1.0, 0.0]})
        allowed = pd.DataFrame({"red": [False, True], "green": [True, False], "blue": [True, True]})
        for policy in (ps.greedy(scores, allowed=allowed), ps.softmax(scores, allowed=allowed),
                       ps.epsilon_greedy(scores, 0.3, allowed=allowed)):
            np.testing.assert_allclose(policy.sum(axis=1), 1.0)
            assert (policy.to_numpy()[~allowed.to_numpy()] == 0).all()
        assert ps.greedy(scores, allowed=allowed).idxmax(axis=1).tolist() == ["green", "red"]
        with pytest.raises(ValueError, match="хотя бы одно"):
            ps.greedy(scores, allowed=allowed & False)

    def test_support_and_exact_uncovered_share(self):
        sim = ps.synthetic.OfferSimulator(seed=6)
        versions = sim.versions(3, drift=0.4, seed=2)
        clients = sim.sample_clients(3000)
        rows = np.array_split(np.arange(len(clients)), 3)
        logs = pd.concat([sim.simulate(clients.iloc[r], versions[m]).assign(month=m) for m, r in zip(versions, rows)],
                         ignore_index=True)
        ev = ps.Evaluator(logs.drop(columns="propensity"), action="offer", reward="accepted",
                          logging_policy=versions, logger="month")
        support = ev.support()
        recommended = sum(v.action_probs(logs) for v in versions.values()) > 0
        assert support.equals(recommended[support.columns])
        other = sim.policy("greedy", seed=9).action_probs(logs)
        rep = ev.evaluate(other)
        expected = float((other * ~recommended).sum(axis=1).mean())
        assert rep.overlap.uncovered == pytest.approx(expected)
        assert "вне покрытия логов" in rep.to_markdown()
        inside = ps.greedy(sim.policy("greedy", seed=9).scores(logs), allowed=support)
        assert ev.evaluate(inside).overlap.uncovered == 0
        with pytest.raises(ValueError, match="logging_policy"):
            ps.Evaluator(logs, action="offer", reward="accepted", propensity="propensity").support()
