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
            # Доли без самой строки (leave-one-out): строки месяца различаются не больше чем на 1/(n−1).
            assert np.ptp(probs.loc[rows].to_numpy(), axis=0).max() <= 1 / (len(rows) - 1) + 1e-12
            counts = logs.loc[rows, "offer"].value_counts()
            own = probs.loc[rows].to_numpy()[np.arange(len(rows)), ev.data.a[logs.index.get_indexer(rows)]]
            np.testing.assert_allclose(own, (counts[logs.loc[rows, "offer"]].to_numpy() - 1) / (len(rows) - 1))
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


class TestReviewFixes:
    """Исправления по математическому ревью: смесь версий, логирующая политика, случайные политики."""

    @pytest.fixture(scope="class")
    @staticmethod
    def months():
        sim = ps.synthetic.OfferSimulator(seed=4)
        versions = sim.versions(3, drift=0.5, seed=1, names=["m1", "m2", "m3"])
        clients = sim.sample_clients(4500)
        sizes = [2500, 1200, 800]  # неравные доли: перестановка меток тогда меняет пропенсити
        bounds = np.cumsum([0, *sizes])
        logs = pd.concat([sim.simulate(clients.iloc[lo:hi], versions[m]).assign(month=m)
                          for m, lo, hi in zip(versions, bounds[:-1], bounds[1:])],
                         ignore_index=True).drop(columns="propensity")
        return sim, versions, logs

    def test_versions_are_covered_and_not_blocked(self, months):
        _, versions, logs = months
        ev = ps.Evaluator(logs, action="offer", reward="accepted", logging_policy=versions, logger="month")
        rep = ev.evaluate(versions["m2"], inference="none")
        codes = {f.code for f in rep.findings}
        assert rep.overlap.uncovered == 0
        assert not codes & {"deterministic_logging", "support_deficit"}
        random = ev.evaluate(ps.uniform(), inference="none")
        blocks = {f.code for f in random.findings if f.severity == "block"}
        assert random.overlap.uncovered > 0.2 and blocks & {"deterministic_logging", "support_deficit"}

    def test_baseline_version_is_not_blocked(self, months):
        _, versions, logs = months
        ev = ps.Evaluator(logs, action="offer", reward="accepted", logging_policy=versions, logger="month")
        rep = ev.evaluate(versions["m3"], baseline=versions["m1"], inference="none")
        assert "deterministic_logging_baseline" not in {f.code for f in rep.findings}

    def test_mislabeled_periods_are_an_error(self, months):
        _, versions, logs = months
        swapped = logs.assign(month=logs["month"].replace({"m1": "m2", "m2": "m1"}))
        with pytest.raises(ValueError, match="политика их периода не могла выбрать"):
            ps.Evaluator(swapped, action="offer", reward="accepted", logging_policy=versions, logger="month")

    def test_logging_policy_may_use_unlogged_actions(self, months):
        sim, versions, logs = months
        probs = sim.policy("softmax", seed=2).action_probs(logs) * 0.9
        probs["new_offer"] = 0.1  # предложение, которого в логах нет
        ev = ps.Evaluator(logs, action="offer", reward="accepted", logging_policy=probs)
        rows = np.arange(len(logs))
        np.testing.assert_allclose(ev.data.propensity, probs[ev.data.actions].to_numpy()[rows, ev.data.a])
        assert "new_offer" not in ev.support().columns
        with pytest.raises(ValueError, match="которых нет в логах"):
            ev.evaluate(probs, inference="none")  # кандидату — нельзя

    def test_logger_shares_ignore_unused_categories(self, months):
        _, versions, logs = months
        cat = logs.assign(month=pd.Categorical(logs["month"], categories=["m0", "m1", "m2", "m3"]))
        ev = ps.Evaluator(cat, action="offer", reward="accepted", logging_policy=versions, logger="month")
        assert set(ev._logger_shares) == {"m1", "m2", "m3"}

    def test_small_by_groups_warn(self, months):
        _, versions, logs = months
        from policyscope.policy import policy_matrix

        ev = ps.Evaluator(logs.assign(tiny=np.arange(len(logs)) // 10), action="offer", reward="accepted",
                          logging_policy=versions, logger="month")
        with pytest.warns(UserWarning, match="самой малой группе"):
            policy_matrix(ps.random_like(by="tiny"), ev.data, ev._frame)

    def test_uniform_deduplicates_labels(self, bandit):
        from policyscope.policy import policy_matrix

        ev = ps.Evaluator(bandit.logs, propensity="propensity", **KW)
        probs = policy_matrix(ps.uniform(["red", "red", "blue"]), ev.data, ev._frame)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0)

    def test_allowed_alignment_is_not_ambiguous(self, bandit):
        shuffled = bandit.logs.sample(frac=1.0, random_state=0)
        ev = ps.Evaluator(shuffled, propensity="propensity", **KW)
        scores = pd.DataFrame(np.random.default_rng(0).normal(size=(len(shuffled), 3)), columns=ACTIONS)
        allowed = pd.DataFrame(True, index=shuffled.index, columns=ACTIONS)
        with pytest.raises(ValueError, match="однозначно"):
            ps.greedy(scores, allowed=allowed)
        assert ps.greedy(scores.set_axis(shuffled.index), allowed=allowed).shape == (len(shuffled), 3)
        del ev

    def test_drift_test_accounts_for_repeat_clients(self):
        """Клиентский эффект и несколько строк клиента в периоде: без кластеров тест врёт."""
        from scipy import stats

        from policyscope.evaluator import _equal_means_test

        rng = np.random.default_rng(3)
        robust, naive = [], []
        for _ in range(200):
            clients = np.repeat(np.arange(300), 8)
            period = rng.integers(0, 3, size=300)[clients]  # клиент целиком в одном периоде
            values = rng.normal(0, 1.0, size=300)[clients] + rng.normal(0, 0.5, size=len(clients))
            robust.append(_equal_means_test(values, period, clients)[0])
            naive.append(stats.f_oneway(*[values[period == k] for k in range(3)]).pvalue)
        assert np.mean(np.array(robust) < 0.05) <= 0.1
        assert np.mean(np.array(naive) < 0.05) > 0.3

    def test_shift_check_sees_non_monotone_composition(self):
        sim = ps.synthetic.OfferSimulator(seed=12)
        versions = sim.versions(2, drift=0.5, seed=1, names=["a", "b"])
        clients = sim.sample_clients(8000)
        extreme = (clients["age"] - clients["age"].median()).abs() > clients["age"].std()
        parts = [sim.simulate(clients[~extreme], versions["a"]).assign(month="a"),
                 sim.simulate(clients[extreme], versions["b"]).assign(month="b")]
        logs = pd.concat(parts, ignore_index=True).drop(columns="propensity")
        ev = ps.Evaluator(logs, action="offer", reward="accepted", features=["age", "income", "loyal"],
                          logging_policy=versions, logger="month")
        assert ev._logger_shift_auc() > 0.9
        assert "logger_shift" in {f.code for f in ev.evaluate(versions["a"], inference="none").findings}


class TestSelectionAndFamilies:
    def test_single_admitted_candidate_is_not_called_significantly_better(self, bandit):
        ev = ps.Evaluator(bandit.logs, propensity="propensity", **KW)
        board = ev.compare({"b": bandit.policy_b, "a": bandit.policy_a}, inference="influence")
        selection = board.best("reward", guardrails=None)
        assert selection.n_admitted == 2
        one = ev.compare({"b": bandit.policy_b}, inference="influence").best("reward")
        assert one.n_admitted == 1
        assert "Других допущенных кандидатов нет" in "\n".join(one.lines())

    def test_adjusted_p_never_below_raw_with_few_clusters(self, bandit):
        logs = bandit.logs.assign(team=np.arange(len(bandit.logs)) % 12)
        ev = ps.Evaluator(logs, propensity="propensity", cluster="team", **KW)
        board = ev.compare({"b": bandit.policy_b, "soft": 0.5 * bandit.policy_b + 0.5 / 3}, inference="influence")
        t = board.table
        assert (t["p_value_adj"] >= t["p_value"] - 1e-12).all()
        assert ((t["delta_high_sim"] - t["delta_low_sim"]) >= (t["delta_high"] - t["delta_low"]) - 1e-12).all()


class TestVerifiedRegressions:
    """Проблемы, найденные проверкой исправлений: каждая — с минимальным воспроизведением."""

    def test_drift_test_is_calibrated_for_rare_events_in_a_small_period(self):
        from policyscope.evaluator import _equal_means_test

        rng = np.random.default_rng(7)
        rejections = []
        for _ in range(400):
            sizes = [2400, 1800, 1500, 100]
            group = np.repeat(np.arange(4), sizes)
            mu = rng.beta(2, 98, size=len(group))
            r = (rng.random(len(group)) < mu).astype(float)
            rejections.append(_equal_means_test(r - mu, group, np.arange(len(group)))[0] < 0.001)
        assert np.mean(rejections) <= 0.01

    def test_drift_test_without_clusters_is_a_slightly_cautious_anova(self):
        from scipy import stats

        from policyscope.evaluator import _equal_means_test

        rng = np.random.default_rng(8)
        values, group = rng.normal(size=600), rng.integers(0, 4, size=600)
        expected = stats.f_oneway(*[values[group == k] for k in range(4)]).pvalue
        p = _equal_means_test(values, group, np.arange(600))[0]
        assert expected - 1e-12 <= p <= expected + 0.05  # пол дисперсии только увеличивает p

    def test_drift_test_handles_client_by_period_effects(self):
        """Несколько строк клиента в месяце и эффект «клиент × месяц»: ложных тревог нет."""
        from policyscope.evaluator import _equal_means_test

        rng = np.random.default_rng(9)
        G, K, r = 300, 4, 8
        clients = np.repeat(np.arange(G), K * r)
        period = np.tile(np.repeat(np.arange(K), r), G)
        p = []
        for _ in range(200):
            values = (rng.normal(0, 1, G)[clients] + rng.normal(0, 0.5, (G, K))[clients, period]
                      + rng.normal(0, 1, len(clients)))
            p.append(_equal_means_test(values, period, clients)[0])
        assert np.mean(np.array(p) < 0.01) <= 0.04

    def test_drift_test_is_calibrated_for_a_small_noisier_period(self):
        from policyscope.evaluator import _equal_means_test

        rng = np.random.default_rng(10)
        group = np.repeat(np.arange(4), [4000, 3000, 2500, 100])
        p = [_equal_means_test(rng.normal(0, np.where(group == 3, 2.0, 1.0)), group, np.arange(len(group)))[0]
             for _ in range(300)]
        assert np.mean(np.array(p) < 0.01) <= 0.03

    def test_random_logging_policy_uses_realised_group_shares(self, bandit):
        rng = np.random.default_rng(1)
        logs = bandit.logs.drop(columns="propensity").assign(seg=rng.integers(0, 40, len(bandit.logs)))
        logs.loc[logs.index[0], "seg"] = 99  # группа из одной строки
        ev = ps.Evaluator(logs, logging_policy=ps.random_like(by="seg"), **KW)
        shares = logs.groupby("seg")["action"].value_counts(normalize=True)
        expected = shares.loc[list(zip(logs["seg"], logs["action"]))].to_numpy()
        np.testing.assert_allclose(ev.data.propensity, expected)
        rep = ev.evaluate(ps.random_like(by="seg"), inference="influence")
        assert rep.overlap.mean_weight == pytest.approx(1.0)
        assert "propensity_check_failed" not in {f.code for f in rep["reward"].findings}

    def test_wrong_known_policy_inflates_weights(self):
        sim = ps.synthetic.OfferSimulator(seed=21)
        true_a = sim.policy("softmax", seed=2, temperature=0.6)
        clients = sim.sample_clients(6000)
        logs = sim.simulate(clients, true_a).drop(columns="propensity")
        stated = sim.policy("softmax", seed=2, temperature=1.0)  # та же модель, но не та температура
        rep = ps.evaluate(logs, sim.policy("greedy", seed=2), action="offer", reward="accepted",
                          logging_policy=stated, inference="influence")
        assert rep.overlap.uncovered == 0
        assert "weights_inflated" in {f.code for f in rep.findings}

    def test_near_zero_logging_probabilities_deflate_weights(self):
        sim = ps.synthetic.OfferSimulator(seed=22)
        cold = sim.policy("softmax", seed=3, temperature=0.05)  # формально все действия возможны
        logs = sim.simulate(sim.sample_clients(4000), cold).drop(columns="propensity")
        rep = ps.evaluate(logs, ps.uniform(), action="offer", reward="accepted", logging_policy=cold,
                          inference="influence")
        assert rep.overlap.uncovered == 0
        found = {f.code: f.severity for f in rep.findings}
        assert "weights_deflated" in found and rep["accepted"].trust != "reliable"

    def test_placebo_cannot_be_a_service_column(self, bandit):
        with pytest.raises(ValueError, match="плацебо"):
            ps.Evaluator(bandit.logs, propensity="propensity", placebo="reward", **KW)

    def test_bootstrap_simultaneous_interval_contains_marginal(self, bandit):
        small = make_bandit(800, seed=52)
        ev = ps.Evaluator(small.logs, propensity="propensity", **KW)
        board = ev.compare({"b": small.policy_b, "soft": 0.5 * small.policy_b + 0.5 / 3},
                           inference="bootstrap", n_boot=50)
        t = board.table
        assert (t["delta_low_sim"] <= t["delta_low"] + 1e-12).all()
        assert (t["delta_high_sim"] >= t["delta_high"] - 1e-12).all()


class TestSecondReview:
    """Проблемы, найденные второй проверкой исправлений."""

    @staticmethod
    def stratified_logs(n_groups=100, per_group=20, seed=0):
        rng = np.random.default_rng(seed)
        b = make_bandit(n_groups * per_group, seed=60 + seed)
        seg = np.repeat(np.arange(n_groups), per_group)
        shares = rng.dirichlet([2, 2, 2], size=n_groups)[seg]
        action = (rng.random(len(seg))[:, None] > shares.cumsum(axis=1)).sum(axis=1).clip(0, 2)
        logs = b.logs.drop(columns="propensity").assign(seg=seg, action=np.asarray(ACTIONS, dtype=object)[action])
        return b, logs

    def test_mixture_member_equal_to_random_logger_uses_logged_shares(self):
        b, logs = self.stratified_logs()
        logger = ps.random_like(by="seg")
        ev = ps.Evaluator(logs, logging_policy=logger, **KW)
        candidate = ps.mixture([ps.random_like(by="seg"), b.policy_b], weights=[0.9, 0.1])
        rep = ev.evaluate(candidate, inference="influence")
        codes = {f.code for f in rep["reward"].findings}
        assert not codes & {"propensity_check_failed", "weights_deflated"}
        same = ev.evaluate(ps.mixture([logger], weights=[1.0]), inference="influence")
        assert same.overlap.mean_weight == pytest.approx(1.0)
        halves = logs.assign(half=logs["seg"] % 2)
        reordered = ps.Evaluator(halves, logging_policy=ps.random_like(by=["half", "seg"]), **KW)
        rep = reordered.evaluate(ps.random_like(by=["seg", "half"]), inference="none")
        assert rep.overlap.mean_weight == pytest.approx(1.0)

    def test_single_row_group_falls_back_to_overall_shares(self):
        from policyscope.policy import policy_matrix

        b, logs = self.stratified_logs()
        logs.loc[logs.index[0], "seg"] = 999
        ev = ps.Evaluator(logs, propensity=None, **KW, n_folds=3)
        with pytest.warns(UserWarning, match="одной строки"):
            probs = policy_matrix(ps.random_like(by="seg"), ev.data, ev._frame)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0)
        counts = pd.Series(logs["action"]).value_counts().reindex(ev.data.actions).to_numpy(dtype=float)
        counts[ev.data.a[0]] -= 1
        np.testing.assert_allclose(probs[0], counts / (len(logs) - 1))

    def test_logger_with_unlogged_actions_is_not_reused_for_a_candidate(self):
        b = make_bandit(300, seed=61)
        logs = b.logs.drop(columns="propensity")
        catalog = [*ACTIONS, "never"]
        ev = ps.Evaluator(logs, logging_policy=ps.uniform(catalog), **KW)
        with pytest.raises(ValueError, match="которых нет в логах"):
            ev.evaluate(ps.uniform(catalog), inference="none")

    def test_mixed_type_logger_labels(self):
        sim = ps.synthetic.OfferSimulator(seed=4)
        versions = sim.versions(2, drift=0.5, seed=1, names=[1, "manual"])
        clients = sim.sample_clients(2000)
        logs = pd.concat([sim.simulate(clients.iloc[:1000], versions[1]).assign(month=1),
                          sim.simulate(clients.iloc[1000:], versions["manual"]).assign(month="manual")],
                         ignore_index=True).drop(columns="propensity")
        logs["month"] = logs["month"].astype(object)
        rep = ps.evaluate(logs, versions[1], action="offer", reward="accepted", logging_policy=versions,
                          logger="month", inference="none")
        assert "logging_mixture" in {f.code for f in rep.findings}

    def test_one_rare_row_does_not_knock_boosting_out(self):
        from policyscope.data import prepare_logs
        from policyscope.nuisance import NuisanceConfig, fit_nuisance

        sim = ps.synthetic.OfferSimulator(seed=0)
        versions = sim.versions(4, drift=0.6, seed=3)
        clients = sim.sample_clients(4000)
        rows = np.array_split(np.arange(len(clients)), 4)
        logs = pd.concat([sim.simulate(clients.iloc[r], v) for v, r in zip(versions.values(), rows)],
                         ignore_index=True).drop(columns="propensity")
        logs.loc[3625, "offer"] = "manual"
        data = prepare_logs(logs, action="offer", reward="accepted",
                            features=["age", "income", "loyal", "risk", "region"])
        assert fit_nuisance(data, NuisanceConfig(), seed=0).propensity_kind == "boosting"

    def test_period_dependent_candidate_is_explained_not_blamed_on_the_logger(self):
        sim = ps.synthetic.OfferSimulator(seed=8)
        months = ["m1", "m2", "m3", "m4"]
        versions = sim.versions(4, drift=0.6, seed=2, names=months)
        logging = {m: ps.mixture([v, ps.uniform()], weights=[0.9, 0.1]) for m, v in versions.items()}
        clients = sim.sample_clients(8000)
        rows = np.array_split(np.arange(len(clients)), 4)
        logs = pd.concat([sim.simulate(clients.iloc[r], (0.9 * versions[m].action_probs(clients.iloc[r]) + 0.025))
                          .assign(month=m) for m, r in zip(months, rows)], ignore_index=True).drop(columns="propensity")
        ev = ps.Evaluator(logs, action="offer", reward="accepted", logging_policy=logging, logger="month")
        rep = ev.evaluate(ps.random_like(by="month"), inference="influence")
        codes = {f.code for f in rep.findings}
        assert "period_dependent_policy" in codes and "weights_inflated" not in codes
        assert rep.overlap.expected_weight != pytest.approx(1.0 - rep.overlap.uncovered, abs=1e-3)
        plain = ev.evaluate(ps.random_like(), inference="influence")
        assert "period_dependent_policy" not in {f.code for f in plain.findings}

    def test_declared_policy_note_does_not_deny_wrong_propensities(self):
        sim = ps.synthetic.OfferSimulator(seed=21)
        logs = sim.simulate(sim.sample_clients(8000), sim.policy("softmax", seed=2, temperature=0.6))
        rep = ps.evaluate(logs.drop(columns="propensity"), sim.policy("greedy", seed=2), action="offer",
                          reward="accepted", logging_policy=sim.policy("softmax", seed=2, temperature=1.0),
                          inference="influence")
        texts = " ".join(f.detail + f.advice for f in rep["accepted"].findings)
        assert "а не что пропенсити неверны" not in texts
        if "propensity_check_failed" in {f.code for f in rep["accepted"].findings}:
            assert "logging_policy" in texts


class TestThirdReview:
    def test_uncovered_decisions_do_not_blame_the_logging_policy(self):
        b = make_bandit(4000, seed=71)
        x = b.logs
        scores = pd.DataFrame({"red": x["x1"], "green": -x["x1"], "blue": x["x2"]}, index=x.index)
        greedy_a = ps.greedy(scores)
        logs = x.assign(action=greedy_a.idxmax(axis=1)).drop(columns="propensity")
        flip = np.random.default_rng(0).random(len(logs)) < 0.04
        candidate = greedy_a.copy()
        candidate.loc[flip] = 1 / 3
        rep = ps.evaluate(logs, candidate, logging_policy=greedy_a, inference="influence", **KW)
        texts = " ".join(f.detail + " " + f.advice for f in rep["reward"].findings)
        assert 0 < rep.overlap.uncovered < 0.05
        if "propensity_check_failed" in {f.code for f in rep["reward"].findings}:
            assert "выбрать не могла" in texts and "Проверьте logging_policy" not in texts

    def test_period_dependent_advice_is_complete(self):
        sim = ps.synthetic.OfferSimulator(seed=8)
        months = ["m1", "m2", "m3", "m4"]
        versions = sim.versions(4, drift=0.6, seed=2, names=months)
        clients = sim.sample_clients(6000)
        rows = np.array_split(np.arange(len(clients)), 4)
        logs = pd.concat([sim.simulate(clients.iloc[r], versions[m]).assign(month=m) for m, r in zip(months, rows)],
                         ignore_index=True).drop(columns="propensity")
        ev = ps.Evaluator(logs, action="offer", reward="accepted", logging_policy=versions, logger="month")
        rep = ev.evaluate(ps.random_like(by="month"), inference="none")
        texts = " ".join(f.advice for f in rep["accepted"].findings)
        assert "Если она верна" not in texts
        assert "ожидаемый при заданной политике" in rep.to_markdown()

    def test_self_normalized_off_support_note_uses_its_own_reference(self):
        b = make_bandit(6000, seed=72)
        x = b.logs
        scores = pd.DataFrame({"red": x["x1"], "green": -x["x1"], "blue": x["x2"]}, index=x.index)
        greedy_a = ps.greedy(scores)
        logs = x.assign(action=greedy_a.idxmax(axis=1)).drop(columns="propensity")
        flip = np.random.default_rng(1).random(len(logs)) < 0.12
        candidate = greedy_a.copy()
        candidate.loc[flip] = 1 / 3
        ev = ps.Evaluator(logs, logging_policy=greedy_a, **KW)
        from policyscope.policy import policy_matrix

        pi = policy_matrix(candidate, ev.data, ev._frame)
        dr_ref, sn_ref = ev._uncovered_mass(pi, None, "reward"), ev._uncovered_mass(pi, None, "reward", sn=True)
        assert np.isfinite(dr_ref) and np.isfinite(sn_ref) and dr_ref != pytest.approx(sn_ref)
        rep = ev.evaluate(candidate, estimator="sndr", inference="influence")
        texts = " ".join(f.detail + " " + f.advice for f in rep["reward"].findings)
        if "propensity_check_failed" in {f.code for f in rep["reward"].findings}:
            assert "SNIPS заменяет" in texts and "Проверьте logging_policy" not in texts
