"""Статистические свойства на многих симуляциях с известной истиной (маркер ``slow``).

Это главное подтверждение работоспособности: unit-тесты проверяют формулы, а здесь —
что оценки несмещены, интервалы покрывают истину с заявленной частотой, а проверки
согласия не поднимают ложных тревог и ловят настоящие проблемы.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import policyscope as ps
from policyscope.estimators import PolicyTerms, contributions
from policyscope.inference import analytic_summary, multiplier_tstats, romano_wolf
from policyscope.synthetic import OfferSimulator

from conftest import ACTIONS, make_bandit

pytestmark = pytest.mark.slow

KW = dict(action="action", reward="reward", features=["x1", "x2"])


def mc_bias_ok(errors, z=3.0):
    errors = np.asarray(errors)
    return abs(errors.mean()) < z * errors.std(ddof=1) / np.sqrt(len(errors))


def test_ips_dr_unbiased_with_logged_propensity():
    errors = {"ips": [], "dr": []}
    for seed in range(40):
        b = make_bandit(2000, seed=100 + seed)
        rep = ps.evaluate(b.logs, b.policy_b, propensity="propensity", inference="none", **KW)
        table = rep["reward"].estimators.set_index("estimator")
        for est in errors:
            errors[est].append(table.loc[est, "delta"] - b.delta)
    for est, e in errors.items():
        assert mc_bias_ok(e), f"{est}: смещение {np.mean(e):.4f}"


def test_error_shrinks_with_sample_size():
    def mean_abs_error(n):
        out = []
        for seed in range(10):
            b = make_bandit(n, seed=500 + seed)
            rep = ps.evaluate(b.logs, b.policy_b, propensity="propensity", inference="none", **KW)
            out.append(abs(rep["reward"].delta.value - b.delta))
        return np.mean(out)

    assert mean_abs_error(20000) < mean_abs_error(500) / 2


class TestDoubleRobustness:
    """DR несмещён, если верна хотя бы одна из моделей; негативный контроль — обе неверны."""

    @staticmethod
    def dr_error(seed, *, good_propensity, good_outcome):
        b = make_bandit(3000, seed=seed)
        rows = np.arange(3000)
        codes = b.logs["action"].map({label: i for i, label in enumerate(ACTIONS)}).to_numpy()
        distorted = b.probs_a ** 0.3
        pa = b.probs_a if good_propensity else distorted / distorted.sum(axis=1, keepdims=True)
        mu = b.mu if good_outcome else np.full_like(b.mu, 0.9)
        pi = b.policy_b.to_numpy()
        terms = PolicyTerms(w=pi[rows, codes] / pa[rows, codes],
                            dm=np.sum(pi * mu, axis=1), mu_taken=mu[rows, codes])
        r = b.logs["reward"].to_numpy(dtype=float)
        return contributions("dr", r, terms).mean() - b.value_b

    @pytest.mark.parametrize("good_propensity,good_outcome", [(True, False), (False, True)])
    def test_one_correct_model_is_enough(self, good_propensity, good_outcome):
        errors = [self.dr_error(s, good_propensity=good_propensity, good_outcome=good_outcome)
                  for s in range(200, 225)]
        assert mc_bias_ok(errors)

    def test_both_wrong_is_biased(self):
        errors = [self.dr_error(s, good_propensity=False, good_outcome=False) for s in range(300, 325)]
        assert not mc_bias_ok(errors)


def test_coverage_with_logged_propensity():
    covered = {"dr": [], "sndr": [], "ips": []}
    for seed in range(100):
        b = make_bandit(1500, seed=1000 + seed)
        rep = ps.Evaluator(b.logs, propensity="propensity", **KW)
        for est in covered:
            m = rep.evaluate(b.policy_b, estimator=est)["reward"]
            covered[est].append(m.delta.low <= b.delta <= m.delta.high)
    for est, c in covered.items():
        assert np.mean(c) >= 0.89, f"{est}: покрытие {np.mean(c):.2f}"


def test_cluster_robust_intervals_account_for_repeat_visits():
    """С повторными визитами строки клиента зависимы: без кластеризации интервал узок.

    Для V(A) и V(B) эффект клиента входит в каждую его строку. Для Δ при повторной
    рандомизации действия на каждом визите он почти сокращается в парной разности
    (E[w - 1 | x] = 0), поэтому проверяется и покрытие Δ, и ширина интервалов для V.
    """
    covered_delta, covered_b, ratio_a, ratio_b = [], [], [], []
    for seed in range(40):
        sim = OfferSimulator(seed=seed, client_effect=1.5)
        clients = sim.sample_clients(600, visits=5)
        a, b = sim.policy("epsilon_greedy", epsilon=0.3, seed=1), sim.policy("softmax", seed=2)
        logs = sim.simulate(clients, a)
        v_a, v_b = sim.true_value(logs, a), sim.true_value(logs, b)
        kw = dict(action="offer", reward="accepted", propensity="propensity",
                  features=["age", "income", "loyal", "risk", "region"])
        clustered = ps.evaluate(logs, b, cluster="client_id", **kw)["accepted"]
        naive = ps.evaluate(logs, b, **kw)["accepted"]
        covered_delta.append(clustered.delta.low <= v_b - v_a <= clustered.delta.high)
        covered_b.append(clustered.v_b.low <= v_b <= clustered.v_b.high)
        width = lambda e: e.high - e.low  # noqa: E731
        ratio_a.append(width(clustered.v_a) / width(naive.v_a))
        ratio_b.append(width(clustered.v_b) / width(naive.v_b))
    assert np.mean(covered_delta) >= 0.85
    assert np.mean(covered_b) >= 0.85
    assert np.mean(ratio_a) > 1.3 and np.mean(ratio_b) > 1.15


def test_estimated_propensity_is_no_longer_catastrophic():
    """Регрессия 0.2: оценённые пропенсити давали RMSE(DR) в сотни раз хуже логированных."""
    err_logged, err_est = [], []
    for seed in range(20):
        sim = OfferSimulator(seed=seed)
        clients = sim.sample_clients(3000)
        a, b = sim.policy("epsilon_greedy", epsilon=0.2, seed=1), sim.policy("softmax", seed=2)
        logs = sim.simulate(clients, a)
        truth = sim.true_value(logs, b) - sim.true_value(logs, a)
        kw = dict(action="offer", reward="accepted", inference="none",
                  features=["age", "income", "loyal", "risk", "region"])
        err_logged.append(ps.evaluate(logs, b, propensity="propensity", **kw)["accepted"].delta.value - truth)
        err_est.append(ps.evaluate(logs, b, **kw)["accepted"].delta.value - truth)
    rmse = lambda e: float(np.sqrt(np.mean(np.square(e))))  # noqa: E731
    assert rmse(err_est) < 2.5 * rmse(err_logged) + 0.01


def test_bootstrap_covers_with_estimated_propensity():
    covered = []
    for seed in range(12):
        sim = OfferSimulator(seed=50 + seed)
        clients = sim.sample_clients(1500)
        a, b = sim.policy("softmax", temperature=0.5, seed=3), sim.policy("softmax", seed=4)
        logs = sim.simulate(clients, a)
        truth = sim.true_value(logs, b) - sim.true_value(logs, a)
        m = ps.evaluate(logs, b, action="offer", reward="accepted", n_boot=60,
                        features=["age", "income", "loyal", "risk", "region"])["accepted"]
        covered.append(m.delta.low <= truth <= m.delta.high)
    assert np.mean(covered) >= 0.75


def test_checks_rarely_false_alarm_and_catch_wrong_propensities():
    false_alarms, detected = 0, 0
    for seed in range(30):
        b = make_bandit(4000, seed=2000 + seed)
        m = ps.evaluate(b.logs, b.policy_b, propensity="propensity", **KW)["reward"]
        false_alarms += any(c.status == "fail" for c in m.checks if c.name == "propensity")
    for seed in range(10):
        b = make_bandit(8000, seed=3000 + seed)
        distorted = b.logs["propensity"] ** 3
        logs = b.logs.assign(propensity=distorted / distorted.max())
        m = ps.evaluate(logs, b.policy_b, propensity="propensity", **KW)["reward"]
        detected += any(c.status == "fail" for c in m.checks if c.name == "propensity")
    assert false_alarms <= 2
    assert detected >= 8


def test_romano_wolf_controls_familywise_error():
    """При верных нулевых гипотезах доля прогонов хотя бы с одним «значимо» ≤ alpha."""
    rng = np.random.default_rng(0)
    rejections = 0
    n, j, sims = 400, 4, 300
    cov = 0.6 * np.ones((j, j)) + 0.4 * np.eye(j)
    for _ in range(sims):
        psi = rng.multivariate_normal(np.zeros(j), cov, size=n)
        s = analytic_summary(psi, np.arange(n), n, 0.05)
        t = s["estimate"] / s["se"]
        null = multiplier_tstats(s["sums"], s["se"], n, 1000, rng)
        rejections += bool((romano_wolf(t, null) < 0.05).any())
    assert rejections / sims <= 0.05 + 2 * np.sqrt(0.05 * 0.95 / sims)


FEATURES = ["age", "income", "loyal", "risk", "region"]


def test_dm_is_more_precise_than_dr_when_outcome_model_is_close():
    """Тезис «DM мощнее»: при близкой к верной модели исхода разброс DM заметно меньше."""
    errors = {"dm": [], "dr": []}
    for seed in range(40):
        sim = OfferSimulator(seed=600 + seed)
        a, b = sim.policy("epsilon_greedy", epsilon=0.2, seed=1), sim.policy("softmax", seed=2)
        logs = sim.simulate(sim.sample_clients(3000), a)
        truth = sim.true_value(logs, b) - sim.true_value(logs, a)
        rep = ps.evaluate(logs, b, action="offer", reward="accepted", propensity="propensity",
                          features=FEATURES, inference="none")
        table = rep["accepted"].estimators.set_index("estimator")["delta"]
        for est in errors:
            errors[est].append(table[est] - truth)
    sd = {est: np.std(e, ddof=1) for est, e in errors.items()}
    assert sd["dm"] < 0.75 * sd["dr"]
    assert mc_bias_ok(errors["dm"]) and mc_bias_ok(errors["dr"])


def test_dm_opt_in_bootstrap_interval_covers():
    """Основной DM со встроенной моделью: bootstrap-интервал покрывает истину.

    Регрессия на два дефекта 0.3: наивная формула дисперсии DM и обрезка/линейность
    модели выручки, смещавшая DM на несколько его стандартных ошибок.
    """
    covered = {"accepted": [], "revenue": []}
    for seed in range(20):
        sim = OfferSimulator(seed=700 + seed)
        a, b = sim.policy("epsilon_greedy", epsilon=0.2, seed=3), sim.policy("softmax", seed=5)
        logs = sim.simulate(sim.sample_clients(2000), a)
        rep = ps.evaluate(logs, b, action="offer", reward=list(covered), propensity="propensity",
                          features=FEATURES, estimator="dm", n_boot=80, random_state=seed)
        for metric, c in covered.items():
            truth = sim.true_value(logs, b, metric) - sim.true_value(logs, a, metric)
            d = rep[metric].delta
            c.append(d.low <= truth <= d.high)
    for metric, c in covered.items():
        assert np.mean(c) >= 0.8, f"{metric}: покрытие {np.mean(c):.2f}"


def test_simultaneous_intervals_and_best_set_cover():
    """Одновременные ДИ накрывают все Δ сразу, а набор лучших — истинно лучшего кандидата."""
    all_cover, marginal_all_cover, best_in_set, reps = 0, 0, 0, 200
    for seed in range(reps):
        b = make_bandit(2000, seed=300 + seed)
        oracle = ps.greedy(pd.DataFrame(b.mu, index=b.logs.index, columns=ACTIONS))
        candidates = {
            "oracle": oracle,
            "oracle_soft": 0.85 * oracle + 0.05,
            "b": b.policy_b,
            "uniform": pd.DataFrame(1 / 3, index=b.logs.index, columns=ACTIONS),
        }
        truth = {name: float(np.mean(np.sum(p[ACTIONS].to_numpy() * b.mu, axis=1))) - b.value_a
                 for name, p in candidates.items()}
        board = ps.Evaluator(b.logs, propensity="propensity", **KW).compare(candidates)
        t = board.table.set_index("candidate")
        all_cover += all(t.loc[k, "delta_low_sim"] <= v <= t.loc[k, "delta_high_sim"] for k, v in truth.items())
        marginal_all_cover += all(t.loc[k, "delta_low"] <= v <= t.loc[k, "delta_high"] for k, v in truth.items())
        best_in_set += max(truth, key=truth.get) in board.best().best_set
    assert all_cover / reps >= 0.91, all_cover / reps
    assert marginal_all_cover <= all_cover
    assert best_in_set / reps >= 0.92, best_in_set / reps


def test_mixture_of_deterministic_loggers_is_unbiased():
    """Две детерминированные модели по периодам: смесь пропенсити даёт честную оценку."""
    sim = OfferSimulator(seed=11)
    v1, v2, v3 = (sim.policy("greedy", seed=s) for s in (1, 3, 5))
    errors, covered, reps = [], 0, 100
    for _ in range(reps):
        clients = sim.sample_clients(4000)
        half = len(clients) // 2
        logs = pd.concat([
            sim.simulate(clients.iloc[:half], v1).assign(period="v1"),
            sim.simulate(clients.iloc[half:], v2).assign(period="v2"),
        ], ignore_index=True)
        # Кандидат выбирает между офферами двух моделей — только там, где у смеси есть данные.
        allowed = (v1.action_probs(logs) + v2.action_probs(logs)) > 0
        target = ps.greedy(v3.scores(logs).where(allowed, -np.inf).clip(lower=-1e9))
        rep = ps.evaluate(logs.drop(columns="propensity"), target, action="offer", reward="accepted",
                          features=["age", "income", "loyal", "risk", "region"],
                          logging_policy={"v1": v1, "v2": v2}, logger="period")
        m = rep["accepted"]
        # V(A) — ожидаемый отклик логирующих политик на своих периодах (не выборочное среднее).
        truth_a = (sim.true_value(logs.iloc[:half], v1, "accepted") * half
                   + sim.true_value(logs.iloc[half:], v2, "accepted") * (len(logs) - half)) / len(logs)
        truth = sim.true_value(logs, target, "accepted") - truth_a
        errors.append(m.delta.value - truth)
        covered += m.delta.low <= truth <= m.delta.high
    assert mc_bias_ok(errors), np.mean(errors)
    assert covered / reps >= 0.88, covered / reps


def _version_logs(seed, n=6000, k=4):
    sim = OfferSimulator(seed=seed)
    versions = sim.versions(k, drift=0.4, seed=1)
    clients = sim.sample_clients(n)
    rows = np.array_split(np.arange(n), k)
    logs = pd.concat([sim.simulate(clients.iloc[r], versions[m]).assign(month=m) for m, r in zip(versions, rows)],
                     ignore_index=True).drop(columns="propensity")
    truth_a = np.mean([sim.true_value(logs.iloc[r], versions[m], "accepted") for m, r in zip(versions, rows)])
    return sim, versions, logs, truth_a


def test_bounds_cover_the_truth():
    """Интервал Imbens–Manski накрывает истинное Δ; для покрытой политики — номинально, как DR."""
    F = ["age", "income", "loyal", "risk", "region"]
    reps, random_cover, version_cover, inside = 80, 0, 0, 0
    for seed in range(reps):
        sim, versions, logs, truth_a = _version_logs(500 + seed)
        ev = ps.Evaluator(logs, action="offer", reward="accepted", features=F,
                          logging_policy=versions, logger="month")
        uniform = pd.DataFrame(0.25, index=logs.index, columns=sim.actions)
        truth_random = sim.true_value(logs, uniform, "accepted") - truth_a
        b = ev.bounds(ps.uniform())["accepted"]
        random_cover += b.ci_low <= truth_random <= b.ci_high
        inside += b.delta_low <= truth_random <= b.delta_high
        truth_v2 = sim.true_value(logs, versions["v2"], "accepted") - truth_a
        bv = ev.bounds(versions["v2"])["accepted"]
        version_cover += bv.ci_low <= truth_v2 <= bv.ci_high
    assert random_cover / reps >= 0.95, random_cover / reps
    assert inside / reps >= 0.9, inside / reps
    assert version_cover / reps >= 0.9, version_cover / reps


def _wrapper_logs(seed, n, informed):
    sim = OfferSimulator(seed=seed, client_effect=1.5)
    both = sim.sample_clients(n, visits=2)
    first, second = both.iloc[::2].reset_index(drop=True), both.iloc[1::2].reset_index(drop=True)
    prev = sim.simulate(first, sim.policy("softmax", temperature=2.0, seed=1))
    rec = sim.policy("greedy", seed=1).action_probs(second).to_numpy()
    u = sim.hidden_trait(second)
    d = 1 / (1 + np.exp(-(1.5 * u - 0.5))) if informed else np.full(n, 0.4)
    probs = (1 - d)[:, None] * rec + d[:, None] * (1 - rec) / 3
    logs = sim.simulate(second, pd.DataFrame(probs, columns=sim.actions, index=second.index))
    logs["prev_accepted"] = prev["accepted"].to_numpy()
    return logs.drop(columns="propensity")


def test_placebo_rarely_false_alarms_and_catches_informed_deviations():
    """Случайные отклонения «обвязки» проходят плацебо, отклонения по скрытой склонности — нет."""
    F = ["age", "income", "loyal", "risk", "region"]
    reps, rates = 40, {}
    for informed in (False, True):
        fails = 0
        for seed in range(reps):
            logs = _wrapper_logs(700 + seed, 5000, informed)
            rep = ps.evaluate(logs, ps.uniform(), action="offer", reward="accepted", features=F,
                              placebo="prev_accepted", inference="influence")
            fails += rep.placebo[0].status == "fail"
        rates[informed] = fails / reps
    assert rates[False] <= 0.1, rates
    assert rates[True] >= 0.8, rates


def _stochastic_version_logs(seed, n=6000, temperature=1.0):
    """Три стохастические версии по периодам; прошлый отклик клиента зависит от сезона периода."""
    from dataclasses import replace

    sim = OfferSimulator(seed=seed, client_effect=1.5)
    versions = {name: replace(v, temperature=temperature)
                for name, v in sim.versions(3, drift=0.8, seed=seed, kind="softmax").items()}
    both = sim.sample_clients(n, visits=2)
    first, second = both.iloc[::2].reset_index(drop=True), both.iloc[1::2].reset_index(drop=True)
    parts = []
    for (name, v), r, shift in zip(versions.items(), np.array_split(np.arange(n), 3), (-1.0, 0.0, 1.0)):
        prev = sim.simulate(first.iloc[r], v, shift=shift)
        cur = sim.simulate(second.iloc[r], v).assign(period=name)
        cur["prev_accepted"] = prev["accepted"].to_numpy()
        parts.append(cur)
    return versions, pd.concat(parts, ignore_index=True).drop(columns="propensity")


@pytest.mark.parametrize("temperature", [1.0, 0.1])
def test_placebo_is_centred_under_version_mixtures(temperature):
    """Смесь версий: в отдельном периоде E[w | x] ≠ 1, но точное центрирование убирает ложные тревоги.

    При низкой температуре версии дают части действий почти нулевую вероятность; без периода
    в прогнозе плацебо проверка срабатывала почти в половине прогонов.
    """
    F = ["age", "income", "loyal", "risk", "region"]
    estimates, fails, reps = [], 0, 20
    for seed in range(reps):
        versions, logs = _stochastic_version_logs(900 + seed, temperature=temperature)
        rep = ps.evaluate(logs, ps.uniform(), action="offer", reward="accepted", features=F,
                          logging_policy=versions, logger="period", placebo="prev_accepted",
                          inference="influence")
        estimates.append(rep.placebo[0].estimate)
        fails += rep.placebo[0].status == "fail"
    assert fails / reps <= 0.1, fails / reps
    if temperature == 1.0:  # при почти нулевых пропенсити разброс огромен, и среднее неинформативно
        assert mc_bias_ok(estimates), np.mean(estimates)
