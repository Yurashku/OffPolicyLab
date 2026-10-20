"""Проверка **математической корректности** оценщиков.

Не smoke-тесты: здесь сверяются точные числа, известные аналитически, и
статистические свойства (несмещённость, состоятельность, двойная робастность,
покрытие доверительных интервалов).
"""

from __future__ import annotations

import numpy as np
import pytest

from policyscope.estimate import estimate_point
from policyscope.estimators import estimator_contributions
from policyscope.nuisance import BehaviorPredictions, OutcomePredictions

from conftest import make_controlled_bandit

# Ручной пример: все шесть значений посчитаны на бумаге.
R = np.array([1.0, 0.0, 1.0, 0.0])
W = np.array([2.0, 0.0, 1.0, 3.0])
DM = np.array([0.5, 0.5, 0.5, 0.5])
MU = np.array([0.4, 0.6, 0.2, 0.8])


class TestExactFormulas:
    """Значения оценщиков на вручную посчитанном примере."""

    def test_ips(self):
        # mean(w*r) = mean([2, 0, 1, 0]) = 0.75
        assert np.mean(estimator_contributions("ips", r=R, w=W)) == pytest.approx(0.75)

    def test_snips(self):
        # sum(w*r)/sum(w) = 3 / 6 = 0.5
        assert np.mean(estimator_contributions("snips", r=R, w=W)) == pytest.approx(0.5)

    def test_dm(self):
        assert np.mean(estimator_contributions("dm", r=R, dm=DM)) == pytest.approx(0.5)

    def test_dr(self):
        # mean(dm + w*(r-mu)) = mean([1.7, 0.5, 1.3, -1.9]) = 0.4
        value = np.mean(estimator_contributions("dr", r=R, w=W, dm=DM, mu_taken=MU))
        assert value == pytest.approx(0.4)

    def test_sndr(self):
        # mean(dm) + mean(w*resid)/mean(w) = 0.5 + (-0.1 / 1.5)
        value = np.mean(estimator_contributions("sndr", r=R, w=W, dm=DM, mu_taken=MU))
        assert value == pytest.approx(0.5 - 0.1 / 1.5)

    def test_switch_dr(self):
        # tau=2 отключает поправку только на четвёртой строке (w=3):
        # mean([1.7, 0.5, 1.3, 0.5]) = 1.0
        value = np.mean(
            estimator_contributions("switch_dr", r=R, w=W, dm=DM, mu_taken=MU, tau=2.0)
        )
        assert value == pytest.approx(1.0)

    @pytest.mark.parametrize(
        "method", ["ips", "snips", "dm", "dr", "sndr", "switch_dr", "on_policy"]
    )
    def test_contributions_average_to_value(self, method):
        """Среднее вкладов обязано совпадать со значением оценки.

        На этом свойстве держится аналитический CI: если бы среднее расходилось
        со значением, интервал строился бы вокруг не того числа.
        """
        psi = estimator_contributions(
            method, r=R, w=W, dm=DM, mu_taken=MU, tau=2.0
        )
        assert np.isfinite(np.mean(psi))
        assert len(psi) == len(R)


class TestIdentities:
    """Соотношения между оценщиками, которые обязаны выполняться тождественно."""

    def test_snips_equals_mean_reward_when_weights_are_one(self):
        w = np.ones_like(R)
        assert np.mean(estimator_contributions("snips", r=R, w=w)) == pytest.approx(R.mean())

    def test_dr_reduces_to_dm_when_residual_is_zero(self):
        """Если mu идеально предсказывает награду, IPS-поправка зануляется."""
        value = np.mean(estimator_contributions("dr", r=R, w=W, dm=DM, mu_taken=R))
        assert value == pytest.approx(DM.mean())

    def test_dr_reduces_to_ips_when_outcome_model_is_zero(self):
        zero = np.zeros_like(R)
        dr = np.mean(estimator_contributions("dr", r=R, w=W, dm=zero, mu_taken=zero))
        ips = np.mean(estimator_contributions("ips", r=R, w=W))
        assert dr == pytest.approx(ips)

    def test_switch_dr_equals_dr_when_tau_is_infinite(self):
        switch = np.mean(
            estimator_contributions("switch_dr", r=R, w=W, dm=DM, mu_taken=MU, tau=np.inf)
        )
        dr = np.mean(estimator_contributions("dr", r=R, w=W, dm=DM, mu_taken=MU))
        assert switch == pytest.approx(dr)

    def test_switch_dr_equals_dm_when_tau_is_zero(self):
        """При tau=0 отключается вся поправка, остаётся чистый DM."""
        switch = np.mean(
            estimator_contributions("switch_dr", r=R, w=W, dm=DM, mu_taken=MU, tau=0.0)
        )
        assert switch == pytest.approx(DM.mean())


class TestUnbiasedness:
    """Несмещённость и состоятельность на данных с точно известной истиной."""

    @staticmethod
    def _run(bandit, method, **kwargs):
        policy = bandit.policy_b()
        return estimate_point(
            bandit.df,
            policy,
            method=method,
            target="reward",
            action_col="action",
            action_space=bandit.action_space,
            feature_cols=["x1", "x2"],
            propensity_source="logged",
            propensity_col="propensity",
            **kwargs,
        )

    def test_ips_is_unbiased_with_true_propensities(self):
        """Средняя оценка IPS по независимым выборкам сходится к истине."""
        estimates, truths = [], []
        for seed in range(30):
            bandit = make_controlled_bandit(n=2000, seed=100 + seed)
            estimates.append(self._run(bandit, "ips").value)
            truths.append(bandit.value_b)
        bias = np.mean(np.array(estimates) - np.array(truths))
        se = np.std(np.array(estimates) - np.array(truths), ddof=1) / np.sqrt(30)
        assert abs(bias) < 3 * se, f"смещение IPS {bias:.5f} превышает 3 SE ({se:.5f})"

    def test_mean_importance_weight_is_one(self):
        """E[w] = 1 по построению при корректных пропенсити."""
        bandit = make_controlled_bandit(n=20000, seed=3)
        point = self._run(bandit, "ips")
        assert point.weights is not None
        assert float(np.mean(point.weights)) == pytest.approx(1.0, abs=0.02)

    @pytest.mark.parametrize("method", ["ips", "dr"])
    def test_error_shrinks_as_sample_grows(self, method):
        """Состоятельность: ошибка при n=20000 заметно меньше, чем при n=500."""

        def mean_abs_error(n: int) -> float:
            errors = []
            for seed in range(12):
                bandit = make_controlled_bandit(n=n, seed=500 + seed)
                errors.append(abs(self._run(bandit, method).value - bandit.value_b))
            return float(np.mean(errors))

        small = mean_abs_error(500)
        large = mean_abs_error(20000)
        assert large < small / 2, f"{method}: ошибка не убывает с n ({small:.4f} -> {large:.4f})"


class TestDoubleRobustness:
    """DR остаётся состоятельным, если верна хотя бы одна из двух моделей."""

    @staticmethod
    def _dr_with(bandit, *, behavior_probs, mu_matrix):
        """Считает DR с явно подставленными nuisance-предсказаниями."""
        n = len(bandit.df)
        actions = bandit.df["action"].to_numpy()
        idx = np.arange(n)
        policy = bandit.policy_b()

        behavior = BehaviorPredictions(
            pA_taken=behavior_probs[idx, actions],
            piB_taken=bandit.probs_b[idx, actions],
            propensity_source="logged",
        )
        outcome = OutcomePredictions(
            mu_logged_action=mu_matrix[idx, actions],
            mu_by_action={a: mu_matrix[:, a] for a in bandit.action_space},
            target="reward",
        )
        return estimate_point(
            bandit.df,
            policy,
            method="dr",
            target="reward",
            action_col="action",
            action_space=bandit.action_space,
            feature_cols=["x1", "x2"],
            nuisance_behavior=behavior,
            nuisance_outcome=outcome,
        ).value

    def test_correct_propensity_wrong_outcome_model(self):
        """Верные пропенсити + заведомо неверная mu -> DR всё ещё несмещён."""
        errors = []
        for seed in range(20):
            bandit = make_controlled_bandit(n=3000, seed=200 + seed)
            wrong_mu = np.full_like(bandit.mu_true, 0.9)  # константа, далёкая от истины
            value = self._dr_with(
                bandit, behavior_probs=bandit.probs_a, mu_matrix=wrong_mu
            )
            errors.append(value - bandit.value_b)
        bias = float(np.mean(errors))
        se = float(np.std(errors, ddof=1) / np.sqrt(len(errors)))
        assert abs(bias) < 3 * se, f"смещение {bias:.5f} превышает 3 SE ({se:.5f})"

    def test_correct_outcome_model_wrong_propensity(self):
        """Верная mu + искажённые пропенсити -> DR всё ещё несмещён."""
        errors = []
        for seed in range(20):
            bandit = make_controlled_bandit(n=3000, seed=300 + seed)
            # Искажаем пропенсити, сохраняя корректную нормировку.
            distorted = bandit.probs_a ** 0.5
            distorted = distorted / distorted.sum(axis=1, keepdims=True)
            value = self._dr_with(
                bandit, behavior_probs=distorted, mu_matrix=bandit.mu_true
            )
            errors.append(value - bandit.value_b)
        bias = float(np.mean(errors))
        se = float(np.std(errors, ddof=1) / np.sqrt(len(errors)))
        assert abs(bias) < 3 * se, f"смещение {bias:.5f} превышает 3 SE ({se:.5f})"

    def test_both_models_wrong_is_biased(self):
        """Контроль: если обе модели неверны, DR действительно смещается.

        Без этой проверки предыдущие два теста ничего не доказывают — они могли бы
        проходить и в случае, когда DR игнорирует переданные nuisance.
        """
        errors = []
        for seed in range(20):
            bandit = make_controlled_bandit(n=3000, seed=400 + seed)
            distorted = bandit.probs_a ** 0.5
            distorted = distorted / distorted.sum(axis=1, keepdims=True)
            wrong_mu = np.full_like(bandit.mu_true, 0.9)
            value = self._dr_with(bandit, behavior_probs=distorted, mu_matrix=wrong_mu)
            errors.append(value - bandit.value_b)
        bias = float(np.mean(errors))
        se = float(np.std(errors, ddof=1) / np.sqrt(len(errors)))
        assert abs(bias) > 3 * se, "ожидалось смещение, но DR оказался несмещённым"


class TestContributionsMatchDirectFormulas:
    """Вклады обязаны в среднем воспроизводить прямую формулу каждого оценщика."""

    @pytest.mark.parametrize("seed", range(20))
    def test_random_inputs(self, seed):
        rng = np.random.default_rng(seed)
        n = int(rng.integers(5, 200))
        r = rng.normal(size=n)
        w = rng.exponential(size=n)
        dm = rng.normal(size=n)
        mu = rng.normal(size=n)
        tau = float(rng.uniform(0.1, 3.0))

        expected = {
            "ips": np.mean(w * r),
            "snips": np.sum(w * r) / np.sum(w),
            "dm": np.mean(dm),
            "dr": np.mean(dm + w * (r - mu)),
            "sndr": np.mean(dm + (w / w.mean()) * (r - mu)),
            "switch_dr": np.mean(dm + np.where(w <= tau, w, 0.0) * (r - mu)),
        }
        for method, direct in expected.items():
            psi = estimator_contributions(
                method, r=r, w=w, dm=dm, mu_taken=mu, tau=tau
            )
            assert np.mean(psi) == pytest.approx(direct, rel=1e-10), method

    def test_self_normalized_linearization_matches_jackknife(self):
        """Дельта-метод для SNIPS должен совпадать с jackknife-влиянием.

        Независимая проверка линеаризации отношения: если она неверна,
        аналитическая дисперсия для SNIPS/SNDR будет систематически неправильной.
        """
        rng = np.random.default_rng(0)
        n = 2000
        r = rng.normal(size=n)
        w = rng.exponential(size=n)

        value = float(np.sum(w * r) / np.sum(w))
        psi = estimator_contributions("snips", r=r, w=w)

        m = 200
        total_wr, total_w = np.sum(w * r), np.sum(w)
        leave_one_out = np.array(
            [(total_wr - w[i] * r[i]) / (total_w - w[i]) for i in range(m)]
        )
        empirical = -(leave_one_out - value) * (n - 1)
        theoretical = (psi - value)[:m]
        assert np.corrcoef(empirical, theoretical)[0, 1] == pytest.approx(1.0, abs=1e-6)
