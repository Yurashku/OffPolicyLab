"""Формулы оценщиков: точные значения, тождества, линеаризация."""

from __future__ import annotations

import numpy as np
import pytest

from policyscope.estimators import ESTIMATORS, PolicyTerms, contributions, ess, policy_terms

# Ручной пример: все значения посчитаны на бумаге.
R = np.array([1.0, 0.0, 1.0, 0.0])
W = np.array([2.0, 0.0, 1.0, 3.0])
DM = np.array([0.5, 0.5, 0.5, 0.5])
MU = np.array([0.4, 0.6, 0.2, 0.8])
T = PolicyTerms(w=W, dm=DM, mu_taken=MU)


def value(name, r=R, t=T):
    return float(np.mean(contributions(name, r, t)))


class TestExactValues:
    def test_ips(self):
        assert value("ips") == pytest.approx(0.75)  # mean([2, 0, 1, 0])

    def test_snips(self):
        assert value("snips") == pytest.approx(0.5)  # 3 / 6

    def test_dm(self):
        assert value("dm") == pytest.approx(0.5)

    def test_dr(self):
        assert value("dr") == pytest.approx(0.4)  # mean([1.7, 0.5, 1.3, -1.9])

    def test_sndr(self):
        assert value("sndr") == pytest.approx(0.5 - 0.1 / 1.5)

    def test_unknown_estimator(self):
        with pytest.raises(ValueError, match="Неизвестный оценщик"):
            contributions("switch_dr", R, T)


class TestIdentities:
    def test_snips_is_mean_reward_when_weights_are_one(self):
        t = PolicyTerms(w=np.ones(4), dm=DM, mu_taken=MU)
        assert value("snips", t=t) == pytest.approx(R.mean())

    def test_dr_is_dm_when_outcome_model_is_exact(self):
        t = PolicyTerms(w=W, dm=DM, mu_taken=R)
        assert value("dr", t=t) == pytest.approx(DM.mean())

    def test_dr_is_ips_when_outcome_model_is_zero(self):
        t = PolicyTerms(w=W, dm=np.zeros(4), mu_taken=np.zeros(4))
        assert value("dr", t=t) == pytest.approx(value("ips"))

    def test_self_normalized_without_overlap_is_nan(self):
        t = PolicyTerms(w=np.zeros(4), dm=DM, mu_taken=MU)
        assert np.isnan(value("snips", t=t)) and np.isnan(value("sndr", t=t))

    @pytest.mark.parametrize("seed", range(10))
    def test_mean_of_contributions_matches_direct_formula(self, seed):
        rng = np.random.default_rng(seed)
        n = int(rng.integers(5, 200))
        r, w, dm, mu = rng.normal(size=n), rng.exponential(size=n), rng.normal(size=n), rng.normal(size=n)
        t = PolicyTerms(w=w, dm=dm, mu_taken=mu)
        direct = {
            "ips": np.mean(w * r),
            "snips": np.sum(w * r) / np.sum(w),
            "dm": np.mean(dm),
            "dr": np.mean(dm + w * (r - mu)),
            "sndr": np.mean(dm) + np.sum(w * (r - mu)) / np.sum(w),
        }
        for name in ESTIMATORS:
            assert value(name, r, t) == pytest.approx(direct[name], rel=1e-10), name


def test_self_normalized_linearization_matches_jackknife():
    """Разброс линеаризованных вкладов SNIPS — это его влияние на оценку.

    Если линеаризация неверна, аналитическая дисперсия SNIPS/SNDR будет
    систематически неправильной; jackknife — независимая проверка.
    """
    rng = np.random.default_rng(0)
    n = 2000
    r, w = rng.normal(size=n), rng.exponential(size=n)
    t = PolicyTerms(w=w, dm=np.zeros(n), mu_taken=np.zeros(n))
    v = np.sum(w * r) / np.sum(w)
    psi = contributions("snips", r, t)
    loo = np.array([(np.sum(w * r) - w[i] * r[i]) / (np.sum(w) - w[i]) for i in range(200)])
    empirical = -(loo - v) * (n - 1)
    assert np.corrcoef(empirical, (psi - v)[:200])[0, 1] == pytest.approx(1.0, abs=1e-6)


def test_policy_terms_shapes_and_values():
    pi = np.array([[0.5, 0.5], [1.0, 0.0]])
    mu = np.array([[0.2, 0.4], [0.6, 0.8]])
    t = policy_terms(pi, np.array([1, 0]), np.array([0.25, 0.5]), mu)
    np.testing.assert_allclose(t.w, [2.0, 2.0])
    np.testing.assert_allclose(t.dm, [0.3, 0.6])
    np.testing.assert_allclose(t.mu_taken, [0.4, 0.6])


class TestESS:
    def test_uniform_weights_give_full_sample(self):
        assert ess(np.ones(100)) == pytest.approx(100)

    def test_one_dominant_weight_collapses(self):
        w = np.r_[np.full(99, 1e-6), 1.0]
        assert ess(w) == pytest.approx(1.0, rel=1e-3)

    def test_degenerate(self):
        assert ess(np.zeros(5)) == 0.0
