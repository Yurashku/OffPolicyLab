"""Инференс: кластерно-робастная SE, bootstrap, Romano–Wolf."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from policyscope.inference import (
    analytic_summary,
    bootstrap_positions,
    centered_pvalue,
    cluster_members,
    cluster_se,
    cluster_sums,
    multiplier_tstats,
    romano_wolf,
    run_bootstrap,
    t_critical,
)


class TestAnalytic:
    def test_unclustered_se_is_sample_sd_over_sqrt_n(self):
        x = np.random.default_rng(0).normal(size=500)
        se = cluster_se(cluster_sums(x, np.arange(500), 500), 500)[0]
        assert se == pytest.approx(np.std(x, ddof=1) / np.sqrt(500))

    def test_clustered_se_matches_manual_cr1(self):
        rng = np.random.default_rng(1)
        clusters = rng.integers(0, 30, 600)
        x = rng.normal(size=600) + rng.normal(size=30)[clusters]
        s = (x - x.mean())
        sums = np.array([s[clusters == g].sum() for g in range(30)])
        expected = np.sqrt(30 / 29 * np.sum(sums**2)) / 600
        assert cluster_se(cluster_sums(x, clusters, 30), 600)[0] == pytest.approx(expected)

    def test_t_critical_uses_cluster_df(self):
        assert t_critical(0.05, 11) == pytest.approx(stats.t.ppf(0.975, 10))

    def test_ci_and_p_value_agree(self):
        rng = np.random.default_rng(2)
        for shift in (0.0, 0.05, 0.2):
            x = rng.normal(shift, 1, 400)
            s = analytic_summary(x, np.arange(400), 400, 0.05)
            excludes = s["low"][0] > 0 or s["high"][0] < 0
            assert excludes == (s["p_value"][0] < 0.05)


class TestRomanoWolf:
    def null(self, rho, j, b=4000, seed=0):
        rng = np.random.default_rng(seed)
        cov = np.full((j, j), rho) + (1 - rho) * np.eye(j)
        return rng.multivariate_normal(np.zeros(j), cov, size=b)

    def test_single_hypothesis_equals_bootstrap_p(self):
        t_null = self.null(0, 1)
        p = romano_wolf(np.array([2.0]), t_null)[0]
        assert p == pytest.approx(2 * stats.norm.sf(2.0), abs=0.01)

    def test_perfectly_correlated_tests_need_no_correction(self):
        t_null = self.null(0.999, 4)
        p = romano_wolf(np.array([2.5, 0.5, 0.3, 0.1]), t_null)
        assert p[0] == pytest.approx(2 * stats.norm.sf(2.5), abs=0.01)

    def test_independent_tests_close_to_sidak(self):
        t_null = self.null(0.0, 5, b=20000)
        p = romano_wolf(np.array([2.8, 0.1, 0.2, 0.3, 0.4]), t_null)
        raw = 2 * stats.norm.sf(2.8)
        assert p[0] == pytest.approx(1 - (1 - raw) ** 5, abs=0.01)

    def test_monotone_in_statistic_and_not_below_raw(self):
        t_obs = np.array([3.0, 2.5, 1.0, 0.2])
        p = romano_wolf(t_obs, self.null(0.3, 4))
        assert np.all(np.diff(p) >= 0)
        assert np.all(p >= 2 * stats.norm.sf(np.abs(t_obs)) - 0.01)

    def test_nan_statistics_are_ignored(self):
        p = romano_wolf(np.array([np.nan, 2.0]), self.null(0, 2))
        assert np.isnan(p[0]) and np.isfinite(p[1])

    def test_multiplier_preserves_correlation(self):
        rng = np.random.default_rng(3)
        z = rng.normal(size=1000)
        psi = np.column_stack([z, z + 0.1 * rng.normal(size=1000)])
        sums = cluster_sums(psi, np.arange(1000), 1000)
        se = cluster_se(sums, 1000)
        t = multiplier_tstats(sums, se, 1000, 3000, rng)
        assert np.std(t, axis=0) == pytest.approx([1, 1], abs=0.05)
        assert np.corrcoef(t.T)[0, 1] > 0.98


class TestBootstrap:
    def test_cluster_resampling_keeps_whole_clusters(self):
        clusters = np.repeat(np.arange(10), [1, 2, 3, 4, 5, 1, 2, 3, 4, 5])
        members = cluster_members(clusters, 10)
        pos = bootstrap_positions(clusters, 10, np.random.default_rng(0), members)
        counts = np.bincount(clusters[pos], minlength=10)
        sizes = np.bincount(clusters, minlength=10)
        assert all(c % s == 0 for c, s in zip(counts, sizes))

    def test_reproducible_and_independent_of_n_jobs(self):
        x = np.random.default_rng(0).normal(size=200)

        def mean_rep(positions, seed):
            return np.array([x[positions].mean()])

        kw = dict(clusters=np.arange(200), n_clusters=200, clustered=False, n_boot=30, seed=5)
        a = run_bootstrap(mean_rep, n_jobs=1, **kw)
        b = run_bootstrap(mean_rep, n_jobs=2, **kw)
        np.testing.assert_allclose(a, b)

    def test_failed_replicates_become_nan(self):
        def flaky(positions, seed):
            if seed % 2:
                raise ValueError("вырожденная реплика")
            return np.array([1.0])

        out = run_bootstrap(flaky, clusters=np.arange(10), n_clusters=10, clustered=False,
                            n_boot=40, seed=0, n_jobs=1)
        assert np.isnan(out).any() and np.isfinite(out).any()

    def test_centered_p_value(self):
        samples = np.random.default_rng(1).normal(0.0, 1.0, size=(2000, 1))
        assert centered_pvalue(samples, np.array([0.0]))[0] == pytest.approx(1.0)
        # Реплики разбросаны вокруг оценки 3.5: ноль далеко в хвосте.
        assert centered_pvalue(samples + 3.5, np.array([3.5]))[0] < 0.01
