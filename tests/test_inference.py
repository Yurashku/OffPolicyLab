"""Проверка слоя инференса: ресэмплинг, интервалы, значимость, покрытие."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from policyscope.comparison import compare_policies
from policyscope.inference import (
    ClusterResampler,
    cluster_robust_se,
    infer_comparison_bootstrap,
    infer_comparison_influence,
)

from conftest import make_controlled_bandit


class TestClusterResampler:
    def test_row_bootstrap_preserves_size(self):
        df = pd.DataFrame({"x": np.arange(50)})
        r = ClusterResampler(df, None)
        part = r.resample(df, np.random.default_rng(0))
        assert len(part) == 50
        assert r.n_clusters is None

    def test_cluster_bootstrap_keeps_clusters_intact(self):
        """Кластер попадает в реплику целиком: это и есть смысл кластерного bootstrap."""
        df = pd.DataFrame({"user_id": [1, 1, 1, 2, 2, 3], "x": range(6)})
        r = ClusterResampler(df, "user_id")
        assert r.n_clusters == 3
        rng = np.random.default_rng(0)
        for _ in range(20):
            part = r.resample(df, rng)
            sizes = part.groupby("user_id").size()
            for user, size in sizes.items():
                original = int((df["user_id"] == user).sum())
                assert size % original == 0, "кластер попал в выборку частично"

    def test_missing_cluster_column_raises(self):
        """Молчаливый откат к построчному bootstrap недопустим: он занижает CI."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        with pytest.raises(ValueError, match="Колонка кластеров"):
            ClusterResampler(df, "user_id")

    def test_resampling_is_reproducible(self):
        df = pd.DataFrame({"user_id": [1, 1, 2, 2, 3, 3], "x": range(6)})
        r = ClusterResampler(df, "user_id")
        a = r.positions(np.random.default_rng(42))
        b = r.positions(np.random.default_rng(42))
        assert np.array_equal(a, b)


class TestClusterRobustSE:
    def test_reduces_to_standard_error_without_clusters(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        expected = float(np.std(x, ddof=0) / np.sqrt(len(x)))
        assert cluster_robust_se(x) == pytest.approx(expected)

    def test_perfectly_correlated_clusters_inflate_se(self):
        """Дублирование строк внутри кластера не должно «создавать» информацию."""
        base = np.array([1.0, 2.0, 3.0, 4.0])
        duplicated = np.repeat(base, 3)
        clusters = np.repeat(np.arange(4), 3)
        naive = cluster_robust_se(duplicated)
        clustered = cluster_robust_se(duplicated, clusters=clusters)
        assert clustered > naive
        # При полной корреляции внутри кластера SE равна SE по 4 независимым точкам.
        assert clustered == pytest.approx(cluster_robust_se(base), rel=1e-9)


class TestInfluenceInference:
    def test_delta_uses_paired_contributions(self):
        """Если вклад B равен награде, delta ровно ноль и SE ноль."""
        r = np.array([1.0, 0.0, 1.0, 1.0])
        out = infer_comparison_influence(r, r)
        assert out.delta == pytest.approx(0.0)
        assert out.se_delta == pytest.approx(0.0)

    def test_detects_obvious_difference(self):
        rng = np.random.default_rng(0)
        r = rng.normal(0, 1, size=2000)
        psi_b = r + 0.5
        out = infer_comparison_influence(psi_b, r)
        assert out.delta == pytest.approx(0.5, abs=1e-9)
        assert out.p_value < 1e-6
        assert out.is_significant
        assert out.ci_excludes_zero

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            infer_comparison_influence(np.zeros(3), np.zeros(4))


class TestBootstrapInference:
    def test_reuses_observed_point_estimate(self):
        """Переданная точечная оценка не должна пересчитываться."""
        df = pd.DataFrame({"user_id": range(60), "r": np.linspace(0, 1, 60)})
        calls = {"n": 0}

        def pair(part):
            calls["n"] += 1
            return float(part["r"].mean()), float(part["r"].mean()) + 0.1

        out = infer_comparison_bootstrap(
            df, pair, cluster_col="user_id", n_boot=20, observed=(0.5, 0.6)
        )
        assert out.v_a == 0.5 and out.v_b == 0.6
        assert calls["n"] == 20, "точечная оценка была посчитана лишний раз"

    def test_failed_replicates_are_dropped_not_propagated(self):
        """Одна упавшая реплика не должна превращать весь интервал в NaN."""
        df = pd.DataFrame({"user_id": range(60), "r": np.linspace(0, 1, 60)})
        state = {"n": 0}

        def pair(part):
            state["n"] += 1
            if state["n"] == 3:
                raise ValueError("смоделированный сбой реплики")
            return 0.5, 0.6

        out = infer_comparison_bootstrap(df, pair, cluster_col="user_id", n_boot=40)
        assert np.isfinite(out.delta_ci.low)
        assert "bootstrap_replicates_with_nan_dropped" in out.warnings

    def test_warns_on_few_clusters_and_low_n_boot(self):
        df = pd.DataFrame({"user_id": [1, 2, 3, 4], "r": [0.0, 1.0, 0.0, 1.0]})
        out = infer_comparison_bootstrap(
            df, lambda p: (0.5, 0.6), cluster_col="user_id", n_boot=10
        )
        assert "few_clusters_inference_unreliable" in out.warnings
        assert "low_n_boot_ci_tails_unstable" in out.warnings


class TestSignificanceConsistency:
    def test_ci_and_p_value_are_reported_separately(self):
        """Два правила значимости приводятся раздельно, а не смешиваются."""
        bandit = make_controlled_bandit(n=800, seed=5)
        policy = bandit.policy_b()
        result = compare_policies(
            bandit.df, policy, estimator="ips", target="reward", action_col="action",
            action_space=bandit.action_space, feature_cols=["x1", "x2"],
            propensity_col="propensity", propensity_source="logged",
            cluster_col=None, inference="influence",
        )
        assert result.is_significant is not None
        assert result.ci_excludes_zero is not None
        assert result.significance_rule


@pytest.mark.slow
class TestCoverage:
    """Фактическое покрытие 95% интервала должно быть близко к номинальному."""

    @pytest.mark.parametrize(
        "method, estimator, n_reps",
        [
            ("influence", "ips", 300),
            ("influence", "dr", 150),
            ("bootstrap", "ips", 150),
        ],
    )
    def test_delta_ci_covers_truth(self, method, estimator, n_reps):
        covered = 0
        for seed in range(n_reps):
            bandit = make_controlled_bandit(n=1500, seed=9000 + seed)
            result = compare_policies(
                bandit.df, bandit.policy_b(), estimator=estimator, target="reward",
                action_col="action", action_space=bandit.action_space,
                feature_cols=["x1", "x2"], propensity_col="propensity",
                propensity_source="logged", cluster_col=None,
                inference=method, n_boot=200, rng_seed=seed * 31 + 1,
                honest_outcome_metrics=False,
            )
            low, high = result.delta_ci
            covered += int(low <= bandit.true_delta <= high)
        coverage = covered / n_reps
        assert 0.88 <= coverage <= 1.0, (
            f"{method}/{estimator}: покрытие {coverage:.3f} вне [0.88, 1.0] "
            f"({covered}/{n_reps})"
        )

    def test_influence_ci_is_anticonservative_for_dm(self):
        """Задокументированное ограничение должно подтверждаться эмпирически.

        Для DM ошибка модели исхода входит в оценку первым порядком, а
        аналитический интервал её не учитывает. Если этот тест начнёт падать
        «в хорошую сторону», предупреждение в отчёте нужно пересмотреть.
        """
        n_reps = 100
        covered = 0
        for seed in range(n_reps):
            bandit = make_controlled_bandit(n=1500, seed=7000 + seed)
            result = compare_policies(
                bandit.df, bandit.policy_b(), estimator="dm", target="reward",
                action_col="action", action_space=bandit.action_space,
                feature_cols=["x1", "x2"], propensity_col="propensity",
                propensity_source="logged", cluster_col=None, inference="influence",
                honest_outcome_metrics=False,
            )
            low, high = result.delta_ci
            covered += int(low <= bandit.true_delta <= high)
        coverage = covered / n_reps
        assert coverage < 0.9, (
            f"покрытие DM {coverage:.3f} неожиданно высоко — проверьте, "
            "актуально ли предупреждение influence_ci_anticonservative_for_dm"
        )
