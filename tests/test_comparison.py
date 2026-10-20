"""Тесты оркестрации: разрешение контракта данных, режимы инференса, схема результата."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from policyscope.comparison import (
    compare_policies,
    compare_policies_multi_estimator,
    compare_policies_multi_target,
)
from policyscope.data import BanditSchema, LoggedBanditDataset
from policyscope.estimate import ALL_ESTIMATORS
from policyscope.evaluator import OPEEvaluator
from policyscope.nuisance import fit_crossfit_nuisance_bundle

from conftest import make_controlled_bandit


@pytest.fixture
def setup():
    bandit = make_controlled_bandit(n=500, seed=21)
    policy = bandit.policy_b()
    kwargs = dict(
        target="reward", action_col="action", action_space=bandit.action_space,
        feature_cols=["x1", "x2"], propensity_col="propensity", cluster_col=None,
    )
    return bandit, policy, kwargs


class TestResultSchema:
    def test_all_keys_present_regardless_of_inference_mode(self, setup):
        """Схема стабильна: ключи не появляются и не исчезают от режима."""
        bandit, policy, kwargs = setup
        with_ci = compare_policies(
            bandit.df, policy, inference="influence", **kwargs
        ).to_dict()
        without = compare_policies(bandit.df, policy, inference="none", **kwargs).to_dict()
        assert set(with_ci) == set(without)
        assert without["Delta_CI"] is None
        assert with_ci["Delta_CI"] is not None

    def test_delta_equals_v_b_minus_v_a(self, setup):
        bandit, policy, kwargs = setup
        r = compare_policies(bandit.df, policy, inference="none", **kwargs)
        assert r.delta == pytest.approx(r.v_b - r.v_a)

    def test_to_frame_is_single_row(self, setup):
        bandit, policy, kwargs = setup
        frame = compare_policies(bandit.df, policy, inference="none", **kwargs).to_frame()
        assert len(frame) == 1
        assert {"estimator", "V_A", "V_B", "Delta"} <= set(frame.columns)

    def test_result_is_json_serializable(self, setup, tmp_path):
        from policyscope.report import dump_json

        bandit, policy, kwargs = setup
        result = compare_policies(bandit.df, policy, inference="influence", **kwargs)
        path = tmp_path / "out.json"
        dump_json(str(path), result)
        assert path.read_text(encoding="utf-8").startswith("{")


class TestDataContractResolution:
    def test_explicit_target_overrides_schema(self):
        """Раньше схема безусловно перезаписывала target, и считалась не та метрика."""
        bandit = make_controlled_bandit(n=300, seed=22)
        df = bandit.df.copy()
        df["other_metric"] = df["reward"] * 10.0
        schema = BanditSchema(
            action_col="action", reward_col="reward",
            feature_cols=["x1", "x2"], propensity_col="propensity", cluster_col=None,
        )
        dataset = LoggedBanditDataset(df=df, schema=schema)
        policy = bandit.policy_b()

        result = compare_policies(
            dataset, policy, target="other_metric",
            action_space=bandit.action_space, inference="none",
        )
        assert result.target == "other_metric"
        assert result.v_a == pytest.approx(df["other_metric"].mean())

    def test_schema_used_when_argument_omitted(self):
        bandit = make_controlled_bandit(n=300, seed=23)
        schema = BanditSchema(
            action_col="action", reward_col="reward",
            feature_cols=["x1", "x2"], propensity_col="propensity", cluster_col=None,
        )
        dataset = LoggedBanditDataset(df=bandit.df, schema=schema)
        policy = bandit.policy_b()
        result = compare_policies(
            dataset, policy, action_space=bandit.action_space, inference="none"
        )
        assert result.target == "reward"

    def test_missing_cluster_column_raises(self, setup):
        bandit, policy, kwargs = setup
        kwargs = {**kwargs, "cluster_col": "no_such_column"}
        with pytest.raises(ValueError, match="Колонка кластеров"):
            compare_policies(bandit.df, policy, inference="none", **kwargs)


class TestPropensitySource:
    def test_logged_source_is_reported(self, setup):
        bandit, policy, kwargs = setup
        r = compare_policies(
            bandit.df, policy, propensity_source="logged", inference="none", **kwargs
        )
        assert r.propensity_source == "logged"
        assert r.propensity_column == "propensity"

    def test_estimated_source_ignores_logged_column(self, setup):
        bandit, policy, kwargs = setup
        r = compare_policies(
            bandit.df, policy, propensity_source="estimated", inference="none", **kwargs
        )
        assert r.propensity_source == "estimated"

    def test_logged_source_requires_column(self, setup):
        bandit, policy, kwargs = setup
        kwargs = {**kwargs, "propensity_col": None}
        with pytest.raises(ValueError, match="propensity_col"):
            compare_policies(
                bandit.df, policy, propensity_source="logged", inference="none", **kwargs
            )

    def test_auto_falls_back_to_estimated_without_column(self, setup):
        bandit, policy, kwargs = setup
        kwargs = {**kwargs, "propensity_col": None}
        r = compare_policies(bandit.df, policy, inference="none", **kwargs)
        assert r.propensity_source == "estimated"
        assert "logged_propensity_missing_fallback_to_estimated" in r.warnings


class TestNoSpuriousWarnings:
    def test_default_path_has_no_external_nuisance_warning(self, setup):
        """Дефолтный прогон не должен сообщать о переиспользовании внешних nuisance."""
        bandit, policy, kwargs = setup
        r = compare_policies(bandit.df, policy, inference="bootstrap", n_boot=20, **kwargs)
        assert not any("external_nuisance" in w for w in r.warnings)


class TestCrossfit:
    def test_crossfit_and_bundle_are_mutually_exclusive(self, setup):
        bandit, policy, kwargs = setup
        bundle = fit_crossfit_nuisance_bundle(
            bandit.df, policy, target="reward", feature_cols=["x1", "x2"],
            action_col="action", action_space=bandit.action_space, n_splits=3,
        )
        with pytest.raises(ValueError, match="либо"):
            compare_policies(
                bandit.df, policy, use_crossfit=True, nuisance_bundle=bundle,
                inference="none", **kwargs
            )

    def test_bundle_target_mismatch_is_rejected(self, setup):
        """Модель исхода обучена под одну метрику и не годится для другой."""
        bandit, policy, kwargs = setup
        bundle = fit_crossfit_nuisance_bundle(
            bandit.df, policy, target="reward", feature_cols=["x1", "x2"],
            action_col="action", action_space=bandit.action_space, n_splits=3,
        )
        df = bandit.df.copy()
        df["other"] = df["reward"] * 2.0
        kwargs = {**kwargs, "target": "other"}
        with pytest.raises(ValueError, match="обучен на цели"):
            compare_policies(
                df, policy, nuisance_bundle=bundle, inference="none", **kwargs
            )

    def test_crossfit_is_applied_inside_bootstrap(self, setup):
        """Точечная оценка и CI должны описывать один и тот же оценщик."""
        bandit, policy, kwargs = setup
        r = compare_policies(
            bandit.df, policy, use_crossfit=True, crossfit_n_splits=3,
            inference="bootstrap", n_boot=10, **kwargs
        )
        assert r.delta_ci is not None
        assert not any("fallback_to_internal_nuisance" in w for w in r.warnings)

    def test_crossfit_folds_respect_clusters(self):
        """Строки одного пользователя не должны попадать и в train, и в holdout."""
        from policyscope.nuisance import make_kfold_indices

        groups = np.repeat(np.arange(30), 4)
        folds = make_kfold_indices(len(groups), n_splits=3, groups=groups, random_state=0)
        for train_idx, holdout_idx in folds:
            assert not set(groups[train_idx]) & set(groups[holdout_idx])


class TestMultiTarget:
    def test_each_target_evaluated_separately(self, setup):
        bandit, policy, kwargs = setup
        df = bandit.df.copy()
        df["reward2"] = 1 - df["reward"]
        kwargs = {k: v for k, v in kwargs.items() if k != "target"}
        multi = compare_policies_multi_target(
            df, policy, targets=["reward", "reward2"], inference="none", **kwargs
        )
        assert set(multi.results) == {"reward", "reward2"}
        assert multi.results["reward"].v_a == pytest.approx(df["reward"].mean())
        assert multi.results["reward2"].v_a == pytest.approx(df["reward2"].mean())
        assert len(multi.to_frame()) == 2

    def test_shared_bundle_across_targets_is_rejected(self, setup):
        bandit, policy, kwargs = setup
        bundle = fit_crossfit_nuisance_bundle(
            bandit.df, policy, target="reward", feature_cols=["x1", "x2"],
            action_col="action", action_space=bandit.action_space, n_splits=3,
        )
        kwargs = {k: v for k, v in kwargs.items() if k != "target"}
        with pytest.raises(ValueError, match="нельзя переиспользовать"):
            compare_policies_multi_target(
                bandit.df, policy, targets=["reward"], nuisance_bundle=bundle, **kwargs
            )


class TestMultiEstimator:
    def test_returns_row_per_estimator(self, setup):
        bandit, policy, kwargs = setup
        table = compare_policies_multi_estimator(
            bandit.df, policy, estimators=("ips", "dm", "dr"),
            inference="influence", **kwargs
        )
        assert len(table) == 3
        assert set(table["estimator"]) == {"ips", "dm", "dr"}

    def test_estimators_broadly_agree_on_clean_data(self, setup):
        """На данных с точными пропенсити оценщики должны давать близкие значения."""
        bandit, policy, kwargs = setup
        table = compare_policies_multi_estimator(
            bandit.df, policy, estimators=("ips", "snips", "dr", "sndr"),
            propensity_source="logged", inference="none", **kwargs
        )
        spread = table["V_B"].max() - table["V_B"].min()
        assert spread < 0.1, f"оценщики разошлись на {spread:.4f}"


class TestEstimatorCoverage:
    @pytest.mark.parametrize("estimator", sorted(set(ALL_ESTIMATORS) - {"on_policy"}))
    def test_every_estimator_runs(self, setup, estimator):
        bandit, policy, kwargs = setup
        r = compare_policies(bandit.df, policy, estimator=estimator, inference="none", **kwargs)
        assert np.isfinite(r.v_b)

    def test_influence_ci_rejected_for_replay(self, setup):
        """Replay не является средним построчных вкладов — аналитический CI неприменим."""
        bandit, policy, kwargs = setup
        with pytest.raises(ValueError, match="Аналитический CI недоступен"):
            compare_policies(
                bandit.df, policy, estimator="replay", inference="influence", **kwargs
            )


class TestReproducibility:
    def test_same_seed_gives_same_ci(self, setup):
        bandit, policy, kwargs = setup
        a = compare_policies(bandit.df, policy, n_boot=20, rng_seed=99, **kwargs)
        b = compare_policies(bandit.df, policy, n_boot=20, rng_seed=99, **kwargs)
        assert a.delta_ci == b.delta_ci

    def test_different_seed_changes_ci(self, setup):
        bandit, policy, kwargs = setup
        a = compare_policies(bandit.df, policy, n_boot=20, rng_seed=1, **kwargs)
        b = compare_policies(bandit.df, policy, n_boot=20, rng_seed=2, **kwargs)
        assert a.delta_ci != b.delta_ci


class TestEvaluator:
    def test_evaluator_matches_function_call(self, setup):
        bandit, policy, kwargs = setup
        ev = OPEEvaluator(
            bandit.df, policy, target="reward", action_col="action",
            action_space=bandit.action_space, feature_cols=["x1", "x2"],
            propensity_col="propensity", cluster_col=None, inference="none",
        )
        assert ev.compare("dr").v_b == pytest.approx(
            compare_policies(bandit.df, policy, estimator="dr", inference="none", **kwargs).v_b
        )

    def test_evaluator_exposes_propensity_and_crossfit(self, setup):
        """Раньше объект не умел пользоваться логированными пропенсити вообще."""
        bandit, policy, _ = setup
        ev = OPEEvaluator(
            bandit.df, policy, target="reward", action_col="action",
            action_space=bandit.action_space, feature_cols=["x1", "x2"],
            propensity_col="propensity", propensity_source="logged",
            cluster_col=None, inference="none", use_crossfit=True, crossfit_n_splits=3,
        )
        assert ev.compare("dr").propensity_source == "logged"

    def test_compare_all_returns_table(self, setup):
        bandit, policy, _ = setup
        ev = OPEEvaluator(
            bandit.df, policy, target="reward", action_col="action",
            action_space=bandit.action_space, feature_cols=["x1", "x2"],
            propensity_col="propensity", cluster_col=None, inference="none",
        )
        table = ev.compare_all(estimators=("ips", "dr"))
        assert isinstance(table, pd.DataFrame) and len(table) == 2


class TestCostWarning:
    def test_expensive_bootstrap_is_announced(self, caplog):
        """Пользователь должен узнать о долгом прогоне до запуска, а не после."""
        from policyscope.comparison import _warn_if_bootstrap_is_expensive

        with caplog.at_level("WARNING"):
            _warn_if_bootstrap_is_expensive(
                n_rows=50_000, n_boot=500, estimator="dr", use_crossfit=False
            )
        assert "influence" in caplog.text

    def test_cheap_run_is_silent(self, caplog):
        from policyscope.comparison import _warn_if_bootstrap_is_expensive

        with caplog.at_level("WARNING"):
            _warn_if_bootstrap_is_expensive(
                n_rows=1000, n_boot=100, estimator="dr", use_crossfit=False
            )
            _warn_if_bootstrap_is_expensive(
                n_rows=50_000, n_boot=500, estimator="ips", use_crossfit=False
            )
        assert caplog.text == ""


class TestClusterColumnDefault:
    def test_plain_dataframe_without_user_id_just_works(self):
        """Дефолт не должен требовать колонку, которой может не быть."""
        bandit = make_controlled_bandit(n=300, seed=61)
        df = bandit.df.drop(columns=["user_id"])
        r = compare_policies(
            df, bandit.policy_b(), estimator="ips", target="reward",
            action_col="action", action_space=bandit.action_space,
            feature_cols=["x1", "x2"], propensity_col="propensity", inference="none",
        )
        assert np.isfinite(r.v_b)
        assert r.n_clusters is None

    def test_repeated_user_id_triggers_a_hint(self, caplog):
        bandit = make_controlled_bandit(n=300, seed=62, rows_per_user=5)
        with caplog.at_level("WARNING"):
            compare_policies(
                bandit.df, bandit.policy_b(), estimator="ips", target="reward",
                action_col="action", action_space=bandit.action_space,
                feature_cols=["x1", "x2"], propensity_col="propensity", inference="none",
            )
        assert "cluster_col" in caplog.text


class TestExternalBundleCoherence:
    def test_external_bundle_rejected_with_bootstrap(self, setup):
        """Внешний bundle нельзя пересчитать на реплике — CI описывал бы другой оценщик."""
        bandit, policy, kwargs = setup
        bundle = fit_crossfit_nuisance_bundle(
            bandit.df, policy, target="reward", feature_cols=["x1", "x2"],
            action_col="action", action_space=bandit.action_space, n_splits=3,
        )
        with pytest.raises(ValueError, match="несовместим с inference='bootstrap'"):
            compare_policies(
                bandit.df, policy, nuisance_bundle=bundle,
                inference="bootstrap", n_boot=10, **kwargs
            )

    def test_external_bundle_works_with_influence(self, setup):
        bandit, policy, kwargs = setup
        bundle = fit_crossfit_nuisance_bundle(
            bandit.df, policy, target="reward", feature_cols=["x1", "x2"],
            action_col="action", action_space=bandit.action_space, n_splits=3,
        )
        r = compare_policies(
            bandit.df, policy, nuisance_bundle=bundle, inference="influence", **kwargs
        )
        assert r.delta_ci is not None and np.isfinite(r.delta_ci[0])
