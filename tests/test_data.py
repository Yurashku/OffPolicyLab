"""Подготовка логов: понятные ошибки вместо молчаливых NaN и утечек."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from policyscope.data import prepare_logs


def frame(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "client_id": np.arange(n),
        "age": rng.integers(18, 70, n),
        "income": rng.normal(50, 10, n),
        "vip": rng.random(n) < 0.3,
        "region": rng.choice(["north", "south", "east"], n),
        "offer": rng.choice(["card", "loan"], n),
        "p": np.full(n, 0.5),
        "accepted": rng.integers(0, 2, n),
        "revenue_noise": rng.normal(size=n),
    })


def prep(df, **kw):
    kw.setdefault("action", "offer")
    kw.setdefault("reward", "accepted")
    return prepare_logs(df, **kw)


class TestErrors:
    def test_missing_column(self):
        with pytest.raises(ValueError, match="нет колонок"):
            prep(frame(), reward="clicks")

    def test_nan_reward_is_refused_with_count(self):
        df = frame()
        df.loc[[1, 5], "accepted"] = np.nan
        with pytest.raises(ValueError, match="2 пропусков"):
            prep(df)

    def test_nan_action(self):
        df = frame()
        df.loc[0, "offer"] = None
        with pytest.raises(ValueError, match="пропусков"):
            prep(df)

    @pytest.mark.parametrize("bad", [0.0, 1.5, np.nan])
    def test_bad_propensity(self, bad):
        df = frame()
        df.loc[3, "p"] = bad
        with pytest.raises(ValueError):
            prep(df, propensity="p")

    def test_service_column_as_feature_is_a_leak(self):
        with pytest.raises(ValueError, match="утечка"):
            prep(frame(), features=["age", "accepted"])

    def test_nan_in_explicit_features(self):
        df = frame()
        df.loc[0, "income"] = np.nan
        with pytest.raises(ValueError, match="пропуски"):
            prep(df, features=["income"])

    def test_non_unique_index(self):
        df = frame()
        df.index = [0] * len(df)
        with pytest.raises(ValueError, match="уникальным"):
            prep(df)

    def test_non_numeric_reward(self):
        df = frame()
        df["accepted"] = df["accepted"].map({0: "no", 1: "yes"})
        with pytest.raises(ValueError, match="числовой"):
            prep(df)


class TestEncoding:
    def test_actions_are_sorted_labels_with_codes(self):
        d = prep(frame())
        assert d.actions == ["card", "loan"]
        assert set(np.unique(d.a)) == {0, 1}

    def test_categorical_feature_is_one_hot(self):
        d = prep(frame(), features=["age", "region", "vip"])
        assert d.feature_names == ["age", "region=east", "region=north", "region=south", "vip"]
        np.testing.assert_allclose(d.X[:, 1:4].sum(axis=1), 1.0)

    def test_binary_detection(self):
        df = frame()
        df["revenue"] = df["accepted"] * 100.0
        d = prep(df, reward=["accepted", "revenue"], features=["age"])
        assert d.binary == {"accepted": True, "revenue": False}

    def test_bool_reward(self):
        df = frame()
        df["accepted"] = df["accepted"].astype(bool)
        assert prep(df, features=["age"]).binary["accepted"]

    def test_unclustered_rows_are_own_clusters(self):
        d = prep(frame(), features=["age"])
        assert d.n_clusters == d.n and not d.is_clustered


class TestAutoFeatures:
    def test_service_id_and_constant_columns_are_skipped(self):
        df = frame()
        df["const"] = 1
        d = prep(df, propensity="p")
        assert "accepted" not in d.features and "p" not in d.features
        assert d.skipped_features["client_id"] == "похоже на идентификатор"
        assert d.skipped_features["const"] == "константа"
        assert {"age", "income", "vip", "region"} <= set(d.features)

    def test_other_outcome_column_is_excluded(self):
        """Другая метрика (выручка) известна только после действия — это утечка."""
        df = frame()
        df["revenue"] = df["accepted"] * np.random.default_rng(1).uniform(50, 150, len(df))
        d = prep(df)
        assert "revenue" not in d.features
        assert "исход" in d.skipped_features["revenue"]

    def test_explicit_outcome_like_feature_is_flagged(self):
        df = frame()
        df["revenue"] = df["accepted"] * 100.0
        d = prep(df, features=["age", "revenue"])
        assert d.suspicious_features == {"revenue": "accepted"}

    def test_high_cardinality_and_nan_columns_skipped(self):
        df = frame()
        df["comment"] = [f"text {i % 70}" for i in range(len(df))]
        df["with_nan"] = np.where(np.arange(len(df)) % 2, 1.0, np.nan)
        d = prep(df)
        assert "comment" in d.skipped_features and "with_nan" in d.skipped_features


class TestClusterHint:
    def test_repeated_client_column_triggers_hint(self):
        df = frame()
        df["client_id"] = np.arange(len(df)) // 3
        assert prep(df, features=["age"]).cluster_hint == "client_id"

    def test_no_hint_when_cluster_is_set(self):
        df = frame()
        df["client_id"] = np.arange(len(df)) // 3
        d = prep(df, features=["age"], cluster="client_id")
        assert d.cluster_hint is None and d.n_clusters == len(df) // 3 + (len(df) % 3 > 0)


class TestDtypes:
    def test_category_columns_with_unused_levels_do_not_warn(self):
        import warnings

        df = frame()
        df["offer"] = pd.Categorical(df["offer"], categories=["card", "loan", "mortgage"])
        df["region"] = pd.Categorical(df["region"], categories=["north", "south", "east", "west"])
        df["age_band"] = pd.cut(df["age"], bins=[0, 30, 50, 100])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            d = prep(df, features=["age", "region", "age_band"])
        assert d.actions == ["card", "loan"]
        assert "region=west" not in d.feature_names  # уровень без наблюдений не кодируется

    def test_action_labels_are_python_scalars(self):
        df = frame().assign(offer=lambda x: np.where(x["vip"], 7, 3))
        d = prep(df, features=["age"])
        assert d.actions == [3, 7] and all(type(a) is int for a in d.actions)

    def test_nullable_propensity_gives_friendly_error(self):
        df = frame()
        df["p"] = pd.array([0.5] * (len(df) - 1) + [None], dtype="Float64")
        with pytest.raises(ValueError, match="пропусков"):
            prep(df, propensity="p")

    def test_huge_explicit_categorical_is_refused(self):
        df = frame(n=1200)
        df["code"] = [f"c{i}" for i in range(len(df))]
        with pytest.raises(ValueError, match="уровней"):
            prep(df, features=["age", "code"])

    def test_auto_features_skip_propensity_like_columns(self):
        df = frame().rename(columns={"p": "propensity_logged"})
        d = prep(df)
        assert "propensity_logged" not in d.features
