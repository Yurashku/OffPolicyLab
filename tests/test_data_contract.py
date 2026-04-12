import pandas as pd
import pytest

from policyscope.data import BanditSchema, LoggedBanditDataset


def _base_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": [1, 1, 2],
            "a_A": [0, 1, 0],
            "accept": [1.0, 0.0, 1.0],
            "f1": [0.1, 0.2, 0.3],
            "f2": [10, 11, 12],
            "propensity_A": [0.5, 0.8, 1.0],
        }
    )


def test_logged_bandit_dataset_validates_and_accessors_work():
    df = _base_df()
    schema = BanditSchema(
        action_col="a_A",
        reward_col="accept",
        feature_cols=["f1", "f2"],
        propensity_col="propensity_A",
        cluster_col="user_id",
    )
    ds = LoggedBanditDataset(df=df, schema=schema)
    assert len(ds.actions) == len(df)
    assert len(ds.rewards) == len(df)
    assert list(ds.features.columns) == ["f1", "f2"]
    assert ds.clusters is not None
    assert ds.logged_propensity is not None


def test_logged_bandit_dataset_empty_df_fails():
    schema = BanditSchema(action_col="a_A", reward_col="accept")
    with pytest.raises(ValueError, match="non-empty DataFrame"):
        LoggedBanditDataset(df=pd.DataFrame(), schema=schema)


def test_logged_bandit_dataset_missing_required_column_fails():
    df = _base_df().drop(columns=["a_A"])
    schema = BanditSchema(action_col="a_A", reward_col="accept")
    with pytest.raises(ValueError, match="Missing required columns"):
        LoggedBanditDataset(df=df, schema=schema)


def test_logged_bandit_dataset_missing_feature_column_fails():
    df = _base_df()
    schema = BanditSchema(action_col="a_A", reward_col="accept", feature_cols=["f1", "f_missing"])
    with pytest.raises(ValueError, match="Missing feature columns"):
        LoggedBanditDataset(df=df, schema=schema)


def test_logged_bandit_dataset_propensity_range_validation_fails():
    df = _base_df().copy()
    df.loc[0, "propensity_A"] = 0.0
    schema = BanditSchema(
        action_col="a_A",
        reward_col="accept",
        propensity_col="propensity_A",
    )
    with pytest.raises(ValueError, match="interval \\(0, 1\\]"):
        LoggedBanditDataset(df=df, schema=schema)


def test_logged_bandit_dataset_cluster_column_optional():
    df = _base_df().drop(columns=["user_id"])
    schema = BanditSchema(action_col="a_A", reward_col="accept", cluster_col=None)
    ds = LoggedBanditDataset(df=df, schema=schema)
    assert ds.clusters is None
