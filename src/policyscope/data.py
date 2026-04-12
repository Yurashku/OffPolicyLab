"""Data contract layer for logged contextual bandit datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import pandas as pd


@dataclass(frozen=True)
class BanditSchema:
    """Schema contract for logged contextual bandit data."""

    action_col: str = "a_A"
    reward_col: str = "accept"
    feature_cols: Optional[Sequence[str]] = None
    propensity_col: Optional[str] = None
    cluster_col: Optional[str] = "user_id"


@dataclass
class LoggedBanditDataset:
    """Validated wrapper around logged contextual bandit data."""

    df: pd.DataFrame
    schema: BanditSchema

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.df is None or self.df.empty:
            raise ValueError("LoggedBanditDataset requires a non-empty DataFrame.")

        required = [self.schema.action_col, self.schema.reward_col]
        missing_required = [c for c in required if c not in self.df.columns]
        if missing_required:
            raise ValueError(f"Missing required columns: {missing_required}")

        feature_cols = list(self.schema.feature_cols or [])
        missing_features = [c for c in feature_cols if c not in self.df.columns]
        if missing_features:
            raise ValueError(f"Missing feature columns: {missing_features}")

        if self.schema.propensity_col is not None:
            p_col = self.schema.propensity_col
            if p_col not in self.df.columns:
                raise ValueError(f"Missing propensity column: {p_col}")
            p = self.df[p_col]
            if p.isna().any():
                raise ValueError("Propensity column contains NaN values.")
            invalid = (~((p > 0.0) & (p <= 1.0))).any()
            if invalid:
                raise ValueError("Propensity values must be in the interval (0, 1].")

        if self.schema.cluster_col is not None and self.schema.cluster_col not in self.df.columns:
            raise ValueError(f"Missing cluster column: {self.schema.cluster_col}")

    @property
    def actions(self) -> pd.Series:
        return self.df[self.schema.action_col]

    @property
    def rewards(self) -> pd.Series:
        return self.df[self.schema.reward_col]

    @property
    def features(self) -> pd.DataFrame:
        cols = list(self.schema.feature_cols or [])
        return self.df[cols]

    @property
    def clusters(self) -> Optional[pd.Series]:
        if self.schema.cluster_col is None:
            return None
        return self.df[self.schema.cluster_col]

    @property
    def logged_propensity(self) -> Optional[pd.Series]:
        if self.schema.propensity_col is None:
            return None
        return self.df[self.schema.propensity_col]
