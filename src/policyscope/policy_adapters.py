"""Reusable policy adapters for common user-provided policy representations."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


class DeterministicActionColumnPolicy:
    """Policy adapter that converts a deterministic action column to one-hot probs."""

    def __init__(self, action_col: str, action_space: Sequence):
        self.action_col = action_col
        self.action_space = list(action_space)
        if len(self.action_space) == 0:
            raise ValueError("action_space must be non-empty")
        if len(set(self.action_space)) != len(self.action_space):
            raise ValueError("action_space must not contain duplicates")
        self._idx = {a: i for i, a in enumerate(self.action_space)}

    def _extract_actions(self, df: pd.DataFrame) -> np.ndarray:
        if self.action_col not in df.columns:
            raise ValueError(f"Missing action column: {self.action_col}")
        a = df[self.action_col].to_numpy()
        unknown = sorted({act for act in np.unique(a) if act not in self._idx})
        if unknown:
            raise ValueError(
                f"Unknown actions in column '{self.action_col}': {unknown}. "
                f"Known action_space: {self.action_space}"
            )
        return a

    def action_probs(self, df: pd.DataFrame) -> np.ndarray:
        actions = self._extract_actions(df)
        probs = np.zeros((len(df), len(self.action_space)), dtype=float)
        for i, action in enumerate(actions):
            probs[i, self._idx[action]] = 1.0
        return probs

    def action_argmax(self, df: pd.DataFrame) -> np.ndarray:
        return self._extract_actions(df)
