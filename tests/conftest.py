"""Общие фикстуры.

:func:`make_bandit` строит логи, где известно всё: истинная функция награды, истинные
пропенсити политики A и истинное значение политики B. Это позволяет проверять не
«код не падает», а корректность: несмещённость, двойную робастность, покрытие.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
ACTIONS = ["red", "green", "blue"]


@dataclass(frozen=True)
class Bandit:
    """Логи и истина.

    Attributes:
        logs: Колонки ``user_id, x1, x2, segment, action, propensity, reward``.
        mu: Истинные ``E[r|x, a]`` ``(n, k)`` (условно на клиентский эффект — 0).
        probs_a: Истинные вероятности политики A ``(n, k)``.
        policy_b: Таблица вероятностей B (колонки — метки действий).
    """

    logs: pd.DataFrame
    mu: np.ndarray
    probs_a: np.ndarray
    policy_b: pd.DataFrame

    @property
    def value_b(self) -> float:
        return float(np.mean(np.sum(self.policy_b.to_numpy() * self.mu, axis=1)))

    @property
    def value_a(self) -> float:
        return float(np.mean(np.sum(self.probs_a * self.mu, axis=1)))

    @property
    def delta(self) -> float:
        return self.value_b - self.value_a

    @property
    def policy_a(self) -> pd.DataFrame:
        return pd.DataFrame(self.probs_a, index=self.logs.index, columns=ACTIONS)


def make_bandit(n: int = 3000, *, seed: int = 0, epsilon: float = 0.3) -> Bandit:
    """Бинарная награда ``Bernoulli(sigmoid(alpha_a + x·beta_a))`` с разными наклонами
    по действиям; A — softmax по своим скорам (полный носитель), B — ε-greedy по другим."""
    rng = np.random.default_rng(seed)
    k = len(ACTIONS)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    X = np.column_stack([np.ones(n), x1, x2])
    coefs = np.column_stack([np.linspace(-0.4, 0.4, k), np.linspace(0.9, -0.9, k),
                             np.linspace(-0.5, 0.7, k)])
    mu = 1.0 / (1.0 + np.exp(-(X @ coefs.T)))

    scores_a = np.column_stack([0.7 * x1 - 0.3 * x2 + 0.2 * j for j in range(k)])
    ea = np.exp(scores_a - scores_a.max(axis=1, keepdims=True))
    probs_a = ea / ea.sum(axis=1, keepdims=True)
    scores_b = np.column_stack([-0.4 * x1 + 0.8 * x2 - 0.1 * j for j in range(k)])
    probs_b = np.full((n, k), epsilon / k)
    probs_b[np.arange(n), scores_b.argmax(axis=1)] += 1.0 - epsilon

    a = (rng.random(n)[:, None] > probs_a.cumsum(axis=1)).sum(axis=1).clip(0, k - 1)
    reward = (rng.random(n) < mu[np.arange(n), a]).astype(int)
    logs = pd.DataFrame({
        "user_id": np.arange(n),
        "x1": x1,
        "x2": x2,
        "segment": np.where(x1 > 0, "high", "low"),
        "action": np.asarray(ACTIONS, dtype=object)[a],
        "propensity": probs_a[np.arange(n), a],
        "reward": reward,
    })
    return Bandit(logs, mu, probs_a, pd.DataFrame(probs_b, columns=ACTIONS))


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def bandit() -> Bandit:
    return make_bandit(3000, seed=7)


@pytest.fixture
def evaluator(bandit):
    import policyscope as ps

    return ps.Evaluator(
        bandit.logs, action="action", reward="reward", propensity="propensity",
        features=["x1", "x2"],
    )
