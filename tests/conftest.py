"""Общие фикстуры для тестов policyscope.

Ключевой объект здесь — :class:`ControlledBandit`: маленькая среда, в которой
**всё известно точно** — истинная функция награды, истинные пропенсити политики A
и истинное значение политики B. Это позволяет проверять не «код не падает», а
собственно корректность: несмещённость, состоятельность, двойную робастность и
фактическое покрытие доверительных интервалов.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Корень репозитория (не зависит от CWD, из которого запущен pytest)."""
    return REPO_ROOT


@dataclass(frozen=True)
class ControlledBandit:
    """Контекстный бандит с полностью известной истиной.

    Attributes:
        df: Логи с колонками ``user_id, x1, x2, action, propensity, reward``.
        mu_true: Матрица ``(n, k)`` истинных ожидаемых наград ``E[r|x, a]``.
        probs_a: Матрица ``(n, k)`` истинных вероятностей логирующей политики.
        probs_b: Матрица ``(n, k)`` вероятностей целевой политики.
        action_space: Метки действий.
    """

    df: pd.DataFrame
    mu_true: np.ndarray
    probs_a: np.ndarray
    probs_b: np.ndarray
    action_space: list

    def policy_b(self):
        """Целевая политика как объект, устойчивый к подвыборкам строк."""
        from policyscope.policies import TabularPolicy

        return TabularPolicy(
            self.probs_b, action_space=self.action_space, index=self.df.index
        )

    @property
    def value_b(self) -> float:
        """Истинное значение политики B на этой выборке (точно, без сэмплирования)."""
        return float(np.mean(np.sum(self.probs_b * self.mu_true, axis=1)))

    @property
    def value_a(self) -> float:
        """Истинное значение логирующей политики A на этой выборке."""
        return float(np.mean(np.sum(self.probs_a * self.mu_true, axis=1)))

    @property
    def true_delta(self) -> float:
        return self.value_b - self.value_a


def make_controlled_bandit(
    n: int = 4000,
    *,
    seed: int = 0,
    k: int = 3,
    epsilon: float = 0.25,
    rows_per_user: int = 1,
) -> ControlledBandit:
    """Строит бандит-датасет с известной истиной.

    Награда бинарная: ``r ~ Bernoulli(sigmoid(alpha_a + x @ beta_a))``. Эффект зависит
    от действия (у каждого действия свой ``beta_a``), поэтому модель исхода без
    взаимодействий ``x*a`` выразить его не может.

    Логирующая политика A — softmax по своим скорам, что даёт положительную
    вероятность каждому действию; её пропенсити известны точно.
    """
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    X = np.column_stack([np.ones(n), x1, x2])

    alpha = np.linspace(-0.4, 0.4, k)
    beta1 = np.linspace(0.9, -0.9, k)
    beta2 = np.linspace(-0.5, 0.7, k)
    coefs = np.column_stack([alpha, beta1, beta2])          # (k, 3)
    mu_true = 1.0 / (1.0 + np.exp(-(X @ coefs.T)))          # (n, k)

    # Политика A: softmax по отдельным скорам (полный носитель).
    scores_a = np.column_stack([0.7 * x1 - 0.3 * x2 + 0.2 * j for j in range(k)])
    ea = np.exp(scores_a - scores_a.max(axis=1, keepdims=True))
    probs_a = ea / ea.sum(axis=1, keepdims=True)

    # Политика B: eps-greedy по другим скорам.
    scores_b = np.column_stack([-0.4 * x1 + 0.8 * x2 - 0.1 * j for j in range(k)])
    probs_b = np.full((n, k), epsilon / k)
    probs_b[np.arange(n), scores_b.argmax(axis=1)] += 1.0 - epsilon

    cumulative = probs_a.cumsum(axis=1)
    action = (rng.random(n)[:, None] > cumulative).sum(axis=1).clip(0, k - 1)
    propensity = probs_a[np.arange(n), action]
    reward = (rng.random(n) < mu_true[np.arange(n), action]).astype(int)

    df = pd.DataFrame({
        "user_id": np.arange(n) // max(rows_per_user, 1),
        "x1": x1,
        "x2": x2,
        "action": action,
        "propensity": propensity,
        "reward": reward,
    })
    return ControlledBandit(
        df=df,
        mu_true=mu_true,
        probs_a=probs_a,
        probs_b=probs_b,
        action_space=list(range(k)),
    )


@pytest.fixture
def bandit() -> ControlledBandit:
    """Средний по размеру контролируемый датасет для большинства тестов."""
    return make_controlled_bandit(n=3000, seed=7)


@pytest.fixture
def small_bandit() -> ControlledBandit:
    """Маленький датасет для быстрых smoke-проверок."""
    return make_controlled_bandit(n=400, seed=11)


@pytest.fixture
def bandit_kwargs() -> dict:
    """Аргументы ``compare_policies`` для датасетов из :func:`make_controlled_bandit`."""
    return {
        "target": "reward",
        "action_col": "action",
        "feature_cols": ["x1", "x2"],
        "propensity_col": "propensity",
        "cluster_col": None,
    }
