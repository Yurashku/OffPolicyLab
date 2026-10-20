"""Синтетический банк с офферами — среда, где истинное значение политики известно.

Нужна, чтобы проверять оценщики против истины (на реальных логах её нет) и чтобы
туториалы были воспроизводимы. Устроена как реальные логи:

- четыре оффера со строковыми метками (``card``, ``deposit``, ``loan``, ``insurance``);
- признаки клиента, включая категориальный регион;
- две метрики: принятие оффера (0/1) и выручка (непрерывная, ноль при отказе);
- клиент может приходить несколько раз (``visits``), и у него есть ненаблюдаемая
  склонность соглашаться (``client_effect``) — строки одного клиента зависимы, как
  в жизни. Именно это проверяет кластерный инференс.

:meth:`OfferSimulator.true_value` считает истинное значение политики **аналитически**:
вероятности принятия интегрируются по распределению клиентского эффекта квадратурой
Гаусса–Эрмита, а не сэмплированием, поэтому у эталона нет собственного шума.

Example:
    >>> sim = ps.synthetic.OfferSimulator(seed=0)
    >>> clients = sim.sample_clients(5000)
    >>> policy_a = sim.policy("epsilon_greedy", epsilon=0.2, seed=1)
    >>> logs = sim.simulate(clients, policy_a)
    >>> sim.true_value(logs, policy_a, "accepted")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from policyscope.policy import epsilon_greedy, greedy, softmax

__all__ = ["OfferSimulator", "SimPolicy"]

ACTIONS = ("card", "deposit", "loan", "insurance")
REGIONS = ("center", "east", "north", "south", "west")
_GH_NODES, _GH_WEIGHTS = np.polynomial.hermite_e.hermegauss(40)
_GH_WEIGHTS = _GH_WEIGHTS / _GH_WEIGHTS.sum()


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


@dataclass
class SimPolicy:
    """Политика симулятора: линейные скоры по признакам клиента + правило выбора.

    Вызов :meth:`action_probs` возвращает таблицу вероятностей (колонки — офферы),
    то есть ровно тот формат, который принимает :meth:`policyscope.Evaluator.evaluate`.
    """

    simulator: "OfferSimulator"
    kind: str
    weights: np.ndarray
    epsilon: float = 0.1
    temperature: float = 1.0

    def scores(self, df: pd.DataFrame) -> pd.DataFrame:
        values = self.simulator.design(df) @ self.weights.T
        return pd.DataFrame(values, index=df.index, columns=list(ACTIONS))

    def action_probs(self, df: pd.DataFrame) -> pd.DataFrame:
        s = self.scores(df)
        if self.kind == "greedy":
            return greedy(s)
        if self.kind == "epsilon_greedy":
            return epsilon_greedy(s, self.epsilon)
        if self.kind == "softmax":
            return softmax(s, self.temperature)
        if self.kind == "uniform":
            return pd.DataFrame(1.0 / len(ACTIONS), index=df.index, columns=list(ACTIONS))
        raise ValueError(f"Неизвестный тип политики: {self.kind}")


class OfferSimulator:
    """Генератор логов и оракул истинного значения политик.

    Args:
        seed: Зерно генерации клиентов и откликов.
        client_effect: Стандартное отклонение ненаблюдаемой склонности клиента
            соглашаться (на шкале логитов). ``0`` — строки независимы.
    """

    actions = ACTIONS

    def __init__(self, seed: int = 0, client_effect: float = 0.0) -> None:
        if client_effect < 0:
            raise ValueError("client_effect не может быть отрицательным")
        self.rng = np.random.default_rng(seed)
        self.client_effect = float(client_effect)
        self._effects = np.zeros(0)
        # Логит принятия: [1, loyal, age_z, risk_z, income_z] + эффект региона, по офферу.
        self.beta = np.array([
            [-1.2, 0.8, -0.4, -0.6, 0.5],    # card
            [-1.6, 1.0, 0.6, -0.2, 0.7],     # deposit
            [-1.4, 0.3, -0.2, 0.9, -0.4],    # loan
            [-1.8, 0.5, 0.5, 0.3, 0.2],      # insurance
        ])
        self.region_effect = np.array([
            [0.0, 0.2, -0.3, 0.1, 0.0],
            [0.0, -0.1, 0.3, -0.2, 0.1],
            [0.0, 0.3, 0.1, 0.2, -0.3],
            [0.0, -0.2, 0.0, 0.3, 0.2],
        ])
        # Выручка при принятии: 100 * exp(gamma_a · [1, loyal, age_z, risk_z, income_z]).
        self.gamma = np.array([
            [0.0, 0.2, -0.1, 0.1, 0.3],
            [0.4, 0.3, 0.2, -0.2, 0.5],
            [0.7, -0.1, -0.2, 0.3, 0.2],
            [-0.2, 0.2, 0.3, 0.0, 0.1],
        ])

    # ------------------------------------------------------------------
    # Клиенты и признаки
    # ------------------------------------------------------------------

    def sample_clients(self, n_clients: int, visits: int = 1) -> pd.DataFrame:
        """Клиенты с признаками; при ``visits > 1`` каждый клиент — несколько строк."""
        if n_clients < 1 or visits < 1:
            raise ValueError("n_clients и visits должны быть положительными")
        r = self.rng
        start = len(self._effects)
        self._effects = np.concatenate([
            self._effects, r.normal(0.0, self.client_effect, size=n_clients)
        ])
        clients = pd.DataFrame({
            "client_id": np.arange(start, start + n_clients),
            "age": r.integers(18, 75, size=n_clients),
            "income": np.round(r.lognormal(mean=11.0, sigma=0.5, size=n_clients), -2),
            "loyal": r.binomial(1, 0.45, size=n_clients),
            "risk": np.round(r.uniform(0, 1, size=n_clients), 3),
            "region": r.choice(REGIONS, size=n_clients),
        })
        if visits > 1:
            clients = clients.loc[clients.index.repeat(visits)].reset_index(drop=True)
        return clients

    def design(self, df: pd.DataFrame) -> np.ndarray:
        """Признаки для скоров политик: ``[1, loyal, age_z, risk_z, income_z, onehot(region)]``."""
        return np.column_stack([self._core(df), self._region_onehot(df)])

    @staticmethod
    def _core(df: pd.DataFrame) -> np.ndarray:
        return np.column_stack([
            np.ones(len(df)),
            df["loyal"].to_numpy(dtype=float),
            (df["age"].to_numpy(dtype=float) - 45.0) / 15.0,
            (df["risk"].to_numpy(dtype=float) - 0.5) / 0.29,
            (np.log(df["income"].to_numpy(dtype=float)) - 11.0) / 0.5,
        ])

    @staticmethod
    def _region_onehot(df: pd.DataFrame) -> np.ndarray:
        codes = pd.Categorical(df["region"], categories=REGIONS).codes
        out = np.zeros((len(df), len(REGIONS)))
        out[np.arange(len(df)), codes] = 1.0
        return out

    def _logits(self, df: pd.DataFrame) -> np.ndarray:
        return self._core(df) @ self.beta.T + self._region_onehot(df) @ self.region_effect.T

    def _client_effect(self, df: pd.DataFrame) -> np.ndarray:
        return self._effects[df["client_id"].to_numpy()]

    def accept_probability(self, df: pd.DataFrame) -> np.ndarray:
        """Средняя по клиентскому эффекту вероятность принятия ``(n, k)``."""
        z = self._logits(df)
        if self.client_effect == 0:
            return _sigmoid(z)
        nodes = self.client_effect * _GH_NODES
        return np.tensordot(_sigmoid(z[..., None] + nodes), _GH_WEIGHTS, axes=([-1], [0]))

    def revenue_if_accepted(self, df: pd.DataFrame) -> np.ndarray:
        """Ожидаемая выручка при принятии ``(n, k)``."""
        return 100.0 * np.exp(self._core(df) @ self.gamma.T)

    # ------------------------------------------------------------------
    # Политики, логи, истина
    # ------------------------------------------------------------------

    def policy(
        self, kind: str, *, seed: int = 0, epsilon: float = 0.1, temperature: float = 1.0
    ) -> SimPolicy:
        """Политика со случайными (по ``seed``) линейными скорами.

        Args:
            kind: ``greedy``, ``epsilon_greedy``, ``softmax`` или ``uniform``.
            seed: Разные зёрна — разные «модели» ранжирования.
        """
        if kind not in ("greedy", "epsilon_greedy", "softmax", "uniform"):
            raise ValueError(f"Неизвестный тип политики: {kind}")
        r = np.random.default_rng(seed)
        weights = r.normal(0.0, 1.0, size=(len(ACTIONS), 5 + len(REGIONS)))
        return SimPolicy(self, kind, weights, epsilon=epsilon, temperature=temperature)

    def _probs(self, policy: Any, df: pd.DataFrame) -> np.ndarray:
        probs = policy.action_probs(df) if hasattr(policy, "action_probs") else policy
        if isinstance(probs, pd.DataFrame):
            return probs.reindex(columns=list(ACTIONS), fill_value=0.0).to_numpy(dtype=float)
        return np.asarray(probs, dtype=float)

    def simulate(self, df: pd.DataFrame, policy: Any, *, log_propensity: bool = True) -> pd.DataFrame:
        """Логи политики: к строкам ``df`` добавляются ``offer``, ``propensity``,
        ``accepted`` и ``revenue``."""
        probs = self._probs(policy, df)
        n = len(df)
        draws = self.rng.random(n)[:, None]
        a = (draws > np.cumsum(probs, axis=1)).sum(axis=1).clip(0, len(ACTIONS) - 1)
        rows = np.arange(n)
        z = self._logits(df)[rows, a] + self._client_effect(df)
        accepted = (self.rng.random(n) < _sigmoid(z)).astype(int)
        noise = self.rng.lognormal(mean=-0.045, sigma=0.3, size=n)  # E[noise] = 1
        revenue = accepted * self.revenue_if_accepted(df)[rows, a] * noise
        logs = df.copy()
        logs["offer"] = np.asarray(ACTIONS, dtype=object)[a]
        if log_propensity:
            logs["propensity"] = probs[rows, a]
        logs["accepted"] = accepted
        logs["revenue"] = np.round(revenue, 2)
        return logs

    def true_value(self, df: pd.DataFrame, policy: Any, metric: str = "accepted") -> float:
        """Истинное значение метрики под политикой на строках ``df``.

        Аналитически: ``mean_i sum_a pi(a|x_i) E[metric | x_i, a]``, где ожидание взято
        по отклику и по распределению клиентского эффекта.
        """
        probs = self._probs(policy, df)
        p = self.accept_probability(df)
        if metric == "accepted":
            per_action = p
        elif metric == "revenue":
            per_action = p * self.revenue_if_accepted(df)
        else:
            raise ValueError("metric: 'accepted' или 'revenue'")
        return float(np.mean(np.sum(probs * per_action, axis=1)))
