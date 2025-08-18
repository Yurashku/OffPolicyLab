"""
policyscope.policies
====================

Определение рекомендательных политик для синтетического эксперимента.
Политики принимают на вход DataFrame с признаками пользователей и возвращают
матрицу вероятностей выбора для каждого действия. Доступны базовые
реализации:

* `GreedyUnknown` — детерминированная политика, выбирающая действие с
  максимальным скрытым скором.
* `EpsilonGreedy` — ε‑жадная политика, добавляющая случайный выбор
  с вероятностью ε.
* `SoftmaxFromScores` — softmax‑распределение по скрытым скоринговым
  функциям с температурой τ.
* `make_policy` — фабрика, упрощающая создание политики по строковому
  идентификатору.

Политики используют внутреннюю модель скрытых скорингов, чтобы имитировать
«чёрный ящик» существующей системы ранжирования. Это позволяет
адаптировать код и для реальных логов, подставив сюда свою модель.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Literal

__all__ = [
    "BasePolicy",
    "GreedyUnknown",
    "EpsilonGreedy",
    "SoftmaxFromScores",
    "make_policy",
]


class BasePolicy:
    """Базовый класс политики. Должен реализовать `action_probs(X)`.
    
    Метод `action_argmax` вычисляет индекс действия с максимальной вероятностью.
    """

    def action_probs(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def action_argmax(self, X: pd.DataFrame) -> np.ndarray:
        probs = self.action_probs(X)
        return probs.argmax(axis=1)


class GreedyUnknown(BasePolicy):
    """Детерминированная «чёрная» политика: выбирает действие с максимальным
    скрытым скором. Скртытые веса генерируются случайно при инициализации.
    
    Для синтетического эксперимента это имитирует текущий работающий
    алгоритм, который мы не можем проанализировать детально, но можем
    использовать для генерации логов.
    """

    def __init__(self, seed: int = 123) -> None:
        rng = np.random.default_rng(seed)
        # Генерируем случайные веса (5 признаков: константа + 4 normalized features)
        self.W = rng.normal(0, 1, size=(4, 5))
        # добавляем смещения для большей разнообразности
        self.W[:, 0] += np.array([0.5, 0.4, 0.3, 0.2])

    def _score(self, X: pd.DataFrame) -> np.ndarray:
        feats = np.stack([
            np.ones(len(X)),
            X["loyal"].values,
            X["age_z"].values,
            X["risk_z"].values,
            X["income_z"].values,
        ], axis=1)  # (n,5)
        return feats @ self.W.T  # (n,4)

    def action_probs(self, X: pd.DataFrame) -> np.ndarray:
        scores = self._score(X)
        a = scores.argmax(axis=1)
        probs = np.zeros_like(scores)
        probs[np.arange(len(X)), a] = 1.0
        return probs


class EpsilonGreedy(BasePolicy):
    """ε‑жадная политика.
    
    C вероятностью `1-ε` выбирает действие, которое бы выбрала базовая
    детерминированная политика, а с вероятностью `ε` — выбирает случайное
    действие равновероятно. Таким образом, логируется пропенсити для всех
    действий, что важно для off‑policy оценивания.
    """

    def __init__(self, base_policy: BasePolicy, epsilon: float = 0.1) -> None:
        self.base = base_policy
        self.eps = float(epsilon)

    def action_probs(self, X: pd.DataFrame) -> np.ndarray:
        base_probs = self.base.action_probs(X)  # (n,4) с единицами
        n, k = base_probs.shape
        # равномерный шум
        p = np.full((n, k), self.eps / k, dtype=float)
        best = base_probs.argmax(axis=1)
        p[np.arange(n), best] += (1.0 - self.eps)
        return p


class SoftmaxFromScores(BasePolicy):
    """Softmax‑политика на основе скрытых скорингов.
    
    Преобразует сырые скоры базовой политики к вероятностям softmax с
    температурой `tau`. Чем выше τ, тем ближе распределение к равномерному.
    """

    def __init__(self, base_policy: GreedyUnknown, tau: float = 1.0) -> None:
        self.base = base_policy
        self.tau = float(tau)

    def action_probs(self, X: pd.DataFrame) -> np.ndarray:
        score = self.base._score(X)
        z = score / max(self.tau, 1e-6)
        z = z - z.max(axis=1, keepdims=True)  # числовая стабильность
        ez = np.exp(z)
        return ez / ez.sum(axis=1, keepdims=True)


def make_policy(kind: Literal["greedy", "epsilon_greedy", "softmax"], seed: int = 123, epsilon: float = 0.1, tau: float = 1.0) -> BasePolicy:
    """Фабрика политик.

    Parameters
    ----------
    kind : {"greedy", "epsilon_greedy", "softmax"}
        Тип политики: жадная, ε‑жадная или softmax.
    seed : int
        Зерно для генерации скрытых скорингов.
    epsilon : float
        Параметр ε для ε‑жадной политики.
    tau : float
        Температура для softmax.

    Returns
    -------
    BasePolicy
        Экземпляр выбранной политики.
    """
    base = GreedyUnknown(seed=seed)
    if kind == "greedy":
        return base
    if kind == "epsilon_greedy":
        return EpsilonGreedy(base, epsilon=epsilon)
    if kind == "softmax":
        return SoftmaxFromScores(base, tau=tau)
    raise ValueError(f"Unknown policy kind: {kind}")