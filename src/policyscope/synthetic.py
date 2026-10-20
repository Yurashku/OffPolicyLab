"""
policyscope.synthetic
=====================

Синтетическая среда для проверки OPE-оценщиков против **известной истины**.

Пользователи описываются признаками (лояльность, возраст, риск, доход), доступны
четыре действия. Для каждого действия задан свой вектор коэффициентов отклика и
приращения CLTV — то есть эффект действительно зависит от действия, и модель
исхода без взаимодействий ``x*a`` выразить его не может (это и проверяется).

Ключевое свойство: :meth:`SyntheticRecommenderEnv.oracle_value` считает истинное
значение политики **аналитически**, а не сэмплированием. Раньше оракул тянул
действие из политики и бернуллиевский отклик, усредняя реализации; при малом числе
прогонов его собственная стандартная ошибка была сопоставима с измеряемым смещением
оценщиков, и все числа bias/coverage были зашумлены самим эталоном.

Example:
    >>> cfg = SynthConfig(n_users=10_000, seed=42)
    >>> env = SyntheticRecommenderEnv(cfg)
    >>> X = env.sample_users()
    >>> logs = env.simulate_logs_A(policyA, X)
    >>> truth = env.oracle_value(policyB, X, metric="accept")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

__all__ = ["SynthConfig", "SyntheticRecommenderEnv"]


@dataclass
class SynthConfig:
    """Конфигурация синтетической среды.

    Attributes:
        n_users: Количество пользователей в выборке.
        horizon_days: Горизонт CLTV в днях (сохранён для совместимости).
        seed: Случайное зерно.
    """

    n_users: int = 50_000
    horizon_days: int = 90
    seed: int = 42


class SyntheticRecommenderEnv:
    """Синтетическая среда с известными истинными значениями политик."""

    #: Четыре действия — например, разные офферы.
    ACTIONS = np.array([0, 1, 2, 3], dtype=int)

    def __init__(self, config: SynthConfig) -> None:
        self.cfg = config
        # Отдельные потоки случайности: вызов оракула не должен сдвигать
        # генерацию данных, иначе результат зависит от порядка вызовов.
        self.rng = np.random.default_rng(config.seed)
        self._oracle_rng = np.random.default_rng(config.seed + 10_000)

        # Истинные коэффициенты вероятности принятия: (n_actions, 1 + n_features).
        self.beta = np.array([
            [-0.2,  0.8,  0.2, -0.4,  0.1],   # действие 0
            [-0.4,  0.5, -0.1, -0.2,  0.4],   # действие 1
            [-0.1,  0.3,  0.1, -0.1,  0.2],   # действие 2
            [-0.3,  0.4,  0.2,  0.1, -0.1],   # действие 3
        ], dtype=float)

        # Истинные коэффициенты приращения CLTV при принятии.
        self.gamma = np.array([
            [200.0, 120.0,  80.0, -60.0,  50.0],
            [300.0, 180.0, -30.0, -90.0,  70.0],
            [150.0,  90.0,  40.0, -20.0,  30.0],
            [100.0,  50.0,  60.0,  30.0, -10.0],
        ], dtype=float)

    # ------------------------------------------------------------------
    # Пользователи и признаки
    # ------------------------------------------------------------------

    def sample_users(self, n: Optional[int] = None) -> pd.DataFrame:
        """Генерирует таблицу пользователей.

        Returns:
            DataFrame с колонками ``user_id, loyal, age, risk, income, region``.
        """
        n = n or self.cfg.n_users
        r = self.rng
        return pd.DataFrame({
            "user_id": np.arange(n, dtype=np.int64),
            "loyal": r.binomial(1, 0.55, size=n),
            "age": r.integers(18, 70, size=n),
            "risk": r.uniform(0, 1, size=n),
            "income": r.lognormal(mean=10.5, sigma=0.5, size=n),
            "region": r.integers(0, 5, size=n),
        })

    def feature_matrix(self, X: pd.DataFrame) -> np.ndarray:
        """Матрица признаков ``(n, 5)``: ``[1, loyal, age_z, risk_z, income_z]``."""
        age_z = (X["age"].to_numpy(dtype=float) - 40) / 12.0
        risk_z = (X["risk"].to_numpy(dtype=float) - 0.5) / 0.25
        income_z = (np.log(X["income"].to_numpy(dtype=float)) - 10.5) / 0.5
        return np.column_stack([
            np.ones(len(X)),
            X["loyal"].to_numpy(dtype=float),
            age_z,
            risk_z,
            income_z,
        ])

    # ------------------------------------------------------------------
    # Истинные величины
    # ------------------------------------------------------------------

    def accept_prob_all_actions(self, X: pd.DataFrame) -> np.ndarray:
        """Истинная ``P(accept=1 | x, a)`` для всех действий: матрица ``(n, k)``."""
        feats = self.feature_matrix(X)
        logits = feats @ self.beta.T
        return np.clip(1.0 / (1.0 + np.exp(-logits)), 1e-6, 1 - 1e-6)

    def true_accept_prob(self, X: pd.DataFrame, a: np.ndarray) -> np.ndarray:
        """Истинная вероятность принятия для фактически выбранных действий."""
        all_probs = self.accept_prob_all_actions(X)
        idx = np.asarray(a, dtype=int)
        return all_probs[np.arange(len(X)), idx]

    def uplift_all_actions(self, X: pd.DataFrame) -> np.ndarray:
        """Приращение CLTV при принятии для всех действий: матрица ``(n, k)``."""
        return np.maximum(0.0, self.feature_matrix(X) @ self.gamma.T)

    def uplift_if_accepted(self, X: pd.DataFrame, a: np.ndarray) -> np.ndarray:
        """Приращение CLTV при принятии выбранного действия."""
        all_uplift = self.uplift_all_actions(X)
        return all_uplift[np.arange(len(X)), np.asarray(a, dtype=int)]

    def draw_accept(self, p: np.ndarray) -> np.ndarray:
        """Сэмплирует бинарный отклик по заданным вероятностям."""
        return self.rng.binomial(1, p).astype(int)

    def _base_cltv_mean(self, X: pd.DataFrame) -> np.ndarray:
        """Детерминированная часть базового CLTV (без шума и усечения)."""
        income_z = (np.log(X["income"].to_numpy(dtype=float)) - 10.5) / 0.5
        return 300.0 + 400.0 * X["loyal"].to_numpy(dtype=float) + 120.0 * np.maximum(income_z, -1)

    def base_cltv(self, X: pd.DataFrame) -> np.ndarray:
        """Базовый CLTV с шумом (до uplift)."""
        noise = self.rng.normal(0, 30, size=len(X))
        return np.maximum(self._base_cltv_mean(X) + noise, 50.0)

    def draw_cltv(self, X: pd.DataFrame, a: np.ndarray, accepted: np.ndarray) -> np.ndarray:
        """Сэмплирует CLTV: ``base + accepted * uplift + N(0, 25)``, усечённый снизу."""
        base = self.base_cltv(X)
        upl = self.uplift_if_accepted(X, a)
        noise = self.rng.normal(0, 25, size=len(X))
        return np.maximum(base + accepted * upl + noise, 10.0)

    # ------------------------------------------------------------------
    # Оракул
    # ------------------------------------------------------------------

    def _policy_probs(self, policy, X: pd.DataFrame) -> np.ndarray:
        probs = np.asarray(policy.action_probs(X), dtype=float)
        if probs.shape != (len(X), len(self.ACTIONS)):
            raise ValueError(
                f"Политика вернула форму {probs.shape}, ожидалось "
                f"{(len(X), len(self.ACTIONS))}"
            )
        return probs

    def oracle_value(
        self,
        policy,
        X: pd.DataFrame,
        metric: str = "accept",
        n_mc: int = 64,
    ) -> float:
        """Истинное значение метрики под политикой на выборке ``X``.

        Для ``accept`` результат **точный**: ``mean_i sum_a pi(a|x_i) p(x_i, a)``.
        Сэмплирование не используется, ``n_mc`` игнорируется.

        Для ``cltv`` интегрирование по действиям и по бернуллиевскому отклику тоже
        аналитическое; методом Монте-Карло берётся только гауссовский шум, который
        не убирается аналитически из-за усечения ``max(..., 10)``. Используется
        отдельный генератор, поэтому вызов оракула не влияет на генерацию данных.

        Args:
            policy: Политика с методом ``action_probs``.
            X: Выборка пользователей.
            metric: ``accept`` или ``cltv``.
            n_mc: Число прогонов Монте-Карло для шума CLTV.

        Returns:
            Истинное ожидаемое значение метрики.
        """
        probs = self._policy_probs(policy, X)

        if metric == "accept":
            return float(np.mean(np.sum(probs * self.accept_prob_all_actions(X), axis=1)))

        if metric != "cltv":
            raise ValueError(f"Неизвестная метрика: {metric}")

        n = len(X)
        base_mean = self._base_cltv_mean(X)
        accept_probs = self.accept_prob_all_actions(X)
        uplift = self.uplift_all_actions(X)

        rng = self._oracle_rng
        totals = np.zeros(n, dtype=float)
        for _ in range(max(n_mc, 1)):
            base = np.maximum(base_mean + rng.normal(0, 30, size=n), 50.0)
            noise = rng.normal(0, 25, size=n)
            # Аналитическое усреднение по действиям и по отклику accept in {0, 1}.
            accepted_value = np.maximum(base[:, None] + uplift + noise[:, None], 10.0)
            rejected_value = np.maximum(base + noise, 10.0)[:, None]
            per_action = accept_probs * accepted_value + (1 - accept_probs) * rejected_value
            totals += np.sum(probs * per_action, axis=1)
        return float(np.mean(totals / max(n_mc, 1)))

    # ------------------------------------------------------------------
    # Логи
    # ------------------------------------------------------------------

    def simulate_logs_A(
        self,
        policyA,
        X: pd.DataFrame,
        *,
        propensity_col: Optional[str] = "propensity_A",
    ) -> pd.DataFrame:
        """Генерирует логи под политикой A.

        Args:
            policyA: Логирующая политика.
            X: Выборка пользователей.
            propensity_col: Куда записать истинное ``pi_A(a|x)``. ``None`` — не писать.
                Логированные пропенсити известны точно, и именно так их и следует
                использовать в OPE: оценка пропенсити моделью — запасной вариант.

        Returns:
            DataFrame с колонками ``a_A``, ``accept``, ``cltv`` (+ пропенсити) и признаками.
        """
        probs = self._policy_probs(policyA, X)
        n = len(X)
        # Векторизованный выбор действия: обратное преобразование по кумулятивной сумме.
        cumulative = np.cumsum(probs, axis=1)
        draws = self.rng.random(n)[:, None]
        a = (draws > cumulative).sum(axis=1).clip(0, probs.shape[1] - 1)

        p_accept = self.true_accept_prob(X, a)
        acc = self.draw_accept(p_accept)
        cltv = self.draw_cltv(X, a, acc)

        df = X.copy()
        df["a_A"] = a.astype(int)
        df["accept"] = acc
        df["cltv"] = cltv
        if propensity_col is not None:
            df[propensity_col] = probs[np.arange(n), a]
        return df
