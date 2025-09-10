"""
policyscope.synthetic
======================

Содержит генератор синтетической среды для офлайн‑оценки рекомендательных
политик. В модели рассматриваются пользователи с базовыми признаками и четыре
возможных действия (например, финансовые предложения). Для каждого действия
определены истинные коэффициенты вероятности отклика и прирост CLTV при
принятии. Также реализованы методы для вычисления «оракульных» ожиданий
метрик под произвольной политикой.

Класс `SyntheticRecommenderEnv` инкапсулирует генерацию данных и откликов.

```
from policyscope.synthetic import SynthConfig, SyntheticRecommenderEnv

cfg = SynthConfig(n_users=10_000, horizon_days=90, seed=42)
env = SyntheticRecommenderEnv(cfg)
X = env.sample_users()
logs = env.simulate_logs_A(policyA, X)
```
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from typing import Optional

__all__ = ["SynthConfig", "SyntheticRecommenderEnv"]


@dataclass
class SynthConfig:
    """Конфигурация синтетической среды.

    Attributes
    ----------
    n_users : int
        Количество пользователей в выборке.
    horizon_days : int
        Горизонт для метрики CLTV (в днях). На синтетике не используется напрямую,
        но сохраняется для совместимости.
    seed : int
        Случайное зерно для генератора.
    """

    n_users: int = 50_000
    horizon_days: int = 90
    seed: int = 42


class SyntheticRecommenderEnv:
    """Синтетическая среда для оценки рекомендательных политик.

    Среда моделирует пользователей с признаками (лояльность, возраст, риск,
    доход) и четыре возможных действия. Каждому действию сопоставлены
    коэффициенты для логистической модели отклика и линейная модель приращения
    CLTV. Эта информация используется для вычисления истинных значений метрик
    и генерации логов.
    """

    #: четыре действия: их можно интерпретировать как разные офферы
    ACTIONS = np.array([0, 1, 2, 3], dtype=int)

    def __init__(self, config: SynthConfig) -> None:
        self.cfg = config
        self.rng = np.random.default_rng(config.seed)

        # Истинные коэффициенты для вероятности принятия (accept)
        # Вектор размера (n_actions, n_features+1). Первый элемент — свободный член.
        self.beta = np.array([
            [-0.2,  0.8,  0.2, -0.4,  0.1],  # действие 0
            [-0.4,  0.5, -0.1, -0.2,  0.4],  # действие 1
            [-0.1,  0.3,  0.1, -0.1,  0.2],  # действие 2
            [-0.3,  0.4,  0.2,  0.1, -0.1],  # действие 3
        ], dtype=float)

        # Истинные коэффициенты для приращения CLTV при принятии
        self.gamma = np.array([
            [200.0, 120.0,  80.0, -60.0,  50.0],   # действие 0
            [300.0, 180.0, -30.0, -90.0,  70.0],   # действие 1
            [150.0,  90.0,  40.0, -20.0,  30.0],   # действие 2
            [100.0,  50.0,  60.0,  30.0, -10.0],   # действие 3
        ], dtype=float)

    def sample_users(self, n: Optional[int] = None) -> pd.DataFrame:
        """Генерирует DataFrame пользователей с признаками.

        Параметры
        ----------
        n : int, optional
            Размер выборки. Если `None`, берётся `self.cfg.n_users`.

        Возвращает
        -------
        pandas.DataFrame
            Таблица пользователей со столбцами:
            user_id, loyal, age, risk, income, region.
        """
        n = n or self.cfg.n_users
        r = self.rng

        loyal = r.binomial(1, 0.55, size=n)
        age = r.integers(18, 70, size=n)
        risk = r.uniform(0, 1, size=n)
        income = r.lognormal(mean=10.5, sigma=0.5, size=n)
        region = r.integers(0, 5, size=n)

        df = pd.DataFrame({
            "user_id": np.arange(n, dtype=np.int64),
            "loyal": loyal,
            "age": age,
            "risk": risk,
            "income": income,
            "region": region,
        })
        return df

    def _features_row(self, row: pd.Series) -> np.ndarray:
        """Вектор признаков для одной строки.

        Нормализация признаков ``age``, ``risk`` и ``income`` выполняется на
        лету, чтобы синтетическая среда оперировала только сырыми значениями.
        Возвращается массив ``(1, loyal, age_z, risk_z, income_z)``.
        """
        age_z = (row["age"] - 40) / 12.0
        risk_z = (row["risk"] - 0.5) / 0.25
        income_z = (np.log(row["income"]) - 10.5) / 0.5
        return np.array([
            1.0,
            row["loyal"],
            age_z,
            risk_z,
            income_z,
        ], dtype=float)

    def true_accept_prob(self, X: pd.DataFrame, a: np.ndarray) -> np.ndarray:
        """Истинная вероятность принятия для каждого пользователя и действия.

        Parameters
        ----------
        X : pd.DataFrame
            Данные пользователей.
        a : np.ndarray
            Массив действий такой же длины, как `X`.

        Returns
        -------
        np.ndarray
            Вектор вероятностей отклика (accept=1). Значения ограничены в (0,1).
        """
        probs = np.empty(len(X), dtype=float)
        for idx, (_, row) in enumerate(X.iterrows()):
            feats = self._features_row(row)
            la = self.beta[a[idx]].dot(feats)
            probs[idx] = 1.0 / (1.0 + np.exp(-la))
        return np.clip(probs, 1e-6, 1 - 1e-6)

    def draw_accept(self, p: np.ndarray) -> np.ndarray:
        """Сэмплирует бинарный отклик (accept) по заданным вероятностям."""
        return self.rng.binomial(1, p).astype(int)

    def base_cltv(self, X: pd.DataFrame) -> np.ndarray:
        """Базовая составляющая CLTV (до uplift)."""
        income_z = (np.log(X["income"]) - 10.5) / 0.5
        base = 300.0 + 400.0 * X["loyal"].values + 120.0 * np.maximum(income_z, -1)
        noise = self.rng.normal(0, 30, size=len(X))
        return np.maximum(base + noise, 50.0)

    def uplift_if_accepted(self, X: pd.DataFrame, a: np.ndarray) -> np.ndarray:
        """Приращение CLTV при принятии действия `a`."""
        upl = np.empty(len(X), dtype=float)
        for idx, (_, row) in enumerate(X.iterrows()):
            feats = self._features_row(row)
            upl[idx] = max(0.0, self.gamma[a[idx]].dot(feats))
        return upl

    def draw_cltv(self, X: pd.DataFrame, a: np.ndarray, accepted: np.ndarray) -> np.ndarray:
        """Сэмплирует CLTV для каждой записи.

        CLTV = base_cltv + accepted * uplift + Gauss(0,25).
        Значение ограничивается снизу, чтобы исключить отрицательные значения.
        """
        base = self.base_cltv(X)
        upl = self.uplift_if_accepted(X, a)
        noise = self.rng.normal(0, 25, size=len(X))
        return np.maximum(base + accepted * upl + noise, 10.0)

    def oracle_value(self, policy, X: pd.DataFrame, metric: str = "accept", n_mc: int = 1) -> float:
        """Вычисляет истинное значение метрики под политикой.

        Для стохастических политик осуществляется MC‑семплирование.

        Parameters
        ----------
        policy : объект политики (должен реализовывать метод `action_probs`)
        X : pd.DataFrame
            Данные пользователей.
        metric : {"accept", "cltv"}
            Целевая метрика.
        n_mc : int
            Число прогонов для MC (если policy стохастическая).

        Returns
        -------
        float
            Ожидаемое значение метрики.
        """
        results = []
        for _ in range(n_mc):
            probs = policy.action_probs(X)  # (n, n_actions)
            a = np.array([
                self.rng.choice(self.ACTIONS, p=probs[i])
                for i in range(len(X))
            ], dtype=int)
            p_accept = self.true_accept_prob(X, a)
            acc = self.draw_accept(p_accept)
            if metric == "accept":
                results.append(acc.mean())
            else:
                cltv = self.draw_cltv(X, a, acc)
                results.append(cltv.mean())
        return float(np.mean(results))

    def simulate_logs_A(self, policyA, X: pd.DataFrame) -> pd.DataFrame:
        """Генерирует логи на основе политики A.

        Параметры
        ----------
        policyA : объект политики
            Политика, генерирующая действие.
        X : pd.DataFrame
            Данные пользователей.

        Returns
        -------
        pd.DataFrame
            Логи со столбцами: a_A, accept, cltv + исходные признаки.
        """
        probs = policyA.action_probs(X)
        a = np.array([
            self.rng.choice(self.ACTIONS, p=probs[i])
            for i in range(len(X))
        ], dtype=int)
        p_accept = self.true_accept_prob(X, a)
        acc = self.draw_accept(p_accept)
        cltv = self.draw_cltv(X, a, acc)
        df = X.copy()
        df["a_A"] = a
        df["accept"] = acc
        df["cltv"] = cltv
        return df
