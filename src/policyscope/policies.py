"""
policyscope.policies
====================

Политика — это объект с методом ``action_probs(df) -> (n, k)``: матрица
вероятностей выбора каждого действия для каждой строки.

Два семейства:

**Адаптеры** (:class:`TabularPolicy`, :class:`DeterministicPolicy`,
:class:`SoftmaxPolicy`, :class:`EpsilonGreedyPolicy`) — то, что нужно на реальных
данных. Они оборачивают уже готовый выход вашей модели (матрицу вероятностей,
скоры или выбранные действия) и не требуют писать свой класс.

**Синтетические политики** (:class:`GreedyUnknown`, :class:`EpsilonGreedy`,
:class:`SoftmaxFromScores`) — только для встроенного симулятора: они зашиты на
признаки синтетического датасета и на реальных логах неприменимы.
"""

from __future__ import annotations

from typing import Literal, Optional, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "BasePolicy",
    "TabularPolicy",
    "DeterministicPolicy",
    "SoftmaxPolicy",
    "EpsilonGreedyPolicy",
    "GreedyUnknown",
    "EpsilonGreedy",
    "SoftmaxFromScores",
    "make_policy",
]


class BasePolicy:
    """Базовый класс политики.

    Наследники обязаны реализовать :meth:`action_probs`. Атрибут
    :attr:`action_space` задаёт порядок действий, соответствующий столбцам
    возвращаемой матрицы; он же передаётся оценщикам, чтобы не догадываться,
    что столбец ``j`` означает действие ``j``.
    """

    action_space: Optional[Sequence] = None

    def action_probs(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def action_argmax(self, X: pd.DataFrame) -> np.ndarray:
        """Наиболее вероятное действие для каждой строки (в метках, не индексах)."""
        probs = self.action_probs(X)
        idx = probs.argmax(axis=1)
        if self.action_space is None:
            return idx
        return np.asarray(list(self.action_space), dtype=object)[idx]


def _normalize_rows(scores: np.ndarray) -> np.ndarray:
    total = scores.sum(axis=1, keepdims=True)
    if np.any(total <= 0):
        raise ValueError("Каждая строка вероятностей должна иметь положительную сумму")
    return scores / total


def _softmax(scores: np.ndarray, temperature: float) -> np.ndarray:
    z = scores / max(temperature, 1e-12)
    z = z - z.max(axis=1, keepdims=True)  # численная стабильность
    ez = np.exp(z)
    return ez / ez.sum(axis=1, keepdims=True)


class _RowAligned:
    """Построчные данные политики, выровненные по индексу исходного DataFrame.

    Cross-fitting и bootstrap передают политике **подвыборки** строк. Массив,
    привязанный к позициям, на них рассыпается. Поэтому при передаче ``index``
    данные хранятся в ``DataFrame`` и извлекаются по меткам ``X.index``, что
    корректно работает и для подмножеств (фолды), и для повторов (bootstrap).

    Без ``index`` остаётся позиционный режим с проверкой длины — он подходит
    только для однократной оценки без ресэмплинга.
    """

    def __init__(self, values: np.ndarray, index: Optional[Sequence], *, label: str) -> None:
        self.label = label
        self._values = np.asarray(values)
        self._ndim = self._values.ndim
        self._frame: Optional[pd.DataFrame] = None
        if index is not None:
            idx = pd.Index(index)
            if len(idx) != len(self._values):
                raise ValueError(
                    f"{label}: index содержит {len(idx)} меток, а данных {len(self._values)}"
                )
            if not idx.is_unique:
                raise ValueError(
                    f"{label}: index исходного DataFrame должен быть уникальным, "
                    "иначе выравнивание строк неоднозначно"
                )
            self._frame = pd.DataFrame(self._values.reshape(len(idx), -1), index=idx)

    def resolve(self, X: pd.DataFrame) -> np.ndarray:
        if self._frame is not None:
            unknown = ~X.index.isin(self._frame.index)
            if unknown.any():
                raise ValueError(
                    f"{self.label}: нет данных для индексов "
                    f"{list(X.index[unknown][:3])} переданного DataFrame."
                )
            out = self._frame.loc[X.index].to_numpy()
            return out.ravel() if self._ndim == 1 else out
        if len(self._values) != len(X):
            raise ValueError(
                f"{self.label} привязан к позициям и содержит {len(self._values)} строк, "
                f"а запрошено {len(X)}. Передайте index=df.index при создании политики — "
                "тогда она переживёт cross-fitting и bootstrap-ресэмплинг."
            )
        return self._values


# ---------------------------------------------------------------------------
# Адаптеры для реальных моделей
# ---------------------------------------------------------------------------


class TabularPolicy(BasePolicy):
    """Политика из готовой матрицы вероятностей ``(n, k)``.

    Самый прямой адаптер: если ваша модель уже выдаёт распределение по действиям,
    просто передайте его сюда.

    Args:
        probs: Матрица ``(n, k)``; строки должны суммироваться в 1.
        action_space: Метки действий в порядке столбцов. По умолчанию ``0..k-1``.
        index: ``df.index`` исходных данных. **Настоятельно рекомендуется**: с ним
            политика выравнивается по меткам строк и потому корректно работает
            внутри cross-fitting и bootstrap-ресэмплинга.
        normalize: Нормировать строки, если они суммируются не точно в 1.

    Example:
        >>> policy = TabularPolicy(
        ...     model.predict_proba(X), action_space=["A", "B", "C"], index=df.index
        ... )
    """

    def __init__(
        self,
        probs: np.ndarray,
        *,
        action_space: Optional[Sequence] = None,
        index: Optional[Sequence] = None,
        normalize: bool = False,
    ) -> None:
        p = np.asarray(probs, dtype=float)
        if p.ndim != 2:
            raise ValueError(f"probs должен быть двумерным, получено {p.shape}")
        if np.isnan(p).any():
            raise ValueError("probs содержит NaN")
        if (p < 0).any():
            raise ValueError("probs содержит отрицательные значения")
        if normalize:
            p = _normalize_rows(p)
        self.probs = p
        self.action_space = (
            list(action_space) if action_space is not None else list(range(p.shape[1]))
        )
        if len(self.action_space) != p.shape[1]:
            raise ValueError(
                f"action_space ({len(self.action_space)}) не соответствует числу "
                f"столбцов probs ({p.shape[1]})"
            )
        self._rows = _RowAligned(p, index, label="TabularPolicy")

    def action_probs(self, X: pd.DataFrame) -> np.ndarray:
        return self._rows.resolve(X)


class DeterministicPolicy(BasePolicy):
    """Детерминированная политика: одно выбранное действие на строку.

    Подходит для argmax-выхода любого ранжировщика.

    .. note::
       Детерминированная целевая политика делает вес важности равным ``0`` везде,
       где её действие не совпало с логированным. Если перекрытие мало, взвешенные
       оценщики становятся очень шумными — смотрите на ESS и ``replay_overlap``.

    Args:
        actions: Выбранное действие для каждой строки, либо имя колонки в ``df``.
            Имя колонки — самый надёжный вариант: он автоматически переживает
            любые подвыборки.
        action_space: Полный список возможных действий.
        index: ``df.index`` исходных данных, если ``actions`` передан массивом.
    """

    def __init__(
        self,
        actions: Sequence | str,
        *,
        action_space: Sequence,
        index: Optional[Sequence] = None,
    ) -> None:
        self.action_space = list(action_space)
        self._column = actions if isinstance(actions, str) else None
        self._rows = (
            None
            if self._column
            else _RowAligned(np.asarray(actions), index, label="DeterministicPolicy")
        )

    def _resolve(self, X: pd.DataFrame) -> np.ndarray:
        if self._column is not None:
            if self._column not in X.columns:
                raise ValueError(f"В данных нет колонки действий политики: {self._column}")
            return X[self._column].to_numpy()
        assert self._rows is not None
        return self._rows.resolve(X)

    def action_probs(self, X: pd.DataFrame) -> np.ndarray:
        actions = self._resolve(X)
        idx_map = {a: i for i, a in enumerate(self.action_space)}
        probs = np.zeros((len(X), len(self.action_space)), dtype=float)
        try:
            cols = np.array([idx_map[a] for a in actions], dtype=int)
        except KeyError as exc:
            raise ValueError(f"Действие {exc} отсутствует в action_space") from exc
        probs[np.arange(len(X)), cols] = 1.0
        return probs


class SoftmaxPolicy(BasePolicy):
    """Softmax-политика над скорами модели.

    Args:
        scores: Матрица скоров ``(n, k)`` либо список колонок ``df`` со скорами.
            Список колонок — самый надёжный вариант: он автоматически переживает
            любые подвыборки.
        temperature: Температура. Чем больше, тем ближе к равномерному распределению.
        action_space: Метки действий в порядке столбцов.
        index: ``df.index`` исходных данных, если ``scores`` передан матрицей.
    """

    def __init__(
        self,
        scores: np.ndarray | Sequence[str],
        *,
        temperature: float = 1.0,
        action_space: Optional[Sequence] = None,
        index: Optional[Sequence] = None,
    ) -> None:
        if temperature <= 0:
            raise ValueError("temperature должна быть положительной")
        self.temperature = float(temperature)
        self._columns: Optional[list[str]] = None
        self._rows: Optional[_RowAligned] = None
        if isinstance(scores, (list, tuple)) and all(isinstance(s, str) for s in scores):
            self._columns = list(scores)
            k = len(self._columns)
        else:
            arr = np.asarray(scores, dtype=float)
            if arr.ndim != 2:
                raise ValueError("scores должен быть двумерным")
            self._rows = _RowAligned(arr, index, label="SoftmaxPolicy")
            k = arr.shape[1]
        self.action_space = list(action_space) if action_space is not None else list(range(k))

    def _resolve(self, X: pd.DataFrame) -> np.ndarray:
        if self._columns is not None:
            missing = [c for c in self._columns if c not in X.columns]
            if missing:
                raise ValueError(f"В данных нет колонок скоров: {missing}")
            return X[self._columns].to_numpy(dtype=float)
        assert self._rows is not None
        return self._rows.resolve(X)

    def action_probs(self, X: pd.DataFrame) -> np.ndarray:
        return _softmax(self._resolve(X), self.temperature)


class EpsilonGreedyPolicy(BasePolicy):
    """ε-жадная политика над скорами: ``1-eps`` на argmax и ``eps/k`` равномерно.

    Это самый практичный способ превратить детерминированный ранжировщик в
    политику с полным носителем: без положительной вероятности у каждого действия
    веса важности вырождаются, а оценка теряет опору в данных.

    Args:
        scores: Матрица скоров ``(n, k)`` либо список колонок ``df``.
        epsilon: Доля равномерного исследования, ``[0, 1]``.
        action_space: Метки действий в порядке столбцов.
        index: ``df.index`` исходных данных, если ``scores`` передан матрицей.
    """

    def __init__(
        self,
        scores: np.ndarray | Sequence[str],
        *,
        epsilon: float = 0.1,
        action_space: Optional[Sequence] = None,
        index: Optional[Sequence] = None,
    ) -> None:
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon должна лежать в [0, 1]")
        self.epsilon = float(epsilon)
        self._inner = SoftmaxPolicy(
            scores, temperature=1.0, action_space=action_space, index=index
        )
        self.action_space = self._inner.action_space

    def action_probs(self, X: pd.DataFrame) -> np.ndarray:
        scores = self._inner._resolve(X)
        n, k = scores.shape
        probs = np.full((n, k), self.epsilon / k, dtype=float)
        probs[np.arange(n), scores.argmax(axis=1)] += 1.0 - self.epsilon
        return probs


# ---------------------------------------------------------------------------
# Синтетические политики (только для встроенного симулятора)
# ---------------------------------------------------------------------------


class GreedyUnknown(BasePolicy):
    """Детерминированная «чёрная» политика синтетического окружения.

    Имитирует существующий продакшн-ранжировщик, устройство которого нам неизвестно.
    Зашита на признаки синтетического датасета (``loyal``, ``age``, ``risk``,
    ``income``) и на реальных логах неприменима — используйте адаптеры выше.
    """

    action_space = [0, 1, 2, 3]

    def __init__(self, seed: int = 123) -> None:
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0, 1, size=(4, 5))
        self.W[:, 0] += np.array([0.5, 0.4, 0.3, 0.2])

    def _score(self, X: pd.DataFrame) -> np.ndarray:
        age_z = (X["age"] - 40) / 12.0
        risk_z = (X["risk"] - 0.5) / 0.25
        income_z = (np.log(X["income"]) - 10.5) / 0.5
        feats = np.stack([
            np.ones(len(X)),
            np.asarray(X["loyal"], dtype=float),
            np.asarray(age_z, dtype=float),
            np.asarray(risk_z, dtype=float),
            np.asarray(income_z, dtype=float),
        ], axis=1)
        return feats @ self.W.T

    def action_probs(self, X: pd.DataFrame) -> np.ndarray:
        scores = self._score(X)
        probs = np.zeros_like(scores)
        probs[np.arange(len(X)), scores.argmax(axis=1)] = 1.0
        return probs


class EpsilonGreedy(BasePolicy):
    """ε-жадная обёртка над синтетической базовой политикой."""

    def __init__(self, base_policy: BasePolicy, epsilon: float = 0.1) -> None:
        self.base = base_policy
        self.eps = float(epsilon)
        self.action_space = base_policy.action_space

    def action_probs(self, X: pd.DataFrame) -> np.ndarray:
        base_probs = self.base.action_probs(X)
        n, k = base_probs.shape
        p = np.full((n, k), self.eps / k, dtype=float)
        p[np.arange(n), base_probs.argmax(axis=1)] += 1.0 - self.eps
        return p


class SoftmaxFromScores(BasePolicy):
    """Softmax над скрытыми скорами синтетической базовой политики."""

    def __init__(self, base_policy: GreedyUnknown, tau: float = 1.0) -> None:
        self.base = base_policy
        self.tau = float(tau)
        self.action_space = base_policy.action_space

    def action_probs(self, X: pd.DataFrame) -> np.ndarray:
        return _softmax(self.base._score(X), self.tau)


def make_policy(
    kind: Literal["greedy", "epsilon_greedy", "softmax"],
    seed: int = 123,
    epsilon: float = 0.1,
    tau: float = 1.0,
) -> BasePolicy:
    """Фабрика синтетических политик (для встроенного симулятора)."""
    base = GreedyUnknown(seed=seed)
    if kind == "greedy":
        return base
    if kind == "epsilon_greedy":
        return EpsilonGreedy(base, epsilon=epsilon)
    if kind == "softmax":
        return SoftmaxFromScores(base, tau=tau)
    raise ValueError(f"Неизвестный тип политики: {kind}")
