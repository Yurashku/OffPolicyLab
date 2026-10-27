"""Политики как таблицы вероятностей.

Политика в Policyscope — это таблица ``(n, k)``: строка — строка логов, колонка —
действие, значение — вероятность, с которой политика выбрала бы это действие в этом
контексте. Самый надёжный формат — ``DataFrame``, у которого **колонки — метки
действий**: тогда сопоставление колонок с действиями не зависит от их порядка.

Политика вычисляется один раз на всех логах; дальше библиотека индексирует матрицу
сама, поэтому кросс-фиттинг и bootstrap не требуют от политики ничего.

Хелперы :func:`greedy`, :func:`softmax`, :func:`epsilon_greedy` превращают скоры
модели (``DataFrame`` с метками действий в колонках) в такие таблицы.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from policyscope.data import LogData

__all__ = ["greedy", "softmax", "epsilon_greedy", "policy_matrix"]

_ROW_SUM_TOL = 1e-6


def _scores_frame(scores: Any) -> pd.DataFrame:
    if isinstance(scores, pd.DataFrame):
        frame = scores
    else:
        arr = np.asarray(scores, dtype=float)
        if arr.ndim != 2:
            raise ValueError(f"Скоры должны быть таблицей (n, k), получена форма {arr.shape}")
        frame = pd.DataFrame(arr)
    values = frame.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Скоры содержат NaN или бесконечности")
    return frame


def greedy(scores: Any) -> pd.DataFrame:
    """Детерминированная политика: действие с максимальным скором.

    Ничьи делятся поровну между лучшими действиями — иначе выбор зависел бы от
    порядка колонок.
    """
    frame = _scores_frame(scores)
    values = frame.to_numpy(dtype=float)
    best = np.isclose(values, values.max(axis=1, keepdims=True), rtol=0.0, atol=1e-12)
    probs = best / best.sum(axis=1, keepdims=True)
    return pd.DataFrame(probs, index=frame.index, columns=frame.columns)


def softmax(scores: Any, temperature: float = 1.0) -> pd.DataFrame:
    """Стохастическая политика ``pi(a|x) ∝ exp(score_a / temperature)``.

    Чем ниже температура, тем ближе к :func:`greedy`; чем выше — к равномерной.
    """
    if temperature <= 0:
        raise ValueError("temperature должна быть положительной")
    frame = _scores_frame(scores)
    z = frame.to_numpy(dtype=float) / temperature
    z = z - z.max(axis=1, keepdims=True)
    ez = np.exp(z)
    return pd.DataFrame(ez / ez.sum(axis=1, keepdims=True), index=frame.index, columns=frame.columns)


def epsilon_greedy(scores: Any, epsilon: float = 0.1) -> pd.DataFrame:
    """С вероятностью ``1 - epsilon`` — лучшее действие, иначе — равномерно случайное."""
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon должна лежать в [0, 1]")
    best = greedy(scores)
    k = best.shape[1]
    return best * (1.0 - epsilon) + epsilon / k


# ---------------------------------------------------------------------------
# Приведение входа к матрице
# ---------------------------------------------------------------------------


def _is_default_index(index: pd.Index, n: int) -> bool:
    return isinstance(index, pd.RangeIndex) and len(index) == n and index.start == 0 and index.step == 1


def _is_permuted_range(index: pd.Index, n: int) -> bool:
    if not pd.api.types.is_integer_dtype(index) or len(index) != n:
        return False
    return bool(np.array_equal(np.sort(index.to_numpy()), np.arange(n)))


def _aligned_rows(frame: pd.DataFrame, logs_index: pd.Index, role: str) -> pd.DataFrame:
    """Сопоставляет строки таблицы политики со строками логов.

    - Индексы совпадают — строки уже выровнены.
    - У таблицы индекс по умолчанию (0..n-1) — это порядок строк логов (типичный результат
      ``pd.DataFrame(model.predict_proba(X))``). Если же индекс логов — перестановка 0..n-1
      (после ``sort_values`` или ``sample``), понять, имелись в виду позиции или метки,
      нельзя: это ошибка, а не молчаливая перестановка строк.
    - Иначе — по меткам индекса, если множества меток совпадают.
    """
    n = len(logs_index)
    if len(frame) == n and frame.index.equals(logs_index):
        return frame
    if _is_default_index(frame.index, n):
        if _is_permuted_range(logs_index, n):
            raise ValueError(
                f"Строки политики {role} нельзя однозначно сопоставить с логами: у таблицы индекс "
                "по умолчанию 0..n-1, а индекс логов — его перестановка (например, после "
                "sort_values или sample). Непонятно, сопоставлять строки по позиции или по метке. "
                "Передайте таблицу с index=logs.index или сбросьте индекс логов: "
                "logs.reset_index(drop=True)."
            )
        return frame.set_axis(logs_index, axis=0)
    if frame.index.is_unique and len(frame) == n and set(frame.index) == set(logs_index):
        return frame.loc[logs_index]
    raise ValueError(
        f"Строки политики {role} не сопоставляются со строками логов: индексы различаются "
        f"({len(frame)} строк против {n}). Передайте таблицу с тем же индексом, что у логов, "
        "или с индексом по умолчанию той же длины."
    )


def _from_actions(chosen: pd.Series, data: LogData, role: str) -> np.ndarray:
    if chosen.isna().any():
        raise ValueError(f"Политика {role}: в выбранных действиях есть пропуски")
    known = {label: i for i, label in enumerate(data.actions)}
    unknown = sorted({v for v in chosen if v not in known}, key=repr)
    if unknown:
        raise ValueError(
            f"Политика {role} выбирает действия, которых нет в логах: {unknown[:5]}. "
            "Для них нет данных, и off-policy оценка невозможна."
        )
    probs = np.zeros((len(chosen), data.k))
    probs[np.arange(len(chosen)), [known[v] for v in chosen]] = 1.0
    return probs


def _from_frame(frame: pd.DataFrame, data: LogData, role: str) -> np.ndarray:
    frame = _aligned_rows(frame, data.index, role)
    try:
        values = frame.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Таблица политики {role} должна быть числовой") from exc
    if not np.isfinite(values).all():
        raise ValueError(f"Таблица политики {role} содержит NaN или бесконечности")
    if (values < 0).any():
        raise ValueError(f"Таблица политики {role} содержит отрицательные вероятности")
    sums = values.sum(axis=1)
    if not np.allclose(sums, 1.0, atol=_ROW_SUM_TOL):
        worst = float(np.max(np.abs(sums - 1.0)))
        raise ValueError(
            f"Строки таблицы политики {role} должны суммироваться в 1 (отклонение до {worst:.3g}). "
            "Если у вас скоры, используйте ps.softmax / ps.greedy / ps.epsilon_greedy."
        )

    columns = list(frame.columns)
    position = {label: i for i, label in enumerate(columns)}
    matched = [label for label in data.actions if label in position]
    if not matched:
        raise ValueError(
            f"Колонки политики {role} {columns[:6]} не совпадают ни с одной меткой действия "
            f"в логах {data.actions[:6]}. Проверьте типы меток (например, '1' и 1 различаются)."
        )
    extra = [label for label in columns if label not in set(data.actions)]
    extra_mass = {label: float(values[:, position[label]].mean()) for label in extra}
    extra_mass = {label: m for label, m in extra_mass.items() if m > 0}
    if extra_mass:
        shown = {label: round(mass, 4) for label, mass in list(extra_mass.items())[:5]}
        raise ValueError(
            f"Политика {role} назначает вероятность действиям, которых нет в логах: "
            f"{shown} (средняя масса). "
            "По ним нет данных, и off-policy оценка невозможна. Уберите эти действия из "
            "кандидата или соберите логи, где они встречаются."
        )
    probs = np.zeros((len(frame), data.k))
    for code, label in enumerate(data.actions):
        if label in position:
            probs[:, code] = values[:, position[label]]
    return probs


def _from_array(arr: np.ndarray, data: LogData, role: str) -> np.ndarray:
    if arr.ndim != 2 or arr.shape[0] != data.n:
        raise ValueError(f"Политика {role}: ожидалась матрица ({data.n}, k), получено {arr.shape}")
    labels_are_codes = all(
        isinstance(label, (int, np.integer)) and 0 <= int(label) < arr.shape[1]
        for label in data.actions
    )
    if not labels_are_codes:
        raise ValueError(
            f"Политика {role} передана массивом без меток действий, а действия в логах — "
            f"{data.actions[:6]}. Передайте DataFrame, у которого колонки — метки действий: "
            "pd.DataFrame(probs, columns=model.classes_)."
        )
    # У массива нет индекса: строки по определению идут в порядке логов.
    frame = pd.DataFrame(arr, index=data.index, columns=list(range(arr.shape[1])))
    return _from_frame(frame, data, role)


def policy_matrix(policy: Any, data: LogData, frame: pd.DataFrame, role: str = "B") -> np.ndarray:
    """Приводит политику к матрице ``(n, k)`` в порядке ``data.actions``.

    Args:
        policy: Одно из:
            - ``DataFrame`` вероятностей с метками действий в колонках (рекомендуется);
            - ``Series`` или имя колонки логов с действием, которое выбрала бы политика;
            - объект с методом ``action_probs(df)`` или функция ``f(df)``, возвращающие
              одно из перечисленного (вызываются один раз на логах без наград и пропенсити);
            - ``ndarray`` — только если действия в логах закодированы числами ``0..k-1``.
        data: Подготовленные логи.
        frame: Логи без колонок наград и пропенсити — то, что видит политика.
        role: Название политики в сообщениях об ошибках.
    """
    if hasattr(policy, "action_probs"):
        return policy_matrix(policy.action_probs(frame), data, frame, role)
    if callable(policy) and not isinstance(policy, (pd.DataFrame, pd.Series, np.ndarray)):
        return policy_matrix(policy(frame), data, frame, role)
    if isinstance(policy, str):
        if policy not in frame.columns:
            raise ValueError(f"Политика {role}: в логах нет колонки '{policy}'")
        return _from_actions(frame[policy], data, role)
    if isinstance(policy, pd.Series):
        aligned = _aligned_rows(policy.to_frame(), data.index, role).iloc[:, 0]
        return _from_actions(aligned, data, role)
    if isinstance(policy, pd.DataFrame):
        return _from_frame(policy, data, role)
    if isinstance(policy, np.ndarray):
        return _from_array(policy, data, role)
    raise TypeError(
        f"Не понимаю политику типа {type(policy).__name__}. Ожидается DataFrame вероятностей "
        "(колонки — метки действий), имя колонки, Series действий или объект с action_probs(df)."
    )
