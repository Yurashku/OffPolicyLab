"""Политики как таблицы вероятностей.

Политика в Policyscope — это таблица ``(n, k)``: строка — строка логов, колонка —
действие, значение — вероятность, с которой политика выбрала бы это действие в этом
контексте. Самый надёжный формат — ``DataFrame``, у которого **колонки — метки
действий**: тогда сопоставление колонок с действиями не зависит от их порядка.

Политика вычисляется один раз на всех логах; дальше библиотека индексирует матрицу
сама, поэтому кросс-фиттинг и bootstrap не требуют от политики ничего.

Хелперы :func:`greedy`, :func:`softmax`, :func:`epsilon_greedy` превращают скоры
модели (``DataFrame`` с метками действий в колонках) в такие таблицы. Случайные политики
скоров не требуют: :func:`uniform` выбирает действия поровну, :func:`random_like` — с тем же
балансом, что у наблюдаемой или любой другой политики, но без привязки к клиенту.

:func:`logging_propensity` решает обратную задачу: по известной политике A (или по
нескольким политикам, работавшим в разные периоды) считает пропенсити залогированных
действий точно, без модели.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Union

import numpy as np
import pandas as pd

from policyscope.data import LogData

__all__ = [
    "greedy", "softmax", "epsilon_greedy", "uniform", "random_like", "mixture", "policy_matrix",
    "logging_propensity",
]

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


@dataclass(frozen=True)
class RandomPolicy:
    """Случайная политика: действие не зависит от клиента (см. :func:`uniform`, :func:`random_like`).

    ``mix`` — откуда взять доли действий: ``None`` — поровну среди ``actions`` (или всех
    действий из логов); ``"logs"`` — доли залогированных действий; словарь «действие -> доля»;
    иначе — политика в любом формате кандидата, её доли усредняются по строкам логов.
    ``by`` — колонки логов, внутри групп которых доли считаются отдельно.
    """

    mix: Any = None
    actions: Optional[tuple] = None
    by: Optional[tuple] = None


def uniform(actions: Optional[Sequence] = None) -> RandomPolicy:
    """Случайная политика: каждое действие с равной вероятностью.

    Args:
        actions: Среди каких действий выбирать. ``None`` — среди всех действий, которые
            встречаются в логах.

    Пример: ``ev.evaluate(ps.uniform(), name="случайная")``.
    """
    return RandomPolicy(None, None if actions is None else tuple(actions))


def random_like(source: Any = "logs", by: Optional[Union[str, Sequence[str]]] = None) -> RandomPolicy:
    """Случайная политика с тем же балансом действий, что у ``source``, но без таргетинга.

    Каждому клиенту действие назначается случайно, с долями ``source``: так проверяется,
    даёт ли таргетинг что-то сверх правильного микса действий. Разница
    ``V(политика) − V(random_like(политика))`` — вклад именно выбора «кому что».

    Args:
        source: ``"logs"`` (по умолчанию) — доли действий в логах, то есть баланс
            наблюдаемой политики A; политика в любом формате кандидата (например,
            скоринговая B) — её средние доли по строкам логов; словарь «действие -> доля».
        by: Колонка (или колонки) логов, внутри групп которой баланс сохраняется отдельно.
            ``by="month"`` — случайная политика, у которой доли действий в каждом месяце те же,
            что у ``source`` в этом месяце.

    Пример: ``ev.evaluate(ps.random_like(by="month"), name="случайная с балансом A по месяцам")``.
    """
    if isinstance(source, RandomPolicy):
        return source
    if by is not None and isinstance(source, Mapping):
        raise ValueError("by= сохраняет баланс логов или политики внутри групп; с явными долями он не нужен")
    groups = None if by is None else ((by,) if isinstance(by, str) else tuple(by))
    return RandomPolicy(dict(source) if isinstance(source, Mapping) else source, by=groups)


def _grouped_shares(policy: RandomPolicy, data: LogData, frame: pd.DataFrame, role: str) -> np.ndarray:
    """Доли действий ``policy.mix`` внутри групп ``policy.by``: строка получает доли своей группы."""
    missing = [col for col in policy.by if col not in frame.columns]
    if missing:
        raise ValueError(f"Политика {role}: в логах нет колонок для by= {missing}")
    groups = frame[list(policy.by)].astype(object).apply(tuple, axis=1).to_numpy()
    if isinstance(policy.mix, str) and policy.mix == "logs":
        source = np.zeros((data.n, data.k))
        source[np.arange(data.n), data.a] = 1.0
    else:
        source = policy_matrix(policy.mix, data, frame, role)
    shares = pd.DataFrame(source).groupby(groups).transform("mean").to_numpy()
    return shares


@dataclass(frozen=True)
class MixturePolicy:
    """Смесь политик (см. :func:`mixture`)."""

    policies: tuple
    weights: tuple


def mixture(policies: Any, weights: Optional[Sequence[float]] = None) -> MixturePolicy:
    """Смешивающая политика: для каждого решения случайно выбирается одна из ``policies``.

    Вероятность действия — взвешенное среднее вероятностей политик. Например, смесь
    ежемесячных версий скоринговой модели — это «каждый раз случайная версия»; её
    сравнение с лучшей версией показывает, стоит ли выбирать версию, а не смешивать.

    Args:
        policies: Словарь «имя -> политика» или список политик в любом формате кандидата.
        weights: Доли политик; ``None`` — поровну.
    """
    members = tuple(policies.values()) if isinstance(policies, Mapping) else tuple(policies)
    if not members:
        raise ValueError("Смесь пуста: передайте хотя бы одну политику")
    w = np.full(len(members), 1.0 / len(members)) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != (len(members),) or (w < 0).any() or not np.isclose(w.sum(), 1.0, atol=_ROW_SUM_TOL):
        raise ValueError("weights: по доле на политику, неотрицательные, в сумме 1")
    return MixturePolicy(members, tuple(float(x) for x in w))


def _from_random(policy: RandomPolicy, data: LogData, frame: pd.DataFrame, role: str) -> np.ndarray:
    labels = data.actions if policy.actions is None else list(policy.actions)
    unknown = [label for label in labels if label not in set(data.actions)]
    if unknown:
        raise ValueError(
            f"Политика {role} выбирает действия, которых нет в логах: {unknown[:5]}. "
            "Для них нет данных, и off-policy оценка невозможна."
        )
    mix = policy.mix
    if mix is None:
        if not labels:
            raise ValueError(f"Политике {role} не из чего выбирать: список действий пуст")
        shares = np.array([1.0 / len(labels) if label in set(labels) else 0.0 for label in data.actions])
    elif policy.by is not None:
        return _grouped_shares(policy, data, frame, role)
    elif isinstance(mix, str) and mix == "logs":
        shares = np.bincount(data.a, minlength=data.k) / data.n
    elif isinstance(mix, Mapping):
        extra = [label for label in mix if label not in set(data.actions)]
        if extra:
            raise ValueError(f"Политика {role}: долей для действий не из логов быть не может: {extra[:5]}")
        shares = np.array([float(mix.get(label, 0.0)) for label in data.actions])
        if (shares < 0).any() or not np.isclose(shares.sum(), 1.0, atol=_ROW_SUM_TOL):
            raise ValueError(f"Политика {role}: доли должны быть неотрицательными и в сумме давать 1")
    else:
        shares = policy_matrix(mix, data, frame, role).mean(axis=0)
    return np.tile(shares, (data.n, 1))


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
            - ``ndarray`` — только если действия в логах закодированы числами ``0..k-1``;
            - :func:`uniform` / :func:`random_like` — случайные политики, :func:`mixture` — смесь.
        data: Подготовленные логи.
        frame: Логи без колонок наград и пропенсити — то, что видит политика.
        role: Название политики в сообщениях об ошибках.
    """
    if isinstance(policy, RandomPolicy):
        return _from_random(policy, data, frame, role)
    if isinstance(policy, MixturePolicy):
        return sum(w * policy_matrix(member, data, frame, f"{role}[{i}]")
                   for i, (member, w) in enumerate(zip(policy.policies, policy.weights)))
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


def logging_propensity(
    policy: Any, data: LogData, frame: pd.DataFrame, logger: Optional[pd.Series] = None,
) -> np.ndarray:
    """Пропенсити залогированных действий по известной логирующей политике.

    Нужна, когда вероятности выбора не сохраняли, но правило A известно (скоринговая
    модель плюс способ выбора): тогда пропенсити считаются точно, а не оцениваются.

    Если логи собраны несколькими политиками — например, разными версиями модели в
    разные периоды, — ``policy`` задаётся словарём «метка -> политика», а ``logger``
    указывает, какая из них записала строку. Пропенсити тогда — смесь
    ``p_i = sum_k (n_k / n) * pi_k(a_i | x_i)`` (balanced IPS, Agarwal et al., 2017):
    оценка несмещена, если клиенты и их отклик в разных периодах сопоставимы. Так даже
    детерминированные политики, которые выбирали по-разному, вместе дают исследование.

    Raises:
        ValueError: если заданная политика не могла выбрать залогированное действие —
            значит, политика или логи заданы неверно.
    """
    return _logging_terms(policy, data, frame, logger)[0]


def _logging_terms(
    policy: Any, data: LogData, frame: pd.DataFrame, logger: Optional[pd.Series] = None,
) -> tuple[np.ndarray, dict]:
    """Пропенсити и матрицы логирующих политик (общая часть :func:`logging_propensity`)."""
    rows = np.arange(data.n)
    matrices: dict = {}
    if logger is None:
        if isinstance(policy, Mapping):
            raise ValueError("Для нескольких логирующих политик укажите колонку logger=")
        matrices[None] = policy_matrix(policy, data, frame, role="A")
        p = matrices[None][rows, data.a]
        where = "заданная политика A"
    else:
        if not isinstance(policy, Mapping):
            raise ValueError(
                "logger= задан, значит, логи собраны несколькими политиками: передайте "
                "logging_policy словарём «метка из колонки logger -> политика»."
            )
        labels = logger.to_numpy(dtype=object)
        present = list(dict.fromkeys(labels))
        missing = [label for label in present if label not in policy]
        if missing:
            raise ValueError(f"Нет политики для меток logger: {missing[:5]}")
        unused = [label for label in policy if label not in set(present)]
        if unused:
            raise ValueError(f"Политики {unused[:5]} не записали ни одной строки логов: проверьте метки")
        p = np.zeros(data.n)
        for label in present:
            share = float(np.mean(labels == label))
            matrices[label] = policy_matrix(policy[label], data, frame, role=f"A[{label}]")
            p += share * matrices[label][rows, data.a]
        where = "ни одна из заданных логирующих политик"
    impossible = int(np.sum(p <= 0))
    if impossible:
        raise ValueError(
            f"В {impossible} строках {where} не могла выбрать залогированное действие "
            "(вероятность 0). Значит, политика задана не та, что работала, или в логах есть "
            "строки из другого источника."
        )
    return np.minimum(p, 1.0), matrices
