"""Подготовка логов: валидация, кодирование действий и признаков.

Всё, что дальше считает библиотека, работает с numpy-массивами из :class:`LogData`.
Здесь же ловятся ошибки входа — с понятным сообщением, а не NaN в результате.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd

__all__ = ["LogData", "prepare_logs"]

#: Категориальный признак с большим числом уровней в авто-режиме пропускается:
#: one-hot на тысячи уровней раздувает модели и почти всегда означает идентификатор.
MAX_AUTO_LEVELS = 50
#: Даже явно заданный категориальный признак с большим числом уровней — почти всегда ошибка.
MAX_EXPLICIT_LEVELS = 500

#: Имена, которые почти всегда обозначают единицу наблюдения (клиента, сессию).
_CLUSTER_NAMES = ("user_id", "userid", "user", "client_id", "clientid", "client",
                  "customer_id", "customer", "session_id", "visitor_id", "account_id")
_ID_PATTERN = re.compile(r"(^|_)id$", flags=re.IGNORECASE)


@dataclass(frozen=True)
class LogData:
    """Логи политики A в виде массивов, готовых к расчёту.

    Attributes:
        n: Число строк.
        X: Матрица признаков ``(n, p)`` после one-hot кодирования категорий.
        feature_names: Имена столбцов ``X``.
        features: Исходные колонки, использованные как признаки.
        skipped_features: Колонки, пропущенные авто-выбором, с причиной.
        features_auto: Признаки выбраны автоматически.
        actions: Метки действий; позиция в списке — код действия.
        a: Коды логированных действий ``(n,)``.
        rewards: Метрика -> вектор наград ``(n,)``.
        binary: Метрика -> является ли награда бинарной.
        propensity: Логированные пропенсити ``(n,)`` или ``None``.
        clusters: Коды кластеров ``(n,)``; без кластеризации каждая строка — свой кластер.
        n_clusters: Число кластеров.
    """

    n: int
    X: np.ndarray
    feature_names: list[str]
    features: list[str]
    skipped_features: dict[str, str]
    features_auto: bool
    actions: list
    a: np.ndarray
    rewards: dict[str, np.ndarray]
    binary: dict[str, bool]
    propensity: Optional[np.ndarray]
    clusters: np.ndarray
    n_clusters: int
    action_col: str
    reward_cols: list[str]
    propensity_col: Optional[str]
    cluster_col: Optional[str]
    cluster_hint: Optional[str]
    suspicious_features: dict[str, str]
    index: pd.Index = field(repr=False)

    @property
    def k(self) -> int:
        return len(self.actions)

    @property
    def is_clustered(self) -> bool:
        return self.cluster_col is not None

    def action_shares(self) -> np.ndarray:
        return np.bincount(self.a, minlength=self.k) / self.n


def _as_list(value: Union[str, Sequence[str]]) -> list[str]:
    if isinstance(value, str):
        return [value]
    return list(value)


def _require_columns(df: pd.DataFrame, cols: Sequence[str], what: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"В логах нет колонок ({what}): {missing}")


def _sorted_labels(values: pd.Series) -> list:
    # Python-скаляры, а не numpy: иначе в сообщениях и JSON видны np.int64(…).
    labels = [v.item() if isinstance(v, np.generic) else v for v in pd.unique(values)]
    try:
        return sorted(labels)
    except TypeError:  # смешанные типы меток — оставляем порядок появления
        return labels


def _reward_vector(df: pd.DataFrame, col: str) -> np.ndarray:
    s = df[col]
    if not (pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s)):
        raise ValueError(f"Награда '{col}' должна быть числовой или булевой, а не {s.dtype}")
    n_nan = int(s.isna().sum())
    if n_nan:
        raise ValueError(
            f"В награде '{col}' {n_nan} пропусков из {len(s)}. Строки не выбрасываются "
            "молча: пропуск часто означает 0 (например, выручка по непринятому офферу). "
            "Заполните их явно или отфильтруйте логи до оценки."
        )
    r = s.to_numpy(dtype=float)
    if not np.isfinite(r).all():
        raise ValueError(f"В награде '{col}' есть бесконечные значения")
    return r


def _is_binary(r: np.ndarray) -> bool:
    return bool(np.isin(np.unique(r), (0.0, 1.0)).all())


def _propensity_vector(df: pd.DataFrame, col: str) -> np.ndarray:
    s = df[col]
    if not pd.api.types.is_numeric_dtype(s):
        raise ValueError(f"Пропенсити '{col}' должны быть числовыми, а не {s.dtype}")
    n_nan = int(s.isna().sum())
    if n_nan:
        raise ValueError(f"В пропенсити '{col}' {n_nan} пропусков")
    p = s.to_numpy(dtype=float)
    bad = int(((p <= 0.0) | (p > 1.0)).sum())
    if bad:
        raise ValueError(
            f"Пропенсити '{col}' должны лежать в (0, 1]; нарушений: {bad}. Пропенсити — это "
            "вероятность, с которой политика A выбрала залогированное действие."
        )
    return p


def _is_id_like(name: str, s: pd.Series) -> bool:
    if _ID_PATTERN.search(str(name)) or str(name).lower() in _CLUSTER_NAMES:
        return True
    return (
        pd.api.types.is_integer_dtype(s)
        and len(s) > 50
        and s.nunique() == len(s)
    )


def _select_features(
    df: pd.DataFrame, service: set[str]
) -> tuple[list[str], dict[str, str]]:
    chosen: list[str] = []
    skipped: dict[str, str] = {}
    for col in df.columns:
        if col in service:
            continue
        s = df[col]
        if pd.api.types.is_datetime64_any_dtype(s) or pd.api.types.is_timedelta64_dtype(s):
            skipped[col] = "дата/время"
        elif s.isna().any():
            skipped[col] = "есть пропуски"
        elif s.nunique() <= 1:
            skipped[col] = "константа"
        elif _is_id_like(col, s):
            skipped[col] = "похоже на идентификатор"
        elif "propens" in str(col).lower():
            skipped[col] = "похоже на пропенсити: передайте её через propensity="
        elif pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
            chosen.append(col)
        elif s.nunique() > MAX_AUTO_LEVELS:
            skipped[col] = f"категориальная, уровней больше {MAX_AUTO_LEVELS}"
        else:
            chosen.append(col)
    return chosen, skipped


#: Признак, который почти идеально предсказывает награду, почти наверняка известен
#: только после действия (другая метрика, факт отклика): в моделях это утечка ответа.
LEAK_AUC = 0.97
LEAK_CORR = 0.95


def _looks_like_outcome(feature: pd.Series, reward: np.ndarray, binary: bool) -> bool:
    if not (pd.api.types.is_numeric_dtype(feature) or pd.api.types.is_bool_dtype(feature)):
        return False
    x = feature.to_numpy(dtype=float)
    if np.std(x) == 0 or np.std(reward) == 0:
        return False
    if binary:
        pos, neg = x[reward == 1], x[reward == 0]
        ranks = pd.Series(x).rank().to_numpy()
        auc = (ranks[reward == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
        return bool(max(auc, 1 - auc) > LEAK_AUC)
    return bool(abs(np.corrcoef(x, reward)[0, 1]) > LEAK_CORR)


def _outcome_like(
    df: pd.DataFrame, cols: list[str], rewards: dict[str, np.ndarray], binary: dict[str, bool]
) -> dict[str, str]:
    found = {}
    for col in cols:
        for metric, r in rewards.items():
            if _looks_like_outcome(df[col], r, binary[metric]):
                found[col] = metric
                break
    return found


def _encode_features(df: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, list[str]]:
    blocks: list[np.ndarray] = []
    names: list[str] = []
    for col in cols:
        s = df[col]
        if pd.api.types.is_datetime64_any_dtype(s):
            raise ValueError(f"Признак '{col}' — дата/время; преобразуйте его в числовой")
        if pd.api.types.is_bool_dtype(s) or pd.api.types.is_numeric_dtype(s):
            values = s.to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(f"Признак '{col}' содержит бесконечные значения")
            blocks.append(values[:, None])
            names.append(str(col))
        else:
            levels = _sorted_labels(s.astype(object))
            if len(levels) > MAX_EXPLICIT_LEVELS:
                raise ValueError(
                    f"Признак '{col}' категориальный, уровней {len(levels)}: one-hot на столько "
                    "уровней раздует модели и почти наверняка означает идентификатор. Сгруппируйте "
                    "редкие уровни или передайте признак в числовом виде."
                )
            # Через object: у category-колонки могут быть неиспользуемые уровни.
            codes = pd.Categorical(s.astype(object), categories=levels).codes
            onehot = np.zeros((len(s), len(levels)), dtype=float)
            onehot[np.arange(len(s)), codes] = 1.0
            blocks.append(onehot)
            names.extend(f"{col}={level}" for level in levels)
    if not blocks:
        return np.zeros((len(df), 0)), []
    return np.hstack(blocks), names


def _cluster_hint(df: pd.DataFrame, service: set[str]) -> Optional[str]:
    for col in df.columns:
        if col in service:
            continue
        if str(col).lower() in _CLUSTER_NAMES or _ID_PATTERN.search(str(col)):
            if df[col].nunique() < len(df):
                return str(col)
    return None


def prepare_logs(
    df: pd.DataFrame,
    *,
    action: str,
    reward: Union[str, Sequence[str]],
    propensity: Optional[str] = None,
    cluster: Optional[str] = None,
    features: Optional[Sequence[str]] = None,
) -> LogData:
    """Проверяет логи и переводит их в массивы.

    Args:
        df: Логи политики A: строка — одно решение.
        action: Колонка с действием, которое выбрала A (любые метки).
        reward: Колонка награды или список колонок (несколько метрик).
        propensity: Колонка с вероятностью, с которой A выбрала залогированное
            действие. ``None`` — пропенсити будут оценены моделью.
        cluster: Колонка единицы наблюдения (например, клиента), если на неё
            приходится несколько строк. ``None`` — строки независимы.
        features: Признаки контекста. ``None`` — авто-выбор: числовые, булевы и
            категориальные колонки, кроме служебных, идентификаторов, констант и колонок
            с пропусками. Категории кодируются one-hot.

    Raises:
        ValueError: при любой проблеме входа — с объяснением, что исправить.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Логи должны быть pandas.DataFrame, а не {type(df).__name__}")
    if df.empty:
        raise ValueError("Логи пусты")
    if not df.index.is_unique:
        raise ValueError("Индекс логов должен быть уникальным: сделайте df.reset_index(drop=True)")

    rewards = _as_list(reward)
    if not rewards:
        raise ValueError("Укажите хотя бы одну метрику в reward")
    if len(set(rewards)) != len(rewards):
        raise ValueError(f"Метрики в reward повторяются: {rewards}")
    service = {action, *rewards}
    if propensity is not None:
        service.add(propensity)
    if cluster is not None:
        service.add(cluster)
    if action in rewards:
        raise ValueError("Колонка действия не может одновременно быть наградой")

    _require_columns(df, [action], "действие")
    _require_columns(df, rewards, "награда")
    if propensity is not None:
        _require_columns(df, [propensity], "пропенсити")
    if cluster is not None:
        _require_columns(df, [cluster], "кластер")

    n_nan = int(df[action].isna().sum())
    if n_nan:
        raise ValueError(f"В колонке действия '{action}' {n_nan} пропусков")
    actions = _sorted_labels(df[action])
    a = pd.Categorical(df[action].astype(object), categories=actions).codes.astype(int)

    reward_map = {col: _reward_vector(df, col) for col in rewards}
    binary = {col: _is_binary(r) for col, r in reward_map.items()}
    p = _propensity_vector(df, propensity) if propensity is not None else None

    if cluster is not None:
        if df[cluster].isna().any():
            raise ValueError(f"В колонке кластера '{cluster}' есть пропуски")
        clusters, uniques = pd.factorize(df[cluster], sort=False)
        n_clusters = len(uniques)
        hint = None
    else:
        clusters = np.arange(len(df))
        n_clusters = len(df)
        hint = _cluster_hint(df, service)

    suspicious: dict[str, str] = {}
    if features is None:
        feature_cols, skipped = _select_features(df, service)
        for col, metric in _outcome_like(df, feature_cols, reward_map, binary).items():
            feature_cols.remove(col)
            skipped[col] = f"почти идеально предсказывает «{metric}» — похоже на исход действия"
        if not feature_cols:
            raise ValueError(
                "Не найдено ни одного пригодного признака. Передайте features явно "
                "(features=[] — оценка без контекста)."
            )
        auto = True
    else:
        feature_cols = list(features)
        _require_columns(df, feature_cols, "признаки")
        leaked = sorted(set(feature_cols) & service)
        if leaked:
            raise ValueError(
                f"Колонки {leaked} служебные (действие/награда/пропенсити/кластер) и не могут "
                "быть признаками: это утечка ответа в модели."
            )
        with_nan = {c: int(df[c].isna().sum()) for c in feature_cols if df[c].isna().any()}
        if with_nan:
            raise ValueError(f"В признаках есть пропуски: {with_nan}. Заполните их до оценки.")
        skipped = {}
        auto = False
        suspicious = _outcome_like(df, feature_cols, reward_map, binary)

    X, names = _encode_features(df, feature_cols)
    return LogData(
        n=len(df),
        X=X,
        feature_names=names,
        features=[str(c) for c in feature_cols],
        skipped_features=skipped,
        features_auto=auto,
        actions=actions,
        a=a,
        rewards=reward_map,
        binary=binary,
        propensity=p,
        clusters=np.asarray(clusters, dtype=int),
        n_clusters=int(n_clusters),
        action_col=action,
        reward_cols=rewards,
        propensity_col=propensity,
        cluster_col=cluster,
        cluster_hint=hint,
        suspicious_features=suspicious,
        index=df.index.copy(),
    )
