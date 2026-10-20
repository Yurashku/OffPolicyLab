"""Слой nuisance-моделей: behavior ``pi_A(a|x)`` и outcome ``mu(x, a)``.

Здесь живут:
- контейнеры предсказаний (``BehaviorPredictions``, ``OutcomePredictions``);
- обучение nuisance «в лоб» (in-sample) и out-of-fold (cross-fitting);
- разрешение источника пропенсити: логированная колонка или оценка моделью.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from policyscope.estimators import (
    DEFAULT_NUISANCE_SPEC,
    NuisanceSpec,
    mu_hat_predict,
    pi_hat_predict,
    prepare_piB_taken,
    resolve_action_space,
    take_action_probabilities_with_fallback,
    train_mu_hat,
    train_pi_hat,
    validate_policy_probs,
)

logger = logging.getLogger(__name__)

PropensitySource = Literal["auto", "logged", "estimated"]

__all__ = [
    "PropensitySource",
    "BehaviorPredictions",
    "OutcomePredictions",
    "CrossFitNuisanceBundle",
    "BehaviorNuisanceBundle",
    "OutcomeNuisanceBundle",
    "fit_behavior_nuisance_bundle",
    "fit_outcome_nuisance_bundle",
    "make_kfold_indices",
    "generate_oof_behavior_predictions",
    "generate_oof_outcome_predictions",
    "fit_crossfit_nuisance_bundle",
    "build_logged_behavior_predictions",
    "resolve_behavior_predictions",
    "validate_behavior_predictions",
    "validate_outcome_predictions",
]


@dataclass(frozen=True)
class BehaviorPredictions:
    """Предсказания behavior-стороны, выровненные по строкам ``df``.

    Attributes:
        pA_taken: ``pi_A(a_logged|x)`` для каждой строки.
        piB_taken: ``pi_B(a_logged|x)`` для каждой строки.
        pA_all: Полная матрица ``(n, k)``, если доступна (только для оценённого пути).
        pA_all_actions: Порядок действий, соответствующий столбцам ``pA_all``.
        unsupported_action_share: Доля строк, где логированное действие оказалось
            неизвестно behavior-модели и получило подставное значение пропенсити.
            Такие строки дают веса порядка ``1/eps`` и должны быть видны в отчёте.
    """

    pA_taken: np.ndarray
    piB_taken: np.ndarray
    pA_all: Optional[np.ndarray] = None
    pA_all_actions: Optional[tuple] = None
    is_out_of_fold: bool = False
    fold_index: Optional[np.ndarray] = None
    propensity_source: Optional[str] = None
    propensity_col: Optional[str] = None
    unsupported_action_share: float = 0.0


@dataclass(frozen=True)
class OutcomePredictions:
    """Предсказания модели исхода, выровненные по строкам ``df``."""

    mu_logged_action: np.ndarray
    mu_by_action: Optional[dict[object, np.ndarray]] = None
    target: Optional[str] = None
    is_out_of_fold: bool = False
    fold_index: Optional[np.ndarray] = None


@dataclass(frozen=True)
class CrossFitNuisanceBundle:
    """Контейнер out-of-fold nuisance-предсказаний для cross-fitting."""

    behavior: Optional[BehaviorPredictions] = None
    outcome: Optional[OutcomePredictions] = None
    n_splits: Optional[int] = None
    target: Optional[str] = None
    notes: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class BehaviorNuisanceBundle:
    """Обученная behavior-модель вместе с её предсказаниями."""

    pi_model: object
    predictions: BehaviorPredictions


@dataclass(frozen=True)
class OutcomeNuisanceBundle:
    """Обученная модель исхода для DM/DR-семейства."""

    mu_model: object


def _validate_length(name: str, values: np.ndarray, n: int) -> None:
    if len(values) != n:
        raise ValueError(f"{name} должен иметь длину {n}, получено {len(values)}")


# ---------------------------------------------------------------------------
# In-sample nuisance
# ---------------------------------------------------------------------------


def fit_behavior_nuisance_bundle(
    df: pd.DataFrame,
    policyB,
    *,
    feature_cols: Sequence[str],
    action_col: str = "a_A",
    action_space: Optional[Sequence] = None,
    spec: NuisanceSpec = DEFAULT_NUISANCE_SPEC,
) -> BehaviorNuisanceBundle:
    """Обучает behavior-модель и выводит переиспользуемые векторы пропенсити."""
    pi_model = train_pi_hat(df, feature_cols=feature_cols, action_col=action_col, spec=spec)
    pA_all = pi_hat_predict(pi_model, df)
    pA_taken, missing_share = take_action_probabilities_with_fallback(
        pA_all,
        df[action_col].to_numpy(),
        action_space=pi_model.classes_,
    )
    piB_taken = prepare_piB_taken(df, policyB, action_col=action_col, action_space=action_space)
    preds = BehaviorPredictions(
        pA_taken=pA_taken,
        piB_taken=piB_taken,
        pA_all=pA_all,
        pA_all_actions=tuple(pi_model.classes_),
        propensity_source="estimated",
        unsupported_action_share=missing_share,
    )
    return BehaviorNuisanceBundle(pi_model=pi_model, predictions=preds)


def fit_outcome_nuisance_bundle(
    df: pd.DataFrame,
    *,
    target: str,
    feature_cols: Sequence[str],
    action_col: str = "a_A",
    action_space: Optional[Sequence] = None,
    spec: NuisanceSpec = DEFAULT_NUISANCE_SPEC,
) -> OutcomeNuisanceBundle:
    """Обучает модель исхода для DM/DR-семейства."""
    mu_model = train_mu_hat(
        df,
        target=target,
        feature_cols=feature_cols,
        action_col=action_col,
        action_space=action_space,
        spec=spec,
    )
    return OutcomeNuisanceBundle(mu_model=mu_model)


# ---------------------------------------------------------------------------
# Cross-fitting
# ---------------------------------------------------------------------------


def make_kfold_indices(
    n_samples: int,
    *,
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int = 123,
    groups: Optional[Sequence] = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Строит train/holdout индексы для cross-fitting.

    Если передан ``groups`` (обычно ``user_id``), фолды формируются **по группам**:
    все строки одного пользователя попадают в один и тот же holdout. Без этого
    строки одного пользователя оказываются и в train, и в holdout, и cross-fitting
    теряет смысл — ровно та утечка, от которой он должен защищать.
    """
    if n_splits < 2:
        raise ValueError("n_splits должен быть не меньше 2")
    if groups is None:
        kf = KFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state if shuffle else None,
        )
        return list(kf.split(np.arange(n_samples)))

    groups_arr = np.asarray(groups)
    if len(groups_arr) != n_samples:
        raise ValueError(f"groups должен иметь длину {n_samples}, получено {len(groups_arr)}")
    unique_groups = pd.unique(pd.Series(groups_arr))
    if len(unique_groups) < n_splits:
        raise ValueError(
            f"Групп ({len(unique_groups)}) меньше, чем фолдов ({n_splits}). "
            "Уменьшите crossfit_n_splits или отключите кластеризацию."
        )
    order = np.arange(len(unique_groups))
    if shuffle:
        order = np.random.default_rng(random_state).permutation(order)
    fold_of_group = {unique_groups[g]: i % n_splits for i, g in enumerate(order)}
    fold_of_row = np.array([fold_of_group[g] for g in groups_arr], dtype=int)

    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for f in range(n_splits):
        holdout = np.flatnonzero(fold_of_row == f)
        train = np.flatnonzero(fold_of_row != f)
        if holdout.size == 0 or train.size == 0:
            raise ValueError(f"Фолд {f} пуст: недостаточно данных для cross-fitting")
        folds.append((train, holdout))
    return folds


def generate_oof_behavior_predictions(
    df: pd.DataFrame,
    policyB,
    *,
    feature_cols: Sequence[str],
    n_splits: int = 5,
    random_state: int = 123,
    action_col: str = "a_A",
    action_space: Optional[Sequence] = None,
    cluster_col: Optional[str] = None,
    spec: NuisanceSpec = DEFAULT_NUISANCE_SPEC,
) -> BehaviorPredictions:
    """Считает out-of-fold behavior-предсказания, выровненные по исходным строкам."""
    n = len(df)
    pA_taken = np.full(n, np.nan, dtype=float)
    piB_taken = np.full(n, np.nan, dtype=float)
    fold_index = np.full(n, -1, dtype=int)
    groups = (
        df[cluster_col].to_numpy()
        if cluster_col is not None and cluster_col in df.columns
        else None
    )
    total_missing = 0.0

    folds = make_kfold_indices(n, n_splits=n_splits, random_state=random_state, groups=groups)
    for fold_id, (train_idx, holdout_idx) in enumerate(folds):
        train_df = df.iloc[train_idx]
        holdout_df = df.iloc[holdout_idx]

        pi_model = train_pi_hat(
            train_df, feature_cols=feature_cols, action_col=action_col, spec=spec
        )
        pA_all_holdout = pi_hat_predict(pi_model, holdout_df)
        pA_taken_holdout, missing_share = take_action_probabilities_with_fallback(
            pA_all_holdout,
            holdout_df[action_col].to_numpy(),
            action_space=pi_model.classes_,
        )
        total_missing += missing_share * len(holdout_idx)

        pA_taken[holdout_idx] = pA_taken_holdout
        piB_taken[holdout_idx] = prepare_piB_taken(
            holdout_df, policyB, action_col=action_col, action_space=action_space
        )
        fold_index[holdout_idx] = fold_id

    if np.isnan(pA_taken).any():
        raise ValueError("Cross-fitting не покрыл все строки: проверьте фолды")

    return BehaviorPredictions(
        pA_taken=pA_taken,
        piB_taken=piB_taken,
        pA_all=None,
        is_out_of_fold=True,
        fold_index=fold_index,
        propensity_source="estimated",
        unsupported_action_share=float(total_missing) / max(n, 1),
    )


def generate_oof_outcome_predictions(
    df: pd.DataFrame,
    *,
    target: str,
    feature_cols: Sequence[str],
    n_splits: int = 5,
    random_state: int = 123,
    action_col: str = "a_A",
    action_space: Optional[Sequence] = None,
    requested_actions: Optional[Sequence] = None,
    cluster_col: Optional[str] = None,
    spec: NuisanceSpec = DEFAULT_NUISANCE_SPEC,
) -> OutcomePredictions:
    """Считает out-of-fold предсказания исхода для логированного и запрошенных действий."""
    n = len(df)
    mu_logged = np.full(n, np.nan, dtype=float)
    fold_index = np.full(n, -1, dtype=int)
    mu_by_action: dict[object, np.ndarray] = {}
    if requested_actions is not None:
        mu_by_action = {a: np.full(n, np.nan, dtype=float) for a in requested_actions}
    groups = (
        df[cluster_col].to_numpy()
        if cluster_col is not None and cluster_col in df.columns
        else None
    )

    folds = make_kfold_indices(n, n_splits=n_splits, random_state=random_state, groups=groups)
    for fold_id, (train_idx, holdout_idx) in enumerate(folds):
        train_df = df.iloc[train_idx]
        holdout_df = df.iloc[holdout_idx]
        mu_model = train_mu_hat(
            train_df,
            target=target,
            feature_cols=feature_cols,
            action_col=action_col,
            action_space=action_space,
            spec=spec,
        )

        mu_logged[holdout_idx] = mu_hat_predict(
            mu_model, holdout_df, holdout_df[action_col].to_numpy()
        )
        for action, values in mu_by_action.items():
            values[holdout_idx] = mu_hat_predict(
                mu_model, holdout_df, np.full(len(holdout_df), action, dtype=object)
            )
        fold_index[holdout_idx] = fold_id

    if np.isnan(mu_logged).any():
        raise ValueError("Cross-fitting не покрыл все строки: проверьте фолды")

    return OutcomePredictions(
        mu_logged_action=mu_logged,
        mu_by_action=mu_by_action or None,
        target=target,
        is_out_of_fold=True,
        fold_index=fold_index,
    )


def fit_crossfit_nuisance_bundle(
    df: pd.DataFrame,
    policyB,
    *,
    target: str,
    feature_cols: Sequence[str],
    n_splits: int = 5,
    random_state: int = 123,
    action_col: str = "a_A",
    action_space: Optional[Sequence] = None,
    cluster_col: Optional[str] = None,
    spec: NuisanceSpec = DEFAULT_NUISANCE_SPEC,
    propensity_source: PropensitySource = "auto",
    propensity_col: Optional[str] = None,
) -> CrossFitNuisanceBundle:
    """Собирает OOF-предсказания обеих nuisance-сторон в один bundle.

    Behavior и outcome используют один ``random_state``, то есть одно и то же
    разбиение на фолды — это стандартное соглашение DML.

    Если пропенсити берутся из логов, behavior-сторона **не** пересчитывается по
    фолдам: логированные пропенсити известны точно, не оцениваются по данным и
    потому не подвержены переобучению, которое лечит cross-fitting. Подменять их
    модельной оценкой значило бы ухудшить точную величину.
    """
    probsB = validate_policy_probs(policyB.action_probs(df), len(df))
    actions = resolve_action_space(probsB, action_space)

    use_logged = propensity_source == "logged" or (
        propensity_source == "auto" and propensity_col is not None and propensity_col in df.columns
    )
    if use_logged and propensity_col is not None:
        behavior = build_logged_behavior_predictions(
            df,
            policyB,
            propensity_col=propensity_col,
            action_col=action_col,
            action_space=action_space,
        )
    else:
        behavior = generate_oof_behavior_predictions(
            df,
            policyB,
            feature_cols=feature_cols,
            n_splits=n_splits,
            random_state=random_state,
            action_col=action_col,
            action_space=action_space,
            cluster_col=cluster_col,
            spec=spec,
        )
    outcome = generate_oof_outcome_predictions(
        df,
        target=target,
        feature_cols=feature_cols,
        n_splits=n_splits,
        random_state=random_state,
        action_col=action_col,
        action_space=actions,
        requested_actions=actions,
        cluster_col=cluster_col,
        spec=spec,
    )
    notes = ("oof_crossfit_bundle",)
    if cluster_col is not None:
        notes = notes + ("crossfit_folds_grouped_by_cluster",)
    if use_logged:
        notes = notes + ("crossfit_skipped_for_logged_propensity",)
    return CrossFitNuisanceBundle(
        behavior=behavior,
        outcome=outcome,
        n_splits=n_splits,
        target=target,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Источник пропенсити
# ---------------------------------------------------------------------------


def _validate_logged_propensity_values(p: pd.Series, col: str) -> None:
    if p.isna().any():
        raise ValueError(f"Колонка пропенсити '{col}' содержит NaN")
    if (~((p > 0.0) & (p <= 1.0))).any():
        raise ValueError(f"Колонка пропенсити '{col}' должна лежать в интервале (0, 1]")


def build_logged_behavior_predictions(
    df: pd.DataFrame,
    policyB,
    *,
    propensity_col: str,
    action_col: str = "a_A",
    action_space: Optional[Sequence] = None,
) -> BehaviorPredictions:
    """Строит behavior-предсказания из логированной колонки пропенсити."""
    if propensity_col not in df.columns:
        raise ValueError(f"В данных нет колонки пропенсити: {propensity_col}")
    _validate_logged_propensity_values(df[propensity_col], propensity_col)
    piB_taken = prepare_piB_taken(df, policyB, action_col=action_col, action_space=action_space)
    return BehaviorPredictions(
        pA_taken=df[propensity_col].to_numpy(dtype=float),
        piB_taken=piB_taken,
        propensity_source="logged",
        propensity_col=propensity_col,
    )


def resolve_behavior_predictions(
    df: pd.DataFrame,
    policyB,
    *,
    feature_cols: Sequence[str],
    propensity_source: PropensitySource = "auto",
    propensity_col: Optional[str] = None,
    action_col: str = "a_A",
    action_space: Optional[Sequence] = None,
    spec: NuisanceSpec = DEFAULT_NUISANCE_SPEC,
) -> tuple[BehaviorPredictions, str, Optional[str], tuple[str, ...]]:
    """Разрешает источник пропенсити: логированная колонка или оценка моделью.

    Returns:
        ``(предсказания, источник, колонка пропенсити, заметки)``.
    """
    notes: list[str] = []

    if propensity_source == "logged":
        if propensity_col is None:
            raise ValueError("propensity_source='logged' требует указать propensity_col")
        preds = build_logged_behavior_predictions(
            df, policyB, propensity_col=propensity_col,
            action_col=action_col, action_space=action_space,
        )
        return preds, "logged", propensity_col, tuple(notes)

    if propensity_source == "estimated":
        bundle = fit_behavior_nuisance_bundle(
            df, policyB, feature_cols=feature_cols,
            action_col=action_col, action_space=action_space, spec=spec,
        )
        notes.append("estimated_propensity_from_behavior_model")
        return bundle.predictions, "estimated", None, tuple(notes)

    # auto: логированные пропенсити, если валидная колонка доступна
    if propensity_col is not None and propensity_col in df.columns:
        try:
            preds = build_logged_behavior_predictions(
                df, policyB, propensity_col=propensity_col,
                action_col=action_col, action_space=action_space,
            )
            notes.append("auto_selected_logged_propensity")
            return preds, "logged", propensity_col, tuple(notes)
        except ValueError as exc:
            logger.warning("Логированные пропенсити невалидны (%s), переключаюсь на оценку", exc)
            notes.append("logged_propensity_invalid_fallback_to_estimated")
    else:
        notes.append("logged_propensity_missing_fallback_to_estimated")

    bundle = fit_behavior_nuisance_bundle(
        df, policyB, feature_cols=feature_cols,
        action_col=action_col, action_space=action_space, spec=spec,
    )
    return bundle.predictions, "estimated", None, tuple(notes)


# ---------------------------------------------------------------------------
# Валидация внешних предсказаний
# ---------------------------------------------------------------------------


def validate_behavior_predictions(predictions: BehaviorPredictions, n: int) -> None:
    """Проверяет, что готовые behavior-предсказания согласованы с длиной данных."""
    _validate_length("pA_taken", predictions.pA_taken, n)
    _validate_length("piB_taken", predictions.piB_taken, n)


def validate_outcome_predictions(
    predictions: OutcomePredictions,
    n: int,
    *,
    required_actions: Optional[Sequence] = None,
) -> None:
    """Проверяет, что готовые предсказания исхода согласованы с длиной данных."""
    _validate_length("mu_logged_action", predictions.mu_logged_action, n)
    if required_actions is None:
        return
    if predictions.mu_by_action is None:
        raise ValueError("Для DM/DR-семейства в cross-fitting нужны предсказания mu_by_action")
    for action in required_actions:
        if action not in predictions.mu_by_action:
            raise ValueError(f"В предсказаниях исхода нет действия {action}")
        _validate_length(f"mu_by_action[{action}]", predictions.mu_by_action[action], n)
