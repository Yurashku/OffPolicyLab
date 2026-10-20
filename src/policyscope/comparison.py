"""
policyscope.comparison
======================

Официальная точка входа: сравнение логирующей политики A и кандидата B.

Один вызов :func:`compare_policies` проходит весь путь — контракт данных, nuisance,
точечная оценка, инференс, диагностика — и возвращает :class:`PolicyComparison`
со **стабильной** схемой полей (ключи всегда присутствуют, отсутствующие значения
равны ``None``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Sequence

import numpy as np
import pandas as pd

from policyscope.data import LoggedBanditDataset
from policyscope.diagnostics import (
    DiagnosticThresholds,
    PolicyDiagnostics,
    compute_policy_diagnostics,
)
from policyscope.estimate import (
    ALL_ESTIMATORS,
    LINEARIZABLE_ESTIMATORS,
    OUTCOME_ESTIMATORS,
    EstimatorName,
    PointEstimate,
    estimate_point,
)
from policyscope.estimators import (
    DEFAULT_NUISANCE_SPEC,
    NuisanceSpec,
    resolve_action_space,
    resolve_feature_cols,
    validate_policy_probs,
)
from policyscope.inference import (
    ComparisonInference,
    infer_comparison_bootstrap,
    infer_comparison_influence,
)
from policyscope.nuisance import (
    CrossFitNuisanceBundle,
    PropensitySource,
    fit_crossfit_nuisance_bundle,
)
from policyscope.nuisance_diagnostics import NuisanceDiagnostics, compute_nuisance_diagnostics

logger = logging.getLogger(__name__)

InferenceMethod = Literal["bootstrap", "influence", "none"]

__all__ = [
    "PolicyComparison",
    "MultiTargetComparison",
    "compare_policies",
    "compare_policies_multi_target",
    "compare_policies_multi_estimator",
]


class _Unset:
    """Сентинел «аргумент не передан», чтобы отличать его от осмысленного ``None``."""

    def __repr__(self) -> str:  # pragma: no cover - отладочное представление
        return "<unset>"


UNSET: Any = _Unset()


@dataclass(frozen=True)
class PolicyComparison:
    """Результат сравнения A vs B со стабильной схемой.

    Все поля присутствуют всегда; недоступные значения равны ``None``. Раньше ключи
    создавались условно, из-за чего набор ключей зависел от оценщика и режима
    инференса, а потребителям приходилось повсюду использовать ``.get()``.
    """

    estimator: str
    target: str
    v_a: float
    v_b: float
    delta: float
    v_a_ci: Optional[tuple[float, float]] = None
    v_b_ci: Optional[tuple[float, float]] = None
    delta_ci: Optional[tuple[float, float]] = None
    p_value: Optional[float] = None
    is_significant: Optional[bool] = None
    ci_excludes_zero: Optional[bool] = None
    significance_rule: Optional[str] = None
    se_delta: Optional[float] = None
    alpha: Optional[float] = None
    n_boot: Optional[int] = None
    n_samples: Optional[int] = None
    n_clusters: Optional[int] = None
    inference_method: Optional[str] = None
    inference_warnings: tuple[str, ...] = field(default_factory=tuple)
    diagnostics: Optional[PolicyDiagnostics] = None
    nuisance_diagnostics: Optional[NuisanceDiagnostics] = None
    propensity_source: Optional[str] = None
    propensity_column: Optional[str] = None
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        """Плоский словарь с вложенными блоками диагностики. Сериализуем в JSON."""
        return {
            "estimator": self.estimator,
            "target": self.target,
            "V_A": self.v_a,
            "V_A_CI": self.v_a_ci,
            "V_B": self.v_b,
            "V_B_CI": self.v_b_ci,
            "Delta": self.delta,
            "Delta_CI": self.delta_ci,
            "p_value": self.p_value,
            "is_significant": self.is_significant,
            "ci_excludes_zero": self.ci_excludes_zero,
            "significance_rule": self.significance_rule,
            "se_delta": self.se_delta,
            "alpha": self.alpha,
            "n_boot": self.n_boot,
            "n_samples": self.n_samples,
            "n_clusters": self.n_clusters,
            "inference_method": self.inference_method,
            "inference_warnings": list(self.inference_warnings),
            "propensity_source": self.propensity_source,
            "propensity_column": self.propensity_column,
            "diagnostics": self.diagnostics.to_dict() if self.diagnostics else {},
            "nuisance_diagnostics": (
                self.nuisance_diagnostics.to_dict() if self.nuisance_diagnostics else {}
            ),
            "warnings": list(self.warnings),
        }

    def to_frame(self) -> pd.DataFrame:
        """Однострочный DataFrame с основными числами (для сборки таблиц)."""
        diag = self.diagnostics.to_dict() if self.diagnostics else {}
        return pd.DataFrame([{
            "estimator": self.estimator,
            "target": self.target,
            "V_A": self.v_a,
            "V_B": self.v_b,
            "Delta": self.delta,
            "Delta_CI_low": self.delta_ci[0] if self.delta_ci else None,
            "Delta_CI_high": self.delta_ci[1] if self.delta_ci else None,
            "p_value": self.p_value,
            "is_significant": self.is_significant,
            "ess_ratio": diag.get("weight_ess_ratio"),
            "weight_mean": diag.get("weight_mean"),
            "weight_p99": diag.get("weight_p99"),
            "replay_overlap": diag.get("replay_overlap"),
            "n_warnings": len(self.warnings),
        }])

    def to_markdown(self, *, metric_name: Optional[str] = None) -> str:
        """Читаемый отчёт с вердиктом доверия (см. :mod:`policyscope.report`)."""
        from policyscope.report import render_summary

        return render_summary(self, metric_name=metric_name)


@dataclass(frozen=True)
class MultiTargetComparison:
    """Результаты по нескольким целевым метрикам."""

    estimator: str
    results: dict[str, PolicyComparison]

    def to_dict(self) -> dict[str, Any]:
        return {
            "estimator": self.estimator,
            "results": {k: v.to_dict() for k, v in self.results.items()},
        }

    def to_frame(self) -> pd.DataFrame:
        if not self.results:
            return pd.DataFrame()
        return pd.concat([r.to_frame() for r in self.results.values()], ignore_index=True)


# ---------------------------------------------------------------------------
# Разрешение контракта данных
# ---------------------------------------------------------------------------


#: Имена, которые почти всегда означают единицу кластеризации.
LIKELY_CLUSTER_COLUMNS = ("user_id", "userid", "user", "customer_id", "session_id", "visitor_id")


def _hint_possible_cluster_column(df: pd.DataFrame, action_col: str, target: str) -> None:
    """Подсказывает про кластеризацию, если данные на неё явно намекают.

    Без кластеризации повторные наблюдения одного пользователя считаются
    независимыми, и доверительный интервал получается заниженным. Молча угадывать
    колонку нельзя, но и промолчать о явном признаке — плохая услуга.
    """
    for name in LIKELY_CLUSTER_COLUMNS:
        if name in df.columns and df[name].nunique() < len(df):
            logger.warning(
                "В данных есть колонка '%s' с повторяющимися значениями, но cluster_col "
                "не задан. Без кластеризации доверительный интервал будет занижен. "
                "Передайте cluster_col='%s', если это единица наблюдения.",
                name, name,
            )
            return
    del action_col, target


@dataclass(frozen=True)
class _ResolvedInputs:
    df: pd.DataFrame
    target: str
    action_col: str
    cluster_col: Optional[str]
    propensity_col: Optional[str]
    feature_cols: list[str]


def _resolve_inputs(
    data: pd.DataFrame | LoggedBanditDataset,
    *,
    target: Any,
    action_col: Any,
    cluster_col: Any,
    propensity_col: Any,
    feature_cols: Any,
) -> _ResolvedInputs:
    """Сливает схему датасета с явными аргументами.

    Правило одно и то же для всех полей: **явный аргумент всегда побеждает схему**.
    Раньше ``target`` и ``action_col`` схемы перезаписывали переданные значения
    безусловно, поэтому ``compare_policies(dataset, policyB, target="cltv")`` молча
    считал метрику из схемы.
    """
    if isinstance(data, LoggedBanditDataset):
        schema = data.schema
        df = data.df
        defaults = {
            "target": schema.reward_col,
            "action_col": schema.action_col,
            "cluster_col": schema.cluster_col,
            "propensity_col": schema.propensity_col,
            "feature_cols": schema.feature_cols,
        }
    else:
        df = data
        # cluster_col по умолчанию None: подставлять имя колонки, которой может не быть,
        # значит ронять простейший вызов. Ниже — подсказка, если колонка похожа на кластер.
        defaults = {
            "target": "accept",
            "action_col": "a_A",
            "cluster_col": None,
            "propensity_col": None,
            "feature_cols": None,
        }

    def pick(value: Any, key: str) -> Any:
        return defaults[key] if isinstance(value, _Unset) else value

    resolved_target = pick(target, "target")
    resolved_action = pick(action_col, "action_col")
    resolved_cluster = pick(cluster_col, "cluster_col")
    resolved_propensity = pick(propensity_col, "propensity_col")
    resolved_features = pick(feature_cols, "feature_cols")

    if resolved_cluster is not None and resolved_cluster not in df.columns:
        raise ValueError(
            f"Колонка кластеров '{resolved_cluster}' отсутствует в данных. "
            "Укажите существующую колонку или cluster_col=None."
        )
    if resolved_cluster is None:
        _hint_possible_cluster_column(df, resolved_action, resolved_target)

    feats = resolve_feature_cols(
        df,
        resolved_features,
        target=resolved_target,
        action_col=resolved_action,
        cluster_col=resolved_cluster,
        propensity_col=resolved_propensity,
    )
    return _ResolvedInputs(
        df=df,
        target=resolved_target,
        action_col=resolved_action,
        cluster_col=resolved_cluster,
        propensity_col=resolved_propensity,
        feature_cols=feats,
    )


#: Порог «дорогого» прогона: число (строк × реплик) для оценщиков с моделью исхода.
#: Каждая реплика переобучает nuisance, поэтому стоимость растёт линейно по обоим.
EXPENSIVE_BOOTSTRAP_BUDGET = 5_000_000


def _warn_if_bootstrap_is_expensive(
    *, n_rows: int, n_boot: int, estimator: str, use_crossfit: bool
) -> None:
    """Предупреждает, если bootstrap с переобучением nuisance займёт очень долго.

    Bootstrap переобучает nuisance-модели внутри каждой реплики — это статистически
    правильно, но на больших данных стоит ощутимого времени. Пользователь должен
    узнать об этом до запуска, а не через двадцать минут ожидания.
    """
    if estimator not in OUTCOME_ESTIMATORS:
        return
    budget = n_rows * n_boot * (5 if use_crossfit else 1)
    if budget <= EXPENSIVE_BOOTSTRAP_BUDGET:
        return
    logger.warning(
        "Bootstrap для '%s' переобучит nuisance-модели %d раз на %d строках — это может "
        "занять минуты или десятки минут. Быстрая альтернатива: inference='influence' "
        "(желательно с use_crossfit=True). Также можно уменьшить n_boot.",
        estimator,
        n_boot * (5 if use_crossfit else 1),
        n_rows,
    )


# ---------------------------------------------------------------------------
# Основная точка входа
# ---------------------------------------------------------------------------


def compare_policies(
    data: pd.DataFrame | LoggedBanditDataset,
    policyB,
    *,
    estimator: EstimatorName = "dr",
    target: Any = UNSET,
    feature_cols: Any = UNSET,
    action_col: Any = UNSET,
    action_space: Optional[Sequence] = None,
    cluster_col: Any = UNSET,
    propensity_col: Any = UNSET,
    propensity_source: PropensitySource = "auto",
    inference: InferenceMethod = "bootstrap",
    n_boot: int = 500,
    alpha: float = 0.05,
    rng_seed: int = 4321,
    weight_clip: Optional[float] = None,
    tau: float = 20.0,
    use_crossfit: bool = False,
    crossfit_n_splits: int = 5,
    crossfit_random_state: int = 123,
    nuisance_bundle: Optional[CrossFitNuisanceBundle] = None,
    nuisance_spec: NuisanceSpec = DEFAULT_NUISANCE_SPEC,
    thresholds: DiagnosticThresholds = DiagnosticThresholds(),
    honest_outcome_metrics: bool = True,
) -> PolicyComparison:
    """Сравнивает логирующую политику A с кандидатом B по логам A.

    Args:
        data: Логи (``DataFrame``) или валидированный ``LoggedBanditDataset``.
        policyB: Кандидат с методом ``action_probs(df) -> (n, k)``. Готовые адаптеры
            для реальных моделей — в :mod:`policyscope.policies`.
        estimator: ``dr`` (по умолчанию), ``ips``, ``snips``, ``dm``, ``sndr``,
            ``switch_dr``, ``on_policy``, ``replay``.
        target: Колонка награды.
        feature_cols: Признаки; по умолчанию автовыбор с исключением служебных колонок.
        action_col: Колонка логированного действия.
        action_space: Порядок действий, соответствующий столбцам политики.
        cluster_col: Колонка кластеров для инференса (``None`` — независимые строки).
        propensity_col: Колонка логированных пропенсити.
        propensity_source: ``auto`` / ``logged`` / ``estimated``.
        inference: ``bootstrap`` (по умолчанию), ``influence`` — быстрый аналитический
            CI, или ``none`` — только точечная оценка и диагностика.
        n_boot: Число bootstrap-реплик.
        alpha: Уровень значимости.
        rng_seed: Зерно генератора для воспроизводимости.
        weight_clip: Верхний порог клиппинга весов важности.
        tau: Порог переключения Switch-DR.
        use_crossfit: Считать nuisance out-of-fold. При ``inference="bootstrap"``
            cross-fitting выполняется **внутри каждой реплики**, поэтому точечная
            оценка и интервал описывают один и тот же оценщик.
        nuisance_bundle: Готовые внешние nuisance-предсказания для исходных строк.
        nuisance_spec: Подключаемые модели nuisance.
        thresholds: Пороги предупреждений диагностики.
        honest_outcome_metrics: Считать метрики качества модели исхода out-of-fold.
            In-sample метрики оптимистичны и потому бесполезны для своей задачи —
            обнаружить слабую модель. Ценой служит дополнительный k-fold fit на
            полных данных (не внутри bootstrap). Отключите на больших данных,
            если диагностика nuisance не нужна: метрики тогда помечаются
            предупреждением `outcome_metrics_in_sample_optimistic`.

    Returns:
        :class:`PolicyComparison`.
    """
    if estimator not in ALL_ESTIMATORS:
        raise ValueError(f"Неизвестный оценщик: {estimator}. Доступны: {list(ALL_ESTIMATORS)}")
    if inference not in ("bootstrap", "influence", "none"):
        raise ValueError(f"Неизвестный метод инференса: {inference}")

    resolved = _resolve_inputs(
        data,
        target=target,
        action_col=action_col,
        cluster_col=cluster_col,
        propensity_col=propensity_col,
        feature_cols=feature_cols,
    )
    df = resolved.df
    probsB = validate_policy_probs(policyB.action_probs(df), len(df))
    actions = resolve_action_space(probsB, action_space)

    if nuisance_bundle is not None and use_crossfit:
        raise ValueError("Укажите либо nuisance_bundle, либо use_crossfit=True, но не оба")
    if (
        nuisance_bundle is not None
        and nuisance_bundle.target is not None
        and nuisance_bundle.target != resolved.target
    ):
        raise ValueError(
            f"nuisance_bundle обучен на цели '{nuisance_bundle.target}', "
            f"а оценка идёт по '{resolved.target}'. Пересоберите bundle."
        )

    if nuisance_bundle is not None and inference == "bootstrap":
        raise ValueError(
            "Внешний nuisance_bundle несовместим с inference='bootstrap': он привязан "
            "к конкретным строкам и не может быть пересчитан на bootstrap-реплике, "
            "поэтому интервал описывал бы другой оценщик, чем точечная оценка. "
            "Используйте inference='influence' (или 'none') либо use_crossfit=True — "
            "cross-fitting выполняется внутри каждой реплики."
        )

    notes: list[str] = []
    if inference == "influence" and estimator not in LINEARIZABLE_ESTIMATORS:
        raise ValueError(
            f"Аналитический CI недоступен для оценщика '{estimator}': "
            "он не является средним построчных вкладов. Используйте inference='bootstrap'."
        )
    if inference == "influence" and estimator == "dm":
        notes.append("influence_ci_anticonservative_for_dm")
    if inference == "influence" and estimator in {"dr", "sndr", "switch_dr"} and not use_crossfit:
        notes.append("influence_ci_ignores_nuisance_uncertainty_without_crossfit")

    def point_on(part: pd.DataFrame, *, is_original: bool) -> PointEstimate:
        behavior = None
        outcome = None
        if is_original and nuisance_bundle is not None:
            behavior, outcome = nuisance_bundle.behavior, nuisance_bundle.outcome
        elif use_crossfit:
            bundle = fit_crossfit_nuisance_bundle(
                part,
                policyB,
                target=resolved.target,
                feature_cols=resolved.feature_cols,
                n_splits=crossfit_n_splits,
                random_state=crossfit_random_state,
                action_col=resolved.action_col,
                action_space=actions,
                cluster_col=resolved.cluster_col,
                spec=nuisance_spec,
                propensity_source=propensity_source,
                propensity_col=resolved.propensity_col,
            )
            behavior, outcome = bundle.behavior, bundle.outcome
        return estimate_point(
            part,
            policyB,
            method=estimator,
            target=resolved.target,
            feature_cols=resolved.feature_cols,
            action_col=resolved.action_col,
            action_space=actions,
            cluster_col=resolved.cluster_col,
            weight_clip=weight_clip,
            tau=tau,
            nuisance_behavior=behavior,
            nuisance_outcome=outcome,
            propensity_source=propensity_source,
            propensity_col=resolved.propensity_col,
            spec=nuisance_spec,
        )

    point = point_on(df, is_original=True)
    v_a = float(df[resolved.target].mean())

    diag = compute_policy_diagnostics(
        df,
        policyB,
        point,
        method=estimator,
        action_col=resolved.action_col,
        action_space=actions,
        cluster_col=resolved.cluster_col,
        thresholds=thresholds,
    )
    nuisance_diag = compute_nuisance_diagnostics(
        df,
        target=resolved.target,
        estimator=estimator,
        action_col=resolved.action_col,
        feature_cols=resolved.feature_cols,
        behavior=point.behavior,
        outcome=point.outcome,
        cluster_col=resolved.cluster_col,
        action_space=actions,
        spec=nuisance_spec,
        honest_outcome_metrics=honest_outcome_metrics,
        n_splits=crossfit_n_splits,
        random_state=crossfit_random_state,
    )

    inf: Optional[ComparisonInference] = None
    if inference == "influence":
        clusters = (
            df[resolved.cluster_col].to_numpy() if resolved.cluster_col is not None else None
        )
        assert point.contributions is not None
        inf = infer_comparison_influence(
            point.contributions,
            df[resolved.target].to_numpy(dtype=float),
            clusters=clusters,
            alpha=alpha,
        )
    elif inference == "bootstrap":
        _warn_if_bootstrap_is_expensive(
            n_rows=len(df), n_boot=n_boot, estimator=estimator, use_crossfit=use_crossfit
        )

        def estimator_pair(part: pd.DataFrame) -> tuple[float, float]:
            return (
                float(part[resolved.target].mean()),
                point_on(part, is_original=False).value,
            )

        inf = infer_comparison_bootstrap(
            df,
            estimator_pair,
            cluster_col=resolved.cluster_col,
            n_boot=n_boot,
            alpha=alpha,
            rng_seed=rng_seed,
            observed=(v_a, point.value),
        )

    all_warnings = list(diag.warnings) + list(nuisance_diag.warnings) + notes
    if inf is not None:
        all_warnings += list(inf.warnings)

    return PolicyComparison(
        estimator=estimator,
        target=resolved.target,
        v_a=v_a,
        v_b=point.value,
        delta=point.value - v_a,
        v_a_ci=inf.v_a_ci.as_tuple() if inf else None,
        v_b_ci=inf.v_b_ci.as_tuple() if inf else None,
        delta_ci=inf.delta_ci.as_tuple() if inf else None,
        p_value=inf.p_value if inf else None,
        is_significant=inf.is_significant if inf else None,
        ci_excludes_zero=inf.ci_excludes_zero if inf else None,
        significance_rule=inf.significance_rule if inf else None,
        se_delta=inf.se_delta if inf else None,
        alpha=alpha if inf else None,
        n_boot=inf.n_boot if inf else None,
        n_samples=len(df),
        n_clusters=diag.n_clusters,
        inference_method=inf.method if inf else "none",
        inference_warnings=tuple(inf.warnings) if inf else (),
        diagnostics=diag,
        nuisance_diagnostics=nuisance_diag,
        propensity_source=point.propensity_source,
        propensity_column=point.propensity_col,
        warnings=tuple(dict.fromkeys(all_warnings)),
    )


def compare_policies_multi_target(
    data: pd.DataFrame | LoggedBanditDataset,
    policyB,
    *,
    targets: Sequence[str],
    **kwargs: Any,
) -> MultiTargetComparison:
    """Повторяет скалярную оценку для каждой целевой метрики.

    ``nuisance_bundle`` здесь запрещён: модель исхода обучается под конкретную
    метрику, и переиспользование одного bundle для другой метрики даёт бессмыслицу.
    Для cross-fitting используйте ``use_crossfit=True`` — он пересобирается на каждую цель.
    """
    if kwargs.get("nuisance_bundle") is not None:
        raise ValueError(
            "nuisance_bundle нельзя переиспользовать между целевыми метриками: "
            "модель исхода обучена под одну метрику. Используйте use_crossfit=True."
        )
    estimator = kwargs.get("estimator", "dr")
    results = {
        t: compare_policies(data, policyB, target=t, **kwargs) for t in targets
    }
    return MultiTargetComparison(estimator=estimator, results=results)


def compare_policies_multi_estimator(
    data: pd.DataFrame | LoggedBanditDataset,
    policyB,
    *,
    estimators: Sequence[EstimatorName] = ("ips", "snips", "dm", "dr", "sndr", "switch_dr"),
    **kwargs: Any,
) -> pd.DataFrame:
    """Считает несколько оценщиков и сводит результат в одну таблицу.

    Согласие оценщиков между собой — самый практичный сигнал доверия в OPE: если
    IPS, DM и DR дают близкие значения, оценка устойчива к выбору метода; если они
    расходятся, ни одному из них верить нельзя, каким бы узким ни был CI.

    ``replay`` в набор по умолчанию не входит: он несмещён только при равномерной
    логирующей политике и не сопоставим с остальными.

    Returns:
        ``DataFrame``, по строке на оценщик.
    """
    kwargs.pop("estimator", None)
    frames = []
    for name in estimators:
        try:
            frames.append(compare_policies(data, policyB, estimator=name, **kwargs).to_frame())
        except ValueError as exc:
            logger.warning("Оценщик '%s' пропущен: %s", name, exc)
            frames.append(pd.DataFrame([{"estimator": name, "error": str(exc)}]))
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    if "Delta" in out.columns:
        spread = out["Delta"].dropna()
        if len(spread) > 1:
            logger.info(
                "Разброс Delta между оценщиками: %.4g .. %.4g",
                float(np.min(spread)),
                float(np.max(spread)),
            )
    return out
