"""
policyscope.report
==================

Рендеринг результата сравнения в читаемый вид.

Главное здесь — **вердикт доверия**: значимый p-value сам по себе ничего не
гарантирует, если данные не поддерживают оценку (нет перекрытия политик, веса
вырождены, модель исхода не работает). Отчёт сводит эти сигналы вместе и
расшифровывает коды предупреждений человеческим текстом.
"""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:  # pragma: no cover
    from policyscope.comparison import PolicyComparison

__all__ = [
    "TRUST_UNRELIABLE",
    "TRUST_CAUTION",
    "TRUST_OK",
    "WARNING_EXPLANATIONS",
    "trust_verdict",
    "render_summary",
    "dump_json",
]

TRUST_OK = "надёжно"
TRUST_CAUTION = "с осторожностью"
TRUST_UNRELIABLE = "ненадёжно"

#: Предупреждения, при которых оценке верить нельзя вообще: нарушены предпосылки,
#: без которых ни один OPE-оценщик не восстановит значение политики из данных.
BLOCKING_WARNINGS = frozenset({
    "policy_b_recommends_unsupported_actions",
    "logged_actions_unknown_to_behavior_model",
    "critically_low_ess",
    "too_many_failed_bootstrap_replicates",
})

#: Предупреждения, требующие внимания, но не отменяющие результат.
CAUTION_WARNINGS = frozenset({
    "low_ess_ratio",
    "low_replay_overlap",
    "extreme_max_weight",
    "heavy_weight_tail",
    "mean_weight_far_from_one",
    "large_clip_share",
    "large_switch_share",
    "few_clusters_inference_unreliable",
    "low_n_boot_ci_tails_unstable",
    "weak_behavior_log_loss",
    "weak_outcome_log_loss",
    "weak_outcome_brier",
    "weak_outcome_auc",
    "weak_outcome_r2",
    "bootstrap_replicates_with_nan_dropped",
    "influence_ci_anticonservative_for_dm",
    "influence_ci_ignores_nuisance_uncertainty_without_crossfit",
    "replay_valid_only_under_uniform_logging",
    "behavior_metrics_in_sample_optimistic",
    "outcome_metrics_in_sample_optimistic",
})

WARNING_EXPLANATIONS: dict[str, str] = {
    "policy_b_recommends_unsupported_actions":
        "Политика B заметную часть массы кладёт на действия, которых политика A почти "
        "не совершала. Данные не содержат информации об этих действиях — оценка "
        "экстраполирует, а не измеряет.",
    "logged_actions_unknown_to_behavior_model":
        "В логах есть действия, неизвестные модели поведения; им подставлено "
        "минимальное пропенсити, что порождает экстремальные веса важности.",
    "low_ess_ratio":
        "Эффективная доля выборки мала: вес сосредоточен в части наблюдений. "
        "Смотрите на абсолютный ESS — при большой выборке это может быть приемлемо.",
    "critically_low_ess":
        "Эффективный размер выборки критически мал: оценка фактически опирается "
        "на несколько наблюдений, и её разброс не контролируется.",
    "low_replay_overlap":
        "Политики A и B редко выбирают одно и то же действие — прямых свидетельств "
        "о поведении B в логах мало.",
    "extreme_max_weight":
        "Есть наблюдение с очень большим весом важности: одна строка способна "
        "сдвинуть всю оценку.",
    "heavy_weight_tail":
        "Тяжёлый хвост весов (99-й процентиль высок): оценка неустойчива, "
        "рассмотрите weight_clip или switch_dr.",
    "mean_weight_far_from_one":
        "Средний вес заметно отличается от 1, хотя для корректных пропенсити "
        "E[w] = 1. Признак несоответствия модели пропенсити или нарушенного перекрытия.",
    "large_clip_share":
        "Обрезана большая доля весов: снижена дисперсия, но добавлено смещение.",
    "large_switch_share":
        "Switch-DR отключил IPS-поправку на большой доле наблюдений — результат "
        "близок к DM и наследует его смещение.",
    "few_clusters_inference_unreliable":
        "Независимых кластеров мало; доверительный интервал оптимистичен.",
    "low_n_boot_ci_tails_unstable":
        "Мало bootstrap-реплик: границы интервала оценены грубо. Увеличьте n_boot.",
    "bootstrap_replicates_with_nan_dropped":
        "Часть bootstrap-реплик не удалось посчитать, они исключены из интервала.",
    "too_many_failed_bootstrap_replicates":
        "Слишком много неудачных bootstrap-реплик — интервал не определён.",
    "weak_behavior_log_loss":
        "Модель поведения почти не лучше равномерного угадывания: оценённые "
        "пропенсити малоинформативны.",
    "weak_outcome_log_loss":
        "Модель исхода почти не лучше константы.",
    "weak_outcome_brier": "Модель исхода плохо откалибрована (высокий Brier score).",
    "weak_outcome_auc": "Модель исхода слабо ранжирует (низкий ROC AUC).",
    "weak_outcome_r2": "Модель исхода объясняет меньше дисперсии, чем константа.",
    "influence_ci_anticonservative_for_dm":
        "Для DM аналитический CI не учитывает ошибку модели исхода и потому слишком "
        "узок. Для честного интервала используйте inference='bootstrap'.",
    "influence_ci_ignores_nuisance_uncertainty_without_crossfit":
        "Аналитический CI не учитывает неопределённость nuisance. Он асимптотически "
        "корректен для DR при use_crossfit=True; иначе интервал может быть занижен.",
    "replay_valid_only_under_uniform_logging":
        "Replay несмещён только при равномерной логирующей политике и не сопоставим "
        "с остальными оценщиками.",
    "behavior_metrics_in_sample_optimistic":
        "Метрики модели поведения посчитаны in-sample и завышены.",
    "outcome_metrics_in_sample_optimistic":
        "Метрики модели исхода посчитаны in-sample и завышены.",
    "estimated_propensity_from_behavior_model":
        "Пропенсити оценены моделью, а не взяты из логов.",
    "logged_propensity_missing_fallback_to_estimated":
        "Колонка логированных пропенсити не задана — использована оценка моделью.",
    "logged_propensity_invalid_fallback_to_estimated":
        "Логированные пропенсити невалидны — использована оценка моделью.",
    "auto_selected_logged_propensity": "Использованы логированные пропенсити из логов.",
    "crossfit_folds_grouped_by_cluster": "Фолды cross-fitting сформированы по кластерам.",
}


def trust_verdict(warnings: tuple[str, ...] | list[str]) -> tuple[str, list[str]]:
    """Возвращает вердикт доверия и список причин.

    Returns:
        ``(вердикт, причины)``, где вердикт — один из :data:`TRUST_OK`,
        :data:`TRUST_CAUTION`, :data:`TRUST_UNRELIABLE`.
    """
    codes = list(dict.fromkeys(warnings))
    blocking = [c for c in codes if c in BLOCKING_WARNINGS]
    caution = [c for c in codes if c in CAUTION_WARNINGS]
    if blocking:
        return TRUST_UNRELIABLE, blocking + caution
    if caution:
        return TRUST_CAUTION, caution
    return TRUST_OK, []


def _fmt(value: Optional[float], digits: int = 6) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "н/д"
    return f"{value:.{digits}f}"


def _fmt_ci(ci: Optional[tuple[float, float]], level: str, digits: int = 6) -> str:
    if ci is None:
        return "(CI не считался)"
    return f"({level} CI: {_fmt(ci[0], digits)} .. {_fmt(ci[1], digits)})"


def render_summary(
    result: "PolicyComparison",
    *,
    metric_name: Optional[str] = None,
    business_threshold: float = 0.0,
) -> str:
    """Формирует текстовый отчёт по результату сравнения.

    Args:
        result: Результат :func:`~policyscope.comparison.compare_policies`.
        metric_name: Человеческое название метрики (по умолчанию — имя колонки).
        business_threshold: Минимальный прирост, который считается значимым для бизнеса.
            Решение принимается по нижней границе CI, а не только по p-value.
    """
    name = metric_name or result.target
    alpha = result.alpha if result.alpha is not None else 0.05
    level = f"{100 * (1 - alpha):.0f}%"
    digits = 6

    header = f"Оценщик: `{result.estimator}`  |  наблюдений: {result.n_samples}"
    if result.n_clusters is not None:
        header += f"  |  кластеров: {result.n_clusters}"

    lines: list[str] = [
        f"## Off-policy оценка: {name}",
        "",
        header,
        "",
        f"- V(A) = {_fmt(result.v_a, digits)} {_fmt_ci(result.v_a_ci, level, digits)}",
        f"- V(B) = {_fmt(result.v_b, digits)} {_fmt_ci(result.v_b_ci, level, digits)}",
        f"- Delta (B − A) = {_fmt(result.delta, digits)} "
        f"{_fmt_ci(result.delta_ci, level, digits)}",
    ]

    if result.p_value is not None:
        lines.append(
            f"- p-value = {_fmt(result.p_value, 4)} "
            f"(значимо: {'да' if result.is_significant else 'нет'}; "
            f"правило: `{result.significance_rule}`)"
        )
        if result.ci_excludes_zero is not None and result.is_significant is not None:
            if result.ci_excludes_zero != result.is_significant:
                lines.append(
                    "- ⚠ p-value и доверительный интервал расходятся: это разные "
                    "процедуры, и на скошенном распределении они могут не совпадать. "
                    "Считайте результат пограничным."
                )
    if result.inference_method:
        lines.append(f"- Метод инференса: `{result.inference_method}`")

    lines.append("")
    lines.append("### Решение")
    if result.delta_ci is None:
        lines.append(
            "Доверительный интервал не считался — статистический вывод сделать нельзя."
        )
    elif result.delta_ci[0] > business_threshold:
        lines.append(
            f"Политика B лучше A: нижняя граница CI превышает порог {business_threshold}."
        )
    elif result.delta_ci[1] < -business_threshold:
        lines.append(
            f"Политика A лучше B: верхняя граница CI ниже −{business_threshold}."
        )
    else:
        lines.append(
            "Значимого отличия не обнаружено либо эффект меньше бизнес-порога."
        )

    verdict, reasons = trust_verdict(result.warnings)
    lines.append("")
    lines.append(f"### Доверие к оценке: **{verdict}**")
    if not reasons:
        lines.append("Проверки перекрытия, весов и качества nuisance-моделей пройдены.")
    else:
        for code in reasons:
            lines.append(f"- {WARNING_EXPLANATIONS.get(code, code)}  \n  `{code}`")

    diag = result.diagnostics.to_dict() if result.diagnostics else {}
    if diag.get("weight_ess_ratio") is not None:
        lines.append("")
        lines.append("### Диагностика")
        lines.append(
            f"- ESS/N = {_fmt(diag.get('weight_ess_ratio'), 3)}  |  "
            f"средний вес = {_fmt(diag.get('weight_mean'), 3)}  |  "
            f"p99 веса = {_fmt(diag.get('weight_p99'), 3)}"
        )
        lines.append(
            f"- Перекрытие действий (replay) = {_fmt(diag.get('replay_overlap'), 3)}  |  "
            f"масса B на неподдержанных действиях = "
            f"{_fmt(diag.get('unsupported_action_mass'), 3)}"
        )
        if result.propensity_source:
            source = result.propensity_source
            col = result.propensity_column
            lines.append(f"- Источник пропенсити: `{source}`" + (f" (`{col}`)" if col else ""))

    return "\n".join(lines)


def dump_json(path: str, obj: Any) -> None:
    """Записывает объект в JSON-файл.

    Принимает как словарь, так и объект с методом ``to_dict``. NaN сериализуются в
    ``null``, чтобы результат оставался валидным JSON.
    """
    if hasattr(obj, "to_dict"):
        obj = obj.to_dict()

    def clean(value: Any) -> Any:
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(v) for v in value]
        return value

    with open(path, "w", encoding="utf-8") as f:
        json.dump(clean(obj), f, ensure_ascii=False, indent=2, allow_nan=False)
