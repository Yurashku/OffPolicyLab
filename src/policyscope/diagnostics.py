"""Диагностика: можно ли верить оценке.

Доверительный интервал отвечает на вопрос «насколько шумна оценка». Диагностика
отвечает на другой: «имеет ли оценка смысл на этих данных». Узкий интервал на
логах, где политика A никогда не делала того, что предлагает B, остаётся бессмысленным.

Каждая проблема — :class:`Finding` с цифрами и советом. Серьёзность:

- ``block`` — вывод сделать нельзя (нарушены предпосылки);
- ``caution`` — оценке можно верить с оговорками;
- ``info`` — к сведению.

Ключевые проверки — **триангуляция оценщиков**. Разность двух оценщиков равна
среднему построчных величин, у которых известно матожидание:

- ``DR − IPS = mean(dm − w·mu(a_i))`` — ноль при верных пропенсити и *любой*
  модели исхода. Значимое отличие — пропенсити неверны или B уходит туда, где A не
  действует.
- ``DR − DM = mean(w·(r − mu(a_i)))`` — ноль при верной модели исхода и *любых*
  весах. Значимое отличие — модель исхода смещена.

DR смещён, только если неверны обе модели, поэтому провал одной проверки — повод
для осторожности, а двух — для отказа от вывода.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from policyscope.data import LogData
from policyscope.estimators import SELF_NORMALIZED, ess
from policyscope.inference import analytic_summary, cluster_sums
from policyscope.nuisance import Nuisance

__all__ = [
    "Thresholds",
    "Finding",
    "Check",
    "Overlap",
    "overlap_summary",
    "policy_findings",
    "triangulate",
    "trust_level",
    "SEVERITY_ORDER",
]

SEVERITY_ORDER = {"info": 0, "caution": 1, "block": 2}


@dataclass(frozen=True)
class Thresholds:
    """Пороги диагностики. Значения по умолчанию — практические ориентиры.

    Attributes:
        min_ess: Эффективный размер выборки, ниже которого вывод не делается.
        low_ess_ratio: Доля ESS/N, ниже которой — осторожность.
        min_clusters: Число кластеров, ниже которого интервалы ненадёжны.
        support_deficit: Доля массы B вне носителя A, выше которой — осторожность.
        support_deficit_block: То же, выше которой вывод не делается.
        deterministic_exploration: Если A в среднем исследует меньше этого числа
            действий (``mean(1/p)``), она считается детерминированной.
        deterministic_match: При детерминированной A вывод возможен, только если B
            кладёт на действие A не меньше этой доли массы.
        floor_share: Доля строк под полом пропенсити, выше которой — осторожность.
        check_z: Критический уровень для проверок согласия оценщиков.
        material_bias: Расхождение оценщиков существенно, если превышает эту долю
            стандартной ошибки Δ.
        max_variance_share: Доля дисперсии Δ от одного кластера, выше которой — осторожность.
        failed_replicates: Доля неудавшихся bootstrap-реплик, выше которой вывод не делается.
    """

    min_ess: float = 30.0
    low_ess_ratio: float = 0.05
    min_clusters: int = 30
    support_deficit: float = 0.05
    support_deficit_block: float = 0.2
    deterministic_exploration: float = 1.2
    deterministic_match: float = 0.95
    floor_share: float = 0.05
    check_z: float = 2.58
    material_bias: float = 0.5
    max_variance_share: float = 0.5
    failed_replicates: float = 0.1


@dataclass(frozen=True)
class Finding:
    """Одна находка диагностики."""

    code: str
    severity: str
    title: str
    detail: str
    advice: str = ""
    metric: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "code": self.code, "severity": self.severity, "metric": self.metric,
            "title": self.title, "detail": self.detail, "advice": self.advice,
        }


@dataclass(frozen=True)
class Check:
    """Проверка согласия двух оценщиков.

    Attributes:
        name: ``propensity`` (DR − IPS) или ``outcome`` (DR − DM).
        label: Человекочитаемое название пары.
        discrepancy: Разность оценщиков.
        low, high: Границы интервала для разности.
        material: Порог существенности.
        status: ``fail`` — значимое и существенное расхождение; ``ok`` — его нет
            (граница ``low…high`` показывает, какое расхождение данные не исключают);
            ``inconclusive`` — данных для проверки мало (низкий ESS).
    """

    name: str
    label: str
    discrepancy: float
    low: float
    high: float
    material: float
    status: str

    def to_dict(self) -> dict:
        return {
            "name": self.name, "label": self.label, "discrepancy": self.discrepancy,
            "low": self.low, "high": self.high, "material": self.material, "status": self.status,
        }


@dataclass(frozen=True)
class Overlap:
    """Перекрытие политик A и B."""

    n: int
    n_clusters: int
    clustered: bool
    k: int
    ess: float
    ess_ratio: float
    mean_weight: float
    mean_weight_low: float
    mean_weight_high: float
    max_weight: float
    p99_weight: float
    exploration: float
    exploration_estimated: bool
    match: float
    propensity_source: str
    floor_share: Optional[float]
    actions: tuple[dict, ...]

    def to_dict(self) -> dict:
        out = {k: v for k, v in self.__dict__.items() if k != "actions"}
        out["actions"] = [dict(row) for row in self.actions]
        return out


def trust_level(findings) -> str:
    """``reliable`` / ``caution`` / ``unreliable`` по самой серьёзной находке."""
    worst = max((SEVERITY_ORDER[f.severity] for f in findings), default=0)
    return {0: "reliable", 1: "caution", 2: "unreliable"}[worst]


# ---------------------------------------------------------------------------
# Перекрытие и находки уровня политики
# ---------------------------------------------------------------------------


def overlap_summary(
    data: LogData, nuis: Nuisance, pi: np.ndarray, alpha: float
) -> Overlap:
    rows = np.arange(data.n)
    w = pi[rows, data.a] / nuis.p_taken
    mw = analytic_summary(w, data.clusters, data.n_clusters, alpha)
    shares = data.action_shares()
    b_mass = pi.mean(axis=0)
    actions = []
    for code, label in enumerate(data.actions):
        mask = data.a == code
        actions.append({
            "action": label,
            "logged_share": float(shares[code]),
            "policy_share": float(b_mass[code]),
            "mean_weight": float(w[mask].mean()) if mask.any() else float("nan"),
        })
    floor_share = None
    if nuis.source == "estimated":
        assert nuis.p_raw is not None and nuis.floor is not None
        floor_share = float(np.mean(nuis.p_raw < nuis.floor[data.a]))
    ess_value = ess(w)
    return Overlap(
        n=data.n,
        n_clusters=data.n_clusters,
        clustered=data.is_clustered,
        k=data.k,
        ess=ess_value,
        ess_ratio=ess_value / data.n,
        mean_weight=float(mw["estimate"][0]),
        mean_weight_low=float(mw["low"][0]),
        mean_weight_high=float(mw["high"][0]),
        max_weight=float(w.max()),
        p99_weight=float(np.percentile(w, 99)),
        exploration=float(np.mean(1.0 / nuis.p_taken)),
        exploration_estimated=nuis.source == "estimated",
        match=float(np.mean(pi[rows, data.a])),
        propensity_source=nuis.source,
        floor_share=floor_share,
        actions=tuple(actions),
    )


def _propensity_validity(data: LogData, nuis: Nuisance, th: Thresholds) -> list[Finding]:
    """Колонка пропенсити — действительно вероятность выбора действия политикой A?

    Для любой политики A: ``E[1{A=a}/p] = P(pi_A(a|x) > 0) <= 1`` для каждого
    действия ``a``. Превышение — верный признак, что в колонке не то (частая ошибка —
    вероятность отклика ``p_accept`` вместо вероятности действия).
    """
    if nuis.source != "logged":
        return []
    inv = 1.0 / nuis.p_taken
    onehot = np.zeros((data.n, data.k))
    onehot[np.arange(data.n), data.a] = inv
    s = analytic_summary(onehot, data.clusters, data.n_clusters, 0.05)
    lower = s["estimate"] - th.check_z * s["se"]
    bad = [
        (data.actions[j], float(s["estimate"][j]))
        for j in range(data.k) if np.isfinite(lower[j]) and lower[j] > 1.02
    ]
    if not bad:
        return []
    label, value = max(bad, key=lambda item: item[1])
    return [Finding(
        code="propensity_not_probability",
        severity="block",
        title="Колонка пропенсити не похожа на вероятность выбора действия",
        detail=(
            f"Для действия «{label}» среднее 1{{A=a}}/p = {value:.2f}, а для настоящих "
            "пропенсити оно не может превышать 1. Веса важности завышены."
        ),
        advice=(
            f"Проверьте колонку «{data.propensity_col}»: в ней должна быть вероятность, с "
            "которой A выбрала залогированное действие, а не вероятность отклика или скор."
        ),
    )]


def policy_findings(
    data: LogData,
    nuis: Nuisance,
    overlap: Overlap,
    th: Thresholds,
    *,
    role: str,
) -> list[Finding]:
    """Находки, не зависящие от метрики: перекрытие, пропенсити, кластеры."""
    out: list[Finding] = []
    suffix = "" if role == "B" else f" (базовая политика «{role}»)"

    deficit = 1.0 - overlap.mean_weight
    deficit_low = 1.0 - overlap.mean_weight_high
    exploration_word = "оценённое" if overlap.exploration_estimated else "по логам"
    deterministic = overlap.exploration < th.deterministic_exploration or (
        nuis.p_max is not None and float(np.mean(nuis.p_max)) > 0.95
    )
    if deterministic and overlap.match < th.deterministic_match:
        out.append(Finding(
            code="deterministic_logging" if role == "B" else "deterministic_logging_baseline",
            severity="block",
            title="Политика A почти не исследует" + suffix,
            detail=(
                f"A в среднем оставляет шанс {overlap.exploration:.2f} из {overlap.k} действий "
                f"({exploration_word}), а политика совпадает с выбором A лишь в "
                f"{100 * overlap.match:.0f}% случаев. О действиях, которые A не выбирает, в "
                "логах нет информации."
            ),
            advice=(
                "Оценить можно только политику, близкую к A. Для честной офлайн-оценки "
                "добавьте в прод исследование (например, ε-greedy) и логируйте пропенсити."
            ),
        ))
    elif deficit_low > th.support_deficit:
        severity = "block" if deficit > th.support_deficit_block else "caution"
        out.append(Finding(
            code="support_deficit",
            severity=severity,
            title="Политика выбирает действия там, где A их не выбирает" + suffix,
            detail=(
                f"Средний вес важности {overlap.mean_weight:.3f} при ожидаемой 1: около "
                f"{100 * deficit:.0f}% массы политики приходится на пары «контекст–действие», "
                "которых нет в логах. Эту часть оценка не измеряет, а экстраполирует моделью "
                "исхода (IPS её просто теряет)."
            ),
            advice=(
                "Смотрите таблицу перекрытия по действиям; сузьте кандидата до действий, "
                "которые A реально выбирает в этих контекстах, или соберите данные с исследованием."
            ),
        ))
    elif overlap.mean_weight_low - 1.0 > th.support_deficit:
        out.append(Finding(
            code="weights_inflated",
            severity="caution",
            title="Веса важности в среднем завышены" + suffix,
            detail=(
                f"Средний вес {overlap.mean_weight:.3f} значимо больше 1. При верных "
                "пропенсити так не бывает: пропенсити занижены."
            ),
            advice="Проверьте колонку пропенсити или модель пропенсити.",
        ))

    if overlap.ess < th.min_ess:
        out.append(Finding(
            code="critically_low_ess",
            severity="block",
            title="Эффективный размер выборки критически мал" + suffix,
            detail=(
                f"ESS = {overlap.ess:.0f} из {overlap.n} строк: оценка держится на нескольких "
                "наблюдениях с огромными весами, и её разброс не контролируется."
            ),
            advice="Возьмите кандидата ближе к A (например, смешайте его с A) или больше данных.",
        ))
    elif overlap.ess_ratio < th.low_ess_ratio:
        out.append(Finding(
            code="low_ess",
            severity="caution",
            title="Малая эффективная доля выборки" + suffix,
            detail=(
                f"ESS = {overlap.ess:.0f} ({100 * overlap.ess_ratio:.1f}% от {overlap.n}): "
                "вес сосредоточен в малой части наблюдений, интервал может быть неточным."
            ),
            advice="Сверьтесь с SNDR в таблице оценщиков: он устойчивее к тяжёлым весам.",
        ))

    if role != "B":
        return out

    out.extend(_propensity_validity(data, nuis, th))
    if nuis.source == "estimated":
        out.append(Finding(
            code="propensity_estimated",
            severity="caution",
            title="Пропенсити оценены моделью",
            detail=(
                "В логах нет вероятностей выбора действия, они восстановлены моделью "
                "(cross-fit, с полом). Оценка верна, если модель воспроизводит политику A "
                "или если верна модель исхода; проверки согласия ниже показывают, так ли это."
            ),
            advice=(
                "Логируйте вероятность выбранного действия в проде: это делает оценку "
                "точнее и в десятки раз быстрее (аналитический интервал вместо bootstrap)."
            ),
        ))
        if overlap.floor_share is not None and overlap.floor_share > th.floor_share:
            out.append(Finding(
                code="propensity_floor_active",
                severity="caution",
                title="Многие оценённые пропенсити упёрлись в пол",
                detail=(
                    f"У {100 * overlap.floor_share:.0f}% строк оценённое пропенсити ниже пола и "
                    "заменено им. Модель уверена, что A почти не выбирает эти действия."
                ),
                advice="Смотрите чувствительность Δ к полу в разделе метрики.",
            ))
        if nuis.unseen_share > 0:
            out.append(Finding(
                code="rare_actions_in_folds",
                severity="caution",
                title="Редкие действия выпадают из фолдов",
                detail=(
                    f"У {100 * nuis.unseen_share:.2f}% строк действие не встретилось в обучающем "
                    "фолде модели пропенсити, им назначен пол."
                ),
                advice="Уменьшите n_folds или объедините редкие действия.",
            ))
    if data.is_clustered and data.n_clusters < th.min_clusters:
        out.append(Finding(
            code="few_clusters",
            severity="caution",
            title="Мало независимых кластеров",
            detail=f"Кластеров всего {data.n_clusters}: интервалы приближённые.",
            advice="Соберите логи по большему числу клиентов.",
        ))
    if data.cluster_hint is not None:
        out.append(Finding(
            code="cluster_not_set",
            severity="caution",
            title="Повторяющиеся клиенты, но кластер не указан",
            detail=(
                f"В колонке «{data.cluster_hint}» значения повторяются. Без кластеризации "
                "строки одного клиента считаются независимыми, и интервал занижен."
            ),
            advice=f"Передайте cluster='{data.cluster_hint}', если это единица наблюдения.",
        ))
    if data.suspicious_features:
        listed = ", ".join(f"{c} → {m}" for c, m in data.suspicious_features.items())
        out.append(Finding(
            code="outcome_like_features",
            severity="caution",
            title="Признаки подозрительно хорошо предсказывают награду",
            detail=(
                f"{listed}: такая связь типична для колонок, которые становятся известны "
                "только после действия (отклик, другая метрика). В моделях это утечка ответа."
            ),
            advice="Уберите эти колонки из features, если они не известны до выбора действия.",
        ))
    if data.features_auto:
        skipped = ", ".join(f"{c} ({why})" for c, why in data.skipped_features.items())
        out.append(Finding(
            code="auto_features",
            severity="info",
            title="Признаки выбраны автоматически",
            detail=(
                f"Использованы: {', '.join(data.features)}."
                + (f" Пропущены: {skipped}." if skipped else "")
            ),
            advice=(
                "Убедитесь, что среди признаков нет колонок, известных только после "
                "действия (отклик, дата контакта и т.п.), иначе передайте features явно."
            ),
        ))
    return out


# ---------------------------------------------------------------------------
# Триангуляция
# ---------------------------------------------------------------------------


def _check(
    name: str, label: str, values: np.ndarray, data: LogData, material: float,
    th: Thresholds, informative: bool,
) -> Check:
    s = analytic_summary(values, data.clusters, data.n_clusters, 0.05)
    est, se = float(s["estimate"][0]), float(s["se"][0])
    low, high = est - th.check_z * se, est + th.check_z * se
    if not informative or not np.isfinite(se):
        status = "inconclusive"
    elif (low > 0 or high < 0) and abs(est) > material:
        status = "fail"
    else:
        status = "ok"  # в отчёте указывается граница, которую данные не исключают
    return Check(name, label, est, low, high, material, status)


def triangulate(
    data: LogData,
    metric: str,
    *,
    estimator: str,
    w: np.ndarray,
    dm: np.ndarray,
    mu_taken: np.ndarray,
    se_delta: float,
    estimated_propensity: bool,
    ess_value: float,
    th: Thresholds,
) -> tuple[tuple[Check, ...], list[Finding]]:
    """Проверки согласия оценщиков для одной метрики и находки по ним."""
    r = data.rewards[metric]
    weights = w / np.mean(w) if estimator in SELF_NORMALIZED and np.mean(w) > 0 else w
    sn = estimator in SELF_NORMALIZED
    material = th.material_bias * se_delta if np.isfinite(se_delta) else np.inf
    informative = ess_value >= th.min_ess
    prop = _check(
        "propensity", "SNDR − SNIPS" if sn else "DR − IPS",
        dm - weights * mu_taken, data, material, th, informative,
    )
    outc = _check(
        "outcome", "SNDR − DM" if sn else "DR − DM",
        weights * (r - mu_taken), data, material, th, informative,
    )

    findings: list[Finding] = []
    prop_fail, out_fail = prop.status == "fail", outc.status == "fail"
    if estimator in ("ips", "snips"):
        if prop_fail:
            findings.append(Finding(
                code="propensity_check_failed",
                severity="block",
                metric=metric,
                title="IPS не согласуется с DR: пропенсити неверны",
                detail=(
                    f"{prop.label} = {prop.discrepancy:+.4g} (интервал {prop.low:+.4g}…{prop.high:+.4g}), "
                    "а при верных пропенсити эта разность в среднем равна нулю."
                ),
                advice="Используйте estimator='dr' и проверьте источник пропенсити.",
            ))
        return (prop, outc), findings

    if prop_fail and out_fail:
        findings.append(Finding(
            code="both_models_rejected",
            severity="block",
            metric=metric,
            title="Неверны и пропенсити, и модель исхода",
            detail=(
                f"{prop.label} = {prop.discrepancy:+.4g}, {outc.label} = {outc.discrepancy:+.4g}: "
                "обе разности значимо и существенно отличаются от нуля. DR защищён, только "
                "пока верна хотя бы одна из моделей."
            ),
            advice=(
                "Улучшите модель исхода (outcome_model=...) или получите логированные пропенсити."
            ),
        ))
    elif prop_fail:
        findings.append(Finding(
            code="propensity_check_failed",
            severity="caution",
            metric=metric,
            title="Пропенсити не согласуются с данными",
            detail=(
                f"{prop.label} = {prop.discrepancy:+.4g} (интервал {prop.low:+.4g}…{prop.high:+.4g}). "
                "Оценка DR держится на модели исхода"
                + ("; её проверка расхождений не выявила." if outc.status == "ok"
                   else ", а её проверить не удалось.")
            ),
            advice="Перепроверьте источник пропенсити; сравните DR с DM в таблице оценщиков.",
        ))
    elif out_fail:
        findings.append(Finding(
            code="outcome_check_failed",
            severity="caution" if estimated_propensity else "info",
            metric=metric,
            title="Модель исхода смещена",
            detail=(
                f"{outc.label} = {outc.discrepancy:+.4g} (интервал {outc.low:+.4g}…{outc.high:+.4g}): "
                "DM смещён. DR это исправляет за счёт весов"
                + (", но веса здесь сами оценены." if estimated_propensity else ", пропенсити верны.")
            ),
            advice="Можно улучшить модель исхода (outcome_model=...) — это сузит интервал DR.",
        ))
    return (prop, outc), findings


def variance_concentration(
    psi_delta: np.ndarray, data: LogData, th: Thresholds, metric: str
) -> Optional[Finding]:
    """Не держится ли разброс Δ на одном клиенте (строке)."""
    sums = cluster_sums(psi_delta, data.clusters, data.n_clusters)[:, 0]
    total = float(np.sum(sums**2))
    if total <= 0:
        return None
    share = float(np.max(sums**2)) / total
    if share <= th.max_variance_share:
        return None
    unit = "Один клиент" if data.is_clustered else "Одна строка"
    return Finding(
        code="variance_concentration",
        severity="caution",
        metric=metric,
        title="Разброс оценки держится на одном наблюдении",
        detail=f"{unit} даёт {100 * share:.0f}% дисперсии Δ: интервал неустойчив.",
        advice="Посмотрите на строки с максимальными весами; SNDR устойчивее к таким выбросам.",
    )
