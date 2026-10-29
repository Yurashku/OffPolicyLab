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
- ``DR − DM = mean((w − 1)·(r − mu(a_i)))`` (для Δ: у DM база — прогноз модели на
  залогированных действиях) — ноль при верной модели исхода и *любых* весах. Значимое
  отличие — модель исхода смещена.

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
            кладёт на действие A не меньше этой доли массы. Оба порога применяются, только
            когда логирующая политика не задана (пропенсити из колонки или оценены). При
            ``logging_policy=`` находка строится по точной доле решений вне носителя с порогами
            ``support_deficit`` / ``support_deficit_block``, а ``deterministic_exploration``
            лишь выбирает код находки.
        floor_share: Доля строк под полом пропенсити, выше которой — осторожность.
        check_z: Критический уровень для проверок согласия оценщиков.
        material_bias: Расхождение оценщиков существенно, если превышает эту долю
            стандартной ошибки Δ. Тот же порог — для плацебо-проверки, в стандартных
            отклонениях плацебо-показателя и метрики.
        max_variance_share: Доля дисперсии Δ от одного кластера, выше которой — осторожность.
        failed_replicates: Доля неудавшихся bootstrap-реплик, выше которой вывод не делается.
        logger_shift_auc: При логах нескольких политик — ROC AUC, с которым период угадывается
            по признакам клиента; выше — составы клиентов в периодах различаются (осторожность).
        logger_drift_p: При логах нескольких политик — p-value различия отклика между периодами
            (на строках, где все политики согласны), ниже которого — осторожность.
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
    logger_shift_auc: float = 0.6
    logger_drift_p: float = 0.001


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
class PlaceboCheck:
    """Плацебо-проверка: «эффект» политики на показателе, известном до решения.

    Решение не может изменить то, что было до него, поэтому при верных пропенсити и без
    скрытых факторов выбора ``estimate`` в среднем равен нулю.

    Attributes:
        column: Плацебо-колонка.
        estimate: Оценка «эффекта».
        low, high: Интервал на уровне ``check_z`` — тот же, что в правиле решения.
        sd: Стандартное отклонение плацебо-показателя.
        material: Порог существенности в единицах показателя (NaN, если проверка неприменима).
        status: ``fail`` — расхождение значимо и существенно; ``ok`` — нет; ``not_applicable`` —
            проверять нечего (выбор внутри периода целиком определён логирующей политикой или
            веса кандидата и базы совпадают).
    """

    column: str
    estimate: float
    low: float
    high: float
    sd: float
    material: float
    status: str

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass(frozen=True)
class Overlap:
    """Перекрытие политик A и B.

    ``uncovered`` — доля решений B (по массе вероятности) на действиях, которые A в этом
    контексте выбрать не могла. Известна точно, только когда логирующая политика задана
    (``logging_policy=``); иначе ``None``, и о покрытии судят по среднему весу.
    ``expected_weight`` — точное ожидание среднего веса при заданной логирующей политике и
    фактическом составе периодов (для политики, не зависящей от периода, при сопоставимых
    периодах это примерно ``1 − uncovered``); иначе ``None``.
    """

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
    uncovered: Optional[float] = None
    expected_weight: Optional[float] = None

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
            if data.propensity_col else
            "Проверьте logging_policy: она должна описывать политику, которая на самом деле "
            "записала эти логи."
        ),
    )]


def policy_findings(
    data: LogData,
    nuis: Nuisance,
    overlap: Overlap,
    th: Thresholds,
    *,
    baseline: Optional[str] = None,
) -> list[Finding]:
    """Находки, не зависящие от метрики: перекрытие, пропенсити, кластеры.

    Args:
        baseline: Имя базовой политики, если находки считаются для неё; ``None`` — для
            кандидата (тогда добавляются и находки о данных и пропенсити).
    """
    out: list[Finding] = []
    suffix = "" if baseline is None else f" (базовая политика «{baseline}»)"

    deficit = 1.0 - overlap.mean_weight
    deficit_low = 1.0 - overlap.mean_weight_high
    exploration_word = "оценённое" if overlap.exploration_estimated else "по логам"
    deterministic = overlap.exploration < th.deterministic_exploration or (
        nuis.p_max is not None and float(np.mean(nuis.p_max)) > 0.95
    )
    if overlap.uncovered is not None:
        # Логирующая политика известна: доля решений вне её носителя посчитана точно, и
        # эвристика «A почти не исследует» не нужна. Для смеси детерминированных версий она
        # ошибалась бы: каждая версия покрыта целиком, хотя строки по отдельности без выбора.
        if overlap.uncovered > th.support_deficit:
            severity = "block" if overlap.uncovered > th.support_deficit_block else "caution"
            if not deterministic:
                code = "support_deficit"
            else:
                code = "deterministic_logging" if baseline is None else "deterministic_logging_baseline"
            out.append(Finding(
                code=code,
                severity=severity,
                title="Часть решений — вне покрытия логов" + suffix,
                detail=(
                    f"{100 * overlap.uncovered:.1f}% решений политики приходится на действия, которые "
                    "логирующая политика в этих контекстах выбрать не могла (доля точная при заданной "
                    "политике A; для смеси версий — при сопоставимых периодах). Об их отклике в логах нет "
                    "информации: эту часть оценка не измеряет, а экстраполирует моделью исхода (IPS её "
                    "теряет), и интервал ошибку экстраполяции не учитывает."
                ),
                advice=(
                    "Сузьте политику до покрытых действий: ps.greedy(scores, allowed=ev.support()). "
                    "Границы без допущений об отклике на непокрытой части даёт ev.bounds(...)."
                ),
            ))
        # При верной логирующей политике средний вес в среднем равен своему точному ожиданию
        # (1 − доля вне покрытия для политики, не зависящей от периода). Значимое отличие —
        # прямой признак того, что пропенсити не те.
        target = overlap.expected_weight if overlap.expected_weight is not None else 1.0 - overlap.uncovered
        if overlap.mean_weight_low - target > th.support_deficit:
            out.append(Finding(
                code="weights_inflated",
                severity="caution",
                title="Веса важности в среднем завышены" + suffix,
                detail=(
                    f"Средний вес {overlap.mean_weight:.3f} значимо больше ожидаемого {target:.3f} "
                    "(точное ожидание при заданной логирующей политике). При верно заданной политике "
                    "так не бывает: её вероятности занижены."
                ),
                advice=(
                    "Проверьте logging_policy: версию модели и правило выбора (температуру, ε), а для "
                    "нескольких версий — разметку периодов."
                ),
            ))
        elif target - overlap.mean_weight_high > th.support_deficit:
            gap = target - overlap.mean_weight
            out.append(Finding(
                code="weights_deflated",
                severity="block" if gap > th.support_deficit_block else "caution",
                title="Веса важности в среднем занижены" + suffix,
                detail=(
                    f"Средний вес {overlap.mean_weight:.3f} значимо меньше ожидаемого {target:.3f}. Так бывает, "
                    "когда логирующая политика даёт части действий почти нулевую вероятность: формально они "
                    "покрыты, но в логах их практически нет, и эту часть оценка экстраполирует моделью "
                    "исхода. Другая причина — задана не та политика, что работала, или неверна разметка "
                    "периодов."
                ),
                advice=(
                    "Проверьте logging_policy; сузьте кандидата до действий, которые A выбирает с "
                    "заметной вероятностью, или соберите данные с исследованием."
                ),
            ))
    elif deterministic and overlap.match < th.deterministic_match:
        out.append(Finding(
            code="deterministic_logging" if baseline is None else "deterministic_logging_baseline",
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

    if overlap.n < th.min_ess:
        out.append(Finding(
            code="too_few_rows",
            severity="block",
            title="Слишком мало данных" + suffix,
            detail=f"В логах {overlap.n} строк: этого не хватает ни для моделей, ни для интервала.",
            advice="Соберите больше логов.",
        ))
    elif overlap.ess < th.min_ess:
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
            advice=(
                "При тяжёлых весах интервал может недопокрывать с одной стороны. Возьмите "
                "кандидата ближе к A или соберите больше данных."
            ),
        ))

    if baseline is not None:
        return out

    out.extend(_propensity_validity(data, nuis, th))
    if nuis.source == "estimated":
        out.append(Finding(
            code="propensity_estimated",
            severity="caution",
            title="Пропенсити оценены моделью",
            detail=(
                "В логах нет вероятностей выбора действия, они восстановлены моделью "
                "(cross-fit, с полом). Пол и неточность модели пропенсити дают смещение, "
                "которое не исчезает с ростом данных, поэтому в этом режиме DR не является "
                "строго дважды робастным."
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
    base_w: Optional[np.ndarray] = None,
    base_dm: Optional[np.ndarray] = None,
    declared_policy: bool = False,
    period_dependent: bool = False,
    uncovered: float = 0.0,
    uncovered_mass: Optional[float] = None,
    uncovered_owner: str = "кандидата",
    mixture: bool = False,
) -> tuple[tuple[Check, ...], list[Finding]]:
    """Проверки согласия оценщиков для одной метрики и находки по ним.

    Сравниваются те Δ, что показаны в отчёте. Для базы «A из логов» DR берёт V(A) = mean(r),
    а DM — mean(mu(x_i, a_i)), поэтому

    - ``DR − IPS`` для Δ = ``mean(dm − w·mu_i)``;
    - ``DR − DM`` для Δ = ``mean((w − 1)(r − mu_i))``.

    Если база — политика, из каждой величины вычитается такая же величина для базы.
    """
    r = data.rewards[metric]
    sn = estimator in SELF_NORMALIZED

    def normalized(weights: np.ndarray) -> np.ndarray:
        return weights / np.mean(weights) if sn and np.mean(weights) > 0 else weights

    wb = normalized(w)
    prop_values = dm - wb * mu_taken
    if base_w is None:
        outcome_values = (wb - 1.0) * (r - mu_taken)
    else:
        wa = normalized(base_w)
        prop_values = prop_values - (base_dm - wa * mu_taken)
        outcome_values = (wb - wa) * (r - mu_taken)
    material = th.material_bias * se_delta if np.isfinite(se_delta) else np.inf
    informative = ess_value >= th.min_ess
    prop = _check(
        "propensity", "SNDR − SNIPS" if sn else "DR − IPS",
        prop_values, data, material, th, informative,
    )
    outc = _check(
        "outcome", "SNDR − DM" if sn else "DR − DM",
        outcome_values, data, material, th, informative,
    )

    findings: list[Finding] = []
    prop_fail, out_fail = prop.status == "fail", outc.status == "fail"
    if estimator == "dm":
        if out_fail:
            findings.append(Finding(
                code="outcome_check_failed",
                severity="block",
                metric=metric,
                title="DM не согласуется с DR: модель исхода смещена",
                detail=(
                    f"{outc.label} = {outc.discrepancy:+.4g} (интервал {outc.low:+.4g}…{outc.high:+.4g}): "
                    "расхождение значимо и существенно, а DM целиком опирается на модель исхода."
                ),
                advice="Используйте estimator='dr' или улучшите модель исхода (outcome_model=...).",
            ))
        return (prop, outc), findings
    # При заданной политике A и решениях вне её покрытия DR − IPS не ноль по построению: IPS эту
    # часть теряет (SNIPS заменяет средним по покрытой), DR её экстраполирует. Если расхождение
    # этим и объясняется (ожидаемая величина — в интервале проверки), политику не винят.
    off_support = declared_policy and not period_dependent and uncovered > 0 and (
        uncovered_mass is None or prop.low <= uncovered_mass <= prop.high
    )
    has_coverage_finding = uncovered > th.support_deficit
    coverage_ref = ("см. находку о покрытии" if has_coverage_finding
                    else "доля решений вне покрытия — в разделе о перекрытии")
    coverage_clause = (
        f"{100 * uncovered:.1f}% решений {uncovered_owner} приходится на действия, которые логирующая "
        "политика в этих контекстах выбрать не могла: "
        + ("SNIPS заменяет эту часть средним по покрытой, а SNDR экстраполирует моделью исхода"
           if sn else "IPS эту часть теряет, а DR экстраполирует моделью исхода")
        + (f" (ожидаемое расхождение при верной политике {uncovered_mass:+.4g})" if uncovered_mass is not None else "")
    )
    if estimated_propensity:
        tail_note = "Оценённые пропенсити, по-видимому, плохо воспроизводят политику A."
    elif off_support:
        tail_note = f"{coverage_clause}. Поэтому расхождение есть и при верной logging_policy ({coverage_ref})."
    elif period_dependent:
        tail_note = (
            "Политика зависит от периода логов: для неё IPS по смеси версий смещён по построению, "
            "так что расхождение ожидаемо (см. находку о политике, зависящей от периода)."
            + (f" Кроме того, {coverage_clause[0].lower()}{coverage_clause[1:]}." if uncovered > 0 else "")
        )
    elif declared_policy:
        tail_note = (
            "Пропенсити посчитаны по заданной logging_policy: так бывает, если она не та, что "
            "записала логи (версия модели, температура, ε"
            + (", разметка периодов" if mixture else "") + ")"
            + (", если кандидат зависит от периода логов (например, таблица долей действий по месяцам)"
               if mixture else "")
            + " или если заметная часть массы весов приходится на редкие сочетания «контекст–действие», "
            "почти не попавшие в выборку (тяжёлый хвост весов)."
        )
    else:
        tail_note = (
            "При логированных пропенсити это обычно значит, что заметная часть массы весов "
            "приходится на редкие сочетания «контекст–действие», почти не попавшие в выборку "
            "(тяжёлый хвост весов), а не что пропенсити неверны."
        )
    check_policy = (
        "Проверьте logging_policy: версию модели, температуру, ε"
        + (" и разметку периодов" if mixture else "") + ". "
    ) if declared_policy and not period_dependent and not off_support else ""
    period_advice = (
        "Для случайной выдачи используйте ps.random_like() без by= по колонке периода: для этой политики "
        "DR опирается на модель исхода (см. проверку DR − DM)."
    )
    extrapolation_advice = (
        "Интервал DR ошибку экстраполяции не учитывает: сузьте политику до покрытых действий "
        "(allowed=ev.support()) или сравните вывод с границами ev.bounds(...)."
    )
    if period_dependent and uncovered > 0:
        period_advice += " " + extrapolation_advice
    if estimator in ("ips", "snips"):
        if prop_fail:
            findings.append(Finding(
                code="propensity_check_failed",
                severity="block",
                metric=metric,
                title="IPS не согласуется с DR",
                detail=(
                    f"{prop.label} = {prop.discrepancy:+.4g} (интервал {prop.low:+.4g}…{prop.high:+.4g}), "
                    "а при верных пропенсити, полной выборке и решениях в пределах покрытия логов эта "
                    f"разность в среднем равна нулю. {tail_note}"
                ),
                advice=(
                    period_advice if period_dependent else
                    "IPS теряет решения вне покрытия логов, DR их экстраполирует: используйте "
                    "estimator='dr' и проверьте вывод границами ev.bounds(...)." if off_support else
                    check_policy + "Используйте estimator='dr': он не зависит от того, попал ли хвост весов в выборку."
                ),
            ))
        return (prop, outc), findings

    if prop_fail and out_fail and (period_dependent or off_support):
        findings.append(Finding(
            code="both_models_rejected",
            severity="caution" if period_dependent else "block",
            metric=metric,
            title="Модель исхода смещена, а веса её не исправляют",
            detail=(
                f"{outc.label} = {outc.discrepancy:+.4g} (интервал {outc.low:+.4g}…{outc.high:+.4g}): модель "
                f"исхода смещена. {prop.label} = {prop.discrepancy:+.4g}. {tail_note} "
                + ("Веса эту модель исправляют не полностью, и DR опирается на неё." if period_dependent
                   else "На решениях вне покрытия DR держится только на этой модели.")
            ),
            advice=(period_advice if period_dependent else extrapolation_advice)
            + " Или улучшите модель исхода (outcome_model=...).",
        ))
    elif prop_fail and out_fail:
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
                check_policy + "Улучшите модель исхода (outcome_model=...)"
                + ("." if declared_policy else " или получите логированные пропенсити.")
            ),
        ))
    elif prop_fail:
        findings.append(Finding(
            code="propensity_check_failed",
            severity="caution",
            metric=metric,
            title="IPS и DR расходятся",
            detail=(
                f"{prop.label} = {prop.discrepancy:+.4g} (интервал {prop.low:+.4g}…{prop.high:+.4g}). "
                f"{tail_note} Там, где весов не хватает, оценка DR держится на модели исхода"
                + ("; её проверка расхождений не выявила." if outc.status == "ok"
                   else ", а её проверить не удалось.")
            ),
            advice=(
                period_advice if period_dependent else
                extrapolation_advice if off_support else
                check_policy + "Если она верна — интервал DR может недопокрывать с одной стороны, "
                "смотрите на таблицу перекрытия по действиям." if declared_policy else
                "При оценённых пропенсити — логируйте настоящие; при логированных — интервал DR "
                "может недопокрывать с одной стороны, смотрите на таблицу перекрытия по действиям."
            ),
        ))
    elif out_fail:
        findings.append(Finding(
            code="outcome_check_failed",
            severity="caution" if estimated_propensity else "info",
            metric=metric,
            title="Модель исхода смещена",
            detail=(
                f"{outc.label} = {outc.discrepancy:+.4g} (интервал {outc.low:+.4g}…{outc.high:+.4g}): "
                "DM смещён. "
                + ("Для политики, зависящей от периода, веса это исправляют не полностью — DR опирается на "
                   "модель исхода." if period_dependent else
                   "DR это исправляет за счёт весов"
                   + (", но веса здесь сами оценены." if estimated_propensity else ", если пропенсити верны."))
            ),
            advice=(period_advice if period_dependent
                    else "Можно улучшить модель исхода (outcome_model=...) — это сузит интервал DR."),
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
        advice="Посмотрите на строки с максимальными весами: оценка и интервал определяются ими.",
    )
