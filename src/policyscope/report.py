"""Результаты оценки и их представление.

:class:`Report` — результат :meth:`Evaluator.evaluate`: по одной политике B и
нескольким метрикам. :class:`Leaderboard` — результат :meth:`Evaluator.compare`.

Представления:

- ``report`` в Jupyter рендерится сам (``_repr_markdown_``);
- :meth:`Report.to_markdown` — отчёт для людей (Confluence, письмо);
- :meth:`Report.summary` — ``DataFrame``, строка на метрику;
- :meth:`Report.to_dict` / :meth:`Report.to_json` — стабильная схема для пайплайнов:
  все ключи присутствуют всегда, отсутствующие значения — ``None``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd

from policyscope.diagnostics import Check, Finding, Overlap

__all__ = ["Estimate", "EstimatorRow", "MetricReport", "Report", "Leaderboard"]

TRUST_LABELS = {"reliable": "надёжно", "caution": "с осторожностью", "unreliable": "ненадёжно"}
DECISION_LABELS = {
    "better": "лучше",
    "worse": "хуже",
    "no_difference": "разница не обнаружена",
    "inconclusive": "вывод сделать нельзя",
    "not_computed": "интервал не считался",
}
ESTIMATOR_LABELS = {
    "dr": "DR (doubly robust)",
    "sndr": "SNDR (самонормированный DR)",
    "ips": "IPS",
    "snips": "SNIPS (самонормированный IPS)",
    "dm": "DM (только модель исхода)",
}
SEVERITY_ICONS = {"block": "⛔", "caution": "⚠️", "info": "ℹ️"}
AGREEMENT_LABELS = {
    "agree": "согласованы",
    "disagree": "расходятся",
    "unclear": "проверка неполная",
}


# ---------------------------------------------------------------------------
# Форматирование
# ---------------------------------------------------------------------------


def _finite(x: Optional[float]) -> bool:
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def _num(x: Optional[float], signed: bool = False) -> str:
    if not _finite(x):
        return "—"
    sign = "+" if signed and x > 0 else ""
    ax = abs(x)
    if ax >= 1000:
        body = f"{x:,.0f}".replace(",", " ")
    elif ax >= 100:
        body = f"{x:.1f}"
    elif ax >= 1:
        body = f"{x:.2f}"
    elif ax == 0:
        body = "0"
    else:
        body = f"{x:.3g}"
    return sign + body.replace("-", "−")


def _value(x: Optional[float], binary: bool) -> str:
    if binary and _finite(x):
        return f"{100 * x:.2f}%"
    return _num(x)


def _diff(x: Optional[float], binary: bool) -> str:
    if binary and _finite(x):
        return f"{100 * x:+.2f} п.п.".replace("-", "−")
    return _num(x, signed=True)


def _pct(x: Optional[float]) -> str:
    if not _finite(x):
        return "—"
    return f"{100 * x:+.1f}%".replace("-", "−")


def _p(x: Optional[float]) -> str:
    if not _finite(x):
        return "—"
    return "<0.001" if x < 0.001 else f"{x:.3f}"


def _interval(low: Optional[float], high: Optional[float], fmt) -> str:
    if not (_finite(low) and _finite(high)):
        return "—"
    return f"[{fmt(low)}; {fmt(high)}]"


def _with_interval(est: "Estimate", binary: bool) -> str:
    """Значение и его интервал в одной ячейке: «18.40% [17.62%; 19.18%]»."""
    fmt = lambda x: _value(x, binary)  # noqa: E731
    ci = _interval(est.low, est.high, fmt)
    return fmt(est.value) if ci == "—" else f"{fmt(est.value)} {ci}"


def _plural(n: int, one: str, few: str, many: str) -> str:
    """Русское склонение существительного после числа: 1 действие, 2 действия, 5 действий."""
    n = abs(int(n))
    if n % 10 == 1 and n % 100 != 11:
        word = one
    elif 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        word = few
    else:
        word = many
    return f"{n:,}".replace(",", " ") + " " + word


def _clean(value: Any) -> Any:
    """Приводит значение к JSON-безопасному виду (NaN -> None, numpy -> python)."""
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            value = value.item()
        except (ValueError, AttributeError):
            pass
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


# ---------------------------------------------------------------------------
# Структуры
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Estimate:
    """Оценка с интервалом. ``low``/``high``/``se`` — ``None``, если не считались."""

    value: float
    low: Optional[float] = None
    high: Optional[float] = None
    se: Optional[float] = None

    def to_dict(self) -> dict:
        return {"value": self.value, "low": self.low, "high": self.high, "se": self.se}


@dataclass(frozen=True)
class EstimatorRow:
    """Строка таблицы согласия оценщиков."""

    estimator: str
    v_b: Estimate
    delta: Estimate
    p_value: Optional[float]


@dataclass(frozen=True)
class MetricReport:
    """Результат по одной метрике.

    Attributes:
        decision: ``better`` / ``worse`` / ``no_difference`` / ``inconclusive`` /
            ``not_computed``. «Лучше» учитывает направление метрики (``lower_is_better``).
        trust: ``reliable`` / ``caution`` / ``unreliable``.
        p_value_adj: p-value с поправкой Romano–Wolf на все метрики отчёта
            (в рейтинге — на все пары «кандидат × метрика»).
        lift: Относительный прирост ``V(B)/V(A) − 1``; ``None``, если V не положительны.
        mde: Минимальный эффект, который был бы обнаружен с мощностью 80%.
        agreement: Согласие оценщиков по проверкам DR − IPS и DR − DM: ``agree`` — обе без
            значимого расхождения, ``disagree`` — хотя бы одна нашла существенное,
            ``unclear`` — проверить не удалось (мало данных).
        agreement_spread: Разброс Δ по всем оценщикам (максимум − минимум).
    """

    metric: str
    estimator: str
    binary: bool
    lower_is_better: bool
    v_a: Estimate
    v_b: Estimate
    delta: Estimate
    p_value: Optional[float]
    p_value_adj: Optional[float]
    lift: Optional[Estimate]
    mde: Optional[float]
    decision: str
    trust: str
    findings: tuple[Finding, ...]
    checks: tuple[Check, ...]
    rows: tuple[EstimatorRow, ...]
    outcome_quality: dict[str, float]
    floor_sensitivity: Optional[dict[str, float]] = None
    agreement: str = "unclear"
    agreement_spread: Optional[float] = None

    @property
    def estimators(self) -> pd.DataFrame:
        """Таблица согласия оценщиков: строка на оценщик."""
        return pd.DataFrame([{
            "estimator": row.estimator,
            "V_B": row.v_b.value,
            "V_B_low": row.v_b.low,
            "V_B_high": row.v_b.high,
            "delta": row.delta.value,
            "delta_low": row.delta.low,
            "delta_high": row.delta.high,
            "p_value": row.p_value,
        } for row in self.rows])

    @property
    def decision_label(self) -> str:
        """Вывод коротко: «лучше», «хуже», «разница не обнаружена» и т.д."""
        return DECISION_LABELS[self.decision]

    def decision_text(self, baseline: str = "A") -> str:
        """Вывод с именем базы: «лучше A», «хуже A» или ``decision_label``."""
        if self.decision in ("better", "worse"):
            return f"{self.decision_label} {baseline}"
        return self.decision_label

    @property
    def trust_label(self) -> str:
        return TRUST_LABELS[self.trust]

    @property
    def agreement_label(self) -> str:
        return AGREEMENT_LABELS[self.agreement]

    @property
    def trust_reasons(self) -> list[str]:
        """Заголовки находок, которые определили уровень доверия (самые серьёзные)."""
        severity = {"caution": "caution", "unreliable": "block"}.get(self.trust)
        titles = [f.title for f in self.findings if f.severity == severity]
        # «Пропенсити …» -> «пропенсити …», но аббревиатуры («IPS и DR …») не трогаем.
        return list(dict.fromkeys(t[:1].lower() + t[1:] if t[1:2].islower() else t for t in titles))

    def sentence(self, policy: str = "B", baseline: str = "A") -> str:
        """Вывод одной фразой."""
        fmt_d = lambda x: _diff(x, self.binary)  # noqa: E731
        delta = fmt_d(self.delta.value)
        lift = f" ({_pct(self.lift.value)})" if self.lift is not None else ""
        ci = _interval(self.delta.low, self.delta.high, fmt_d)
        if self.decision == "inconclusive":
            return (f"вывод сделать нельзя: оценка Δ = {delta}{lift} ненадёжна "
                    "(причины ниже).")
        if self.decision == "not_computed":
            return f"Δ = {delta}{lift}; доверительный интервал не считался."
        if self.decision in ("better", "worse"):
            word = "лучше" if self.decision == "better" else "хуже"
            return f"{policy} {word} {baseline}: Δ = {delta}{lift}, ДИ {ci}."
        mde = f"; эффект больше ±{_diff(self.mde, self.binary).lstrip('+')} был бы обнаружен" \
            if _finite(self.mde) else ""
        return f"значимой разницы нет: Δ = {delta}{lift}, ДИ {ci}{mde}."

    def to_dict(self) -> dict:
        return {
            "metric": self.metric,
            "estimator": self.estimator,
            "binary": self.binary,
            "lower_is_better": self.lower_is_better,
            "V_A": self.v_a.to_dict(),
            "V_B": self.v_b.to_dict(),
            "delta": self.delta.to_dict(),
            "p_value": self.p_value,
            "p_value_adj": self.p_value_adj,
            "lift": self.lift.to_dict() if self.lift is not None else None,
            "mde": self.mde,
            "decision": self.decision,
            "trust": self.trust,
            "findings": [f.to_dict() for f in self.findings],
            "checks": [c.to_dict() for c in self.checks],
            "estimators": [{
                "estimator": r.estimator, "V_B": r.v_b.to_dict(),
                "delta": r.delta.to_dict(), "p_value": r.p_value,
            } for r in self.rows],
            "outcome_quality": dict(self.outcome_quality),
            "floor_sensitivity": self.floor_sensitivity,
            "agreement": {"status": self.agreement, "spread": self.agreement_spread},
        }


@dataclass(frozen=True)
class Report:
    """Отчёт об оценке политики B по логам политики A.

    Доступ к метрике: ``report["revenue"]`` -> :class:`MetricReport`.
    """

    policy: str
    baseline: str
    metrics: dict[str, MetricReport]
    overlap: Overlap
    findings: tuple[Finding, ...]
    settings: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, metric: str) -> MetricReport:
        try:
            return self.metrics[metric]
        except KeyError:
            raise KeyError(f"Метрики '{metric}' нет в отчёте; есть: {list(self.metrics)}") from None

    @property
    def trust(self) -> str:
        """Худший уровень доверия среди метрик."""
        order = ["reliable", "caution", "unreliable"]
        return max((m.trust for m in self.metrics.values()), key=order.index)

    def summary(self) -> pd.DataFrame:
        """Строка на метрику: значения с интервалами, Δ, p-value, вывод, доверие, согласие оценщиков."""
        rows = []
        for m in self.metrics.values():
            rows.append({
                "metric": m.metric,
                "V_A": m.v_a.value,
                "V_A_low": m.v_a.low,
                "V_A_high": m.v_a.high,
                "V_B": m.v_b.value,
                "V_B_low": m.v_b.low,
                "V_B_high": m.v_b.high,
                "delta": m.delta.value,
                "delta_low": m.delta.low,
                "delta_high": m.delta.high,
                "lift": m.lift.value if m.lift else None,
                "lift_low": m.lift.low if m.lift else None,
                "lift_high": m.lift.high if m.lift else None,
                "p_value": m.p_value,
                "p_value_adj": m.p_value_adj,
                "mde": m.mde,
                "decision": m.decision,
                "trust": m.trust,
                "agreement": m.agreement,
                "estimator": m.estimator,
            })
        return pd.DataFrame(rows).set_index("metric")

    def to_dict(self) -> dict:
        return _clean({
            "policy": self.policy,
            "baseline": self.baseline,
            "trust": self.trust,
            "metrics": {k: v.to_dict() for k, v in self.metrics.items()},
            "overlap": self.overlap.to_dict(),
            "findings": [f.to_dict() for f in self.findings],
            "settings": self.settings,
        })

    def to_json(self, path: Optional[str] = None) -> str:
        """JSON-представление; при ``path`` ещё и записывает его в файл."""
        text = json.dumps(self.to_dict(), ensure_ascii=False, indent=2, allow_nan=False)
        if path is not None:
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
        return text

    def _repr_markdown_(self) -> str:
        return self.to_markdown()

    def __repr__(self) -> str:
        lines = [f"Report({self.policy} vs {self.baseline}, доверие: {TRUST_LABELS[self.trust]})"]
        for m in self.metrics.values():
            lines.append(f"  {m.metric}: {m.sentence(self.policy, self.baseline)}")
        return "\n".join(lines)

    # -- markdown -----------------------------------------------------------------

    def to_markdown(self) -> str:
        """Отчёт для людей: вывод, доверие, согласие оценщиков, перекрытие, настройки."""
        s = self.settings
        level = f"{100 * (1 - s.get('alpha', 0.05)):.0f}%"
        out = [f"## Off-policy оценка: {self.policy} против {self.baseline}", ""]
        for m in self.metrics.values():
            icon = {"reliable": "✅", "caution": "⚠️", "unreliable": "⛔"}[m.trust]
            why = f" ({'; '.join(m.trust_reasons)})" if m.trust_reasons else ""
            out.append(
                f"- **{m.metric}** — {m.sentence(self.policy, self.baseline)} "
                f"Доверие: {icon} {m.trust_label}{why}."
            )
        out += ["", self._metric_table(level), "", self._settings_line()]
        if len(self.metrics) > 1 and s.get("inference") != "none":
            out.append(
                f"p (RW) — p-value с поправкой Romano–Wolf на "
                f"{_plural(len(self.metrics), 'метрику', 'метрики', 'метрик')}: "
                "вероятность хотя бы одного ложного «значимо» по всем метрикам не выше α."
            )
        out += ["", "### Можно ли верить оценке", *self._findings_block()]
        out += ["", "### Согласие оценщиков", *self._estimators_block(level)]
        out += ["", "### Перекрытие политик", *self._overlap_block()]
        out += ["", "### Настройки и предпосылки", *self._settings_block()]
        return "\n".join(out)

    def _metric_table(self, level: str) -> str:
        head = (f"| Метрика | V({self.baseline}) [{level} ДИ] | V({self.policy}) [{level} ДИ] | Δ "
                f"| {level} ДИ для Δ | Δ, % | p | p (RW) | Вывод | Доверие | Оценщики |")
        lines = [head, "|---|---|---|---|---|---|---|---|---|---|---|"]
        for m in self.metrics.values():
            fd = lambda x, b=m.binary: _diff(x, b)  # noqa: E731
            lines.append(
                f"| {m.metric} | {_with_interval(m.v_a, m.binary)} | {_with_interval(m.v_b, m.binary)} "
                f"| {fd(m.delta.value)} | {_interval(m.delta.low, m.delta.high, fd)} "
                f"| {_pct(m.lift.value) if m.lift else '—'} | {_p(m.p_value)} "
                f"| {_p(m.p_value_adj)} | {m.decision_text(self.baseline)} | {m.trust_label} "
                f"| {m.agreement_label} |"
            )
        return "\n".join(lines)

    def _settings_line(self) -> str:
        s = self.settings
        inference = {
            "influence": "аналитический ДИ",
            "bootstrap": f"bootstrap ДИ ({s.get('n_boot')} реплик с переобучением моделей)",
            "none": "без ДИ",
        }[s["inference"]]
        clusters = (f", кластеры «{s['cluster']}»: {self.overlap.n_clusters}"
                    if s.get("cluster") else "")
        source = (f"пропенсити из логов («{s['propensity']}»)" if s.get("propensity")
                  else "пропенсити оценены моделью")
        chosen = ", выбран автоматически" if s.get("estimator_auto") else ""
        return (f"Оценщик {ESTIMATOR_LABELS[s['estimator']]}{chosen} · {inference}{clusters} · {source} · "
                f"{_plural(self.overlap.n, 'строка', 'строки', 'строк')}, "
                f"{_plural(self.overlap.k, 'действие', 'действия', 'действий')}.")

    def _findings_block(self) -> list[str]:
        items: list[Finding] = list(self.findings)
        for m in self.metrics.values():
            items.extend(f for f in m.findings if f.metric is not None)
        items.sort(key=lambda f: {"block": 0, "caution": 1, "info": 2}[f.severity])
        if not any(f.severity != "info" for f in items):
            lines = ["Проверки перекрытия, пропенсити и модели исхода серьёзных проблем не выявили."]
        else:
            lines = []
        for f in items:
            where = f" [{f.metric}]" if f.metric and len(self.metrics) > 1 else ""
            advice = f" *Что делать:* {f.advice}" if f.advice else ""
            lines.append(f"- {SEVERITY_ICONS[f.severity]} **{f.title}**{where}. {f.detail}{advice}")
        return lines

    def _estimators_block(self, level: str) -> list[str]:
        s = self.settings
        name = s["estimator"].upper()
        if s.get("estimator_auto"):
            choice = f"Основной оценщик — {name}, выбран автоматически: {s.get('estimator_reason')}."
        else:
            choice = f"Основной оценщик — {name}, задан явно."
        lines = [
            choice,
            "",
            "IPS опирается только на пропенсити, DM — только на модель исхода, DR — на обе. "
            "Если они расходятся, хотя бы одна из моделей неверна; согласие исключает только "
            "грубые ошибки.",
            "",
        ]
        for m in self.metrics.values():
            se = m.delta.se
            spread = m.agreement_spread
            scale = (f" ({spread / se:.1f} стандартной ошибки основного оценщика)"
                     if _finite(spread) and _finite(se) and se > 0 else "")
            lines.append(
                f"- **{m.metric}: {m.agreement_label}.** Δ по {len(m.rows)} оценщикам — от "
                f"{_diff(min(r.delta.value for r in m.rows), m.binary)} до "
                f"{_diff(max(r.delta.value for r in m.rows), m.binary)}, разброс "
                f"{_diff(spread, m.binary).lstrip('+')}{scale}."
            )
        lines += [
            "",
            f"| Метрика | Оценщик | V({self.policy}) | Δ | {level} ДИ для Δ |",
            "|---|---|---|---|---|",
        ]
        for m in self.metrics.values():
            for row in m.rows:
                fd = lambda x, b=m.binary: _diff(x, b)  # noqa: E731
                mark = " (основной)" if row.estimator == m.estimator else ""
                lines.append(
                    f"| {m.metric} | {row.estimator}{mark} | {_value(row.v_b.value, m.binary)} "
                    f"| {fd(row.delta.value)} | {_interval(row.delta.low, row.delta.high, fd)} |"
                )
        lines.append("")
        if self.settings.get("estimator") == "dm":
            lines.append(
                "Интервал DM описывает разброс оценки при верной модели исхода; её смещение "
                "в интервал не входит."
            )
        else:
            lines.append(
                "DM показан без интервала: его точность держится на модели исхода, смещение "
                "которой по логам не проверяется. Интервал DM считается, если выбрать его "
                "основным явно (estimator='dm')."
            )
        lines.append(
            "Отсутствие расхождения не подтверждает модели: проверки замечают лишь грубые "
            "ошибки, сравнимые с несколькими стандартными ошибками."
        )
        meaning = {"propensity": "пропенсити", "outcome": "модель исхода"}
        for m in self.metrics.values():
            for c in m.checks:
                fd = lambda x, b=m.binary: _diff(x, b)  # noqa: E731
                bound = fd(max(abs(c.low), abs(c.high))).lstrip("+")
                verdict = {
                    "ok": f"значимого расхождения нет (не исключено до ±{bound})",
                    "fail": "значимое и существенное расхождение",
                    "inconclusive": "мало данных для проверки",
                }[c.status]
                lines.append(
                    f"- {m.metric}, {meaning[c.name]} ({c.label}): {fd(c.discrepancy)} "
                    f"[{fd(c.low)}; {fd(c.high)}] — {verdict}."
                )
            if m.floor_sensitivity:
                fs = m.floor_sensitivity
                lines.append(
                    f"- {m.metric}, чувствительность Δ к полу пропенсити: ×0.5 → {_diff(fs['x0.5'], m.binary)}, "
                    f"×1 → {_diff(fs['x1'], m.binary)}, ×2 → {_diff(fs['x2'], m.binary)}."
                )
        return lines

    def _overlap_block(self) -> list[str]:
        o = self.overlap
        explore = "оценка по модели" if o.exploration_estimated else "по логированным пропенсити"
        lines = [
            f"- Эффективный размер выборки: {o.ess:,.0f} из {o.n:,} ({100 * o.ess_ratio:.0f}%).".replace(",", " "),
            f"- Средний вес важности {o.mean_weight:.3f} [{o.mean_weight_low:.3f}; {o.mean_weight_high:.3f}] "
            f"(при полном перекрытии — 1); максимальный {_num(o.max_weight)}, 99-й перцентиль {_num(o.p99_weight)}.",
            f"- A в среднем оставляет шанс {min(o.exploration, o.k):.1f} "
            f"из {o.k} действий ({explore}); {self.policy} совпадает с залогированным действием "
            f"в {100 * o.match:.0f}% случаев.",
            "",
            f"| Действие | Доля в логах | Доля у {self.policy} | Средний вес |",
            "|---|---|---|---|",
        ]
        for row in o.actions:
            lines.append(
                f"| {row['action']} | {100 * row['logged_share']:.1f}% | {100 * row['policy_share']:.1f}% "
                f"| {_num(row['mean_weight'])} |"
            )
        return lines

    def _settings_block(self) -> list[str]:
        s = self.settings
        features = ", ".join(s.get("features", [])) or "нет (оценка без контекста)"
        models = s.get("outcome_model") or {}
        outcome = ("; ".join(f"{m} — {name}" for m, name in models.items())
                   if isinstance(models, dict) else str(models))
        auto = " (выбраны автоматически)" if s.get("features_auto") else ""
        lines = [
            f"- Признаки{auto}: {features}.",
            f"- Модель пропенсити: {s.get('propensity_model')}; модель исхода: {outcome}; "
            f"cross-fitting: {s.get('n_folds')} фолдов.",
        ]
        quality = []
        for m in self.metrics.values():
            q = m.outcome_quality
            if "roc_auc" in q:
                quality.append(f"{m.metric}: ROC AUC {_num(q['roc_auc'])}")
            elif "r2" in q:
                quality.append(f"{m.metric}: R² {_num(q['r2'])}")
        if quality:
            lines.append("- Качество модели исхода (out-of-fold): " + "; ".join(quality) + ".")
        lines += [
            "- Предпосылки, которые нельзя проверить по данным: все факторы, влиявшие и на "
            "выбор A, и на метрику, есть среди признаков; политика B не обучалась на этих же "
            "логах (иначе оценка завышена).",
        ]
        return lines


@dataclass(frozen=True)
class Leaderboard:
    """Сравнение нескольких кандидатов с одной базой.

    ``table`` — строка на пару «кандидат × метрика», кандидаты упорядочены по Δ первой
    метрики (лучший — первым); ``reports`` — полный отчёт по каждому кандидату. p-value
    скорректированы методом Romano–Wolf на всё семейство «кандидат × метрика».
    """

    table: pd.DataFrame
    reports: dict[str, Report]
    baseline: str
    alpha: float

    def __getitem__(self, name: str) -> Report:
        return self.reports[name]

    def __repr__(self) -> str:
        lines = [f"Leaderboard({len(self.reports)} кандидатов против {self.baseline})"]
        for name, report in self.reports.items():
            for m in report.metrics.values():
                lines.append(f"  {name} / {m.metric}: Δ = {_diff(m.delta.value, m.binary)}, "
                             f"{m.decision_text(self.baseline)}, доверие: {m.trust_label}")
        return "\n".join(lines)

    def _repr_markdown_(self) -> str:
        return self.to_markdown()

    def to_markdown(self) -> str:
        level = f"{100 * (1 - self.alpha):.0f}%"
        lines = [
            f"## Сравнение кандидатов с {self.baseline}",
            "",
            f"| Кандидат | Метрика | V({self.baseline}) [{level} ДИ] | V(кандидата) [{level} ДИ] | Δ "
            f"| {level} ДИ для Δ | Δ, % | p | p (RW) | Вывод | Доверие | Оценщики |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        adjusted = False
        for name, report in self.reports.items():
            for m in report.metrics.values():
                adjusted = adjusted or m.p_value_adj is not None
                fd = lambda x, b=m.binary: _diff(x, b)  # noqa: E731
                lines.append(
                    f"| {name} | {m.metric} | {_with_interval(m.v_a, m.binary)} "
                    f"| {_with_interval(m.v_b, m.binary)} | {fd(m.delta.value)} "
                    f"| {_interval(m.delta.low, m.delta.high, fd)} | {_pct(m.lift.value) if m.lift else '—'} "
                    f"| {_p(m.p_value)} | {_p(m.p_value_adj)} | {m.decision_text(self.baseline)} "
                    f"| {m.trust_label} | {m.agreement_label} |"
                )
        lines.append("")
        if adjusted:
            pairs = _plural(len(self.table), "пару", "пары", "пар")
            lines.append(f"p (RW) скорректированы методом Romano–Wolf на {pairs} «кандидат × метрика».")
        lines.append(
            "Оценка лучшего из многих кандидатов систематически завышена (winner's curse): "
            "перед внедрением подтвердите победителя на свежих логах или в A/B-тесте."
        )
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return _clean({
            "baseline": self.baseline,
            "alpha": self.alpha,
            "table": self.table.to_dict(orient="records"),
            "reports": {k: v.to_dict() for k, v in self.reports.items()},
        })

    def to_json(self, path: Optional[str] = None) -> str:
        """JSON-представление; при ``path`` ещё и записывает его в файл."""
        text = json.dumps(self.to_dict(), ensure_ascii=False, indent=2, allow_nan=False)
        if path is not None:
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
        return text
