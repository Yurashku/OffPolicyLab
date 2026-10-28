"""Точка входа: :class:`Evaluator` и :func:`evaluate`.

Сценарий::

    ev = ps.Evaluator(logs, action="offer", reward=["accepted", "revenue"],
                      propensity="p_offer", cluster="client_id")
    report = ev.evaluate(policy_b)              # B против политики A из логов
    board = ev.compare({"мягкий": b1, "жёсткий": b2})   # несколько кандидатов

Nuisance-модели (пропенсити и модель исхода) от кандидата не зависят: они обучаются
один раз при первом вызове и переиспользуются, поэтому каждый следующий кандидат
оценивается за доли секунды (при аналитическом интервале).
"""

from __future__ import annotations

import logging
import os
import sys
import warnings
from dataclasses import replace
from functools import partial
from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

from policyscope.data import LogData, prepare_logs
from policyscope.diagnostics import (
    Finding,
    Thresholds,
    overlap_summary,
    policy_findings,
    triangulate,
    trust_level,
    variance_concentration,
)
from policyscope.estimators import ESTIMATORS, PolicyTerms, contributions, ess, policy_terms
from policyscope.inference import (
    analytic_summary,
    centered_pvalue,
    cluster_sums,
    multiplier_tstats,
    nan_std,
    percentile_interval,
    romano_wolf,
    run_bootstrap,
    t_critical,
)
from policyscope.nuisance import (
    OUTCOME_KIND_LABELS,
    Nuisance,
    NuisanceConfig,
    fit_nuisance,
    outcome_kind,
)
from policyscope.policy import policy_matrix
from policyscope.report import Estimate, EstimatorRow, Leaderboard, MetricReport, Report

logger = logging.getLogger(__name__)

__all__ = ["Evaluator", "evaluate", "PRIMARY_ESTIMATORS"]

#: Оценщики, которые могут быть основными. ``dm`` — только явным выбором: он точнее DR,
#: если модель исхода верна, но его смещение целиком определяется этой моделью и по
#: логам не проверяется (см. docs/estimators.md).
PRIMARY_ESTIMATORS = ("dr", "sndr", "ips", "snips", "dm")
#: Почему ``estimator="auto"`` выбирает DR. Выбор по самому узкому интервалу не делается:
#: он занижает интервал (выбор по шуму) и отдавал бы предпочтение DM, чьё смещение
#: интервал не отражает (см. docs/methodology.md, «Выбор основного оценщика»).
AUTO_ESTIMATOR_REASONS = {
    "logged": (
        "пропенсити залогированы, поэтому DR несмещён при любой модели исхода, а модель "
        "исхода уменьшает его разброс по сравнению с IPS и SNIPS"
    ),
    "estimated": (
        "пропенсити оценены моделью, а IPS и SNIPS переносят её ошибку в оценку целиком, DM — "
        "ошибку модели исхода; DR смещается, только если ошиблись обе модели"
    ),
}
#: Объём, начиная с которого даже небольшая ошибка спецификации модели исхода сравнима
#: с интервалом DM (симуляции: покрытие падает до ~0.8 к 3·10⁴ строк).
DM_LARGE_N = 10_000
_Z_POWER_80 = 0.8416
_RW_DRAWS = 2000


def _model_name(model: Any, default: str) -> str:
    if model is None:
        return default
    steps = getattr(model, "steps", None)
    if steps:
        return " → ".join(type(step).__name__ for _, step in steps)
    return type(model).__name__


def _user_stacklevel() -> int:
    """``stacklevel`` для warnings: первый кадр стека вне пакета, то есть код пользователя."""
    frame, level = sys._getframe(1), 1
    while frame is not None and frame.f_globals.get("__name__", "").startswith("policyscope"):
        frame, level = frame.f_back, level + 1
    return level


def _terms(
    data: LogData, nuis: Nuisance, idx: np.ndarray, pi: np.ndarray, metric: str,
    p_taken: Optional[np.ndarray] = None,
) -> PolicyTerms:
    p = nuis.p_taken if p_taken is None else p_taken
    return policy_terms(pi[idx], data.a[idx], p, nuis.mu[metric])


def _baseline_contributions(
    data: LogData, nuis: Nuisance, idx: np.ndarray, metric: str, est: str,
    base_terms: Optional[PolicyTerms],
) -> np.ndarray:
    """Вклады базы для оценщика ``est``.

    База — политика A из логов (``base_terms is None``): для всех оценщиков, кроме DM,
    её значение — средняя награда. Для DM — прогноз той же модели исхода на
    залогированных действиях, ``mean(mu(x_i, a_i))``: тогда Δ у DM — контраст двух
    прогнозов одной модели, и её систематическая ошибка частично сокращается. В
    симуляциях доля убранного смещения зависела от пары политик (от нуля до ~0.45,
    иногда с перелётом), а RMSE контраста не был хуже, чем у базы ``mean(r)``.
    """
    r = data.rewards[metric][idx]
    if base_terms is not None:
        return contributions(est, r, base_terms)
    if est == "dm":
        return nuis.mu[metric][np.arange(len(idx)), data.a[idx]]
    return r


def _values(
    data: LogData, nuis: Nuisance, idx: np.ndarray, pis: list[np.ndarray], base: Optional[np.ndarray]
) -> np.ndarray:
    """Значения (V_B, V_base) для всех кандидатов, метрик и оценщиков: ``(P, M, E, 2)``."""
    out = np.empty((len(pis), len(data.reward_cols), len(ESTIMATORS), 2))
    for m, metric in enumerate(data.reward_cols):
        r = data.rewards[metric][idx]
        base_terms = _terms(data, nuis, idx, base, metric) if base is not None else None
        for e, est in enumerate(ESTIMATORS):
            out[:, m, e, 1] = _baseline_contributions(data, nuis, idx, metric, est, base_terms).mean()
        for i, pi in enumerate(pis):
            terms = _terms(data, nuis, idx, pi, metric)
            for e, est in enumerate(ESTIMATORS):
                out[i, m, e, 0] = contributions(est, r, terms).mean()
    return out


def _replicate(
    data: LogData, config: NuisanceConfig, pis: list[np.ndarray], base: Optional[np.ndarray],
    positions: np.ndarray, seed: int,
) -> np.ndarray:
    """Одна bootstrap-реплика: переобучение nuisance на реплике и все оценки."""
    nuis = fit_nuisance(data, config, seed=seed, positions=positions)
    return _values(data, nuis, positions, pis, base).reshape(-1)


class Evaluator:
    """Оценка политик-кандидатов по логам политики A.

    Args:
        logs: Логи A: строка — одно решение (контекст, действие, награда).
        action: Колонка с действием, которое выбрала A.
        reward: Колонка метрики или список колонок.
        propensity: Колонка с вероятностью, с которой A выбрала залогированное действие.
            ``None`` — пропенсити оцениваются моделью (тогда по умолчанию включается
            bootstrap: аналитический интервал не учитывает ошибку их оценки).
        cluster: Колонка единицы наблюдения (клиент), если на неё приходится несколько
            строк. ``None`` — строки независимы.
        features: Признаки контекста. ``None`` — авто-выбор (список попадает в отчёт).
        propensity_model: sklearn-классификатор для пропенсити (по умолчанию —
            логистическая регрессия).
        outcome_model: sklearn-модель для ``E[r|x, a]`` или словарь «метрика -> модель». По
            умолчанию выбирается по данным: логистическая регрессия (бинарная награда),
            двухчастная модель (неотрицательная с нулями, например выручка) или Ridge — все с
            взаимодействиями признаков и действия.
        n_folds: Число фолдов cross-fitting.
        propensity_floor: Пол оценённого пропенсити как доля частоты действия.
        lower_is_better: Метрики, где меньше — лучше (отток, расходы).
        thresholds: Пороги диагностики.
        random_state: Зерно (фолды, bootstrap, поправка на множественность).
        n_jobs: Параллелизм bootstrap (как в joblib; ``-1`` — все ядра).
    """

    def __init__(
        self,
        logs: pd.DataFrame,
        *,
        action: str,
        reward: Union[str, Sequence[str]],
        propensity: Optional[str] = None,
        cluster: Optional[str] = None,
        features: Optional[Sequence[str]] = None,
        propensity_model: Any = None,
        outcome_model: Any = None,
        n_folds: int = 5,
        propensity_floor: float = 0.1,
        lower_is_better: Union[str, Sequence[str]] = (),
        thresholds: Optional[Thresholds] = None,
        random_state: int = 0,
        n_jobs: Optional[int] = -1,
    ) -> None:
        self.data: LogData = prepare_logs(
            logs, action=action, reward=reward, propensity=propensity,
            cluster=cluster, features=features,
        )
        if isinstance(lower_is_better, str):
            lower_is_better = [lower_is_better]
        if isinstance(random_state, bool) or not isinstance(random_state, (int, np.integer)):
            raise ValueError("random_state должен быть целым числом (нужен для воспроизводимости)")
        groups = self.data.n_clusters
        if groups < n_folds:
            unit = f"кластеров («{cluster}»)" if cluster else "строк"
            raise ValueError(
                f"Для cross-fitting на {n_folds} фолдов нужно не меньше {n_folds} {unit}, а их {groups}. "
                "Уменьшите n_folds или соберите больше данных."
            )
        unknown = sorted(set(lower_is_better) - set(self.data.reward_cols))
        if unknown:
            raise ValueError(f"lower_is_better содержит метрики, которых нет в reward: {unknown}")
        self.lower_is_better = frozenset(lower_is_better)
        if isinstance(outcome_model, Mapping):
            extra = sorted(set(outcome_model) - set(self.data.reward_cols))
            if extra:
                raise ValueError(f"outcome_model задан для метрик, которых нет в reward: {extra}")
        hidden = list(self.data.reward_cols) + ([propensity] if propensity else [])
        # Политике показываются логи без наград и пропенсити — она не должна их видеть.
        self._frame = logs.drop(columns=hidden).copy()
        self.config = NuisanceConfig(
            propensity_model=propensity_model,
            outcome_model=outcome_model,
            n_folds=n_folds,
            propensity_floor=propensity_floor,
        )
        self.thresholds = thresholds or Thresholds()
        self.random_state = int(random_state)
        self.n_jobs = n_jobs
        self._nuisance: Optional[Nuisance] = None

    # ------------------------------------------------------------------
    # Публичный API
    # ------------------------------------------------------------------

    @property
    def nuisance(self) -> Nuisance:
        """Out-of-fold прогнозы nuisance-моделей (обучаются при первом обращении)."""
        if self._nuisance is None:
            self._nuisance = fit_nuisance(self.data, self.config, seed=self.random_state)
        return self._nuisance

    def evaluate(
        self,
        policy: Any,
        *,
        baseline: Any = None,
        estimator: str = "auto",
        inference: str = "auto",
        alpha: float = 0.05,
        n_boot: int = 200,
        name: str = "B",
        baseline_name: Optional[str] = None,
    ) -> Report:
        """Оценивает политику ``policy`` относительно ``baseline``.

        Args:
            policy: Политика-кандидат: ``DataFrame`` вероятностей с метками действий в
                колонках (см. :func:`policyscope.softmax` и др.), имя колонки с действием
                кандидата, ``Series`` действий или объект с ``action_probs(df)``.
            baseline: С чем сравнивать. ``None`` — с политикой A, как она есть в логах
                (``V(A)`` = средняя награда; у DM — прогноз модели на залогированных действиях).
                Политика в том же формате — с ней (например, с прод-версией A без
                исследования); её значение оценивается тем же оценщиком.
            estimator: Основной оценщик. ``auto`` (по умолчанию) выбирает его по тому,
                залогированы ли пропенсити, и объясняет выбор в отчёте; сейчас это всегда DR
                (почему — в ``report.settings["estimator_reason"]``). Явно: ``dr``, ``sndr``, ``ips``,
                ``snips`` или ``dm`` (точнее при верной модели исхода, но её смещение не
                проверяется; только со встроенной моделью и bootstrap-интервалом).
                Остальные оценщики считаются для сверки на тех же моделях.
            inference: ``auto`` — аналитический интервал при логированных пропенсити;
                bootstrap при оценённых пропенсити и при ``estimator="dm"``. Также
                ``influence``, ``bootstrap`` или ``none``.
            alpha: Уровень значимости (интервалы ``1 - alpha``).
            n_boot: Число bootstrap-реплик.
            name: Имя кандидата в отчёте.
            baseline_name: Имя базы в отчёте (по умолчанию ``A`` или ``база``).
        """
        board = self._run(
            {name: policy}, baseline=baseline, estimator=estimator, inference=inference,
            alpha=alpha, n_boot=n_boot, baseline_name=baseline_name,
        )
        return board.reports[name]

    def compare(
        self,
        candidates: Mapping[str, Any],
        *,
        baseline: Any = None,
        estimator: str = "auto",
        inference: str = "auto",
        alpha: float = 0.05,
        n_boot: int = 200,
        baseline_name: Optional[str] = None,
    ) -> Leaderboard:
        """Оценивает несколько кандидатов за один проход и строит рейтинг.

        Все кандидаты считаются на одних и тех же моделях и bootstrap-репликах, а
        p-value корректируются методом Romano–Wolf на всё семейство
        «кандидат × метрика».
        """
        if not candidates:
            raise ValueError("Передайте хотя бы одного кандидата")
        return self._run(
            dict(candidates), baseline=baseline, estimator=estimator, inference=inference,
            alpha=alpha, n_boot=n_boot, baseline_name=baseline_name,
        )

    # ------------------------------------------------------------------
    # Расчёт
    # ------------------------------------------------------------------

    def _resolve_estimator(self, estimator: str) -> tuple[str, Optional[str]]:
        """Основной оценщик и причина выбора (``None`` — оценщик задан явно)."""
        if estimator != "auto":
            return estimator, None
        source = "logged" if self.data.propensity is not None else "estimated"
        return "dr", AUTO_ESTIMATOR_REASONS[source]

    def _replicate_config(self) -> NuisanceConfig:
        """Настройки для bootstrap-реплик: ``C`` пропенсити и дизайн модели исхода уже выбраны."""
        nuis = self.nuisance
        return replace(self.config, propensity_C=nuis.propensity_C, interactions=dict(nuis.interactions))

    def _validate(self, estimator: str, inference: str, alpha: float, n_boot: int) -> str:
        if estimator not in PRIMARY_ESTIMATORS:
            raise ValueError(
                f"Неизвестный оценщик '{estimator}'. Доступны: 'auto', {', '.join(map(repr, PRIMARY_ESTIMATORS))}"
            )
        if self.data.propensity is None and estimator in ("ips", "snips"):
            raise ValueError(
                f"При оценённых пропенсити '{estimator}' не может быть основным: смещение от "
                "ошибки модели пропенсити входит в него напрямую. Используйте 'dr' или 'sndr'."
            )
        if estimator == "dm":
            custom = [m for m in self.data.reward_cols if self.config.model_for(m) is not None]
            if custom:
                raise ValueError(
                    f"Основной DM доступен только со встроенной моделью исхода, а для {custom} "
                    "задана своя. Для гибких ML-моделей у DM нет валидного интервала: в симуляциях "
                    "bootstrap покрывал истину в 30–70% случаев. DM всё равно показывается точкой "
                    "в таблице согласия; основным возьмите 'dr'."
                )
            if inference == "influence":
                raise ValueError(
                    "Аналитический интервал для DM не реализован: формула по вкладам dm_i "
                    "игнорирует ошибку оценки модели исхода и занижает разброс. Используйте "
                    "inference='bootstrap' (или 'auto')."
                )
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha должна лежать в (0, 1)")
        if inference == "auto":
            if estimator == "dm" or self.data.propensity is None:
                inference = "bootstrap"
            else:
                inference = "influence"
        if inference not in ("influence", "bootstrap", "none"):
            raise ValueError("inference: 'auto', 'influence', 'bootstrap' или 'none'")
        if inference == "bootstrap" and n_boot < 50:
            raise ValueError("n_boot должен быть не меньше 50")
        return inference

    def _bootstrap_eta(self, n_boot: int, estimator: str) -> None:
        if self.n_jobs is None:
            jobs = 1
        elif self.n_jobs < 0:
            jobs = max(1, (os.cpu_count() or 1) + 1 + self.n_jobs)
        else:
            jobs = max(1, int(self.n_jobs))
        # Реплики на всех ядрах мешают друг другу, плюс старт процессов-воркеров.
        speedup = max(1.0, jobs / 2) if jobs > 1 else 1.0
        eta = self.nuisance.fit_seconds * n_boot / speedup + (5.0 if jobs > 1 else 0.0)
        if eta > 30:
            # warnings, а не logging: пакет не настраивает логирование, и сообщение
            # иначе не дошло бы до пользователя в Jupyter.
            if self.data.propensity is None:
                hint = "Логированные пропенсити позволили бы обойтись аналитическим интервалом."
            elif estimator == "dm":
                hint = "Для DM честный интервал — только bootstrap."
            else:
                hint = "При логированных пропенсити быстрее inference='influence'."
            warnings.warn(
                f"Bootstrap: {n_boot} реплик с переобучением моделей, ориентировочно "
                f"{eta / 60:.1f} мин. {hint}",
                stacklevel=_user_stacklevel(),
            )

    def _run(
        self,
        candidates: dict[str, Any],
        *,
        baseline: Any,
        estimator: str,
        inference: str,
        alpha: float,
        n_boot: int,
        baseline_name: Optional[str],
    ) -> Leaderboard:
        estimator, estimator_reason = self._resolve_estimator(estimator)
        inference = self._validate(estimator, inference, alpha, n_boot)
        data, th = self.data, self.thresholds
        base_label = baseline_name or ("A" if baseline is None else "база")
        if base_label in candidates:
            raise ValueError(f"Имя базы '{base_label}' совпадает с именем кандидата")
        names = list(candidates)
        pis = [policy_matrix(candidates[n], data, self._frame, role=n) for n in names]
        base = policy_matrix(baseline, data, self._frame, role=base_label) if baseline is not None else None

        nuis = self.nuisance
        full = np.arange(data.n)
        metrics = data.reward_cols
        e_primary = ESTIMATORS.index(estimator)
        P, M, E = len(names), len(metrics), len(ESTIMATORS)

        # Построчные вклады на полной выборке.
        psi_b = np.empty((P, M, E, data.n))
        psi_a = np.empty((M, E, data.n))
        terms_b: dict[tuple[int, int], PolicyTerms] = {}
        for m, metric in enumerate(metrics):
            r = data.rewards[metric]
            base_terms = _terms(data, nuis, full, base, metric) if base is not None else None
            for e, est in enumerate(ESTIMATORS):
                psi_a[m, e] = _baseline_contributions(data, nuis, full, metric, est, base_terms)
            for i, pi in enumerate(pis):
                terms_b[i, m] = _terms(data, nuis, full, pi, metric)
                for e, est in enumerate(ESTIMATORS):
                    psi_b[i, m, e] = contributions(est, r, terms_b[i, m])

        v_b = psi_b.mean(axis=-1)
        v_a = psi_a.mean(axis=-1)
        stats: dict[str, np.ndarray] = {}
        extra_findings: list[Finding] = []
        rw_t_obs = np.full((P, M), np.nan)
        rw_null: Optional[np.ndarray] = None
        rng = np.random.default_rng(self.random_state + 1)

        if inference == "influence":
            stats = self._influence(psi_b, psi_a, v_b, v_a, alpha)
            se_family = stats["delta_se"][:, :, e_primary]
            delta_family = v_b[:, :, e_primary] - v_a[None, :, e_primary]
            with np.errstate(divide="ignore", invalid="ignore"):
                rw_t_obs = np.where(se_family > 0, delta_family / se_family, np.nan)
            if P * M > 1:
                psi_family = (psi_b[:, :, e_primary] - psi_a[None, :, e_primary]).reshape(P * M, -1)
                sums = cluster_sums(psi_family.T, data.clusters, data.n_clusters)
                rw_null = multiplier_tstats(sums, se_family.reshape(-1), data.n, _RW_DRAWS, rng)
            if nuis.source == "estimated":
                extra_findings.append(Finding(
                    code="influence_with_estimated_propensity",
                    severity="caution",
                    title="Аналитический интервал при оценённых пропенсити",
                    detail="Интервал не учитывает ошибку оценки пропенсити и может быть занижен.",
                    advice="Используйте inference='auto' (bootstrap) для честного интервала.",
                ))
        elif inference == "bootstrap":
            self._bootstrap_eta(n_boot, estimator)

            samples = run_bootstrap(
                partial(_replicate, data, self._replicate_config(), pis, base),
                clusters=data.clusters, n_clusters=data.n_clusters,
                clustered=data.is_clustered, n_boot=n_boot,
                seed=self.random_state + 2, n_jobs=self.n_jobs,
            ).reshape(n_boot, P, M, E, 2)
            stats = self._bootstrap_stats(samples, v_b, v_a, alpha)
            failed = float(np.mean(~np.isfinite(samples).all(axis=(1, 2, 3, 4))))
            if failed > 0:
                extra_findings.append(Finding(
                    code="bootstrap_failures",
                    severity="block" if failed > th.failed_replicates else "caution",
                    title="Часть bootstrap-реплик не посчиталась",
                    detail=f"Не удалось посчитать {100 * failed:.0f}% реплик (вырожденные выборки).",
                    advice="Увеличьте объём данных или уменьшите n_folds.",
                ))
            d = samples[..., e_primary, 0] - samples[..., e_primary, 1]
            delta_hat = v_b[:, :, e_primary] - v_a[None, :, e_primary]
            sd = nan_std(d)
            with np.errstate(divide="ignore", invalid="ignore"):
                rw_t_obs = np.where(sd > 0, delta_hat / sd, np.nan)
                rw_null = ((d - delta_hat) / sd).reshape(n_boot, -1)

        p_primary = stats.get("p_value")
        if p_primary is not None:
            p_family = p_primary[:, :, e_primary]
            p_adj = p_family.copy() if rw_null is None else romano_wolf(
                rw_t_obs.reshape(-1), rw_null
            ).reshape(P, M)
        else:
            p_adj = None

        settings = self._settings(estimator, estimator_reason, inference, alpha, n_boot)
        reports = {}
        for i, name in enumerate(names):
            reports[name] = self._report(
                i, name, base_label, pis[i], base, psi_b, psi_a, v_b, v_a, stats, p_adj,
                terms_b, estimator, inference, alpha, extra_findings, settings,
            )
        # Рейтинг: лучший по первой метрике — первым (с учётом её направления).
        first = metrics[0]
        sign = -1.0 if first in self.lower_is_better else 1.0

        def rank_key(name: str) -> float:
            value = reports[name][first].delta.value
            return -sign * value if np.isfinite(value) else np.inf

        reports = {name: reports[name] for name in sorted(reports, key=rank_key)}
        table = pd.concat(
            {name: rep.summary() for name, rep in reports.items()}, names=["candidate"]
        ).reset_index()
        return Leaderboard(table=table, reports=reports, baseline=base_label, alpha=alpha)

    # ------------------------------------------------------------------
    # Инференс
    # ------------------------------------------------------------------

    def _influence(self, psi_b, psi_a, v_b, v_a, alpha) -> dict[str, np.ndarray]:
        data = self.data
        P, M, E, _ = psi_b.shape
        keys = ("v_b_low", "v_b_high", "v_a_low", "v_a_high", "delta_low", "delta_high",
                "delta_se", "p_value", "lift_low", "lift_high")
        out = {k: np.full((P, M, E), np.nan) for k in keys}
        tc = t_critical(alpha, data.n_clusters)
        for m in range(M):
            for e in range(E):
                sa = analytic_summary(psi_a[m, e], data.clusters, data.n_clusters, alpha)
                for i in range(P):
                    cols = [psi_b[i, m, e], psi_b[i, m, e] - psi_a[m, e]]
                    vb, va = v_b[i, m, e], v_a[m, e]
                    if vb > 0 and va > 0:
                        cols.append(psi_b[i, m, e] / vb - psi_a[m, e] / va)
                    s = analytic_summary(np.column_stack(cols), data.clusters, data.n_clusters, alpha)
                    out["v_b_low"][i, m, e], out["v_b_high"][i, m, e] = s["low"][0], s["high"][0]
                    out["v_a_low"][i, m, e], out["v_a_high"][i, m, e] = sa["low"][0], sa["high"][0]
                    out["delta_low"][i, m, e], out["delta_high"][i, m, e] = s["low"][1], s["high"][1]
                    out["delta_se"][i, m, e] = s["se"][1]
                    out["p_value"][i, m, e] = s["p_value"][1]
                    if len(cols) == 3:
                        log_ratio, se = np.log(vb / va), s["se"][2]
                        out["lift_low"][i, m, e] = np.exp(log_ratio - tc * se) - 1
                        out["lift_high"][i, m, e] = np.exp(log_ratio + tc * se) - 1
        return out

    @staticmethod
    def _bootstrap_stats(samples, v_b, v_a, alpha) -> dict[str, np.ndarray]:
        sb, sa = samples[..., 0], samples[..., 1]
        d = sb - sa
        delta = v_b - v_a[None]
        out: dict[str, np.ndarray] = {}
        out["v_b_low"], out["v_b_high"] = percentile_interval(sb, alpha)
        a_low, a_high = percentile_interval(sa, alpha)
        out["v_a_low"], out["v_a_high"] = a_low, a_high
        out["delta_low"], out["delta_high"] = percentile_interval(d, alpha)
        out["delta_se"] = nan_std(d)
        out["p_value"] = centered_pvalue(d, delta)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where((sb > 0) & (sa > 0), sb / sa - 1, np.nan)
        out["lift_low"], out["lift_high"] = percentile_interval(ratio, alpha)
        positive = (v_b > 0) & (v_a[None] > 0)
        out["lift_low"] = np.where(positive, out["lift_low"], np.nan)
        out["lift_high"] = np.where(positive, out["lift_high"], np.nan)
        return out

    # ------------------------------------------------------------------
    # Сборка отчёта
    # ------------------------------------------------------------------

    def _settings(
        self, estimator: str, estimator_reason: Optional[str], inference: str, alpha: float, n_boot: int,
    ) -> dict:
        d, nuis = self.data, self.nuisance
        return {
            "estimator": estimator,
            "estimator_auto": estimator_reason is not None,
            "estimator_reason": estimator_reason,
            "inference": inference,
            "alpha": alpha,
            "n_boot": n_boot if inference == "bootstrap" else None,
            "n_folds": self.config.n_folds,
            "action": d.action_col,
            "rewards": list(d.reward_cols),
            "propensity": d.propensity_col,
            "propensity_source": self.nuisance.source,
            "propensity_floor": self.config.propensity_floor if d.propensity is None else None,
            "cluster": d.cluster_col,
            "features": list(d.features),
            "features_auto": d.features_auto,
            "skipped_features": dict(d.skipped_features),
            "propensity_model": (
                "логированные" if d.propensity is not None else
                _model_name(
                    self.config.propensity_model,
                    f"LogisticRegression (OOF, C = {nuis.propensity_C:g} по out-of-fold log-loss)"
                    if nuis.propensity_C is not None else "LogisticRegression (OOF)",
                )
            ),
            "propensity_C": nuis.propensity_C,
            "outcome_model": {
                m: _model_name(
                    self.config.model_for(m),
                    OUTCOME_KIND_LABELS[outcome_kind(d.rewards[m], d.binary[m])]
                    + (" с x×a" if nuis.interactions.get(m, True) else " без x×a"),
                )
                for m in d.reward_cols
            },
            "outcome_design": dict(nuis.design_notes),
            "random_state": self.random_state,
            "actions": list(d.actions),
        }

    def _outcome_quality(self, metric: str) -> dict[str, float]:
        r = self.data.rewards[metric]
        mu = self.nuisance.mu[metric][np.arange(self.data.n), self.data.a]
        if self.data.binary[metric]:
            if 0 < r.mean() < 1:
                from sklearn.metrics import roc_auc_score

                return {"roc_auc": float(roc_auc_score(r, mu))}
            return {}
        sst = float(np.sum((r - r.mean()) ** 2))
        return {"r2": float(1 - np.sum((r - mu) ** 2) / sst)} if sst > 0 else {}

    def _floor_sensitivity(self, pi, base, metric, estimator) -> Optional[dict[str, float]]:
        nuis = self.nuisance
        if nuis.source != "estimated" or estimator == "dm":  # DM пропенсити не использует
            return None
        data = self.data
        r = data.rewards[metric]
        full = np.arange(data.n)
        out = {}
        for label, factor in (("x0.5", 0.5), ("x1", 1.0), ("x2", 2.0)):
            p = nuis.floored(data.a, factor)
            vb = contributions(estimator, r, _terms(data, nuis, full, pi, metric, p)).mean()
            va = r.mean() if base is None else contributions(
                estimator, r, _terms(data, nuis, full, base, metric, p)
            ).mean()
            out[label] = float(vb - va)
        return out

    def _report(
        self, i, name, base_label, pi, base, psi_b, psi_a, v_b, v_a, stats, p_adj,
        terms_b, estimator, inference, alpha, extra_findings, settings,
    ) -> Report:
        data, th, nuis = self.data, self.thresholds, self.nuisance
        e_primary = ESTIMATORS.index(estimator)
        overlap = overlap_summary(data, nuis, pi, alpha)
        findings = policy_findings(data, nuis, overlap, th)
        if base is not None:
            base_overlap = overlap_summary(data, nuis, base, alpha)
            findings += policy_findings(data, nuis, base_overlap, th, baseline=base_label)
        findings += extra_findings
        if estimator == "dm":
            findings.append(Finding(
                code="model_based_estimate",
                severity="caution",
                title="Оценка держится только на модели исхода (DM)",
                detail=(
                    "DM точнее DR, если модель исхода верна, но её смещение по логам не "
                    "проверяется: сверка с DR замечает лишь грубые расхождения, порядка "
                    "нескольких стандартных ошибок DM. "
                    + ("При таком объёме данных (от 10 000 строк) остаточная ошибка спецификации "
                       "становится заметной долей стандартной ошибки; в симуляциях к 3·10⁴ строк "
                       "покрытие падало до ~0.8. " if data.n >= DM_LARGE_N else "")
                    + "Интервал описывает разброс оценки, но не смещение модели."
                ),
                advice=(
                    "Принимайте решение по DM, только если уверены в модели исхода и DR в таблице "
                    "оценщиков ему не противоречит; иначе основным возьмите 'dr'."
                ),
            ))
        ess_value = ess(terms_b[i, 0].w)
        has_ci = inference != "none"

        def get(key, m, e):
            return float(stats[key][i, m, e]) if has_ci else None

        metric_reports = {}
        for m, metric in enumerate(data.reward_cols):
            binary = data.binary[metric]
            delta_value = float(v_b[i, m, e_primary] - v_a[m, e_primary])
            se_delta = get("delta_se", m, e_primary)
            terms = terms_b[i, m]
            base_terms = _terms(data, nuis, np.arange(data.n), base, metric) if base is not None else None
            checks, metric_findings = triangulate(
                data, metric, estimator=estimator, w=terms.w, dm=terms.dm,
                mu_taken=terms.mu_taken,
                base_w=base_terms.w if base_terms is not None else None,
                base_dm=base_terms.dm if base_terms is not None else None,
                se_delta=se_delta if se_delta is not None else float(
                    analytic_summary(psi_b[i, m, e_primary] - psi_a[m, e_primary],
                                     data.clusters, data.n_clusters, alpha)["se"][0]
                ),
                estimated_propensity=nuis.source == "estimated",
                ess_value=ess_value, th=th,
            )
            if not nuis.interactions.get(metric, True) and nuis.design_notes.get(metric, "").startswith("без x×a: "):
                metric_findings.append(Finding(
                    code="outcome_without_interactions",
                    severity="info",
                    metric=metric,
                    title="Модель исхода без взаимодействий x×a",
                    detail=(
                        f"{nuis.design_notes[metric].removeprefix('без x×a: ').capitalize()}. Модель "
                        "различает действия только сдвигом: DM теряет неоднородность эффекта, DR — "
                        "часть точности (смещения при логированных пропенсити это не даёт)."
                    ),
                    advice="Передайте features= с меньшим числом признаков или задайте outcome_model=.",
                ))
            r_metric = data.rewards[metric]
            if np.ptp(r_metric) == 0:
                metric_findings.append(Finding(
                    code="constant_metric",
                    severity="block",
                    metric=metric,
                    title="Метрика в логах не меняется",
                    detail=(
                        f"Все {data.n} значений «{metric}» равны {r_metric[0]:g}: по таким логам "
                        "нельзя оценить, изменит ли её другая политика."
                    ),
                    advice="Проверьте период логов и определение метрики.",
                ))
            concentration = variance_concentration(
                psi_b[i, m, e_primary] - psi_a[m, e_primary], data, th, metric
            )
            if concentration is not None:
                metric_findings.append(concentration)
            floor = self._floor_sensitivity(pi, base, metric, estimator)
            if floor is not None and se_delta is not None and (
                max(floor.values()) - min(floor.values()) > se_delta
            ):
                metric_findings.append(Finding(
                    code="floor_sensitive",
                    severity="caution",
                    metric=metric,
                    title="Оценка чувствительна к полу пропенсити",
                    detail=(
                        "При изменении пола вдвое Δ меняется больше чем на стандартную ошибку: "
                        f"{floor['x0.5']:+.4g} / {floor['x1']:+.4g} / {floor['x2']:+.4g}."
                    ),
                    advice="Политика A плохо воспроизводится моделью; нужны логированные пропенсити.",
                ))

            primary_values = [v_b[i, m, e_primary], v_a[m, e_primary]]
            if has_ci:
                primary_values += [get("delta_low", m, e_primary), get("delta_high", m, e_primary)]
            if not np.all(np.isfinite(primary_values)):
                metric_findings.append(Finding(
                    code="non_finite_estimate",
                    severity="block",
                    metric=metric,
                    title="Оценка не посчиталась",
                    detail="Основной оценщик дал нечисловое значение (например, нет перекрытия политик).",
                    advice="Проверьте таблицу перекрытия и модель исхода.",
                ))
            all_findings = tuple(findings) + tuple(metric_findings)
            trust = trust_level(all_findings)
            low, high = get("delta_low", m, e_primary), get("delta_high", m, e_primary)
            if not has_ci:
                decision = "not_computed"
            elif trust == "unreliable" or not (np.isfinite(low) and np.isfinite(high)):
                decision = "inconclusive"
            elif low is not None and low > 0:
                decision = "worse" if metric in self.lower_is_better else "better"
            elif high is not None and high < 0:
                decision = "better" if metric in self.lower_is_better else "worse"
            else:
                decision = "no_difference"

            lift = None
            vb, va = float(v_b[i, m, e_primary]), float(v_a[m, e_primary])
            if vb > 0 and va > 0:
                lift = Estimate(vb / va - 1, get("lift_low", m, e_primary), get("lift_high", m, e_primary))
            mde = None
            if se_delta is not None and np.isfinite(se_delta):
                mde = (t_critical(alpha, data.n_clusters) + _Z_POWER_80) * se_delta

            rows = []
            for e, est in enumerate(ESTIMATORS):
                # Интервал DM показывается, только если DM выбран основным явно.
                show_ci = has_ci and (est != "dm" or estimator == "dm")
                rows.append(EstimatorRow(
                    estimator=est,
                    v_b=Estimate(float(v_b[i, m, e]), get("v_b_low", m, e) if show_ci else None,
                                 get("v_b_high", m, e) if show_ci else None),
                    delta=Estimate(float(v_b[i, m, e] - v_a[m, e]),
                                   get("delta_low", m, e) if show_ci else None,
                                   get("delta_high", m, e) if show_ci else None,
                                   get("delta_se", m, e) if show_ci else None),
                    p_value=get("p_value", m, e) if show_ci else None,
                ))

            statuses = [c.status for c in checks]
            if "fail" in statuses:
                agreement = "disagree"
            elif statuses and all(s == "ok" for s in statuses):
                agreement = "agree"
            else:
                agreement = "unclear"
            point_deltas = [row.delta.value for row in rows if np.isfinite(row.delta.value)]

            metric_reports[metric] = MetricReport(
                metric=metric,
                estimator=estimator,
                binary=binary,
                lower_is_better=metric in self.lower_is_better,
                v_a=Estimate(va, get("v_a_low", m, e_primary), get("v_a_high", m, e_primary)),
                v_b=Estimate(vb, get("v_b_low", m, e_primary), get("v_b_high", m, e_primary)),
                delta=Estimate(delta_value, low, high, se_delta),
                p_value=get("p_value", m, e_primary),
                p_value_adj=float(p_adj[i, m]) if p_adj is not None else None,
                lift=lift,
                mde=mde,
                decision=decision,
                trust=trust,
                findings=all_findings,
                checks=checks,
                rows=tuple(rows),
                outcome_quality=self._outcome_quality(metric),
                floor_sensitivity=floor,
                agreement=agreement,
                agreement_spread=float(np.ptp(point_deltas)) if point_deltas else None,
            )
        return Report(
            policy=name,
            baseline=base_label,
            metrics=metric_reports,
            overlap=overlap,
            findings=tuple(findings),
            settings=settings,
        )


def evaluate(
    logs: pd.DataFrame,
    policy: Any,
    *,
    action: str,
    reward: Union[str, Sequence[str]],
    propensity: Optional[str] = None,
    cluster: Optional[str] = None,
    features: Optional[Sequence[str]] = None,
    baseline: Any = None,
    estimator: str = "auto",
    inference: str = "auto",
    alpha: float = 0.05,
    n_boot: int = 200,
    name: str = "B",
    baseline_name: Optional[str] = None,
    **evaluator_options: Any,
) -> Report:
    """Одноразовая оценка: ``Evaluator(logs, ...).evaluate(policy, ...)``.

    Аргументы данных (``action``, ``reward``, ``propensity``, ``cluster``, ``features``)
    и ``evaluator_options`` (``propensity_model``, ``outcome_model``, ``n_folds``,
    ``propensity_floor``, ``lower_is_better``, ``thresholds``, ``random_state``,
    ``n_jobs``) — как у :class:`Evaluator`; аргументы оценки (``baseline``, ``estimator``,
    ``inference``, ``alpha``, ``n_boot``, ``name``, ``baseline_name``) — как у
    :meth:`Evaluator.evaluate`.

    Для нескольких кандидатов на одних логах создайте :class:`Evaluator` один раз —
    модели не будут переобучаться.
    """
    ev = Evaluator(
        logs, action=action, reward=reward, propensity=propensity, cluster=cluster,
        features=features, **evaluator_options,
    )
    return ev.evaluate(
        policy, baseline=baseline, estimator=estimator, inference=inference,
        alpha=alpha, n_boot=n_boot, name=name, baseline_name=baseline_name,
    )
