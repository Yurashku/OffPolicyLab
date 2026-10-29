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
    cluster_se,
    cluster_sums,
    multiplier_tstats,
    nan_std,
    percentile_interval,
    romano_wolf,
    run_bootstrap,
    simultaneous_critical,
    t_critical,
)
from policyscope.nuisance import (
    OUTCOME_KIND_LABELS,
    Nuisance,
    NuisanceConfig,
    fit_nuisance,
    make_folds,
    outcome_kind,
)
from policyscope.policy import _logging_terms, policy_matrix
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
        "пропенсити известны точно, поэтому DR несмещён при любой модели исхода, а модель "
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


def _propensity_label(nuis: Nuisance) -> str:
    """Подпись модели пропенсити по умолчанию для отчёта."""
    losses = nuis.propensity_losses
    logistic = [v for key, v in losses.items() if key != "boosting"]
    if nuis.propensity_kind == "boosting":
        if not (losses and logistic):
            return "градиентный бустинг (OOF)"
        return (f"градиентный бустинг (OOF, выбран по out-of-fold log-loss: {losses['boosting']:.3f} "
                f"против {min(logistic):.3f} у логистической регрессии)")
    if nuis.propensity_C is not None:
        return f"LogisticRegression (OOF, C = {nuis.propensity_C:g}, подобрано по out-of-fold log-loss)"
    return "LogisticRegression (OOF)"


def _fmt_number(x: float) -> str:
    return f"{x:.3g}"


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
            bootstrap: аналитический интервал не учитывает ошибку их оценки) или
            считаются по ``logging_policy``.
        logging_policy: Известная политика A — вместо колонки пропенсити, если их не
            логировали, но правило выбора известно (скоринговая модель плюс способ выбора);
            формат — как у кандидата. Если логи собраны несколькими политиками (разные версии
            модели в разные периоды) — словарь «метка из колонки ``logger`` -> политика»:
            пропенсити считаются по их смеси (см. :func:`policyscope.policy.logging_propensity`).
        logger: Колонка с меткой политики, записавшей строку; нужна вместе со словарём
            ``logging_policy``.
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
        logging_policy: Any = None,
        logger: Optional[str] = None,
        propensity_model: Any = None,
        outcome_model: Any = None,
        n_folds: int = 5,
        propensity_floor: float = 0.1,
        lower_is_better: Union[str, Sequence[str]] = (),
        thresholds: Optional[Thresholds] = None,
        random_state: int = 0,
        n_jobs: Optional[int] = -1,
    ) -> None:
        if logging_policy is not None and propensity is not None:
            raise ValueError("Укажите что-то одно: колонку propensity или logging_policy")
        if logger is not None and logging_policy is None:
            raise ValueError("logger= нужен только вместе с logging_policy (словарём политик)")
        self.data: LogData = prepare_logs(
            logs, action=action, reward=reward, propensity=propensity,
            cluster=cluster, features=features, logger=logger,
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
        self.propensity_origin = "column" if propensity else None
        self._logger_matrices: dict = {}
        self._logging_matrix: Optional[np.ndarray] = None
        if logging_policy is not None:
            p, self._logger_matrices = _logging_terms(
                logging_policy, self.data, self._frame, logs[logger] if logger else None,
            )
            shares = logs[logger].value_counts(normalize=True) if logger else {None: 1.0}
            self._logging_matrix = sum(shares[label] * m for label, m in self._logger_matrices.items())
            self.data = replace(self.data, propensity=p)
            self.propensity_origin = "mixture" if logger else "policy"
        self.logger_col = logger
        self._logger_labels = logs[logger].to_numpy(dtype=object) if logger else None
        self._logger_shares = logs[logger].value_counts(normalize=True).to_dict() if logger else {}
        self.config = NuisanceConfig(
            propensity_model=propensity_model,
            outcome_model=outcome_model,
            n_folds=n_folds,
            propensity_floor=propensity_floor,
        )
        self.thresholds = thresholds or Thresholds()
        self._mixture_cache: Optional[list[Finding]] = None
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
                известны ли пропенсити, и объясняет выбор в отчёте; сейчас это всегда DR
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

    def support(self) -> pd.DataFrame:
        """Какие действия логирующая политика могла выбрать для каждой строки логов.

        Таблица ``bool`` (строки — логи, колонки — действия). Известна только при
        ``logging_policy=``. Пригодится, чтобы оценить вариант новой модели, который выбирает
        лишь среди уже показанных клиенту действий: ``ps.greedy(scores, allowed=ev.support())``.
        """
        if self._logging_matrix is None:
            raise ValueError(
                "Покрытие известно точно, только когда логирующая политика задана (logging_policy=). "
                "С колонкой пропенсити известна лишь вероятность залогированного действия."
            )
        return pd.DataFrame(self._logging_matrix > 0, index=self.data.index, columns=self.data.actions)

    def _replicate_config(self) -> NuisanceConfig:
        """Настройки для bootstrap-реплик: ``C`` пропенсити и дизайн модели исхода уже выбраны."""
        nuis = self.nuisance
        return replace(self.config, propensity_C=nuis.propensity_C, propensity_kind=nuis.propensity_kind,
                       interactions=dict(nuis.interactions))

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
        # Реплики на всех ядрах мешают друг другу (память, частота ядер), плюс старт
        # процессов-воркеров: на 8-ядерном ноутбуке реальное ускорение около 2× — так на
        # Hillstrom (13 тыс. строк) и на Open Bandit Dataset (100 тыс. строк, 34 действия).
        speedup = max(1.0, jobs / 4) if jobs > 1 else 1.0
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
        for role, policy in [*candidates.items(), (base_label, baseline)]:
            if isinstance(policy, str) and policy in data.features:
                raise ValueError(
                    f"Колонка «{policy}» задаёт политику {role}, но она же — признак моделей. Решение "
                    "политики не может быть признаком клиента: передайте features= без неё."
                )
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
        samples: Optional[np.ndarray] = None
        extra_findings: list[Finding] = []
        if self.propensity_origin == "mixture":
            if self._mixture_cache is None:  # от кандидата не зависит — считаем один раз
                self._mixture_cache = self._mixture_findings()
            extra_findings += self._mixture_cache
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

        simultaneous, pairwise = (
            self._competition(names, psi_b, psi_a, v_b, v_a, samples, inference, alpha, e_primary, rng)
            if inference != "none" else (None, None)
        )
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
        if simultaneous is not None:
            table = table.merge(simultaneous, on=["candidate", "metric"], how="left")
        return Leaderboard(
            table=table, reports=reports, baseline=base_label, alpha=alpha,
            pairwise=pairwise if pairwise is not None else pd.DataFrame(),
        )

    def _logger_shift_auc(self) -> Optional[float]:
        """ROC AUC, с которым источник логов угадывается по признакам клиента (out-of-fold).

        Около 0.5 — составы клиентов в периодах разных политик не различаются; заметно
        выше — различаются, и доли периодов как пропенсити смеси неверны для части клиентов.
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        data = self.data
        labels, y = np.unique(self._logger_labels, return_inverse=True)
        if len(labels) < 2 or data.X.shape[1] == 0:
            return None
        groups = data.clusters if data.is_clustered else np.arange(data.n)
        folds = make_folds(groups, self.config.n_folds, np.random.default_rng(self.random_state))
        probs = np.zeros((data.n, len(labels)))
        for f in range(self.config.n_folds):
            train, hold = folds != f, folds == f
            if len(np.unique(y[train])) < len(labels):
                return None
            model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(data.X[train], y[train])
            probs[hold] = model.predict_proba(data.X[hold])
        if len(labels) == 2:
            return float(roc_auc_score(y, probs[:, 1]))
        return float(roc_auc_score(y, probs, multi_class="ovr"))

    def _logger_drift(self) -> tuple[int, list[tuple[str, float, float]]]:
        """Различается ли отклик между периодами при одном и том же решении.

        Берутся строки, где все логирующие политики выбирают одно и то же действие: там
        период на решение не влияет, и остаток ``r − mu(x, a)`` модели исхода (период в неё
        не входит) не должен от него зависеть. Различие средних остатков между периодами
        (однофакторный дисперсионный анализ) — признак сезонности или изменений продукта.

        Returns:
            Число таких строк и список «метрика, p-value, разброс средних остатков»; пустой,
            если строк мало (нужно не меньше 30 в двух периодах).
        """
        from scipy import stats

        data, nuis = self.data, self.nuisance
        stacked = np.stack(list(self._logger_matrices.values()))
        agree = np.flatnonzero(stacked.min(axis=0).max(axis=1) >= 0.99)
        labels = self._logger_labels[agree]
        groups = [agree[labels == label] for label in np.unique(labels)]
        groups = [g for g in groups if len(g) >= 30]
        if len(groups) < 2:
            return len(agree), []
        out = []
        for metric in data.reward_cols:
            resid = data.rewards[metric] - nuis.mu[metric][np.arange(data.n), data.a]
            samples = [resid[g] for g in groups]
            if all(np.ptp(s) == 0 for s in samples):
                continue
            means = [float(s.mean()) for s in samples]
            out.append((metric, float(stats.f_oneway(*samples).pvalue), max(means) - min(means)))
        return len(agree), out

    def _mixture_findings(self) -> list[Finding]:
        shares = ", ".join(f"{label} — {100 * share:.0f}%" for label, share in self._logger_shares.items())
        auc = self._logger_shift_auc()
        shift = auc is not None and auc > self.thresholds.logger_shift_auc
        n_agree, drift = self._logger_drift()
        drifted = [(m, p, spread) for m, p, spread in drift if p < self.thresholds.logger_drift_p]
        if drift and not drifted:
            drift_text = (f" Сезонность проверена на {n_agree} строках, где все политики выбирают одно и "
                          "то же: отклик между периодами не различается.")
        elif not drift:
            drift_text = (" Проверить сезонность не на чем: мало строк, где все политики выбирают одно и "
                          "то же.")
        else:
            drift_text = ""
        findings = [Finding(
            code="logging_mixture",
            severity="info",
            title="Пропенсити посчитаны по смеси логирующих политик",
            detail=(
                f"Логи записаны {len(self._logger_shares)} политиками (колонка «{self.logger_col}»: "
                f"{shares}). Оценка несмещена, если клиенты и их отклик в периодах разных политик "
                "сопоставимы."
                + (f" Период по признакам клиента угадывается с ROC AUC {auc:.2f}: составы не "
                   "различаются." if auc is not None and not shift else "")
                + drift_text
            ),
            advice="Убедитесь, что между периодами не менялись продукт и цены.",
        )]
        if drifted:
            worst = min(drifted, key=lambda item: item[1])
            findings.append(Finding(
                code="logger_drift",
                severity="caution",
                metric=None,
                title="Отклик различается между периодами",
                detail=(
                    f"На {n_agree} строках, где все политики выбирают одно и то же, средний отклик "
                    f"«{worst[0]}» с поправкой на признаки различается между периодами на "
                    f"{_fmt_number(worst[2])} (p = {worst[1]:.2g}). Похоже на сезонность или изменения "
                    "продукта: смесь периодов смешивает и эти различия."
                ),
                advice="Возьмите соседние, сопоставимые периоды или оценивайте сезоны по отдельности.",
            ))
        if shift:
            findings.append(Finding(
                code="logger_shift",
                severity="caution",
                title="Клиенты в периодах разных политик различаются",
                detail=(
                    f"Период по признакам клиента угадывается с ROC AUC {auc:.2f} (порог "
                    f"{self.thresholds.logger_shift_auc:.2f}). Доли периодов тогда неверны как "
                    "пропенсити смеси для части клиентов, и оценка смещается."
                ),
                advice="Возьмите периоды с похожим составом клиентов или оценивайте по одному периоду.",
            ))
        return findings

    def _competition(
        self, names, psi_b, psi_a, v_b, v_a, samples, inference, alpha, e, rng,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Одновременные интервалы Δ кандидатов и попарные сравнения — по каждой метрике.

        Семейство — кандидаты одной метрики (для Δ к базе) и все пары кандидатов (для
        разностей между ними). Одновременные интервалы остаются честными для кандидата,
        выбранного по данным, а попарные позволяют понять, отличим ли лидер от остальных.
        Совместное распределение — то же, что у основного интервала: multiplier bootstrap
        по кластерным суммам вкладов или реплики bootstrap с переобучением моделей.
        """
        data = self.data
        P = len(names)
        pairs = [(i, j) for i in range(P) for j in range(i + 1, P)]
        sim_rows, pair_rows = [], []
        for m, metric in enumerate(data.reward_cols):
            delta_hat = v_b[:, m, e] - v_a[m, e]
            pair_hat = np.array([v_b[i, m, e] - v_b[j, m, e] for i, j in pairs])
            families = {}
            if inference == "influence":
                cols = {
                    "delta": (psi_b[:, m, e] - psi_a[m, e]).T,
                    "pair": np.column_stack([psi_b[i, m, e] - psi_b[j, m, e] for i, j in pairs]) if pairs else None,
                }
                for key, c in cols.items():
                    if c is None:
                        continue
                    sums = cluster_sums(c, data.clusters, data.n_clusters)
                    se = cluster_se(sums, data.n)
                    families[key] = (se, multiplier_tstats(sums, se, data.n, _RW_DRAWS, rng))
            else:
                s = samples[:, :, m, e, :]
                stars = {
                    "delta": s[..., 0] - s[..., 1],
                    "pair": np.column_stack([s[:, i, 0] - s[:, j, 0] for i, j in pairs]) if pairs else None,
                }
                hats = {"delta": delta_hat, "pair": pair_hat}
                for key, star in stars.items():
                    if star is None:
                        continue
                    se = nan_std(star)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        families[key] = (se, (star - hats[key]) / se)
            se_d, null_d = families["delta"]
            c_d = simultaneous_critical(null_d, alpha)
            for i, name in enumerate(names):
                sim_rows.append({
                    "candidate": name, "metric": metric,
                    "delta_low_sim": delta_hat[i] - c_d * se_d[i],
                    "delta_high_sim": delta_hat[i] + c_d * se_d[i],
                })
            if pairs:
                se_p, null_p = families["pair"]
                c_p = simultaneous_critical(null_p, alpha)
                with np.errstate(divide="ignore", invalid="ignore"):
                    t_obs = np.where(se_p > 0, pair_hat / se_p, np.nan)
                p_adj = romano_wolf(t_obs, null_p)
                for q, (i, j) in enumerate(pairs):
                    pair_rows.append({
                        "metric": metric, "candidate": names[i], "other": names[j],
                        "diff": pair_hat[q], "se": se_p[q],
                        "low": pair_hat[q] - c_p * se_p[q], "high": pair_hat[q] + c_p * se_p[q],
                        "p_value_adj": p_adj[q],
                    })
        columns = ["metric", "candidate", "other", "diff", "se", "low", "high", "p_value_adj"]
        return pd.DataFrame(sim_rows), pd.DataFrame(pair_rows, columns=columns)

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
            "propensity_origin": self.propensity_origin,
            "logger": self.logger_col,
            "propensity_floor": self.config.propensity_floor if d.propensity is None else None,
            "cluster": d.cluster_col,
            "features": list(d.features),
            "features_auto": d.features_auto,
            "skipped_features": dict(d.skipped_features),
            "propensity_model": (
                {"column": "логированные", "policy": "по заданной политике A",
                 "mixture": f"по смеси логирующих политик (колонка «{self.logger_col}»)"}
                [self.propensity_origin] if d.propensity is not None else
                _model_name(self.config.propensity_model, _propensity_label(nuis))
            ),
            "propensity_C": nuis.propensity_C,
            "propensity_kind": nuis.propensity_kind,
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
        if self._logging_matrix is not None:
            overlap = replace(overlap, uncovered=float(np.mean(np.sum(pi * (self._logging_matrix <= 0), axis=1))))
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
