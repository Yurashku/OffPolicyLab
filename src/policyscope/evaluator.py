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
from scipy import stats
from sklearn.base import clone
from threadpoolctl import threadpool_limits

from policyscope.data import LogData, prepare_logs
from policyscope.diagnostics import (
    Finding,
    PlaceboCheck,
    Thresholds,
    overlap_summary,
    policy_findings,
    triangulate,
    trust_level,
    variance_concentration,
)
from policyscope.estimators import ESTIMATORS, SELF_NORMALIZED, PolicyTerms, contributions, ess, policy_terms
from policyscope.inference import (
    analytic_summary,
    centered_pvalue,
    imbens_manski_critical,
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
    _boosting_propensity_model,
    fit_nuisance,
    make_folds,
    outcome_kind,
)
from policyscope.policy import MixturePolicy, RandomPolicy, _logging_terms, policy_matrix
from policyscope.report import (
    BoundsReport,
    Estimate,
    EstimatorRow,
    Leaderboard,
    MetricBounds,
    MetricReport,
    Report,
)

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
    "policy": (
        "пропенсити посчитаны по заданной политике A: если она задана верно (а для смеси версий "
        "периоды сопоставимы), DR несмещён при любой модели исхода на решениях в покрытии логов для "
        "политики, не зависящей от периода, а модель исхода уменьшает его разброс по сравнению с IPS "
        "и SNIPS"
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


def _sign_change(f, saturation: float) -> Optional[float]:
    """Наименьшее λ ≥ 0, где ``f`` меняет знак (бисекция), или ``None``.

    После ``saturation`` все прогнозы упираются в границу диапазона, и ``f`` больше не
    меняется, поэтому ``None`` означает «не меняет знак ни при каком λ».
    """
    upper = max(saturation, 1e-9)
    grid = np.linspace(0.0, min(upper, 10.0), 201)
    if upper > 10.0:  # дальше — по 20 точек на порядок, до самого насыщения
        points = int(min(400, max(20, 20 * np.log10(upper / 10.0))))
        grid = np.concatenate([grid, np.geomspace(10.0, upper, points)[1:]])
    values = np.array([f(x) for x in grid])
    signs = np.sign(values)
    for j in range(len(grid) - 1):
        if signs[j] != 0 and signs[j] != signs[j + 1]:
            a, b = grid[j], grid[j + 1]
            for _ in range(50):
                mid = (a + b) / 2
                if np.sign(f(mid)) == signs[j]:
                    a = mid
                else:
                    b = mid
            return float((a + b) / 2)
    return None


def _propensity_label(nuis: Nuisance) -> str:
    """Подпись модели пропенсити по умолчанию для отчёта."""
    losses = nuis.propensity_losses
    logistic = [v for key, v in losses.items() if key != "boosting"]
    if nuis.propensity_kind == "boosting":
        if not (losses and logistic):
            return "градиентный бустинг (OOF)"
        return (f"градиентный бустинг (OOF, {nuis.propensity_iterations} итераций; выбран по out-of-fold "
                f"log-loss: {losses['boosting']:.3f} против {min(logistic):.3f} у логистической регрессии)")
    if nuis.propensity_C is not None:
        note = f"; {nuis.propensity_note}" if nuis.propensity_note else ""
        return f"LogisticRegression (OOF, C = {nuis.propensity_C:g}, подобрано по out-of-fold log-loss{note})"
    return "LogisticRegression (OOF)"


def _equal_means_test(values: np.ndarray, group: np.ndarray, clusters: np.ndarray) -> tuple[float, float]:
    """Тест Вальда «средние ``values`` по группам равны» с учётом повторов кластеров.

    Ковариация средних групп — модель со случайными эффектами: эффект клиента (общий для всех
    его строк), эффект «клиент × период» (общий для строк клиента внутри одного периода) и шум.
    Дисперсии эффектов оцениваются методом моментов по суммам остатков клиента — по периодам и
    в целом; кластер (клиент) может входить в несколько групп, это даёт ковариацию средних.
    Дисперсия шума общая (как в дисперсионном анализе), но у группы она не меньше её
    собственной: отдельная оценка по маленькой группе с редкими событиями почти нулевая и давала
    бы ложные различия, а общая занижала бы разброс маленькой, но более шумной группы. Без
    кластеров это однофакторный дисперсионный анализ, лишь чуть более осторожный из-за пола
    дисперсии. Статистика ``W / q`` сравнивается с ``F(q, G − K)``, ``q`` — ранг ограничений.

    Returns:
        p-value и разброс средних (максимум минус минимум).
    """
    n, K = len(values), int(group.max()) + 1
    n_k = np.bincount(group, minlength=K).astype(float)
    means = np.bincount(group, weights=values, minlength=K) / n_k
    resid = values - means[group]
    G = int(clusters.max()) + 1
    spread = float(means.max() - means.min())
    if n <= K or G <= K:
        return float("nan"), spread
    counts = np.zeros((G, K))
    np.add.at(counts, (clusters, group), 1.0)
    cell = np.zeros((G, K))
    np.add.at(cell, (clusters, group), resid)
    total = float(resid @ resid) / (n - K)
    within = float(np.sum(counts**2)) - n                                  # пары строк: клиент и период те же
    across = float(np.sum(counts.sum(axis=1) ** 2) - np.sum(counts**2))    # клиент тот же, период другой
    a = float(np.sum(cell**2)) - total * n                                 # ≈ within · (s_u + s_v)
    b = float(cell.sum(axis=1) @ cell.sum(axis=1)) - total * n             # ≈ (within + across) · s_u + within · s_v
    if within > 0 and across > 0:
        both = min(max(a / within, 0.0), total)
        s_u = min(max((b - a) / across, 0.0), both)
        s_v = both - s_u
    elif across > 0:  # строка на клиента в периоде: эффект «клиент × период» неотделим от шума
        s_u, s_v = min(max(b / across, 0.0), total), 0.0
    elif within > 0:  # клиент целиком в одном периоде
        s_u, s_v = min(max(a / within, 0.0), total), 0.0
    else:
        s_u = s_v = 0.0
    own = np.bincount(group, weights=resid**2, minlength=K) / np.maximum(n_k - 1, 1)
    noise = np.maximum(own, total) - s_u - s_v
    pairs = counts.T @ counts
    cov = s_u * pairs / np.outer(n_k, n_k) + np.diag(s_v * np.diag(pairs) / n_k**2 + noise / n_k)
    contrast = np.hstack([-np.ones((K - 1, 1)), np.eye(K - 1)])
    d = contrast @ means
    middle = contrast @ cov @ contrast.T
    q = int(np.linalg.matrix_rank(middle))
    if q == 0:
        return float("nan"), spread
    wald = float(d @ np.linalg.pinv(middle) @ d)
    return float(stats.f.sf(wald / q, q, G - K)), spread


def _same_random_policy(policy: Any, other: Any) -> bool:
    """Одна и та же случайная политика (сравнение без поэлементного ``==`` таблиц pandas)."""
    if not (isinstance(policy, RandomPolicy) and isinstance(other, RandomPolicy)):
        return False
    if policy is other:
        return True
    a, b = policy.mix, other.mix
    same_mix = a is b or (
        isinstance(a, (str, type(None))) and isinstance(b, (str, type(None))) and a == b
    ) or (isinstance(a, Mapping) and isinstance(b, Mapping) and dict(a) == dict(b))
    same_by = set(policy.by or ()) == set(other.by or ()) and (policy.by is None) == (other.by is None)
    return same_mix and policy.actions == other.actions and same_by


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
        placebo: Колонка (или колонки) с показателем, известным до решения (например,
            отклик клиента в прошлом месяце). Решение не может его изменить, поэтому оценка
            «эффекта» политики на нём должна быть нулевой; значимое и существенное отличие —
            признак скрытых факторов выбора действия (менеджеры, каналы) или неверной модели
            пропенсити. В признаки не входит, поэтому правило A не должно его использовать:
            иначе это не плацебо, а признак. При логах нескольких политик (``logger=``)
            проверяется, зависит ли выбор от показателя внутри периода; если версии
            детерминированы, проверять нечего — проверка помечается неприменимой.
        cluster: Колонка единицы наблюдения (клиент), если на неё приходится несколько
            строк. ``None`` — строки независимы.
        features: Признаки контекста. ``None`` — авто-выбор (список попадает в отчёт).
        propensity_model: sklearn-классификатор для пропенсити. По умолчанию выбирается по
            out-of-fold log-loss: логистическая регрессия с подобранным ``C`` или бустинг.
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
        placebo: Union[str, Sequence[str], None] = None,
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
        placebo_cols = [placebo] if isinstance(placebo, str) else list(placebo or [])
        self.data: LogData = prepare_logs(
            logs, action=action, reward=reward, propensity=propensity,
            cluster=cluster, features=features, logger=logger, exclude=placebo_cols,
        )
        self._placebo: dict[str, np.ndarray] = {}
        service = {action, *([reward] if isinstance(reward, str) else reward), propensity, cluster, logger}
        clash = [col for col in placebo_cols if col in service]
        if clash:
            raise ValueError(
                f"Колонки {clash} служебные (действие/награда/пропенсити/кластер/источник логов) и не могут "
                "быть плацебо: плацебо — показатель, известный до решения."
            )
        for col in placebo_cols:
            if col not in logs.columns:
                raise ValueError(f"В логах нет плацебо-колонки '{col}'")
            values = pd.to_numeric(logs[col], errors="coerce").to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(f"Плацебо-колонка '{col}' должна быть числовой, без пропусков и бесконечностей")
            if np.ptp(values) == 0:
                raise ValueError(f"Плацебо-колонка '{col}' постоянна: проверка на ней ничего не покажет")
            self._placebo[col] = values
        self._placebo_resid: Optional[dict[str, np.ndarray]] = None
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
        self.logger_col = logger
        self._logging_policy = logging_policy
        self._logger_labels = logs[logger].to_numpy(dtype=object) if logger else None
        # Доли — по меткам, что встречаются в логах (у категориальной колонки бывают пустые уровни).
        present = pd.Series(self._logger_labels).value_counts(normalize=True) if logger else None
        if present is not None:
            try:
                present = present.sort_index()
            except TypeError:  # метки разных типов не сравниваются — оставляем порядок частот
                pass
        self._logger_shares = present.to_dict() if logger else {}
        if logging_policy is not None:
            p, self._logger_matrices = _logging_terms(
                logging_policy, self.data, self._frame, logs[logger] if logger else None,
            )
            shares = present if logger else {None: 1.0}
            self._logging_matrix = sum(shares[label] * m for label, m in self._logger_matrices.items())
            self.data = replace(self.data, propensity=p)
            self.propensity_origin = "mixture" if logger else "policy"
        self.config = NuisanceConfig(
            propensity_model=propensity_model,
            outcome_model=outcome_model,
            n_folds=n_folds,
            propensity_floor=propensity_floor,
        )
        self.thresholds = thresholds or Thresholds()
        self._mixture_cache: Optional[list[Finding]] = None
        self._own_cache: Optional[np.ndarray] = None
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

    def bounds(
        self,
        policy: Any,
        *,
        baseline: Any = None,
        name: str = "B",
        baseline_name: Optional[str] = None,
        low: Optional[Mapping[str, float]] = None,
        high: Optional[Mapping[str, float]] = None,
        sensitivity: Sequence[float] = (0.5, 0.75, 1.0, 1.25, 1.5),
        alpha: float = 0.05,
    ) -> BoundsReport:
        """Границы для кандидата, часть решений которого вне покрытия логов.

        Когда логирующая политика известна (``logging_policy=``), известно и то, какие
        действия она могла показать каждому клиенту. Решения кандидата делятся на две части:

        - **покрытая** — оценивается как обычно (DR с точными пропенсити; несмещённо при любой
          модели исхода, если политика не зависит от периода логов);
        - **непокрытая** — отклик на эти действия в логах не наблюдался. Без допущений он
          лишь лежит в диапазоне метрики ``[low, high]``.

        Отсюда границы для V(кандидата) и Δ без допущений об отклике на непокрытых действиях
        (в пределах ``[low, high]``; частичная идентификация, Manski), интервал Imbens–Manski и
        односторонние оценки «Δ не больше» и «Δ не меньше». Покрытая часть опирается на
        точные пропенсити заданной политики, а для смеси версий — ещё и на сопоставимость
        периодов. Таблица чувствительности добавляет допущение: отклик на непоказанные
        действия равен λ × прогноз модели исхода; ``breakdown`` — λ, при котором Δ меняет знак.

        Args:
            policy: Кандидат в любом формате (например, ``ps.random_like()``).
            baseline: База; ``None`` — политика A, как она есть в логах.
            low, high: Диапазон среднего отклика на действие по метрикам, например
                ``high={"revenue": 500}``. По умолчанию: [0, 1] для бинарных метрик, иначе
                наблюдаемые минимум и максимум (для денежных метрик лучше задать явно).
            sensitivity: Значения λ для таблицы чувствительности.
            alpha: Уровень значимости.
        """
        if self._logging_matrix is None:
            raise ValueError(
                "Границы считаются, когда логирующая политика известна (logging_policy=): только "
                "тогда известно, какие действия она могла показать каждому клиенту."
            )
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha должна лежать в (0, 1)")
        data, nuis = self.data, self.nuisance
        base_label = baseline_name or ("A" if baseline is None else "база")
        pi = self._matrix(policy, name)
        base = self._matrix(baseline, base_label) if baseline is not None else None
        covered = self._logging_matrix > 0
        rows = np.arange(data.n)
        df = max(data.n_clusters - 1, 1)
        t_one = float(stats.t.ppf(1.0 - alpha, df))
        # Масса вне покрытия по действиям: у кандидата и у базы. Неизвестный отклик на одно и то
        # же непоказанное действие у них общий, поэтому в Δ входит только разность масс.
        out_b = pi * ~covered
        out_a = base * ~covered if base is not None else np.zeros_like(pi)
        diff = out_b - out_a
        more, less = np.clip(diff, 0.0, None), np.clip(-diff, 0.0, None)
        metrics: dict[str, MetricBounds] = {}
        for metric in data.reward_cols:
            r = data.rewards[metric]
            mu = nuis.mu[metric]
            lo, hi, note, assumed = self._reward_range(metric, low, high)

            def covered_part(P: np.ndarray) -> np.ndarray:
                """DR по действиям в покрытии: несмещён при точных пропенсити и любой модели исхода."""
                w = P[rows, data.a] / nuis.p_taken
                return (P * covered * mu).sum(axis=1) + w * (r - mu[rows, data.a])

            ib = covered_part(pi)
            ia = r.astype(float) if base is None else covered_part(base)
            cols = np.column_stack([
                ib + out_b.sum(axis=1) * lo, ib + out_b.sum(axis=1) * hi,
                ib - ia + (more * lo - less * hi).sum(axis=1),
                ib - ia + (more * hi - less * lo).sum(axis=1),
            ])
            s = analytic_summary(cols, data.clusters, data.n_clusters, alpha)
            est, se = s["estimate"], s["se"]
            c = imbens_manski_critical(est[3] - est[2], se[2], se[3], alpha, df=df)

            def delta_at(lam: float, ib=ib, ia=ia, mu=mu, lo=lo, hi=hi) -> float:
                # Отклик на каждое непоказанное действие — λ × прогноз модели, в пределах диапазона.
                return float(np.mean(ib - ia + (diff * np.clip(lam * mu, lo, hi)).sum(axis=1)))

            # После насыщения λ·μ̂ на всех непокрытых ячейках упирается в границу диапазона:
            # при μ̂ > 0 — в верхнюю, при μ̂ < 0 и отрицательной нижней — в нижнюю.
            unseen = mu[diff != 0]
            limits = [hi / unseen[unseen > 0]] if hi > 0 else []
            if lo < 0:
                limits.append(lo / unseen[unseen < 0])
            limits = np.concatenate(limits) if limits else np.zeros(0)
            saturation = float(np.max(limits)) if limits.size else 0.0
            metrics[metric] = MetricBounds(
                metric=metric,
                binary=data.binary[metric],
                reward_range=(lo, hi),
                range_note=note,
                range_assumed=assumed,
                v_low=float(est[0]),
                v_high=float(est[1]),
                delta_low=float(est[2]),
                delta_high=float(est[3]),
                ci_low=float(est[2] - c * se[2]),
                ci_high=float(est[3] + c * se[3]),
                at_most=float(est[3] + t_one * se[3]),
                at_least=float(est[2] - t_one * se[2]),
                sensitivity={float(lam): delta_at(lam) for lam in sensitivity},
                breakdown=_sign_change(delta_at, saturation),
            )
        findings: tuple[Finding, ...] = ()
        if self.propensity_origin == "mixture":
            if self._mixture_cache is None:
                self._mixture_cache = self._mixture_findings()
            findings = tuple(self._mixture_cache)
        if self._depends_on_logger(policy) or self._depends_on_logger(baseline):
            findings += (self._period_finding(),)
        return BoundsReport(
            policy=name, baseline=base_label, uncovered=float(np.mean(out_b.sum(axis=1))),
            baseline_uncovered=float(np.mean(out_a.sum(axis=1))), alpha=alpha, metrics=metrics,
            findings=findings,
        )

    def _reward_range(
        self, metric: str, low: Optional[Mapping[str, float]], high: Optional[Mapping[str, float]],
    ) -> tuple[float, float, str, bool]:
        """Диапазон среднего отклика на действие, откуда он взят и есть ли в нём допущение.

        Для бинарной метрики диапазон [0, 1] верен всегда. Для остальных сторона, не заданная
        явно, берётся из наблюдаемых значений — это допущение: средний отклик на непоказанные
        действия не выходит за наблюдённые минимум и максимум.
        """
        r = self.data.rewards[metric]
        given_low, given_high = (low or {}).get(metric), (high or {}).get(metric)
        if self.data.binary[metric] and given_low is None and given_high is None:
            return 0.0, 1.0, "бинарная", False
        lo = float(given_low) if given_low is not None else float(r.min())
        hi = float(given_high) if given_high is not None else float(r.max())
        if lo > hi:
            raise ValueError(f"Диапазон отклика для «{metric}»: low больше high")
        binary = self.data.binary[metric]
        sides = [
            "low задан" if given_low is not None else ("low = 0" if binary else "low — наблюдаемый min"),
            "high задан" if given_high is not None else ("high = 1" if binary else "high — наблюдаемый max"),
        ]
        assumed = not binary and (given_low is None or given_high is None)
        note = "задан" if given_low is not None and given_high is not None else ", ".join(sides)
        return lo, hi, note, assumed

    # ------------------------------------------------------------------
    # Расчёт
    # ------------------------------------------------------------------

    def _resolve_estimator(self, estimator: str) -> tuple[str, Optional[str]]:
        """Основной оценщик и причина выбора (``None`` — оценщик задан явно)."""
        if estimator != "auto":
            return estimator, None
        if self.data.propensity is None:
            source = "estimated"
        else:
            source = "policy" if self.propensity_origin in ("policy", "mixture") else "logged"
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
                       propensity_iterations=nuis.propensity_iterations, interactions=dict(nuis.interactions))

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
        pis = [self._matrix(candidates[n], n) for n in names]
        base = self._matrix(baseline, base_label) if baseline is not None else None

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
            # Поправка на множественность не может сделать гипотезу значимее, чем без неё.
            with np.errstate(invalid="ignore"):
                p_adj = np.where(np.isfinite(p_adj) & np.isfinite(p_family), np.fmax(p_adj, p_family), p_adj)
        else:
            p_adj = None

        simultaneous, pairwise = (
            self._competition(names, psi_b, psi_a, v_b, v_a, samples, inference, alpha, e_primary, rng)
            if inference != "none" else (None, None)
        )
        if simultaneous is not None and len(simultaneous):
            # Одновременный интервал симметричен, а перцентильный маргинальный при bootstrap бывает
            # асимметричным: интервал для семейства не должен быть уже маргинального ни с одной стороны.
            marginal_low = stats["delta_low"][:, :, e_primary].T.ravel()
            marginal_high = stats["delta_high"][:, :, e_primary].T.ravel()
            low, high = simultaneous["delta_low_sim"].to_numpy(), simultaneous["delta_high_sim"].to_numpy()
            ok = np.isfinite(low) & np.isfinite(marginal_low)
            simultaneous["delta_low_sim"] = np.where(ok, np.minimum(low, marginal_low), low)
            ok = np.isfinite(high) & np.isfinite(marginal_high)
            simultaneous["delta_high_sim"] = np.where(ok, np.maximum(high, marginal_high), high)
        settings = self._settings(estimator, estimator_reason, inference, alpha, n_boot)
        reports = {}
        for i, name in enumerate(names):
            dependent = self._depends_on_logger(candidates[name]) or self._depends_on_logger(baseline)
            reports[name] = self._report(
                i, name, base_label, pis[i], base, psi_b, psi_a, v_b, v_a, stats, p_adj,
                terms_b, estimator, inference, alpha,
                extra_findings + ([self._period_finding()] if dependent else []), settings,
                period_dependent=dependent,
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

    def _matrix(self, policy: Any, role: str) -> np.ndarray:
        """Матрица политики-кандидата или базы.

        Если кандидат (или член смеси) — та же случайная политика, что задана логирующей, или
        сама логирующая смесь, берётся матрица логирующей политики. Доли случайного кандидата
        считаются без самой строки, а логирующей — с ней, и без этой подмены одна и та же
        политика давала бы веса, отличные от 1. Подмена делается, только если строки
        логирующей матрицы в сумме дают 1: иначе логирующая политика кладёт массу на действия,
        которых нет в логах, и кандидату так нельзя.
        """
        complete = self._logging_matrix is not None and np.allclose(self._logging_matrix.sum(axis=1), 1.0)
        if complete and isinstance(policy, MixturePolicy) and policy is self._logging_policy:
            return self._logging_matrix.copy()
        if isinstance(policy, MixturePolicy):
            return sum(w * self._matrix(member, f"{role}[{i}]")
                       for i, (member, w) in enumerate(zip(policy.policies, policy.weights)))
        if complete and _same_random_policy(policy, self._logging_policy):
            return self._logging_matrix.copy()
        return policy_matrix(policy, self.data, self._frame, role=role)

    @property
    def _own_matrix(self) -> np.ndarray:
        """Для каждой строки — вероятности действий у политики, записавшей её (своего периода)."""
        if self._own_cache is None:
            data = self.data
            own = np.zeros((data.n, data.k))
            for label, matrix in self._logger_matrices.items():
                mask = np.ones(data.n, dtype=bool) if label is None else self._logger_labels == label
                own[mask] = matrix[mask]
            self._own_cache = own
        return self._own_cache

    def _exact_uncovered(self, overlap, pi):
        """Добавляет к перекрытию точную долю решений вне носителя и точное ожидание среднего веса.

        Ожидание — ``mean_i Σ_a π_{k_i}(a|x_i) π(a|i) / p(a|x_i)`` по действиям в носителе: строку
        записала политика своего периода. Для политики, не зависящей от периода, при сопоставимых
        периодах это примерно ``1 − uncovered``; для зависящей (например, случайной с балансом по
        месяцам при логах версий) — нет, и сравнивать средний вес с ``1 − uncovered`` было бы ошибкой.
        """
        if self._logging_matrix is None:
            return overlap
        P = self._logging_matrix
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(P > 0, pi / P, 0.0)
        return replace(
            overlap,
            uncovered=float(np.mean(np.sum(pi * (P <= 0), axis=1))),
            expected_weight=float(np.mean(np.sum(self._own_matrix * ratio, axis=1))),
        )

    def _uncovered_mass(self, pi, base, metric: str, sn: bool = False) -> Optional[float]:
        """Ожидаемое расхождение DR − IPS (SNDR − SNIPS) при верной политике A из-за решений вне покрытия.

        IPS теряет решения вне покрытия, поэтому ``E[DR − IPS]`` — их вклад по модели исхода. SNIPS
        заменяет эту часть средним по покрытой, и ``E[SNDR − SNIPS]`` — разность всего прогноза и
        прогноза покрытой части, нормированного на её долю. Для базы-политики вычитается то же для базы.
        """
        if self._logging_matrix is None:
            return None
        covered, mu = self._logging_matrix > 0, self.nuisance.mu[metric]

        def expected(P: np.ndarray) -> float:
            total = float(np.mean(np.sum(P * mu, axis=1)))
            inside = float(np.mean(np.sum(P * covered * mu, axis=1)))
            if not sn:
                return total - inside
            share = float(np.mean(np.sum(P * covered, axis=1)))
            return total - inside / share if share > 0 else float("nan")

        return expected(pi) - expected(base) if base is not None else expected(pi)

    def _depends_on_logger(self, policy: Any) -> bool:
        """Зависит ли политика от колонки источника логов (периода) — например, баланс по месяцам."""
        if self.logger_col is None:
            return False
        if isinstance(policy, RandomPolicy):
            return policy.by is not None and self.logger_col in policy.by
        if isinstance(policy, MixturePolicy):
            return any(self._depends_on_logger(member) for member in policy.policies)
        return False

    def _period_finding(self) -> Finding:
        return Finding(
            code="period_dependent_policy",
            severity="info",
            title="Политика зависит от периода логов",
            detail=(
                f"Политика зависит от колонки «{self.logger_col}», по которой смешаны логирующие политики. "
                "Для такой политики IPS по смеси версий смещён по построению, расхождение DR − IPS не ноль "
                "даже при верных пропенсити, а DR сильнее опирается на модель исхода. При сопоставимых "
                "периодах — предпосылке смеси — та же политика без привязки к периоду имеет то же значение. "
                "Распознаётся только ps.random_like(by=...) с колонкой периода (в том числе внутри смеси)."
            ),
            advice="Для случайной выдачи используйте ps.random_like() без by= по колонке периода.",
        )

    def _placebo_residuals(self) -> dict[str, np.ndarray]:
        """Плацебо минус его out-of-fold прогноз ``ĝ`` по признакам (и периоду при ``logger=``).

        Прогноз не меняет среднего статистики, но убирает шум. При логах нескольких политик в
        прогноз входит и период. Без этого проверка ложно срабатывала бы, когда версии дают
        части действий почти нулевую вероятность: такие действия почти никогда не попадают в
        выборку, и в типичной выборке среднее весов в разных периодах разное, а плацебо вроде
        прошлого отклика от периода зависит. Под нулевой гипотезой ``E[ω | x, период, z] = 0``,
        поэтому прогноз по ``(x, период)`` допустим так же, как по ``x``.
        """
        if self._placebo_resid is None:
            from sklearn.linear_model import Ridge
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler

            data = self.data
            X = data.X if data.X.shape[1] else np.zeros((data.n, 1))
            if self._logger_labels is not None:
                codes = pd.factorize(pd.Series(self._logger_labels))[0]
                X = np.column_stack([X, np.eye(codes.max() + 1)[codes]])
            groups = data.clusters if data.is_clustered else np.arange(data.n)
            folds = make_folds(groups, self.config.n_folds, np.random.default_rng(self.random_state))
            self._placebo_resid = {}
            for col, z in self._placebo.items():
                pred = np.zeros(data.n)
                for f in range(self.config.n_folds):
                    train, hold = folds != f, folds == f
                    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(X[train], z[train])
                    pred[hold] = model.predict(X[hold])
                self._placebo_resid[col] = z - pred
        return self._placebo_resid

    def _placebo_weights(self, pi, base) -> tuple[np.ndarray, bool]:
        """Веса плацебо-теста ``ω = w_B − w_база − E[w_B − w_база | x, политика строки]``.

        Если логирующая политика задана, условное среднее известно точно: строку записала
        политика ``π_k`` своего периода (для одной политики — она сама), и
        ``E[w_B | x, k] = Σ_a π_k(a|x) π_B(a|x) / p(a|x)``. Тогда ``E[ω | x, k] = 0`` и
        среднее ``ω·(z − ĝ(x, k))`` при отсутствии скрытых факторов выбора равно нулю при любом
        прогнозе ``ĝ`` — даже для смеси версий, где ``E[w_B | x, k] ≠ 1`` в отдельном периоде.
        Одного нулевого среднего мало для калибровки, если версии дают части действий почти
        нулевую вероятность, — поэтому ``ĝ`` включает период (см. :meth:`_placebo_residuals`).
        Иначе (пропенсити из колонки или оценены) поправка нулевая: при верных пропенсити
        ``E[w | x] = 1`` для политики в носителе A.

        Returns:
            Веса и признак того, что внутри периода выбор целиком определён логирующей
            политикой (тогда ``ω ≡ 0`` и проверять нечего).
        """
        data, nuis = self.data, self.nuisance
        rows = np.arange(data.n)
        w = pi[rows, data.a] / nuis.p_taken
        w0 = np.ones(data.n) if base is None else base[rows, data.a] / nuis.p_taken
        if self._logging_matrix is None:
            return w - w0, False
        diff = pi - (base if base is not None else self._logging_matrix)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(self._logging_matrix > 0, diff / self._logging_matrix, 0.0)
        own = self._own_matrix
        centre = np.sum(own * ratio, axis=1)
        determined = bool(np.all(own.max(axis=1) >= 1.0 - 1e-12))
        return (w - w0) - centre, determined

    def _placebo_checks(self, pi, base, se_by_metric) -> tuple[tuple[PlaceboCheck, ...], list[Finding]]:
        """Оценка «эффекта» политики на показателях до решения: ``mean(ω·(z − ĝ(x, период)))``.

        Решение не может изменить то, что было до него, поэтому без скрытых факторов выбора
        среднее равно нулю (веса ``ω`` — см. :meth:`_placebo_weights`). Расхождение считается
        найденным, если оно значимо на уровне ``check_z`` и существенно: в стандартных
        отклонениях «z» не меньше ``material_bias`` стандартной ошибки Δ самой точной метрики
        в её стандартных отклонениях. Так порог сужается вместе с интервалом Δ, как у
        проверок согласия оценщиков, а не остаётся постоянным при любом объёме данных.
        """
        if not self._placebo:
            return (), []
        data, th = self.data, self.thresholds
        omega, determined = self._placebo_weights(pi, base)
        precision = [
            se / float(np.std(data.rewards[m], ddof=1)) for m, se in se_by_metric.items()
            if np.isfinite(se) and np.std(data.rewards[m]) > 0
        ]
        checks, findings = [], []
        for col, resid in self._placebo_residuals().items():
            sd = float(np.std(self._placebo[col], ddof=1))
            if determined or not np.any(np.abs(omega) > 1e-12):
                checks.append(PlaceboCheck(col, 0.0, 0.0, 0.0, sd, float("nan"), "not_applicable"))
                findings.append(Finding(
                    code="placebo_not_applicable",
                    severity="info",
                    title="Плацебо-проверка неприменима",
                    detail=(
                        "Действие целиком определено детерминированной логирующей политикой (при "
                        "нескольких версиях — версией своего периода): выбор не может зависеть от "
                        "скрытых факторов, и проверять нечего. То, что логи совпадают с её выбором, "
                        "проверено при расчёте пропенсити."
                        if determined else
                        "Веса кандидата и базы совпадают на всех строках: проверка ничего не сообщает."
                    ),
                ))
                continue
            s = analytic_summary(omega * resid, data.clusters, data.n_clusters, 0.05)
            est, se = float(s["estimate"][0]), float(s["se"][0])
            low, high = est - th.check_z * se, est + th.check_z * se
            material = th.material_bias * min(precision) * sd if precision else 0.0
            significant = low > 0 or high < 0
            failed = significant and abs(est) > material
            checks.append(PlaceboCheck(col, est, low, high, sd, material, "fail" if failed else "ok"))
            effect = (
                f"Δ = {est:+.3g} (интервал проверки {low:+.3g}…{high:+.3g}), "
                f"это {est / sd:+.3f} стандартного отклонения «{col}»"
            )
            if failed:
                findings.append(Finding(
                    code="placebo_failed",
                    severity="caution",
                    title="Плацебо-проверка не пройдена",
                    detail=(
                        f"Политика «меняет» «{col}» — показатель, известный до решения: {effect}. "
                        "Решение не могло на него повлиять: значит, выбор действия зависел от того, "
                        "чего нет в признаках (скрытый фактор), или неверна модель пропенсити. Если "
                        "плацебо и метрика связаны со скрытыми факторами одинаково, смещение Δ не "
                        "меньше половины его стандартной ошибки (у самой точной метрики)."
                    ),
                    advice=(
                        "Найдите, что ещё влияло на выбор действия (менеджеры, каналы, ручные "
                        "правила), и добавьте это в признаки — или логируйте пропенсити. Если правило "
                        f"A само использовало «{col}», это не плацебо: уберите placebo= и оставьте "
                        "колонку признаком."
                    ),
                ))
            else:
                bound = max(abs(low), abs(high))
                if precision and bound <= material:
                    reach = (f"Если плацебо и метрика связаны со скрытыми факторами одинаково, факторы, "
                             f"связанные с «{col}», смещают Δ меньше чем на половину его стандартной ошибки. ")
                elif precision:
                    reach = (f"Если плацебо и метрика связаны со скрытыми факторами одинаково, не исключено "
                             f"смещение Δ до ≈{bound / (sd * min(precision)):.1f} его стандартной ошибки: "
                             f"проверка ловит только сильные скрытые факторы, связанные с «{col}», и "
                             "прохождение не доказывает, что их нет. ")
                else:
                    reach = ""
                findings.append(Finding(
                    code="placebo_ok",
                    severity="info",
                    title="Плацебо-проверка пройдена",
                    detail=(
                        f"«{col}» известен до решения: {effect}. "
                        + ("Расхождение значимо, но мало: в пересчёте на метрики оно меньше половины "
                           "стандартной ошибки Δ. " if significant else "Значимого расхождения нет. ")
                        + f"Данные исключают «эффект» на «{col}» больше {bound:.3g} "
                        f"({bound / sd:.3f} стандартного отклонения). " + reach
                        + f"Факторы, не связанные с «{col}», эта проверка не видит."
                    ),
                ))
        return tuple(checks), findings

    def _logger_shift_auc(self) -> Optional[float]:
        """ROC AUC, с которым источник логов угадывается по признакам клиента (out-of-fold).

        Классификатор — логистическая регрессия или градиентный бустинг, что даст меньший
        out-of-fold log-loss: линейная модель не видит немонотонных различий состава. Около
        0.5 — сдвиг состава клиентов между периодами не обнаружен; заметно выше — составы
        различаются, и доли периодов как пропенсити смеси неверны для части клиентов.
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        data = self.data
        y, labels = pd.factorize(pd.Series(self._logger_labels))  # метки могут быть разных типов
        if len(labels) < 2 or data.X.shape[1] == 0:
            return None
        groups = data.clusters if data.is_clustered else np.arange(data.n)
        folds = make_folds(groups, self.config.n_folds, np.random.default_rng(self.random_state))
        if any(len(np.unique(y[folds != f])) < len(labels) for f in range(self.config.n_folds)):
            return None
        candidates = {
            "logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
            "boosting": _boosting_propensity_model(),
        }
        best: Optional[tuple[float, np.ndarray]] = None
        with threadpool_limits(limits=1):
            for model in candidates.values():
                probs = np.zeros((data.n, len(labels)))
                try:
                    for f in range(self.config.n_folds):
                        train, hold = folds != f, folds == f
                        probs[hold] = clone(model).fit(data.X[train], y[train]).predict_proba(data.X[hold])
                except ValueError:  # бустингу не хватило строк редкого периода на раннюю остановку
                    continue
                loss = float(-np.mean(np.log(np.clip(probs[np.arange(data.n), y], 1e-12, None))))
                if best is None or loss < best[0]:
                    best = (loss, probs)
        if best is None:
            return None
        probs = best[1]
        if len(labels) == 2:
            return float(roc_auc_score(y, probs[:, 1]))
        return float(roc_auc_score(y, probs, multi_class="ovr"))

    def _logger_drift(self) -> tuple[int, list[tuple[str, float, float]]]:
        """Различается ли отклик между периодами при одном и том же решении.

        Берутся строки, где все логирующие политики выбирают одно и то же действие: там
        период на решение не влияет, и остаток ``r − mu(x, a)`` модели исхода (период в неё
        не входит) не должен от него зависеть. Равенство средних остатков по периодам
        проверяется тестом Вальда (:func:`_equal_means_test`), который учитывает повторы
        клиентов: строки одного клиента зависимы, и обычный дисперсионный анализ давал бы
        ложные сигналы сезонности.

        Returns:
            Число таких строк и список «метрика, p-value, разброс средних остатков»; пустой,
            если строк мало (нужно не меньше 30 в двух периодах).
        """
        data, nuis = self.data, self.nuisance
        stacked = np.stack(list(self._logger_matrices.values()))
        agree = np.flatnonzero(stacked.min(axis=0).max(axis=1) >= 0.99)
        codes = pd.factorize(pd.Series(self._logger_labels))[0][agree]  # метки могут быть разных типов
        sizes = np.bincount(codes)
        kept = np.flatnonzero(sizes >= 30)
        if len(kept) < 2:
            return len(agree), []
        rows = agree[np.isin(codes, kept)]
        period = np.unique(codes[np.isin(codes, kept)], return_inverse=True)[1]
        clusters = np.unique(data.clusters[rows], return_inverse=True)[1]
        out = []
        for metric in data.reward_cols:
            resid = data.rewards[metric][rows] - nuis.mu[metric][rows, data.a[rows]]
            if np.ptp(resid) == 0:
                continue
            p_value, spread = _equal_means_test(resid, period, clusters)
            if np.isfinite(p_value):
                out.append((metric, p_value, spread))
        return len(agree), out

    def _mixture_findings(self) -> list[Finding]:
        shares = ", ".join(f"{label} — {100 * share:.0f}%" for label, share in self._logger_shares.items())
        auc = self._logger_shift_auc()
        shift = auc is not None and auc > self.thresholds.logger_shift_auc
        n_agree, drift = self._logger_drift()
        drifted = [(m, p, spread) for m, p, spread in drift if p < self.thresholds.logger_drift_p]
        if drift and not drifted:
            drift_text = (f" Сезонность проверена на {n_agree} строках, где все политики выбирают одно и "
                          "то же: различий отклика между периодами не обнаружено.")
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
                + (f" Период по признакам клиента угадывается с ROC AUC {auc:.2f}: сдвиг состава "
                   "клиентов не обнаружен." if auc is not None and not shift else "")
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
                    f"{_fmt_number(worst[2])} (p = {worst[1]:.2g}"
                    + (", с учётом повторов клиентов" if self.data.is_clustered else "")
                    + "). Похоже на сезонность или изменения продукта: смесь периодов смешивает и эти "
                    "различия."
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
            # Интервал для семейства не может быть уже интервала для одной гипотезы.
            floor = t_critical(alpha, data.n_clusters)
            c_d = max(simultaneous_critical(null_d, alpha), floor)
            for i, name in enumerate(names):
                sim_rows.append({
                    "candidate": name, "metric": metric,
                    "delta_low_sim": delta_hat[i] - c_d * se_d[i],
                    "delta_high_sim": delta_hat[i] + c_d * se_d[i],
                })
            if pairs:
                se_p, null_p = families["pair"]
                c_p = max(simultaneous_critical(null_p, alpha), floor)
                with np.errstate(divide="ignore", invalid="ignore"):
                    t_obs = np.where(se_p > 0, pair_hat / se_p, np.nan)
                p_raw = 2.0 * stats.t.sf(np.abs(t_obs), df=max(data.n_clusters - 1, 1))
                p_adj = romano_wolf(t_obs, null_p)
                with np.errstate(invalid="ignore"):
                    p_adj = np.where(np.isfinite(p_adj) & np.isfinite(p_raw), np.fmax(p_adj, p_raw), p_adj)
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
        terms_b, estimator, inference, alpha, extra_findings, settings, period_dependent=False,
    ) -> Report:
        data, th, nuis = self.data, self.thresholds, self.nuisance
        e_primary = ESTIMATORS.index(estimator)
        overlap = self._exact_uncovered(overlap_summary(data, nuis, pi, alpha), pi)
        findings = policy_findings(data, nuis, overlap, th)
        base_overlap = None
        if base is not None:
            base_overlap = self._exact_uncovered(overlap_summary(data, nuis, base, alpha), base)
            findings += policy_findings(data, nuis, base_overlap, th, baseline=base_label)
        shares = [o.uncovered for o in (overlap, base_overlap) if o is not None and o.uncovered is not None]
        uncovered_share = max(shares, default=0.0)
        candidate_share = overlap.uncovered or 0.0
        base_share = base_overlap.uncovered if base_overlap is not None and base_overlap.uncovered else 0.0
        if candidate_share > 0 and base_share > 0:
            uncovered_owner = f"кандидата или базы «{base_label}»"
        elif base_share > 0:
            uncovered_owner = f"базы «{base_label}»"
        else:
            uncovered_owner = "кандидата"
        findings += extra_findings
        has_ci = inference != "none"
        se_by_metric = {}
        for m, metric in enumerate(data.reward_cols):
            se_by_metric[metric] = float(stats["delta_se"][i, m, e_primary]) if has_ci else float(
                analytic_summary(psi_b[i, m, e_primary] - psi_a[m, e_primary],
                                 data.clusters, data.n_clusters, alpha)["se"][0]
            )
        placebo_checks, placebo_findings = self._placebo_checks(pi, base, se_by_metric)
        findings += placebo_findings
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
                se_delta=se_by_metric[metric],
                estimated_propensity=nuis.source == "estimated",
                ess_value=ess_value, th=th,
                declared_policy=self.propensity_origin in ("policy", "mixture"),
                period_dependent=period_dependent,
                uncovered=uncovered_share,
                uncovered_mass=self._uncovered_mass(pi, base, metric, sn=estimator in SELF_NORMALIZED),
                uncovered_owner=uncovered_owner,
                mixture=self.propensity_origin == "mixture",
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
            placebo=placebo_checks,
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
    и ``evaluator_options`` (``logging_policy``, ``logger``, ``placebo``, ``propensity_model``,
    ``outcome_model``, ``n_folds``, ``propensity_floor``, ``lower_is_better``, ``thresholds``,
    ``random_state``, ``n_jobs``) — как у :class:`Evaluator`; аргументы оценки (``baseline``, ``estimator``,
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
