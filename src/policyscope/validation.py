"""
policyscope.validation
======================

Обвязка систематической проверки оценщиков на синтетике с **известной истиной**.

Прогоняет повторяемые эксперименты и сводит по каждой паре «режим × оценщик»
смещение, RMSE, фактическое покрытие доверительного интервала и диагностику.
Это единственный способ утверждать, что библиотека работает: точечные unit-тесты
проверяют формулы, а эта обвязка — статистические свойства.

Эталон берётся из аналитического оракула
:meth:`~policyscope.synthetic.SyntheticRecommenderEnv.oracle_value`, поэтому измеряемое
смещение — это смещение оценщика, а не шум самого эталона.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from policyscope.comparison import InferenceMethod, compare_policies
from policyscope.estimators import DEFAULT_NUISANCE_SPEC, NuisanceSpec
from policyscope.policies import make_policy
from policyscope.synthetic import SynthConfig, SyntheticRecommenderEnv

logger = logging.getLogger(__name__)

__all__ = [
    "ValidationMode",
    "ValidationRunRow",
    "ValidationResult",
    "ValidationThresholds",
    "run_simulation_validation",
    "DEFAULT_MODES",
]


@dataclass(frozen=True)
class ValidationMode:
    """Один режим прогона: как получаются пропенсити и что делается с весами."""

    name: str
    propensity_source: str = "auto"
    propensity_col: Optional[str] = "propensity_A"
    use_crossfit: bool = False
    weight_clip: Optional[float] = None
    inference: InferenceMethod = "bootstrap"


DEFAULT_MODES: tuple[ValidationMode, ...] = (
    ValidationMode(name="logged_propensity", propensity_source="logged"),
    ValidationMode(name="estimated_propensity", propensity_source="estimated"),
    ValidationMode(name="crossfit_estimated", propensity_source="estimated", use_crossfit=True),
    ValidationMode(
        name="logged_influence_ci", propensity_source="logged", inference="influence"
    ),
)


@dataclass(frozen=True)
class ValidationThresholds:
    """Пороги приёмки для автоматической проверки.

    Attributes:
        max_abs_delta_bias: Допустимое смещение delta для несмещённых оценщиков,
            в долях стандартного отклонения оценки по прогонам.
        coverage_low: Нижняя допустимая граница фактического покрытия 95% CI.
        coverage_high: Верхняя допустимая граница (сильно завышенное покрытие
            означает бессмысленно широкий интервал).
    """

    max_abs_delta_bias: float = 3.0
    coverage_low: float = 0.85
    coverage_high: float = 1.0


@dataclass(frozen=True)
class ValidationRunRow:
    """Строка результата одного прогона."""

    run_id: int
    seed: int
    estimator: str
    mode: str
    oracle_v_a: float
    oracle_v_b: float
    oracle_delta: float
    v_a_hat: float
    v_b_hat: float
    delta_hat: float
    v_b_bias: float
    delta_bias: float
    abs_delta_error: float
    delta_ci_low: Optional[float]
    delta_ci_high: Optional[float]
    delta_ci_covers_oracle: Optional[bool]
    delta_ci_width: Optional[float]
    is_significant: Optional[bool]
    p_value: Optional[float]
    propensity_source_used: Optional[str]
    ess_ratio: Optional[float]
    weight_mean: Optional[float]
    weight_p99: Optional[float]
    n_warnings: int


@dataclass(frozen=True)
class ValidationResult:
    """Результат всей серии прогонов."""

    config: dict
    run_rows: list[ValidationRunRow]
    aggregate: pd.DataFrame
    thresholds: ValidationThresholds = field(default_factory=ValidationThresholds)

    def to_dict(self) -> dict:
        return {
            "config": self.config,
            "run_rows": [asdict(r) for r in self.run_rows],
            "aggregate": self.aggregate.to_dict(orient="records"),
        }

    def check(self, *, estimators: Sequence[str] = ("ips", "dr")) -> list[str]:
        """Проверяет агрегаты на пороги и возвращает список нарушений.

        По умолчанию проверяются только оценщики, несмещённые **в конечной выборке**
        при верных пропенсити: IPS и DR. Из проверки исключены:

        - ``snips`` и ``sndr`` — самонормированные, а значит смещённые в конечной
          выборке по построению (это их сознательный размен смещения на дисперсию);
        - ``dm`` — его корректность целиком определяется тем, насколько верно
          специфицирована модель исхода, и не является свойством библиотеки;
        - ``replay`` — оценивает другую величину и несопоставим с остальными.

        Returns:
            Список текстовых описаний нарушений; пустой список — всё в порядке.
        """
        problems: list[str] = []
        if self.aggregate.empty:
            return ["Агрегат пуст: прогоны не выполнялись"]

        for _, row in self.aggregate.iterrows():
            if row["estimator"] not in estimators:
                continue
            label = f"{row['mode']}/{row['estimator']}"

            std = row.get("delta_std")
            bias = row.get("delta_bias_mean")
            runs = row.get("runs", 0)
            if std and std > 0 and runs > 1 and np.isfinite(bias):
                se = std / np.sqrt(runs)
                if abs(bias) > self.thresholds.max_abs_delta_bias * se:
                    problems.append(
                        f"{label}: смещение delta {bias:.5f} превышает "
                        f"{self.thresholds.max_abs_delta_bias} SE ({se:.5f})"
                    )

            coverage = row.get("delta_ci_coverage")
            if coverage is not None and np.isfinite(coverage) and runs > 1:
                # Само покрытие оценено по конечному числу прогонов, поэтому
                # сравнивать его с порогом «в лоб» нельзя: при 12 повторениях
                # разброс оценки покрытия сопоставим с шириной допустимого коридора,
                # и проверка начала бы срабатывать случайно. Нарушением считается
                # только статистически значимый выход за границу.
                se = np.sqrt(max(coverage * (1 - coverage), 1e-4) / runs)
                if coverage < self.thresholds.coverage_low - 2 * se:
                    problems.append(
                        f"{label}: покрытие CI {coverage:.3f} значимо ниже "
                        f"{self.thresholds.coverage_low} (SE {se:.3f}, прогонов {runs})"
                    )
                elif coverage > self.thresholds.coverage_high + 2 * se:
                    problems.append(
                        f"{label}: покрытие CI {coverage:.3f} значимо выше "
                        f"{self.thresholds.coverage_high} (SE {se:.3f}, прогонов {runs})"
                    )
        return problems


def _aggregate_rows(rows: list[ValidationRunRow]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([asdict(r) for r in rows])
    return (
        df.groupby(["mode", "estimator"], dropna=False)
        .agg(
            runs=("run_id", "count"),
            v_b_bias_mean=("v_b_bias", "mean"),
            v_b_rmse=("v_b_bias", lambda x: float(np.sqrt(np.mean(np.square(x))))),
            delta_bias_mean=("delta_bias", "mean"),
            delta_std=("delta_hat", "std"),
            delta_rmse=("delta_bias", lambda x: float(np.sqrt(np.mean(np.square(x))))),
            delta_ci_coverage=("delta_ci_covers_oracle", "mean"),
            delta_ci_width=("delta_ci_width", "mean"),
            significance_rate=("is_significant", "mean"),
            mean_ess_ratio=("ess_ratio", "mean"),
            mean_weight=("weight_mean", "mean"),
            mean_weight_p99=("weight_p99", "mean"),
        )
        .reset_index()
    )


def run_simulation_validation(
    *,
    target: str = "accept",
    estimators: Sequence[str] = ("ips", "snips", "dm", "dr", "sndr", "switch_dr"),
    modes: Optional[Sequence[ValidationMode]] = None,
    n_runs: int = 20,
    n_users: int = 2000,
    base_seed: int = 2026,
    n_boot: int = 200,
    alpha: float = 0.05,
    feature_cols: Sequence[str] = ("loyal", "age", "risk", "income"),
    action_col: str = "a_A",
    cluster_col: Optional[str] = None,
    tau: float = 20.0,
    policyA_epsilon: float = 0.2,
    policyB_tau: float = 0.8,
    thresholds: Optional[ValidationThresholds] = None,
    nuisance_spec: NuisanceSpec = DEFAULT_NUISANCE_SPEC,
    honest_outcome_metrics: bool = False,
) -> ValidationResult:
    """Прогоняет серию синтетических экспериментов и сравнивает с оракулом.

    Args:
        target: Целевая метрика (``accept`` или ``cltv``).
        estimators: Какие оценщики проверять.
        modes: Режимы прогона; по умолчанию :data:`DEFAULT_MODES`.
        n_runs: Число независимых повторений (определяет точность оценки покрытия).
        n_users: Размер выборки в одном прогоне.
        base_seed: Базовое зерно; каждый прогон использует своё.
        n_boot: Число bootstrap-реплик.
        alpha: Уровень значимости.
        cluster_col: Колонка кластеров. В этой среде один пользователь = одна строка,
            поэтому по умолчанию ``None`` (кластеризация не нужна и лишь замедляет).
        thresholds: Пороги приёмки для :meth:`ValidationResult.check`.
        nuisance_spec: Конфигурация nuisance-моделей. Позволяет, например, отключить
            взаимодействия ``x*a`` и посмотреть, как оценщики ведут себя при
            неверно специфицированной модели исхода.
        honest_outcome_metrics: По умолчанию ``False``: обвязка измеряет свойства
            **оценщиков**, а метрики качества nuisance в результат не записываются,
            поэтому дополнительный k-fold фит на каждую ячейку был бы чистой тратой
            (примерно ×5 ко времени прогона).

    Returns:
        :class:`ValidationResult`.
    """
    modes = tuple(modes) if modes is not None else DEFAULT_MODES
    rows: list[ValidationRunRow] = []

    for run_id in range(n_runs):
        seed = base_seed + run_id
        env = SyntheticRecommenderEnv(SynthConfig(n_users=n_users, seed=seed))
        X = env.sample_users()
        policyA = make_policy("epsilon_greedy", seed=seed, epsilon=policyA_epsilon)
        policyB = make_policy("softmax", seed=seed + 1, tau=policyB_tau)

        logs = env.simulate_logs_A(policyA, X, propensity_col="propensity_A")

        # Аналитический оракул: без шума эталона.
        oracle_v_a = env.oracle_value(policyA, X, metric=target)
        oracle_v_b = env.oracle_value(policyB, X, metric=target)
        oracle_delta = float(oracle_v_b - oracle_v_a)

        for mode in modes:
            for estimator in estimators:
                try:
                    summary = compare_policies(
                        logs,
                        policyB,
                        estimator=estimator,
                        target=target,
                        feature_cols=list(feature_cols),
                        action_col=action_col,
                        action_space=list(env.ACTIONS),
                        cluster_col=cluster_col,
                        propensity_col=mode.propensity_col,
                        propensity_source=mode.propensity_source,  # type: ignore[arg-type]
                        inference=mode.inference,
                        n_boot=n_boot,
                        alpha=alpha,
                        # Разное зерно на прогон: иначе ошибка ресэмплинга
                        # систематически повторяется во всех повторениях.
                        rng_seed=seed * 7919 + 13,
                        weight_clip=mode.weight_clip,
                        tau=tau,
                        use_crossfit=mode.use_crossfit,
                        nuisance_spec=nuisance_spec,
                        honest_outcome_metrics=honest_outcome_metrics,
                    )
                except ValueError as exc:
                    logger.warning("%s/%s пропущен: %s", mode.name, estimator, exc)
                    continue

                ci = summary.delta_ci
                covers = None
                width = None
                if ci is not None and np.isfinite(ci[0]) and np.isfinite(ci[1]):
                    covers = bool(ci[0] <= oracle_delta <= ci[1])
                    width = float(ci[1] - ci[0])

                diag = summary.diagnostics.to_dict() if summary.diagnostics else {}
                rows.append(
                    ValidationRunRow(
                        run_id=run_id,
                        seed=seed,
                        estimator=estimator,
                        mode=mode.name,
                        oracle_v_a=float(oracle_v_a),
                        oracle_v_b=float(oracle_v_b),
                        oracle_delta=oracle_delta,
                        v_a_hat=float(summary.v_a),
                        v_b_hat=float(summary.v_b),
                        delta_hat=float(summary.delta),
                        v_b_bias=float(summary.v_b - oracle_v_b),
                        delta_bias=float(summary.delta - oracle_delta),
                        abs_delta_error=float(abs(summary.delta - oracle_delta)),
                        delta_ci_low=ci[0] if ci else None,
                        delta_ci_high=ci[1] if ci else None,
                        delta_ci_covers_oracle=covers,
                        delta_ci_width=width,
                        is_significant=summary.is_significant,
                        p_value=summary.p_value,
                        propensity_source_used=summary.propensity_source,
                        ess_ratio=diag.get("weight_ess_ratio"),
                        weight_mean=diag.get("weight_mean"),
                        weight_p99=diag.get("weight_p99"),
                        n_warnings=len(summary.warnings),
                    )
                )

    config = {
        "target": target,
        "estimators": list(estimators),
        "modes": [asdict(m) for m in modes],
        "n_runs": n_runs,
        "n_users": n_users,
        "base_seed": base_seed,
        "n_boot": n_boot,
        "alpha": alpha,
        "cluster_col": cluster_col,
    }
    return ValidationResult(
        config=config,
        run_rows=rows,
        aggregate=_aggregate_rows(rows),
        thresholds=thresholds or ValidationThresholds(),
    )
