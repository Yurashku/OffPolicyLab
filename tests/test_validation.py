"""Тесты валидационной обвязки.

Быстрые тесты проверяют структуру результата, медленные (маркер ``slow``) —
собственно статистические свойства: смещение около нуля и покрытие CI близко
к номинальному. Раньше обвязка считала эти числа, но никто на них не смотрел.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from policyscope.validation import (
    ValidationMode,
    ValidationThresholds,
    run_simulation_validation,
)


FAST_MODES = (ValidationMode(name="logged", propensity_source="logged"),)


def _restrict(result, *, mode: str):
    """Возвращает копию результата только с одним режимом."""
    from policyscope.validation import ValidationResult

    rows = [r for r in result.run_rows if r.mode == mode]
    agg = result.aggregate[result.aggregate["mode"] == mode]
    return ValidationResult(
        config=result.config, run_rows=rows, aggregate=agg, thresholds=result.thresholds
    )


@pytest.fixture(scope="module")
def small_result():
    return run_simulation_validation(
        n_runs=2, n_users=300, n_boot=20,
        modes=FAST_MODES, estimators=("ips", "dr"),
    )


class TestStructure:
    def test_row_per_run_mode_estimator(self, small_result):
        assert len(small_result.run_rows) == 2 * 1 * 2

    def test_aggregate_has_expected_columns(self, small_result):
        cols = set(small_result.aggregate.columns)
        assert {"mode", "estimator", "runs", "delta_bias_mean", "delta_ci_coverage"} <= cols

    def test_oracle_and_estimates_are_finite(self, small_result):
        for row in small_result.run_rows:
            assert np.isfinite(row.oracle_delta)
            assert np.isfinite(row.delta_hat)
            assert np.isfinite(row.delta_bias)

    def test_bias_is_estimate_minus_oracle(self, small_result):
        for row in small_result.run_rows:
            assert row.delta_bias == pytest.approx(row.delta_hat - row.oracle_delta)

    def test_serializable(self, small_result):
        payload = small_result.to_dict()
        assert {"config", "run_rows", "aggregate"} <= set(payload)

    def test_different_runs_use_different_seeds(self, small_result):
        """Одно зерно на все повторения сделало бы оценку покрытия скоррелированной."""
        seeds = {row.seed for row in small_result.run_rows}
        assert len(seeds) == 2

    def test_logged_propensity_mode_is_actually_used(self, small_result):
        assert all(r.propensity_source_used == "logged" for r in small_result.run_rows)


class TestCheck:
    def test_empty_result_is_reported_as_problem(self):
        result = run_simulation_validation(n_runs=0, modes=FAST_MODES, estimators=("ips",))
        assert result.check()

    def test_impossible_thresholds_produce_violations(self, small_result):
        strict = ValidationThresholds(max_abs_delta_bias=0.0, coverage_low=0.999)
        result = type(small_result)(
            config=small_result.config,
            run_rows=small_result.run_rows,
            aggregate=small_result.aggregate,
            thresholds=strict,
        )
        assert result.check(estimators=("dr",))


class TestCoverageCheckAccountsForMonteCarloError:
    """Оценка покрытия сама шумная; порог нельзя применять «в лоб».

    Иначе проверка при малом числе прогонов срабатывала бы случайно, и её
    пришлось бы отключить — то есть остаться без автоматического контроля.
    """

    @staticmethod
    def _result(coverage: float, runs: int):
        from policyscope.validation import ValidationResult

        agg = pd.DataFrame([{
            "mode": "m", "estimator": "ips", "runs": runs,
            "delta_bias_mean": 0.0, "delta_std": 0.1,
            "delta_ci_coverage": coverage,
        }])
        return ValidationResult(config={}, run_rows=[], aggregate=agg)

    def test_noisy_low_coverage_on_few_runs_is_tolerated(self):
        assert not self._result(0.667, runs=6).check(estimators=("ips",))

    def test_same_coverage_on_many_runs_is_flagged(self):
        problems = self._result(0.667, runs=200).check(estimators=("ips",))
        assert problems and "значимо ниже" in problems[0]

    def test_nominal_coverage_is_never_flagged(self):
        assert not self._result(0.95, runs=200).check(estimators=("ips",))


@pytest.mark.slow
class TestStatisticalProperties:
    """Медленная проверка: оценщики действительно работают на синтетике."""

    @pytest.fixture(scope="class")
    def result(self):
        # Инференс здесь аналитический: проверяются свойства **оценщиков**
        # (смещение, RMSE), а покрытие интервалов отдельно проверяется
        # в tests/test_inference.py. Bootstrap с переобучением nuisance на каждой
        # из сотен реплик сделал бы этот набор непригодным для регулярного запуска.
        return run_simulation_validation(
            n_runs=30, n_users=2000, n_boot=100,
            modes=(
                ValidationMode(
                    name="logged", propensity_source="logged", inference="influence"
                ),
                ValidationMode(
                    name="estimated", propensity_source="estimated", inference="influence"
                ),
            ),
            estimators=("ips", "snips", "dm", "dr", "sndr"),
        )

    def test_unbiased_estimators_pass_thresholds(self, result):
        """IPS и DR несмещены в конечной выборке при **верных** пропенсити.

        SNIPS и SNDR сюда не входят: самонормировка вносит смещение конечной
        выборки по построению, обменивая его на снижение дисперсии.

        Режим `estimated` тоже исключён — см.
        :meth:`test_estimated_propensity_failure_is_flagged`.
        """
        logged_only = _restrict(result, mode="logged")
        problems = logged_only.check(estimators=("ips", "dr"))
        assert not problems, "\n".join(problems)

    def test_coverage_is_near_nominal_with_logged_propensity(self, result):
        agg = result.aggregate
        logged = agg[agg["mode"] == "logged"]
        for _, row in logged.iterrows():
            if row["estimator"] in ("ips", "dr", "sndr", "snips"):
                assert 0.85 <= row["delta_ci_coverage"] <= 1.0, (
                    f"{row['estimator']}: покрытие {row['delta_ci_coverage']:.3f}"
                )

    def test_estimated_propensity_failure_is_flagged(self, result):
        """Оценка пропенсити на почти детерминированной логирующей политике ломается —
        и библиотека обязана об этом сообщать, а не молча выдавать число.

        Логистическая модель не может воспроизвести ступенчатые истинные пропенсити
        ε-greedy политики, из-за чего веса важности достигают тысяч. Само по себе
        это свойство данных, а не библиотеки; проверяется здесь то, что такой прогон
        не выглядит благополучным.
        """
        agg = result.aggregate.set_index(["mode", "estimator"])
        logged_rmse = agg.loc[("logged", "ips"), "delta_rmse"]
        estimated_rmse = agg.loc[("estimated", "ips"), "delta_rmse"]
        assert estimated_rmse > 10 * logged_rmse, (
            "ожидалась деградация IPS на оценённых пропенсити; "
            f"logged={logged_rmse:.4f}, estimated={estimated_rmse:.4f}"
        )

        estimated = [r for r in result.run_rows
                     if r.mode == "estimated" and r.estimator == "ips"]
        logged = [r for r in result.run_rows
                  if r.mode == "logged" and r.estimator == "ips"]
        assert all(r.n_warnings > 0 for r in estimated), (
            "прогоны с непригодными весами прошли без единого предупреждения"
        )
        # Прогоны с точными пропенсити должны быть заметно «тише».
        assert (np.mean([r.n_warnings for r in estimated])
                > np.mean([r.n_warnings for r in logged]))

    def test_dr_is_insensitive_to_outcome_model_specification(self):
        """При верных пропенсити оценка DR почти не зависит от модели исхода.

        Это наблюдаемое следствие двойной робастности: IPS-поправка компенсирует
        ошибку ``mu``, поэтому лишение модели исхода ключевых признаков смещает её
        предсказания, но не итоговую оценку DR.

        Утверждение «DR точнее DM» здесь **не** проверяется: на этой среде DM
        остаётся почти несмещённым даже при грубой модели исхода, потому что он
        усредняет ``mu`` по популяции, а поточечные ошибки в среднем гасятся.
        Строгое доказательство двойной робастности — с подстановкой заведомо
        неверных nuisance и негативным контролем — живёт в
        ``tests/test_estimator_correctness.py::TestDoubleRobustness``.
        """
        def dr_bias(feature_cols):
            result = run_simulation_validation(
                n_runs=12, n_users=2000, n_boot=50, feature_cols=feature_cols,
                modes=(
                    ValidationMode(
                        name="logged", propensity_source="logged", inference="influence"
                    ),
                ),
                estimators=("dr",),
            )
            return float(result.aggregate.set_index("estimator").loc["dr", "delta_bias_mean"])

        rich = dr_bias(("loyal", "age", "risk", "income"))
        starved = dr_bias(("loyal",))
        assert abs(rich - starved) < 0.005, (
            f"оценка DR заметно изменилась при обеднении модели исхода: "
            f"{rich:.5f} -> {starved:.5f}"
        )

    def test_mean_weight_is_near_one_with_logged_propensity(self, result):
        agg = result.aggregate
        logged = agg[(agg["mode"] == "logged") & (agg["estimator"] == "ips")]
        assert float(logged["mean_weight"].iloc[0]) == pytest.approx(1.0, abs=0.05)
