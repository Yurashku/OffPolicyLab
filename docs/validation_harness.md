# Simulation validation harness (internal methodology check)

`policyscope.validation.run_simulation_validation(...)` запускает повторяемые synthetic-эксперименты и сравнивает OPE-оценки с oracle-значениями из `SyntheticRecommenderEnv`.

## Что измеряется

На уровне run:
- oracle `V_A`, `V_B`, `delta`;
- оценённые `V_B`, `delta`;
- bias и absolute error;
- `Delta_CI` coverage (если CI рассчитан);
- частота significance decision (`is_significant`);
- diagnostics-поля (например, `weight_ess_ratio`, `weight_p99`);
- provenance (`propensity_source_used`, `propensity_column_used`);
- nuisance-quality summaries (например behavior log-loss, outcome log-loss/RMSE) для сравнения режимов.

На уровне aggregate (по `mode` и `estimator`):
- mean bias, std, RMSE для `V_B` и `delta`;
- mean absolute error;
- CI coverage;
- significance rate;
- средние diagnostic summary метрики.

## Зачем это нужно

Это **внутренняя validation harness** для методологической уверенности и регрессий при развитии кода.

Это **не гарантия** корректности на любых реальных данных:
- synthetic environment упрощает реальный мир;
- bootstrap inference и diagnostics полезны, но не заменяют предметную валидацию конкретного прод-кейса.

## Быстрый пример

```python
from policyscope.validation import run_simulation_validation, ValidationMode

result = run_simulation_validation(
    n_runs=3,
    n_users=300,
    n_boot=20,
    estimators=("ips", "dr", "sndr"),
    modes=(
        ValidationMode(name="estimated", propensity_source="estimated"),
        ValidationMode(name="logged", propensity_source="logged", propensity_col="propensity_A"),
        ValidationMode(name="crossfit", propensity_source="estimated", use_crossfit=True),
    ),
)

print(result.aggregate)
```
