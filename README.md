# Policyscope

Библиотека для **contextual bandit off-policy evaluation (OPE)**: оценка target/candidate policy **B** по логам behavior/logging policy **A**.

Главные выходы сравнения: `V_A`, `V_B`, `Delta = V_B - V_A`, CI/p-value и diagnostics/trust metadata (включая `weight_ess_ratio`, replay overlap и warning flags).

## Быстрый навигатор

### 1) Я хочу запустить на своих данных
- **Quickstart notebook (основной путь)**: `examples/quickstart_own_data_ru.ipynb`
- Покрывает: `BanditSchema`, `LoggedBanditDataset`, `compare_policies(...)`, режимы propensity (logged/estimated/auto), чтение `Delta`, CI, `p_value`, diagnostics и `trust_level`.
- Если у вас уже есть колонка `action_B`, используйте `policyscope.policy_adapters.DeterministicActionColumnPolicy`.

### 2) Я хочу понять, как методы ведут себя относительно oracle
- **Synthetic comparison notebook**: `examples/compare_estimators_vs_oracle_ru.ipynb`
- Покрывает сравнение Replay / IPS / SNIPS / DM / DR / SNDR / Switch-DR в контролируемых synthetic-сценариях.

### 3) Я хочу понять, когда OPE результату можно (и нельзя) доверять
- **Практический интерпретационный гайд (RU)**: `docs/how_to_interpret_ope_outputs_ru.md`

### 4) Я хочу понять устройство библиотеки
- **Architecture doc**: `docs/architecture.md`

### 5) Я хочу системно валидировать поведение оценщиков
- **Validation harness doc**: `docs/validation_harness.md`

## Установка

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# или
pip install -e .
```

## Что реализовано

- Replay
- IPS / SNIPS
- DM
- DR / SNDR / Switch-DR
- Bootstrap inference (clustered и non-clustered)
- Data contract слой: `BanditSchema`, `LoggedBanditDataset`


Инференс significance в `compare_policies` опирается на **centered paired bootstrap** p-value для проверки **H0: Delta = 0**.

## Официальный high-level entrypoint

```python
from policyscope.comparison import compare_policies

summary = compare_policies(
    logged_dataset,
    policyB,
    estimator="dr",
    target="accept",
    propensity_source="auto",
)

print(summary.to_dict())
```

## Про experiment runners (скрипты)

Файлы в `examples/`:
- `run_synthetic_experiment.py`
- `run_validation_experiment.py`

Это **script-like experiment runners** для пакетных прогонов/артефактов, а не основной обучающий путь. Для обучения и first-run используйте notebook'и из раздела «Быстрый навигатор».

## Дополнительный tutorial

- `examples/tutorial.ipynb` сохранён как расширенный walkthrough.
- Для первого запуска рекомендуется `examples/quickstart_own_data_ru.ipynb`.

Для multi-metric сценариев используйте `compare_policies_multi_target` (повторная scalar-оценка по нескольким target).

Propensity source modes: `auto`, `logged`, `estimated` (для взвешенных estimators).

Nuisance model diagnostics возвращаются в summary (`nuisance_diagnostics`) и дополняют overlap/weight diagnostics.

Важно: OPE + diagnostics/trust metadata помогают с offline screening, но не являются автоматической заменой A/B-теста для high-stakes решений.
