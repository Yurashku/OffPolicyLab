# Policyscope: офлайн‑оценка политик рекомендаций (переиспользуемый пайплайн)

`Policyscope` помогает оценивать новую политику **B** по логам текущей политики **A** без онлайн A/B‑теста.

Главное в текущей версии:
- API стал **универсальным**: названия колонок (`a_A`, `a_B`, целевая метрика, `user_id`) и список признаков задаются аргументами.
- Туториал стал короче и практичнее: есть компактный сценарий «взял свой DataFrame → получил все OPE‑оценки».
- Bootstrap CI считается через единый API: `OPEEvaluator(...).evaluate(method)` или `estimate_value_with_ci(..., method=...)`.
- Все основные OPE‑оценщики снабжены подробными docstring на русском (аргументы, возвращаемые значения, интерпретация).

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
- DM (Direct Method)
- DR / SNDR / Switch-DR
- Кластерный и обычный бутстрэп (если `cluster_col=None`)
- Data contract слой для логов contextual bandit (`BanditSchema`, `LoggedBanditDataset`)

## Архитектура

- Краткое описание архитектурных границ и доменной модели: `docs/architecture.md`.
- Библиотека разделяет слои: `data contract -> point estimators -> inference -> diagnostics/reporting`.

## Минимальный формат данных

Нужны:
- колонка действия, зафиксированного в логах A (по умолчанию `a_A`),
- целевая метрика (например, `accept`, `cltv` или ваша `reward`),
- признаки (`feature_cols`),
- опционально `user_id` для кластерного бутстрэпа.

Дополнительно можно хранить `a_B` (действие рекомендованное B) для диагностики и таблиц в туториале.

## Универсальный пример на своих данных

```python
import numpy as np
import pandas as pd
from policyscope.data import BanditSchema, LoggedBanditDataset
from policyscope.estimators import (
    train_pi_hat,
    pi_hat_predict,
    train_mu_hat,
    prepare_piB_taken,
    take_action_probabilities,
    ips_value,
    snips_value,
    dm_value,
    dr_value,
)
from policyscope.ci import estimate_value_with_ci
from policyscope.evaluator import OPEEvaluator

# ваш датасет
# df columns example:
# user_col, logged_action, candidate_action, reward, f1, f2, f3

df = pd.read_csv("my_logs.csv")

# (Опционально, но рекомендуется) валидируем контракт логов до OPE
schema = BanditSchema(
    action_col="logged_action",
    reward_col="reward",
    feature_cols=["f1", "f2", "f3"],
    cluster_col="user_col",
)
logged = LoggedBanditDataset(df=df, schema=schema)
df = logged.df

feature_cols = ["f1", "f2", "f3"]
action_col = "logged_action"
target_col = "reward"

policyB = ...  # объект с методом action_probs(df) -> (n, k)

# 1) Вероятность того, что B выбрала бы логированное действие
piB_taken = prepare_piB_taken(df, policyB, action_col=action_col)

# 2) Оценка модели поведения A: pA(a|x)
pi_model = train_pi_hat(df, feature_cols=feature_cols, action_col=action_col)
pA_all = pi_hat_predict(pi_model, df)
pA_taken = take_action_probabilities(
    pA_all,
    df[action_col].values,
    action_space=pi_model.classes_,
)

# 3) Модель исхода mu(x, a)
mu = train_mu_hat(df, target=target_col, feature_cols=feature_cols, action_col=action_col)

# 4) OPE-оценки
v_ips, ess_ips, clip_ips = ips_value(df, piB_taken, pA_taken, target=target_col, action_col=action_col)
v_snips, ess_snips, clip_snips = snips_value(df, piB_taken, pA_taken, target=target_col, action_col=action_col)
v_dm = dm_value(df, policyB, mu, target=target_col)
v_dr, ess_dr, clip_dr = dr_value(df, policyB, mu, pA_taken, target=target_col, action_col=action_col)

# 5) Единый слой CI для любого встроенного OPE-эстиматора
ips_ci = estimate_value_with_ci(
    df,
    policyB,
    method="ips",          # "snips", "dm", "dr", "sndr", "switch_dr", ...
    target=target_col,
    action_col=action_col,
    feature_cols=feature_cols,
    cluster_col="user_col",
    n_boot=300,
)
print(ips_ci)

# 6) Единый абстрактный объект (CI включён по умолчанию)
evaluator = OPEEvaluator(
    df,
    policyB,
    target=target_col,
    feature_cols=feature_cols,
    action_col=action_col,
    cluster_col="user_col",
    n_boot=300,
    alpha=0.05,
    weight_clip=20.0,
)
dr_report = evaluator.evaluate("dr")  # также: "ips", "snips", "dm", "sndr", "switch_dr", ...
print(dr_report)  # V_A, V_B, Delta + CI
```

## Теория и ссылки на статьи (RU)

### Нотация в формулах

Используем единый набор обозначений: `π_A`, `π_B`, `μ̂`, `V̂`, `Δ`, `τ`, `w̄`.

Добавлен отдельный подробный гайд по математической интуиции и строгим источникам для всех реализованных OPE-методов:

- `docs/ope_methods_math_guide_ru.md`

В гайде разобраны: On-policy baseline, Replay, IPS, SNIPS, DM, DR, SNDR, Switch-DR и методы построения доверительных интервалов (CI).
Формулы в гайде приведены в plain-text формате (без обязательной LaTeX-разметки), чтобы корректно читаться и на GitHub, и в локальных Markdown-просмотрщиках.
Отдельно поясняется важный момент: bootstrap CI — это не отдельный OPE-эстиматор, а inference-обёртка поверх выбранного оценщика.
Гайд также содержит отдельное сравнение: какие CI-подходы используются в OPE-литературе/практике и что именно реализовано в текущем репозитории.
В API есть единый CI-слой `estimate_value_with_ci` для встроенных OPE-эстиматоров и совместимая низкоуровневая обёртка `estimator_with_bootstrap_ci` для произвольных `estimator_fn`.
Нотация в формулах гайда унифицирована: `π_A`, `π_B`, `μ̂`, `V̂`, `Δ`, `τ`, `w̄` — для более читаемого и однозначного математического стиля.
Также добавлен `OPEEvaluator` — единый абстрактный объект, который унифицированно вызывает эстиматоры по имени и возвращает CI по умолчанию.

## Постоянные инструкции для AI-агентов

В репозиторий добавлены постоянные инструкции для агентных инструментов (в т.ч. Codex):

- `AGENTS.md` — каноничный набор постоянных правил и workflow для Codex/агентов.


## Быстрый синтетический запуск

```bash
python examples/run_synthetic_experiment.py --n_users 50000 --seed 42 --policyA epsilon_greedy --policyB softmax
```

Скрипт сохраняет артефакты в `artifacts/`.

## Туториал

- Основной notebook: `examples/tutorial.ipynb`
- В нём показано:
  1. генерация синтетики,
  2. валидация data contract через `BanditSchema` / `LoggedBanditDataset`,
  3. вывод oracle ground-truth (`V_A`, `V_B`, `Delta`) сразу после генерации,
  4. явная остановка использования синтезатора после шага ground-truth (anti data-leakage),
  5. проверка логов и таблица фич + `a_A` + `a_B`,
  6. компактный расчёт всех метрик,
  7. унифицированный вызов через `OPEEvaluator` (переключается только имя эстиматора),
  8. bootstrap CI для **всех** методов через `OPEEvaluator`,
  9. итоговая таблица сравнения методов с колонками CI (`V_B_CI`, `Delta_CI`) для каждого метода.

Для переноса на реальный кейс в туториале отдельно показано, какие 3-4 строки обычно нужно заменить (`df/logs`, `feature_cols`, `action_col`, `target_col`).

После `OPEEvaluator(...).evaluate("dr")` вы получите словарь:

```python
{
  'V_A': ...,
  'V_A_CI': (..., ...),
  'V_B': ...,
  'V_B_CI': (..., ...),
  'Delta': ...,
  'Delta_CI': (..., ...),
  'n_boot': ...
}
```

## Проверки перед коммитом

```bash
python -m flake8 src tests
pytest
jupyter nbconvert --to notebook --execute examples/tutorial.ipynb --inplace
```
