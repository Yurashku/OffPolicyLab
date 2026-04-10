# Policyscope: офлайн‑оценка политик рекомендаций (переиспользуемый пайплайн)

`Policyscope` помогает оценивать новую политику **B** по логам текущей политики **A** без онлайн A/B‑теста.

Главное в текущей версии:
- API стал **универсальным**: названия колонок (`a_A`, `a_B`, целевая метрика, `user_id`) и список признаков задаются аргументами.
- Туториал стал короче и практичнее: есть компактный сценарий «взял свой DataFrame → получил все OPE‑оценки».
- Бутстрэп для DR можно вызывать одной функцией (`dr_with_bootstrap_ci`) без ручной сборки циклов.
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
    dr_with_bootstrap_ci,
)

# ваш датасет
# df columns example:
# user_col, logged_action, candidate_action, reward, f1, f2, f3

df = pd.read_csv("my_logs.csv")

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

# 5) DR + bootstrap CI одной функцией
dr_ci = dr_with_bootstrap_ci(
    df,
    policyB,
    target=target_col,
    feature_cols=feature_cols,
    action_col=action_col,
    cluster_col="user_col",   # либо None
    n_boot=300,
)
print(v_ips, v_snips, v_dm, v_dr, dr_ci)
```

## Теория и ссылки на статьи (RU)

Добавлен отдельный подробный гайд по математической интуиции и строгим источникам для всех реализованных OPE-методов:

- `docs/ope_methods_math_guide_ru.md`

В гайде разобраны: On-policy baseline, Replay, IPS, SNIPS, DM, DR, SNDR, Switch-DR и методы построения доверительных интервалов (CI).
Формулы в гайде приведены в plain-text формате (без обязательной LaTeX-разметки), чтобы корректно читаться и на GitHub, и в локальных Markdown-просмотрщиках.
Отдельно поясняется важный момент: bootstrap CI — это не отдельный OPE-эстиматор, а inference-обёртка поверх выбранного оценщика.

## Постоянные инструкции для AI-агентов

В репозиторий добавлены постоянные инструкции для агентных инструментов (в т.ч. Codex):

- `AGENT.md` — каноничный набор постоянных правил и workflow,
- `AGENTS.md` — указатель на `AGENT.md` для совместимости с авто-подхватом инструкций.


## Быстрый синтетический запуск

```bash
python examples/run_synthetic_experiment.py --n_users 50000 --seed 42 --policyA epsilon_greedy --policyB softmax
```

Скрипт сохраняет артефакты в `artifacts/`.

## Туториал

- Основной notebook: `examples/tutorial.ipynb`
- В нём показано:
  1. проверка логов,
  2. таблица с фичами + `a_A` + `a_B`,
  3. компактный расчёт всех метрик,
  4. bootstrap через `dr_with_bootstrap_ci`,
  5. итоговая «красивая» таблица сравнения методов на текущих данных (Replay/IPS/SNIPS/DM/DR/SNDR/Switch‑DR + ESS/clip/switch + CI для `V_A` и `V_B`).

После `dr_with_bootstrap_ci` вы получите словарь:

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
