# Policyscope: офлайн-оценка рекомендательных систем

**Policyscope** помогает сравнивать рекомендательные модели без запуска дорогостоящих A/B‑тестов. 
Библиотека переиспользует логи текущей политики и оценивает, насколько другая политика могла бы увеличить целевую метрику.

## Как это работает

1. **Собираем логи** текущей политики A: какое действие показали, с какой вероятностью и как реагировал пользователь.
2. **Определяем новую политику B** — например, другую модель рекомендаций.
3. **Пере‑взвешиваем** наблюдения из логов A и получаем приближённое значение метрики под политикой B.

## Реализованные алгоритмы

- **Replay** — учитывает только те логи, где B совпадает с A.
- **IPS** — взвешивает отклики по отношению вероятностей выбора в B и A.
- **SNIPS** — нормализует веса IPS для меньшей дисперсии.
- **Doubly Robust** — комбинирует модель отклика и IPS; достаточно корректности хотя бы одной из них.

## Установка

```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

или как пакет:

```bash
pip install -e .
```

## Пример: синтетический эксперимент

В репозитории есть скрипт, который генерирует пользователей и сравнивает две политики.

```bash
python examples/run_synthetic_experiment.py \
  --n_users 50000 \
  --seed 42 \
  --policyA epsilon_greedy --epsilon 0.15 \
  --policyB softmax --tau 0.7 \
  --horizon 90 \
  --weight_clip 20
```

После запуска создаётся папка `artifacts` с логами, оценками и коротким текстовым отчётом.

## Использование на собственных данных

Логи политики A должны содержать:

- `user_id` — идентификатор пользователя;
- `a_A` — действие, которое показала политика A;
- `propensity_A` — вероятность показа этого действия;
- `accept` и `cltv` — отклик и ценность;
- признаки пользователя (например, возраст, доход и т.п.).

Пример кода:

```python
import pandas as pd
from policyscope.estimators import train_mu_hat, prepare_piB_taken, ips_value, snips_value, dr_value
from policyscope.policies import make_policy

df = pd.read_csv("logs_with_propensity.csv")
policyB = make_policy("softmax", tau=0.7)
piB_taken = prepare_piB_taken(df, policyB)
mu_hat = train_mu_hat(df, target="accept")
V_B_dr, ess, clip = dr_value(df, policyB, mu_hat, target="accept")
print(V_B_dr)
```

## Ссылки

- [Counterfactual Evaluation for Recommendation Systems](https://eugeneyan.com/writing/offline-recsys/)
- Farajtabar et al., *More Robust Doubly Robust Off-policy Evaluation* (arXiv:2205.13421)

Policyscope распространяется по лицензии MIT.
