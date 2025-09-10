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
- **Direct Method** — строит модель отклика и прогнозирует исходы под политикой B.
- **Doubly Robust** — комбинирует Direct Method и IPS; достаточно корректности хотя бы одной из них.
- **SN-DR** — нормализует поправку Doubly Robust, что снижает дисперсию.
- **Switch-DR** — применяет IPS-поправку только при малых весах, иначе полагается на модель.

## Предположения и ограничения

- **Replay** — новая политика должна часто совпадать со старой, иначе большинство логов отбрасывается.
- **IPS** — требует точного знания вероятностей действий в обеих политиках; большие веса увеличивают дисперсию.
- **SNIPS** — нормализует веса IPS и снижает дисперсию, но остаётся чувствительным к ошибкам вероятностей и малым объёмам данных.
- **Direct Method** — зависит от точности модели отклика и может смещаться вне обучающей области.
- **Doubly Robust** — корректность достигается, если верна хотя бы модель отклика или пропенсити, но метод чувствителен к ошибкам обеих моделей и выбору клиппинга.
- **SN-DR** — уменьшает дисперсию DR за счёт нормализации весов, но наследует его предположения.
- **Switch-DR** — отбрасывает экстремальные веса, сочетая DM и DR, но выбор порога влияет на смещение.

## Jupyter-туториал

Интерактивный ноутбук с теорией и примером расчёта ATE доступен в файле [examples/tutorial.ipynb](examples/tutorial.ipynb).

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

При работе на синтетике полезно сравнивать офлайн‑оценки с истинным эффектом (oracle). Такое сравнение позволяет проверить состоятельность методов OPE и убедиться, что оценщики не дают систематического смещения.

## Требования к входным данным

Логи политики A должны содержать обязательные поля:

- `user_id` — идентификатор пользователя;
- `a_A` — действие, которое показала политика A;
- `accept` и/или `cltv` — отклик и ценность;
- признаки пользователя (возраст, доход и др.), используемые моделью.

Числовые поля `age`, `risk` и `income` можно передавать в исходном масштабе:
функции обучения (`train_pi_hat`, `train_mu_hat`) автоматически выполняют их
нормализацию.

## Пример применения на своих данных

Функции обучения выполняют внутреннюю нормализацию числовых признаков,
поэтому в DataFrame достаточно сырых столбцов `age`, `risk` и `income`.

```python
import numpy as np
import pandas as pd
from policyscope.estimators import (
    train_pi_hat,
    pi_hat_predict,
    train_mu_hat,
    prepare_piB_taken,
    replay_value,
    ips_value,
    snips_value,
    dm_value,
    dr_value,
    sndr_value,
    switch_dr_value,
)
from policyscope.policies import make_policy

df = pd.read_csv("logs_without_propensity.csv")
policyB = make_policy("softmax", tau=0.7)
piB_taken = prepare_piB_taken(df, policyB)
pi_model = train_pi_hat(df)
pA_all = pi_hat_predict(pi_model, df)
pA_taken = pA_all[np.arange(len(df)), df["a_A"].values]
mu_hat = train_mu_hat(df, target="accept")
V_replay = replay_value(df, policyB, target="accept")
V_ips, ess_ips, clip_ips = ips_value(df, piB_taken, pA_taken, target="accept")
V_snips, ess_snips, clip_snips = snips_value(df, piB_taken, pA_taken, target="accept")
V_dm = dm_value(df, policyB, mu_hat, target="accept")
V_dr, ess_dr, clip_dr = dr_value(df, policyB, mu_hat, pA_taken, target="accept")
V_sndr, ess_sndr, clip_sndr = sndr_value(df, policyB, mu_hat, pA_taken, target="accept")
V_switch, ess_switch, share_switch = switch_dr_value(df, policyB, mu_hat, pA_taken, tau=20, target="accept")
print(V_replay, V_ips, V_snips, V_dm, V_dr, V_sndr, V_switch)
```

## Валидация оценок

- **ESS** — проверяйте эффективный размер выборки, чтобы убедиться в достаточном покрытии новой политики.
- **Клиппинг** — ограничивайте большие веса IPS, чтобы уменьшить дисперсию и влияние выбросов.
- **Бутстрэп** — оценивайте доверительные интервалы путём повторной выборки логов.

## Логирование

Функции‑оценщики выводят подробные сообщения на русском языке. Для каждого
алгоритма логируется начало работы, проверки корректности пропенсити,
значение ESS с предупреждением при низком покрытии, доля клиппинга
и итоговое значение метрики. По умолчанию логирование настроено (формат
`%(message)s`), поэтому дополнительные настройки не требуются.

## Разработка

Перед коммитом выполните проверки стиля и тесты:

```bash
python -m flake8 src tests
pytest
```

CI также запускает синтетический эксперимент `examples/run_synthetic_experiment.py`, чтобы убедиться в корректной работе библиотеки.

## Ссылки

- Joachims et al., *Unbiased Learning-to-Rank with Biased Feedback* (WSDM 2017)
- Dudík et al., *Doubly Robust Policy Evaluation and Learning* (ICML 2011)
- Farajtabar et al., *More Robust Doubly Robust Off-policy Evaluation* (arXiv:2205.13421)
- [Counterfactual Evaluation for Recommendation Systems](https://eugeneyan.com/writing/offline-recsys/)

Policyscope распространяется по лицензии MIT.
