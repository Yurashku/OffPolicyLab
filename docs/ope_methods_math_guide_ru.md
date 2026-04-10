# Методы офлайн-оценки политик в `Policyscope`: интуиция, формулы, доверительные интервалы

> Важно про отображение: формулы в документе даны в **plain-text формате** (без обязательного LaTeX `$$...$$`), чтобы всё одинаково читалось на GitHub и в локальных Markdown-просмотрщиках.

## Что важно сразу

- `IPS/SNIPS/DM/DR/SNDR/Switch-DR/Replay` — это **оценщики значения политики** (point estimators).
- `Bootstrap CI` — это **не отдельный OPE-оценщик**, а процедура статистического вывода (inference), которая строит доверительный интервал вокруг выбранного оценщика.

Иными словами, `dr_with_bootstrap_ci` — это обёртка «оценщик + интервал», а не «ещё один метод оценки политики» наравне с IPS/DR.

---

## 1) Единая постановка задачи

Есть логи политики A: `(x_i, a_i, r_i)`, где:
- `x_i` — контекст,
- `a_i` — действие, выбранное логирующей политикой `pi_A`,
- `r_i` — наблюдаемая награда.

Нужно оценить ценность новой политики `pi_B`:

```text
V(pi_B) = E_x [ E_{a~pi_B(.|x)} [ r(x, a) ] ]
```

Ключевые обозначения:
- `mu(x,a) = E[r|x,a]` — модель ожидаемой награды,
- `p_A(a|x) = pi_A(a|x)` — propensity логирующей политики,
- `w_i = pi_B(a_i|x_i) / p_A(a_i|x_i)` — importance weight.

---

## 2) Point-estimators (что оценивает `V_hat`)

### 2.1 On-policy baseline (`value_on_policy`)

```text
V_A_hat = (1/n) * sum_i r_i
```

Это baseline текущей политики A.

### 2.2 Replay (`replay_value`)

```text
I = { i : a_i == a_i^B }
V_replay_hat = (1/|I|) * sum_{i in I} r_i
```

- Плюс: «честная» фильтрация без моделирования контрфактов.
- Минус: может иметь большую дисперсию, если совпадений мало.

Источник: Li et al. (unbiased offline evaluation) — https://arxiv.org/abs/1003.5956

### 2.3 IPS (`ips_value`)

```text
V_IPS_hat = (1/n) * sum_i [w_i * r_i]
where w_i = pi_B(a_i|x_i) / p_A(a_i|x_i)
```

- Несмещён при корректных propensity и overlap.
- Нестабилен при тяжёлом хвосте весов.

Источник: Dudík, Langford, Li — https://arxiv.org/abs/1103.4601

### 2.4 SNIPS (`snips_value`)

```text
V_SNIPS_hat = sum_i (w_i * r_i) / sum_i w_i
```

- Обычно более стабилен, чем IPS.
- Может вносить смещение.

Источник: Swaminathan & Joachims — https://arxiv.org/abs/1502.02362

### 2.5 DM (`dm_value`)

```text
V_DM_hat = (1/n) * sum_i sum_a [ pi_B(a|x_i) * mu_hat(x_i,a) ]
```

- Низкая дисперсия.
- Чувствителен к ошибкам модели `mu_hat`.

Источник: https://arxiv.org/abs/1103.4601

### 2.6 DR (`dr_value`)

```text
V_DR_hat = (1/n) * sum_i [
  sum_a pi_B(a|x_i) * mu_hat(x_i,a)
  + (pi_B(a_i|x_i)/p_A(a_i|x_i)) * (r_i - mu_hat(x_i,a_i))
]
```

- Doubly robust: консистентность при корректности хотя бы одной части (`p_A` или `mu_hat`).

Источники:
- https://arxiv.org/abs/1103.4601
- https://arxiv.org/abs/1503.02834

### 2.7 SNDR (`sndr_value`)

```text
w_bar = (1/n) * sum_i w_i
V_SNDR_hat = (1/n) * sum_i [
  sum_a pi_B(a|x_i) * mu_hat(x_i,a)
  + (w_i/w_bar) * (r_i - mu_hat(x_i,a_i))
]
```

- Стабилизирует DR при тяжёлых весах.
- Может увеличить смещение.

Основано на идеях self-normalization + DR:
- https://arxiv.org/abs/1502.02362
- https://arxiv.org/abs/1103.4601

### 2.8 Switch-DR (`switch_dr_value`)

```text
V_SwitchDR_hat = (1/n) * sum_i [
  sum_a pi_B(a|x_i) * mu_hat(x_i,a)
  + 1[w_i <= tau] * w_i * (r_i - mu_hat(x_i,a_i))
]
```

- Контролирует variance через отключение опасно больших весов.

Источник: https://arxiv.org/abs/1612.01205

---

## 3) Почему `Bootstrap CI` — не отдельный OPE-оценщик

`Bootstrap CI` не задаёт новую формулу `V_hat`.
Он берёт **любой выбранный `V_hat`** (например DR/IPS/SNIPS) и оценивает его неопределённость:

```text
Результат inference: [L, U] для неизвестного V(pi_B)
```

Поэтому корректно мыслить так:
- «Estimator» отвечает на вопрос: *какая точечная оценка?*
- «CI method» отвечает на вопрос: *насколько эта оценка статистически неопределённа?*

---

## 4) Как считать CI для остальных OPE-оценщиков

Ниже — основные подходы, которые используются в литературе.

### 4.1 Nonparametric bootstrap (универсальный практический baseline)

Подходит почти для любого point-estimator (Replay/IPS/SNIPS/DM/DR/SNDR/Switch-DR):

1. `b=1..B`: ресемплируем строки (или кластеры `user_id`) с возвращением.
2. Считаем `V_hat^(b)` тем же оценщиком.
3. CI берём по квантилям `alpha/2` и `1-alpha/2`.

Плюсы:
- просто,
- единая процедура для разных оценщиков.

Минусы:
- чувствителен к тяжёлым хвостам,
- в адаптивных данных (bandit logs) может давать некорректное покрытие без доп. поправок.

Классика bootstrap: Efron & Tibshirani (книга).

### 4.2 Асимптотические Wald/IF-интервалы (для IPS/DR и их вариантов)

Идея: представить оценщик как среднее influence-like термов `psi_i`, затем:

```text
SE_hat = sqrt( Var_hat(psi_i) / n )
CI = V_hat +/- z_(1-alpha/2) * SE_hat
```

Плюсы:
- дешево по вычислениям,
- удобно для онлайн-отчётности.

Минусы:
- требуют условий асимптотики,
- плохи при малом `n` и тяжелых весах.

Связанные источники:
- DR/OPE базис: https://arxiv.org/abs/1103.4601
- Пост-bandit inference и корректность интервалов в адаптивных дизайнах:
  https://proceedings.neurips.cc/paper/2021/file/eff3058117fd4cf4d4c3af12e273a40f-Paper.pdf

### 4.3 High-confidence bounds (концентрационные, «гарантийные»)

Вместо симметричного CI строят нижнюю/верхнюю границу с гарантиями вероятности.
Часто используют эмпирические Bernstein/концентрационные подходы и клиппинг.

Подходит для risk-averse запуска, когда важнее «не переоценить» качество.

Источник: High Confidence OPE (Thomas et al., AAAI 2015):
- PDF: https://people.cs.umass.edu/~pthomas/papers/Thomas2015.pdf

### 4.4 Cluster bootstrap (когда есть зависимость внутри пользователя)

Если на одного `user_id` приходится много наблюдений, стандартный row-bootstrap занижает неопределённость.
Нужно ресемплировать кластеры целиком.

Источник: Cameron, Gelbach, Miller (cluster bootstrap):
- https://direct.mit.edu/rest/article/90/3/414/57731/Bootstrap-Based-Improvements-for-Inference-with

---

## 5) Практическая рекомендация для `Policyscope`

1. Для всех OPE-оценщиков начните с **cluster bootstrap CI** (если есть повторные наблюдения на пользователя).
2. Для отчёта держите и point-estimate, и CI.
3. Если веса тяжёлые (низкий ESS), добавляйте диагностику стабильности:
   - IPS vs SNIPS,
   - DR vs Switch-DR,
   - sensitivity по `weight_clip` / `tau`.
4. Для критичных решений используйте high-confidence lower bound как дополнительный guardrail.

---

## 6) Мини-ответ на ваш ключевой вопрос

**Нет, `Bootstrap CI для DR` не является отдельным OPE-эстиматором наравне с IPS/DR.**
Это слой статистического вывода поверх выбранного point-estimator.

То, что в коде есть отдельная функция `dr_with_bootstrap_ci`, действительно может выглядеть как «ещё один метод», но математически это скорее «pipeline для инференса», а не новый `V_hat`.


---

## 7) Что используется в литературе и на практике (конкретно для OPE)

Ниже — практичный список CI-подходов, которые чаще всего встречаются в OPE-работах и прикладных пайплайнах:

1. **Percentile bootstrap / cluster bootstrap**  
   Часто берут как дефолт в прикладных системах: просто, универсально, подходит для большинства оценщиков.

2. **BCa / bootstrap-t (studentized) bootstrap**  
   Используют, когда хотят более точное покрытие, чем простой percentile bootstrap, особенно при асимметрии распределения оценки.

3. **Асимптотические Wald CI через influence-function (IF)**  
   Стандартны для DR/AIPW-подобных оценщиков в больших выборках; быстры и удобны в продакшн-отчётах.

4. **High-confidence lower/upper bounds (концентрационные гарантии)**  
   Применяются для более консервативных решений «катить/не катить» (особенно когда важно не переоценить эффект).

5. **Инференс для адаптивных/бандитных логов**  
   Отдельная тема: стандартные CI могут ломаться из-за адаптивности; нужны специальные поправки/оценки.

Ключевые источники:
- Dudík, Langford, Li (DR и базис OPE): https://arxiv.org/abs/1103.4601
- Swaminathan & Joachims (SNIPS/CRM): https://arxiv.org/abs/1502.02362
- Thomas et al., High Confidence OPE: https://people.cs.umass.edu/~pthomas/papers/Thomas2015.pdf
- Hadad et al., Post-Contextual-Bandit Inference (NeurIPS 2021): https://proceedings.neurips.cc/paper/2021/file/eff3058117fd4cf4d4c3af12e273a40f-Paper.pdf
- Efron & Tibshirani (bootstrap практика): https://www.routledge.com/An-Introductionto-the-Bootstrap/Efron-Tibshirani/p/book/9780412042317

---

## 8) Сравнение с тем, что реализовано у нас сейчас

Сейчас в репозитории реализовано:

- `estimate_value_with_ci` — единый CI-слой для встроенных OPE-оценщиков (`ips/snips/dm/dr/sndr/switch_dr/...`).  
- `estimator_with_bootstrap_ci` — низкоуровневая универсальная обёртка «любой estimator_fn + bootstrap CI».  
- `cluster_bootstrap_ci` — **percentile bootstrap CI** (по строкам или кластерам).  
- `paired_bootstrap_ci` — парный percentile bootstrap для `(V_A, V_B, Delta)`.  
- `dr_with_bootstrap_ci` — специализированный convenience-wrapper для DR + парный bootstrap CI (backward-compatible API).

Что **не реализовано** (но встречается в литературе/практике):

- BCa / bootstrap-t интервалы,
- аналитические Wald/IF CI для IPS/SNIPS/DR/Switch-DR,
- high-confidence concentration bounds,
- специализированные CI для сильно адаптивных логов (bandit-aware inference).

Практический вывод: текущая реализация покрывает рабочий baseline (percentile cluster bootstrap), но до «полного набора индустриальных/академических CI-подходов» ещё есть пространство для расширения.


Рекомендация по API: **не удалять `dr_with_bootstrap_ci` прямо сейчас**.
Лучше оставить его как совместимый shortcut для DR, а единый слой развивать через `estimate_value_with_ci`.
