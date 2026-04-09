# Методы офлайн-оценки политик в `Policyscope`: интуиция, формулы и ссылки на статьи

> Важно про отображение: в этом документе формулы даны в **plain-text формате** (без LaTeX-разметки `$$...$$`), чтобы текст одинаково читался на GitHub и в локальных Markdown-просмотрщиках без MathJax/KaTeX.

Этот гайд покрывает все методы, реализованные в репозитории:

- On-policy baseline (`value_on_policy`)
- Replay (`replay_value`)
- IPS (`ips_value`)
- SNIPS (`snips_value`)
- DM / Direct Method (`dm_value`)
- DR / Doubly Robust (`dr_value`)
- SNDR / Self-Normalized DR (`sndr_value`)
- Switch-DR (`switch_dr_value`)
- Bootstrap CI для DR (`dr_with_bootstrap_ci`)

---

## 1) Единая постановка задачи

Есть логи политики A: `(x_i, a_i, r_i)`, где:

- `x_i` — контекст,
- `a_i` — действие, выбранное логирующей политикой `pi_A`,
- `r_i` — наблюдаемая награда.

Нужно оценить ожидаемую ценность новой политики `pi_B`:

```text
V(pi_B) = E_x [ E_{a~pi_B(.|x)} [ r(x, a) ] ]
```

Ключевые обозначения:

- `mu(x, a) = E[r | x, a]` — модель ожидаемой награды,
- `p_A(a|x) = pi_A(a|x)` — propensity логирующей политики,
- `w_i = pi_B(a_i|x_i) / p_A(a_i|x_i)` — importance weight.

---

## 2) On-policy baseline (`value_on_policy`)

Средняя награда на логах A:

```text
V_A_hat = (1/n) * sum_i r_i
```

Это baseline для сравнения с `V_B`.

---

## 3) Replay (`replay_value`)

Берём только строки, где действия A и B совпали:

```text
I = { i : a_i == a_i^B }
V_replay_hat = (1 / |I|) * sum_{i in I} r_i
```

Интуиция:
- максимально «честно» (без моделирования контрфактов),
- но может иметь высокую дисперсию при малом числе совпадений.

Литература:
- Li et al. (unbiased offline evaluation / replay): https://arxiv.org/abs/1003.5956

---

## 4) IPS (`ips_value`)

```text
V_IPS_hat = (1/n) * sum_i (w_i * r_i)
where w_i = pi_B(a_i|x_i) / p_A(a_i|x_i)
```

Интуиция:
- корректно перевзвешивает логи A «под» политику B,
- несмещён при корректных propensity и overlap,
- чувствителен к большим весам.

Литература:
- Dudík, Langford, Li: https://arxiv.org/abs/1103.4601
- Практический контекст bandits: https://arxiv.org/abs/1003.5956

---

## 5) SNIPS (`snips_value`)

```text
V_SNIPS_hat = sum_i (w_i * r_i) / sum_i w_i
```

Интуиция:
- нормировка стабилизирует оценки,
- обычно снижает дисперсию,
- может добавлять смещение.

Литература:
- Swaminathan & Joachims: https://arxiv.org/abs/1502.02362
- JMLR version: https://jmlr.org/papers/v16/swaminathan15a.html

---

## 6) DM / Direct Method (`dm_value`)

Сначала строится модель `mu_hat(x,a)`, затем:

```text
V_DM_hat = (1/n) * sum_i sum_a [ pi_B(a|x_i) * mu_hat(x_i, a) ]
```

Интуиция:
- низкая дисперсия,
- зависит от качества модели награды.

Литература:
- Dudík et al. (DM/IPS/DR): https://arxiv.org/abs/1103.4601

---

## 7) DR / Doubly Robust (`dr_value`)

```text
V_DR_hat = (1/n) * sum_i [
    sum_a pi_B(a|x_i) * mu_hat(x_i, a)
    + (pi_B(a_i|x_i)/p_A(a_i|x_i)) * (r_i - mu_hat(x_i, a_i))
]
```

Интуиция:
- сочетает DM + IPS-коррекцию,
- консистентен, если корректна хотя бы одна часть: propensity или `mu_hat`.

Литература:
- Классика DR: https://arxiv.org/abs/1103.4601
- Доп. анализ DR: https://arxiv.org/abs/1503.02834

---

## 8) SNDR / Self-Normalized DR (`sndr_value`)

В реализации `Policyscope` нормируется коррекционный член DR:

```text
w_bar = (1/n) * sum_i w_i
V_SNDR_hat = (1/n) * sum_i [
    sum_a pi_B(a|x_i) * mu_hat(x_i, a)
    + (w_i / w_bar) * (r_i - mu_hat(x_i, a_i))
]
```

Интуиция:
- меньше вариативность при «тяжёлых» весах,
- ценой некоторого смещения.

Литература:
- Self-normalization: https://arxiv.org/abs/1502.02362
- DR фундамент: https://arxiv.org/abs/1103.4601

---

## 9) Switch-DR (`switch_dr_value`)

Для весов выше порога `tau` IPS-поправка выключается:

```text
V_SwitchDR_hat = (1/n) * sum_i [
    sum_a pi_B(a|x_i) * mu_hat(x_i, a)
    + 1[w_i <= tau] * w_i * (r_i - mu_hat(x_i, a_i))
]
```

Интуиция:
- снижает дисперсию за счёт контролируемого bias-variance trade-off.

Литература:
- SWITCH estimator: https://proceedings.mlr.press/v70/wang17a.html
- arXiv: https://arxiv.org/abs/1612.01205

---

## 10) Bootstrap CI для DR (`dr_with_bootstrap_ci`)

Интервалы строятся бутстрэпом (по строкам или по кластерам `user_id`).

Интуиция:
- эмпирически оцениваем распределение оценки,
- учитываем зависимость внутри кластера при cluster bootstrap.

Литература:
- Efron & Tibshirani: https://www.routledge.com/An-Introductionto-the-Bootstrap/Efron-Tibshirani/p/book/9780412042317
- Cluster bootstrap: https://direct.mit.edu/rest/article/90/3/414/57731/Bootstrap-Based-Improvements-for-Inference-with

---

## 11) Практический чек-лист

1. Проверяйте overlap: нет ли зон, где `p_A(a|x)` очень мал, а `pi_B(a|x)` велик.
2. Контролируйте ESS: низкий ESS = мало эффективных наблюдений.
3. Сравнивайте IPS/SNIPS/DR/Switch-DR, а не одну точечную оценку.
4. Всегда смотрите доверительные интервалы.
5. Решение о выкладке делайте по `V_B` и `Delta = V_B - V_A`.

---

## 12) Важные допущения

- Positivity (overlap): действия B должны иметь поддержку в логах A.
- Корректное логирование действий и propensity.
- Стабильность определения награды.

Нарушение допущений обычно критичнее, чем выбор конкретного OPE-оценщика.

---

## 13) Что использовать первым

- Нужен прозрачный baseline: **IPS**.
- Нужна устойчивая практика: **DR**.
- Тяжёлые веса/нестабильность: **SNDR** или **Switch-DR**.
- Обязательный слой для принятия решения: **bootstrap CI**.
