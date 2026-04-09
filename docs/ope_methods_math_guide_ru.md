# Методы офлайн-оценки политик в `Policyscope`: математическая интуиция и путь к глубокому пониманию

Этот материал объясняет **все реализованные в репозитории методы OPE** (Off-Policy Evaluation):

- On-policy baseline (`value_on_policy`)
- Replay (`replay_value`)
- IPS (`ips_value`)
- SNIPS (`snips_value`)
- DM / Direct Method (`dm_value`)
- DR / Doubly Robust (`dr_value`)
- SNDR / Self-Normalized DR (`sndr_value`)
- Switch-DR (`switch_dr_value`)
- Bootstrap CI для DR (`dr_with_bootstrap_ci`)

Цель: дать новичку «рабочую» интуицию, а затем указать, куда читать для строгого обоснования.

---

## 1) Формализация задачи (единый язык для всех методов)

Пусть в логах политики **A** есть наблюдения:
\[
(x_i, a_i, r_i), \quad i=1,\dots,n,
\]
где:
- \(x_i\) — контекст (признаки пользователя/сессии),
- \(a_i\) — действие, выбранное логирующей политикой \(\pi_A\),
- \(r_i\) — наблюдаемая награда (например, `accept` или `cltv`).

Мы хотим оценить качество новой политики \(\pi_B\):
\[
V(\pi_B) = \mathbb{E}_{x\sim D}\,\mathbb{E}_{a\sim \pi_B(\cdot|x)}[r(x,a)].
\]

Проблема: в логах наблюдаем награду только для \(a_i\), а не для всех действий.

Ключевые обозначения:
- \(\mu(x,a)=\mathbb{E}[r\mid x,a]\) — модель ожидаемой награды,
- \(p_A(a\mid x)=\pi_A(a\mid x)\) — propensity логирующей политики,
- \(w_i = \pi_B(a_i\mid x_i)/p_A(a_i\mid x_i)\) — важностный вес.

---

## 2) On-policy baseline (`value_on_policy`)

**Идея:** средняя награда в логах A:
\[
\hat V_A = \frac{1}{n}\sum_i r_i.
\]

Это не OPE для B, но важная база сравнения (дельта \(V_B-V_A\)).

---

## 3) Replay (`replay_value`)

**Идея:** оставить только строки, где B выбрала бы то же действие, что и A:
\[
\hat V_{\text{Replay}} = \frac{1}{|\mathcal I|}\sum_{i\in\mathcal I} r_i,
\quad
\mathcal I=\{i: a_i = a^B_i\}.
\]

### Интуиция
- Это «честный» способ не додумывать контрфакты.
- Но при малом пересечении действий дисперсия взлетает, а оценка может стать неустойчивой/неопределённой.

### База в литературе
- Li et al. (2010/2011), unbiased offline evaluation / replay-подход в contextual bandits.
  - arXiv: https://arxiv.org/abs/1003.5956

---

## 4) IPS (`ips_value`)

\[
\hat V_{\text{IPS}} = \frac{1}{n}\sum_i w_i r_i,
\quad
w_i=\frac{\pi_B(a_i\mid x_i)}{p_A(a_i\mid x_i)}.
\]

### Почему это работает
При корректных propensity и overlap (\(p_A>0\) там, где \(\pi_B>0\)) IPS несмещён: лог A можно «перевзвесить», чтобы имитировать B.

### Плюсы / минусы
- + Теоретически прозрачный и несмещённый.
- − Очень чувствителен к большим весам (редкие действия \(\Rightarrow\) большая дисперсия).

### База в литературе
- Dudík, Langford, Li (DR paper, разделы с IPS): https://arxiv.org/abs/1103.4601
- Практика OPE в bandits: https://arxiv.org/abs/1003.5956

---

## 5) SNIPS (`snips_value`)

\[
\hat V_{\text{SNIPS}} = \frac{\sum_i w_i r_i}{\sum_i w_i}.
\]

### Интуиция
Нормировка на \(\sum_i w_i\) стабилизирует масштаб весов:
- обычно уменьшает дисперсию,
- но вводит смещение (особенно в малой выборке).

### Когда полезен
Когда IPS «шумит» из-за тяжёлого хвоста весов.

### База в литературе
- Swaminathan & Joachims, Counterfactual Risk Minimization (self-normalized weighting):
  - arXiv: https://arxiv.org/abs/1502.02362
  - JMLR версия: https://jmlr.org/papers/v16/swaminathan15a.html

---

## 6) DM / Direct Method (`dm_value`)

Сначала оцениваем \(\mu(x,a)\), затем:
\[
\hat V_{\text{DM}} = \frac{1}{n}\sum_i\sum_a \pi_B(a\mid x_i)\,\hat\mu(x_i,a).
\]

### Интуиция
DM полностью опирается на модель награды:
- если \(\hat\mu\) хороша — низкая дисперсия,
- если \(\hat\mu\) смещена — оценка смещена.

### База в литературе
- Dudík et al. (сравнение DM/IPS/DR): https://arxiv.org/abs/1103.4601

---

## 7) DR / Doubly Robust (`dr_value`)

\[
\hat V_{\text{DR}} =
\frac{1}{n}\sum_i
\left[
\sum_a \pi_B(a\mid x_i)\hat\mu(x_i,a)
+
\frac{\pi_B(a_i\mid x_i)}{p_A(a_i\mid x_i)}\bigl(r_i-\hat\mu(x_i,a_i)\bigr)
\right].
\]

### Главная идея
DR = DM + IPS-коррекция ошибки модели на наблюдаемом действии.

### Почему «doubly robust»
Оценка консистентна, если корректна **хотя бы одна** часть:
1. либо propensity \(p_A\),
2. либо outcome-модель \(\mu\).

### База в литературе
- Классическая работа: https://arxiv.org/abs/1103.4601
- Развитие и анализ: https://arxiv.org/abs/1503.02834

---

## 8) SNDR / Self-Normalized DR (`sndr_value`)

В `Policyscope` нормируется именно коррекционный член DR:
\[
\hat V_{\text{SNDR}} =
\frac{1}{n}\sum_i
\left[
\sum_a \pi_B(a\mid x_i)\hat\mu(x_i,a)
+
\frac{w_i}{\bar w}\bigl(r_i-\hat\mu(x_i,a_i)\bigr)
\right],
\quad \bar w=\frac{1}{n}\sum_i w_i.
\]

### Интуиция
- DR может быть нестабилен при тяжёлых весах.
- Нормировка поправки снижает вариативность ценой дополнительного смещения.

### Что читать
SNDR как инженерная стабилизация опирается на идеи self-normalization и DR:
- Self-normalization в bandit-оценивании: https://arxiv.org/abs/1502.02362
- DR фундамент: https://arxiv.org/abs/1103.4601

---

## 9) Switch-DR (`switch_dr_value`)

Вводится порог \(\tau\): если \(w_i>\tau\), IPS-поправку отключаем.

\[
\hat V_{\text{Switch-DR}} =
\frac{1}{n}\sum_i
\left[
\sum_a \pi_B(a\mid x_i)\hat\mu(x_i,a)
+
\mathbb{1}(w_i\le \tau)\,w_i\bigl(r_i-\hat\mu(x_i,a_i)\bigr)
\right].
\]

### Интуиция
- Большие веса дают огромную дисперсию.
- Switch-DR «обрезает» опасные точки, оставляя там только DM.
- Это управляемый bias–variance trade-off через \(\tau\).

### База в литературе
- Wang et al., *Optimal and Adaptive Off-policy Evaluation in Contextual Bandits* (SWITCH):
  - PMLR: https://proceedings.mlr.press/v70/wang17a.html
  - arXiv: https://arxiv.org/abs/1612.01205

---

## 10) Bootstrap CI для DR (`dr_with_bootstrap_ci`)

`Policyscope` даёт доверительные интервалы через бутстрэп по строкам или кластерам (`cluster_col`).

### Интуиция
- Вместо аналитической дисперсии эмпирически приближаем распределение оценок,
  многократно ресемплируя данные.
- Кластерный бутстрэп нужен, если наблюдения внутри `user_id` зависимы.

### База в литературе
- Efron & Tibshirani (классика bootstrap):
  - издатель: https://www.routledge.com/An-Introductionto-the-Bootstrap/Efron-Tibshirani/p/book/9780412042317
- Cameron, Gelbach, Miller (bootstrap для clustered inference):
  - https://direct.mit.edu/rest/article/90/3/414/57731/Bootstrap-Based-Improvements-for-Inference-with

---

## 11) Как читать результаты на практике

Короткая памятка:
1. **Проверяйте overlap**: нет ли областей, где \(p_A(a|x)\) очень мал, а \(\pi_B(a|x)\) велик.
2. Смотрите **ESS**: низкий ESS = фактически мало «информативных» наблюдений.
3. Сравнивайте **IPS vs SNIPS vs DR vs Switch-DR**:
   - если IPS сильно шумит, а DR/Switch-DR стабильны — это нормально.
4. Всегда добавляйте **CI** (bootstrap), а не смотрите на одну точку.
5. Для решения «катить/не катить» оценивайте и \(V_B\), и \(\Delta=V_B-V_A\).

---

## 12) Что важно помнить про допущения

Любой OPE-метод требует минимум:
- **SUTVA / корректная постановка награды**: награда зависит от своего контекста и действия,
- **Positivity / overlap**: если B может выбрать действие, A иногда тоже должна была его выбирать,
- **Корректность логов**: action и propensity соответствуют одному и тому же процессу логирования.

Нарушение этих пунктов чаще всего важнее выбора между IPS/DR/Switch-DR.

---

## 13) Карта «какой метод брать первым»

- Хотите самый простой и прозрачный baseline → **IPS**.
- Нужна практическая устойчивость на реальных логах → **DR**.
- Весы «тяжёлые», DR шумит → **SNDR** или **Switch-DR**.
- Есть хорошая модель \(\mu\), но сомнения в propensity → сравнивайте **DM + DR**.
- Для отчёта и принятия решения обязательно → **DR/Switch-DR + bootstrap CI**.

