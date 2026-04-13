# Архитектура Policyscope (contextual bandit OPE)

## 1) Назначение

`Policyscope` — библиотека для **off-policy evaluation** в постановке contextual bandit:

- есть логи поведения политики `A` (behavior/logging policy),
- нужно оценить значение политики `B` (target/candidate policy),
- целевые артефакты: `policy value`, `delta = value(B) - value(A)`, CI, significance metadata (`p-value` или CI-based decision) и диагностика.

## 2) Доменная модель (phase 1)

### 2.1 Core entities

1. **BanditSchema**  
   Декларирует контракт колонок: действие, награда, признаки, опционально propensity и cluster id.

2. **LoggedBanditDataset**  
   Лёгкая обёртка над `DataFrame` + `BanditSchema` с базовой валидацией и удобными accessors.

3. **Nuisance models**  
   Модели `pi_hat` и `mu_hat`, используемые point-estimators (IPS/SNIPS/DR/...).
   Внутренний reusable bundle-слой собран в `policyscope.nuisance`, чтобы orchestration/inference/diagnostics не дублировали сборку nuisance-объектов.

4. **Point estimators**  
   Оценщики значения политики (`replay`, `ips`, `snips`, `dm`, `dr`, `sndr`, `switch_dr`).

5. **Inference**  
   Слой построения CI и significance metadata (percentile bootstrap CI + centered paired bootstrap test), независимо от point-estimator.

6. **Diagnostics**  
   ESS, clipping/share, устойчивость и sanity-check метрики.
   Диагностика обязательна рядом с инференсом, но не заменяет CI/p-value.
   Базовые trust/stability diagnostics включают: `ESS`, `ESS/N`, replay overlap, max/p95/p99 importance weights, clip/switch share и простые warning rules.

7. **Comparison result**  
   Сводка по `V_A`, `V_B`, `delta`, CI, `p-value` и диагностике для сравнения A vs B.
   Официальный orchestration path: `policyscope.comparison.compare_policies(...)`.

8. **Scalar target metric (core abstraction)**  
   Базовая единица оценки — одна скалярная метрика награды. Несколько метрик поддерживаются как повторные запуски оценки для разных target-колонок, а не как native vector-valued reward.
   Для multi-target используется `compare_policies_multi_target(...)` как mapping из target -> single-target summary.

## 3) Границы слоёв

```text
Data contract -> Point estimation -> Inference -> Diagnostics/Reporting
```

- **Data contract** не знает про алгоритмы оценивания.
- **Point estimation** не должен дублировать логику инференса.
- **Inference** принимает estimator как функцию/метод и не меняет математику estimator.
- **Reporting** собирает результаты в человекочитаемую форму.

## 4) Статус текущей миграции

В этой фазе добавлен новый data contract слой (`BanditSchema`, `LoggedBanditDataset`) без массового переписывания estimator-кода.

Это означает:

- текущие estimator API остаются рабочими;
- новый слой готов для поэтапного внедрения в точках входа;
- математическая реализация существующих estimators не меняется.

## 5) Migration strategy (без API-ломки)

1. Добавить data contract и покрыть тестами.
2. Использовать data contract на внешних границах (tutorial/examples/high-level API).
3. Постепенно адаптировать внутренние вызовы estimator/inference к новому объекту.
4. Удалять дублирующие проверки только после стабилизации миграции.

## 6) Non-goals этой фазы

- Полное переписывание всех estimators на новый объект данных.
- Изменение математических формул/предположений существующих OPE методов.
- Большая переработка публичного API за один релиз.


## 7) Cross-fitting readiness (incremental)

Почему это важно:
- при использовании одних и тех же данных для обучения nuisance-моделей (`pi_hat`, `mu_hat`) и финальной оценки возможен дополнительный finite-sample bias;
- out-of-fold предсказания уменьшают этот эффект и подготавливают почву для более строгой cross-fitting схемы.

Что добавлено сейчас:
- явные структуры предсказаний: `BehaviorPredictions`, `OutcomePredictions`;
- контейнер `CrossFitNuisanceBundle` для fold-aware nuisance артефактов;
- утилиты первого уровня: `make_kfold_indices`, `generate_oof_behavior_predictions`, `generate_oof_outcome_predictions`;
- cross-fitting mode применим для `dm`, `dr`, `sndr`, `switch_dr` (outcome + behavior nuisance) и для `ips`, `snips` (behavior nuisance).
- основной orchestration path (`compare_policies`) поддерживает `use_crossfit=True` и/или внешний `nuisance_bundle` (additive path, без обязательного использования).

Что пока не делаем (future work):
- полноценный cross-fitting rollout для каждого estimator и bootstrap-веток;
- жёсткий API-контракт для всех вариантов multi-fold обучения;
- изменение математики уже реализованных OPE-оценщиков.

Краткая migration note:
- текущий дефолтный workflow не меняется (внутренний fit nuisance работает как раньше);
- при необходимости можно заранее посчитать OOF nuisance-предсказания и передать их в `compare_policies(..., nuisance_bundle=...)`;
- на текущем этапе это интегрировано инкрементально и совместимо с существующим API.


## 8) Logged vs estimated propensity (first-class modes)

Для weighted-estimators (`ips`, `snips`, `dr`, `sndr`, `switch_dr`) источник propensity теперь является явной частью orchestration:
- `propensity_source="auto"` — по умолчанию: сначала пытаемся взять logged propensity column, затем fallback к estimated `pi_hat`;
- `propensity_source="logged"` — строгий режим только по колонке логов;
- `propensity_source="estimated"` — всегда через behavior model.

Это важно методологически: logged и estimated propensity отражают разные допущения о качестве данных и модели поведения, поэтому provenance отражается в structured outputs и diagnostics.

Migration note: текущий API совместим назад (по умолчанию `auto`), а дополнительная metadata (`propensity_source`, `propensity_column`, fallback notes) доступна без изменения формул estimators.


## 9) Simulation validation harness

Добавлен lightweight слой `policyscope.validation` для повторяемой проверки поведения estimators на synthetic-среде с oracle-значениями.

Harness поддерживает сравнение методов (`replay`, `ips`, `snips`, `dm`, `dr`, `sndr`, `switch_dr`) в разных режимах (logged/estimated propensity, cross-fit mode, clipping settings) и возвращает структурированные run-level и aggregate метрики (bias/std/RMSE, coverage, significance rate, diagnostic summaries).

Назначение — внутренний validation/regression инструмент для развития библиотеки. Это не универсальная гарантия корректности на произвольных real-world логах.

Подробнее: `docs/validation_harness.md`.


## 10) Nuisance-model quality diagnostics

Дополнительно к overlap/weight diagnostics введён отдельный слой `nuisance diagnostics` для качества моделей `pi_hat` и `mu_hat`.

Зачем: хорошие ESS/overlap сами по себе не гарантируют, что nuisance-модели адекватны. Поэтому в structured outputs добавляются behavior/outcome quality метрики и warning rules.

Ключевые принципы:
- logged propensity path: behavior model quality помечается как not applicable;
- estimated propensity path: behavior quality считается и добавляется в summary;
- cross-fit mode: diagnostics отмечаются как OOF (fold-aware provenance).

Этот слой не меняет формулы estimators и служит для trust-quality интерпретации результатов.
