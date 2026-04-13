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

8. **Scalar target metric (core abstraction)**  
   Базовая единица оценки — одна скалярная метрика награды. Несколько метрик поддерживаются как повторные запуски оценки для разных target-колонок, а не как native vector-valued reward.

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
