# Архитектура Policyscope (contextual bandit OPE)

## 1) Назначение

`Policyscope` — библиотека для **off-policy evaluation** в постановке contextual bandit:

- есть логи поведения политики `A` (behavior/logging policy),
- оценивается политика `B` (target/candidate policy),
- основные выходы: `policy value`, `delta = value(B) - value(A)`, CI, significance metadata и diagnostics.

## 2) Доменная модель

1. **BanditSchema**  
   Контракт колонок: action, reward, features, опционально propensity и cluster id.

2. **LoggedBanditDataset**  
   Обёртка над `DataFrame` + `BanditSchema` с валидацией и accessors.

3. **Nuisance models**  
   `pi_hat` и `mu_hat`, используемые в weighted/model-based OPE-оценках.

4. **Point estimators**  
   `replay`, `ips`, `snips`, `dm`, `dr`, `sndr`, `switch_dr`.

5. **Inference**  
   CI + significance metadata (включая centered paired bootstrap для проверки `delta=0`).

6. **Diagnostics**  
   Replay overlap, ESS/ESS ratio, weight tails, clip/switch share и warning flags.

7. **Comparison summary**  
   Официальный orchestration path: `compare_policies(...)`.
   Summary включает `V_A`, `V_B`, `Delta`, CI/p-value, diagnostics, `trust_level`, notes/warnings.

8. **Scalar target abstraction**  
   Базовая единица — одна скалярная reward-метрика. Multi-target режим реализуется как повторные scalar-оценки через `compare_policies_multi_target(...)`.

## 3) Границы слоёв

```text
Data contract -> Point estimation -> Inference -> Diagnostics/Reporting
```

- **Data contract**: схема и валидация входов.
- **Point estimation**: оценка `V_B` выбранным estimator.
- **Inference**: uncertainty/significance поверх point-estimator.
- **Diagnostics/Reporting**: устойчивость и интерпретация результата.

## 4) Официальный high-level orchestration path

`compare_policies(...)` — основной entrypoint для сравнения A vs B на пользовательских логах.

Практический default-путь:
- estimator: `dr`,
- propensity mode: `propensity_source="auto"`,
- вместе читать `Delta`, CI/p-value, diagnostics и trust metadata.

## 5) Propensity source modes (logged vs estimated propensity)

Для weighted methods (`ips`, `snips`, `dr`, `sndr`, `switch_dr`) поддерживаются first-class режимы:

- `propensity_source="auto"`: использовать logged propensity при валидной колонке, иначе fallback на estimated behavior model;
- `propensity_source="logged"`: строго использовать logged propensity column;
- `propensity_source="estimated"`: всегда использовать оценённую behavior model.

Эти режимы отражаются в structured output (`propensity_source`, `propensity_column`, warnings/notes).

## 6) Nuisance-model quality diagnostics

Помимо overlap/weight diagnostics, summary содержит **nuisance-model quality diagnostics**:

- behavior-side качество для estimated propensity path;
- outcome-side качество для DM/DR-family;
- OOF/provenance markers для cross-fit path.

Диагностика — это support-сигналы для интерпретации, а не доказательство корректности результата.

## 7) Cross-fitting support и ограничения

Поддерживаются:
- `BehaviorPredictions`, `OutcomePredictions`, `CrossFitNuisanceBundle`;
- OOF utilities для behavior/outcome nuisance;
- `compare_policies(..., use_crossfit=True)` и передача внешнего `nuisance_bundle`.

Текущее ограничение:
- внешний nuisance не переиспользуется в bootstrap-resamples при несовпадении row identity/order (безопасный fallback на внутренний fit).

## 8) Validation harness: роль и ограничения

`policyscope.validation.run_simulation_validation(...)` — инструмент synthetic-regression проверки estimator behavior против oracle.

Что даёт:
- run-level и aggregate метрики (bias/error/coverage/significance/diagnostics).

Что не даёт:
- не является универсальной гарантией корректности для произвольных real-world логов.

## 9) Recommended defaults и trust metadata

High-level summary содержит guidance metadata:
- preferred estimator: `dr`,
- preferred propensity mode with logged column: `auto`,
- fallback: `estimated`,
- trust metadata: `diagnostic_warnings`, `inference_warnings`, `trust_notes`, `trust_level`.

Важно: trust metadata — эвристическая поддержка принятия решений, а не замена A/B-тестам.

## 10) Non-goals (scope boundaries)

`Policyscope` не пытается:
- автоматически заменять продуктовые A/B-эксперименты одним OPE-числом;
- выдавать «гарантии безопасности» только по CI/diagnostics;
- менять математическую сущность существующих estimators без явного запроса.
