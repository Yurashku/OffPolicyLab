"""
Пример запуска синтетического эксперимента с off‑policy оценкой.

Скрипт генерирует пользователей, симулирует логи политики A,
расчитывает оффлайн‑оценки новой политики B (Replay, IPS, SNIPS, DM, DR)
по метрикам «Отклик» и «CLTV», строит бутстрэп‑интервалы для DR и
сравнивает с истинными ожиданиями (оракулом).

Пример использования:

```
python examples/run_synthetic_experiment.py \
  --n_users 50000 \
  --seed 42 \
  --policyA epsilon_greedy --epsilon 0.15 \
  --policyB softmax --tau 0.7 \
  --horizon 90 \
  --weight_clip 20
```
"""

from __future__ import annotations

import argparse
import os
import json
import numpy as np
import pandas as pd
from typing import Tuple

from policyscope.synthetic import SynthConfig, SyntheticRecommenderEnv
from policyscope.policies import make_policy
from policyscope.estimators import (
    value_on_policy,
    replay_value,
    prepare_piB_taken,
    ips_value,
    snips_value,
    dm_value,
    dr_value,
    train_mu_hat,
    ate_from_values,
)
from policyscope.bootstrap import paired_bootstrap_ci
from policyscope.report import decision_summary, dump_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_users", type=int, default=50_000, help="Количество пользователей в синтетике")
    parser.add_argument("--seed", type=int, default=42, help="Зерно генератора")
    parser.add_argument("--horizon", type=int, default=90, help="Горизонт для CLTV")
    parser.add_argument("--policyA", type=str, default="epsilon_greedy", choices=["greedy", "epsilon_greedy", "softmax"], help="Тип политики A")
    parser.add_argument("--epsilon", type=float, default=0.15, help="Параметр ε для ε‑жадной политики A")
    parser.add_argument("--tau", type=float, default=0.7, help="Температура softmax политики A")
    parser.add_argument("--policyB", type=str, default="softmax", choices=["greedy", "epsilon_greedy", "softmax"], help="Тип политики B")
    parser.add_argument("--weight_clip", type=float, default=20.0, help="Порог клиппинга весов для IPS/SNIPS/DR")
    parser.add_argument("--business_threshold_accept", type=float, default=0.0, help="Бизнес‑порог для метрики отклика")
    parser.add_argument("--business_threshold_cltv", type=float, default=1.0, help="Бизнес‑порог для метрики CLTV")
    parser.add_argument("--oracle_users", type=int, default=200_000, help="Число пользователей для оракула")
    parser.add_argument("--oracle_mc", type=int, default=1, help="Число MC‑прогонов для оракула")
    parser.add_argument("--artifacts_dir", type=str, default="artifacts", help="Каталог для сохранения артефактов")
    args = parser.parse_args()

    os.makedirs(args.artifacts_dir, exist_ok=True)

    # Создаём синтетическую среду
    cfg = SynthConfig(n_users=args.n_users, horizon_days=args.horizon, seed=args.seed)
    env = SyntheticRecommenderEnv(cfg)
    X = env.sample_users(args.n_users)

    # Создаём политики A и B
    policyA = make_policy(args.policyA, seed=111, epsilon=args.epsilon, tau=args.tau)
    policyB = make_policy(args.policyB, seed=222, epsilon=0.1, tau=0.7)

    # Генерируем логи A
    logsA = env.simulate_logs_A(policyA, X)

    # On‑policy значения для A
    vA_accept = value_on_policy(logsA, target="accept")
    vA_cltv = value_on_policy(logsA, target="cltv")

    # Расчёт вероятностей выбора текущих действий под B
    piB_taken = prepare_piB_taken(logsA, policyB)

    # Обучаем модели исхода для DM/DR
    mu_accept = train_mu_hat(logsA, target="accept")
    mu_cltv = train_mu_hat(logsA, target="cltv")

    # Replay (на совпадающих действиях)
    vB_replay_accept = replay_value(logsA, policyB.action_argmax(X), target="accept")
    vB_replay_cltv = replay_value(logsA, policyB.action_argmax(X), target="cltv")

    # IPS и SNIPS
    vB_ips_accept, ess_ips_accept, clip_ips_accept = ips_value(logsA, piB_taken, target="accept", weight_clip=args.weight_clip)
    vB_ips_cltv, ess_ips_cltv, clip_ips_cltv = ips_value(logsA, piB_taken, target="cltv", weight_clip=args.weight_clip)

    vB_snips_accept, ess_snips_accept, clip_snips_accept = snips_value(logsA, piB_taken, target="accept", weight_clip=args.weight_clip)
    vB_snips_cltv, ess_snips_cltv, clip_snips_cltv = snips_value(logsA, piB_taken, target="cltv", weight_clip=args.weight_clip)

    # Direct Method
    vB_dm_accept = dm_value(logsA, policyB, mu_accept, target="accept")
    vB_dm_cltv = dm_value(logsA, policyB, mu_cltv, target="cltv")

    # Doubly Robust
    vB_dr_accept, ess_dr_accept, clip_dr_accept = dr_value(
        logsA, policyB, mu_accept, target="accept", weight_clip=args.weight_clip
    )
    vB_dr_cltv, ess_dr_cltv, clip_dr_cltv = dr_value(
        logsA, policyB, mu_cltv, target="cltv", weight_clip=args.weight_clip
    )

    # Oracle (истинные ожидания)
    X_oracle = env.sample_users(args.oracle_users)
    vA_accept_true = env.oracle_value(
        policyA, X_oracle, metric="accept", n_mc=args.oracle_mc
    )
    vB_accept_true = env.oracle_value(
        policyB, X_oracle, metric="accept", n_mc=args.oracle_mc
    )
    vA_cltv_true = env.oracle_value(
        policyA, X_oracle, metric="cltv", n_mc=args.oracle_mc
    )
    vB_cltv_true = env.oracle_value(
        policyB, X_oracle, metric="cltv", n_mc=args.oracle_mc
    )

    dr_abs_error_accept = abs(vB_dr_accept - vB_accept_true)
    dr_abs_error_cltv = abs(vB_dr_cltv - vB_cltv_true)

    # Paired bootstrap for DR
    def estimator_pair_accept(df_part: pd.DataFrame) -> Tuple[float, float, float]:
        mu_acc = train_mu_hat(df_part, target="accept")
        vA = value_on_policy(df_part, target="accept")
        vB, _, _ = dr_value(df_part, policyB, mu_acc, target="accept", weight_clip=args.weight_clip)
        return vA, vB, ate_from_values(vB, vA)

    def estimator_pair_cltv(df_part: pd.DataFrame) -> Tuple[float, float, float]:
        mu = train_mu_hat(df_part, target="cltv")
        vA = value_on_policy(df_part, target="cltv")
        vB, _, _ = dr_value(df_part, policyB, mu, target="cltv", weight_clip=args.weight_clip)
        return vA, vB, ate_from_values(vB, vA)

    res_accept = paired_bootstrap_ci(logsA, estimator_pair_accept, cluster_col="user_id", n_boot=300, alpha=0.05)
    res_cltv = paired_bootstrap_ci(logsA, estimator_pair_cltv, cluster_col="user_id", n_boot=300, alpha=0.05)

    # Диагностика весов
    diagnostics = {
        "ips": {
            "ess_accept": ess_ips_accept,
            "clip_share_accept": clip_ips_accept,
            "ess_cltv": ess_ips_cltv,
            "clip_share_cltv": clip_ips_cltv,
        },
        "snips": {
            "ess_accept": ess_snips_accept,
            "clip_share_accept": clip_snips_accept,
            "ess_cltv": ess_snips_cltv,
            "clip_share_cltv": clip_snips_cltv,
        },
        "dr": {
            "ess_accept": ess_dr_accept,
            "clip_share_accept": clip_dr_accept,
            "ess_cltv": ess_dr_cltv,
            "clip_share_cltv": clip_dr_cltv,
        },
        "n_logs": int(len(logsA)),
        "policyA": args.policyA,
        "policyB": args.policyB,
        "epsilon": args.epsilon,
        "tau": args.tau,
        "weight_clip": args.weight_clip,
    }

    summary = {
        "accept": {
            "on_policy_A": vA_accept,
            "replay_B": vB_replay_accept,
            "ips_B": vB_ips_accept,
            "snips_B": vB_snips_accept,
            "dm_B": vB_dm_accept,
            "dr_B": vB_dr_accept,
            "oracle_A": vA_accept_true,
            "oracle_B": vB_accept_true,
            "dr_abs_error": dr_abs_error_accept,
            "bootstrap_DR": res_accept,
        },
        "cltv": {
            "on_policy_A": vA_cltv,
            "replay_B": vB_replay_cltv,
            "ips_B": vB_ips_cltv,
            "snips_B": vB_snips_cltv,
            "dm_B": vB_dm_cltv,
            "dr_B": vB_dr_cltv,
            "oracle_A": vA_cltv_true,
            "oracle_B": vB_cltv_true,
            "dr_abs_error": dr_abs_error_cltv,
            "bootstrap_DR": res_cltv,
        },
    }

    # Отчёт
    report_accept = decision_summary(res_accept, "Отклик", business_threshold=args.business_threshold_accept)
    report_cltv = decision_summary(res_cltv, "CLTV", business_threshold=args.business_threshold_cltv)

    # Сохраняем артефакты
    logs_path = os.path.join(args.artifacts_dir, "logs_A.csv")
    logsA.to_csv(logs_path, index=False)
    dump_json(os.path.join(args.artifacts_dir, "diagnostics.json"), diagnostics)
    dump_json(os.path.join(args.artifacts_dir, "summary.json"), summary)
    # записываем отчёт
    with open(os.path.join(args.artifacts_dir, "report.txt"), "w", encoding="utf-8") as f:
        f.write("# Итоговый отчёт (DR как основной оценщик)\n\n")
        f.write("## Метрика: Отклик\n")
        f.write(report_accept + "\n")
        f.write(f"Абсолютная ошибка DR: {dr_abs_error_accept:.6f}\n\n")
        f.write("## Метрика: CLTV\n")
        f.write(report_cltv + "\n")
        f.write(f"Абсолютная ошибка DR: {dr_abs_error_cltv:.6f}\n")

    # Выводим кратко в консоль
    print("==== DR (Отклик) ====")
    print(report_accept)
    print("\n==== DR (CLTV) ====")
    print(report_cltv)
    print("\n==== Оракул ====")
    print(f"Accept: V_A_true={vA_accept_true:.6f}, V_B_true={vB_accept_true:.6f}, Delta_true={vB_accept_true - vA_accept_true:.6f}")
    print(f"CLTV  : V_A_true={vA_cltv_true:.6f}, V_B_true={vB_cltv_true:.6f}, Delta_true={vB_cltv_true - vA_cltv_true:.6f}")
    print(f"\nАртефакты сохранены в: {args.artifacts_dir}")


if __name__ == "__main__":
    main()
