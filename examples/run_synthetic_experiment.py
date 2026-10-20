"""Сквозной синтетический эксперимент: оценка политики B по логам политики A.

Скрипт генерирует пользователей, симулирует логи политики A, оценивает политику B
всеми доступными оценщиками и сравнивает результат с **аналитически известной**
истиной. Это самый быстрый способ увидеть, как выглядит полный отчёт.

Пример:
    python examples/run_synthetic_experiment.py --n-users 20000 --seed 42
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

import policyscope as ps
from policyscope.policies import make_policy
from policyscope.synthetic import SynthConfig, SyntheticRecommenderEnv

FEATURES = ["loyal", "age", "risk", "income"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-users", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target", default="accept", choices=["accept", "cltv"])
    parser.add_argument(
        "--policy-a", default="epsilon_greedy",
        choices=["greedy", "epsilon_greedy", "softmax"],
        help="логирующая политика: для OPE ей нужен полный носитель",
    )
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--policy-b", default="softmax",
                        choices=["greedy", "epsilon_greedy", "softmax"])
    parser.add_argument("--tau-b", type=float, default=0.8)
    parser.add_argument("--n-boot", type=int, default=300)
    parser.add_argument("--weight-clip", type=float, default=None)
    parser.add_argument("--artifacts-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    env = SyntheticRecommenderEnv(
        SynthConfig(n_users=args.n_users, seed=args.seed)
    )
    X = env.sample_users()
    policy_a = make_policy(args.policy_a, seed=111, epsilon=args.epsilon)
    policy_b = make_policy(args.policy_b, seed=222, tau=args.tau_b)

    logs = env.simulate_logs_A(policy_a, X, propensity_col="propensity_A")

    # Истина известна аналитически — на реальных данных её, разумеется, нет.
    oracle_a = env.oracle_value(policy_a, X, metric=args.target)
    oracle_b = env.oracle_value(policy_b, X, metric=args.target)

    common = dict(
        target=args.target,
        action_col="a_A",
        action_space=list(env.ACTIONS),
        feature_cols=FEATURES,
        propensity_col="propensity_A",
        propensity_source="logged",
        cluster_col=None,
        weight_clip=args.weight_clip,
    )

    print(f"\n=== Истина (оракул), метрика «{args.target}» ===")
    print(f"  V(A) = {oracle_a:.6f}")
    print(f"  V(B) = {oracle_b:.6f}")
    print(f"  Delta = {oracle_b - oracle_a:.6f}")

    table = ps.compare_policies_multi_estimator(
        logs, policy_b,
        estimators=("ips", "snips", "dm", "dr", "sndr", "switch_dr"),
        inference="influence", **common,
    )
    table["oracle_Delta"] = oracle_b - oracle_a
    table["delta_error"] = table["Delta"] - table["oracle_Delta"]

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print("\n=== Оценщики против истины ===")
    print(
        table[
            ["estimator", "V_B", "Delta", "oracle_Delta", "delta_error",
             "ess_ratio", "weight_mean", "n_warnings"]
        ].to_string(index=False)
    )
    print(
        "\nСогласие оценщиков между собой — практичный признак доверия: "
        f"разброс Delta = {table['Delta'].max() - table['Delta'].min():.6f}"
    )

    print("\n=== Полный отчёт (DR, bootstrap CI) ===")
    result = ps.compare_policies(
        logs, policy_b, estimator="dr",
        inference="bootstrap", n_boot=args.n_boot, **common,
    )
    print(result.to_markdown(metric_name=args.target))
    print(f"\nОшибка Delta относительно оракула: {result.delta - (oracle_b - oracle_a):+.6f}")

    if args.artifacts_dir:
        args.artifacts_dir.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.artifacts_dir / "estimators.csv", index=False)
        ps.dump_json(str(args.artifacts_dir / "dr_result.json"), result)
        (args.artifacts_dir / "report.md").write_text(
            result.to_markdown(metric_name=args.target), encoding="utf-8"
        )
        print(f"\nАртефакты сохранены в {args.artifacts_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
