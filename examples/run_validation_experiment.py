"""Систематическая проверка OPE-оценщиков на синтетике с известной истиной.

Прогоняет несколько независимых экспериментов по каждой паре «режим × оценщик» и
сравнивает результат с аналитическим оракулом. Возвращает ненулевой код возврата,
если нарушены пороги приёмки, — поэтому скрипт пригоден для запуска в CI.

Пример:
    python examples/run_validation_experiment.py --n-runs 20 --n-users 2000
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from policyscope.validation import ValidationMode, run_simulation_validation

# Cross-fitting внутри bootstrap стоит 5*n_boot переобучений на ячейку, поэтому
# режимы с ним используют аналитический инференс — там он и нужен для валидности.
DEFAULT_MODES = (
    ValidationMode(name="logged_bootstrap", propensity_source="logged"),
    ValidationMode(
        name="logged_influence", propensity_source="logged", inference="influence"
    ),
    ValidationMode(
        name="estimated_influence", propensity_source="estimated", inference="influence"
    ),
    ValidationMode(
        name="crossfit_influence", propensity_source="estimated",
        use_crossfit=True, inference="influence",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-runs", type=int, default=10, help="число независимых прогонов")
    parser.add_argument("--n-users", type=int, default=1500, help="размер выборки в прогоне")
    parser.add_argument("--n-boot", type=int, default=150, help="число bootstrap-реплик")
    parser.add_argument("--target", default="accept", help="целевая метрика")
    parser.add_argument(
        "--estimators",
        nargs="+",
        default=["ips", "snips", "dm", "dr", "sndr", "switch_dr"],
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=None,
        help="куда сохранить агрегат (CSV) и полный отчёт (JSON)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="вернуть ненулевой код возврата при нарушении порогов приёмки",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    result = run_simulation_validation(
        target=args.target,
        estimators=tuple(args.estimators),
        modes=DEFAULT_MODES,
        n_runs=args.n_runs,
        n_users=args.n_users,
        n_boot=args.n_boot,
    )

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    columns = [
        "mode", "estimator", "runs", "delta_bias_mean", "delta_rmse",
        "delta_ci_coverage", "mean_ess_ratio", "mean_weight",
    ]
    available = [c for c in columns if c in result.aggregate.columns]
    print("\n=== Агрегат по режимам и оценщикам ===")
    print(result.aggregate[available].to_string(index=False))

    problems = result.check()
    print("\n=== Пороги приёмки ===")
    if problems:
        for problem in problems:
            print(f"  ! {problem}")
    else:
        print("  все проверки пройдены")

    if args.artifacts_dir:
        args.artifacts_dir.mkdir(parents=True, exist_ok=True)
        result.aggregate.to_csv(args.artifacts_dir / "aggregate.csv", index=False)
        (args.artifacts_dir / "validation.json").write_text(
            json.dumps(result.to_dict(), ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        print(f"\nАртефакты сохранены в {args.artifacts_dir}")

    return 1 if (problems and args.strict) else 0


if __name__ == "__main__":
    sys.exit(main())
