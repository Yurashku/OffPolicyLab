"""Валидация оценщиков на синтетике с аналитически известной истиной.

Для каждого сценария генерирует много независимых логов, оценивает политику B и
сравнивает результат с истинным Δ = V(B) − V(A): смещение, RMSE, фактическое
покрытие 95% интервала основного оценщика (DR).

Примеры:
    python examples/validate.py                       # все сценарии, таблица в консоль
    python examples/validate.py --runs 20 --strict    # с проверкой порогов (для CI)
    python examples/validate.py --output docs/validation_results.md
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import policyscope as ps
from policyscope.synthetic import OfferSimulator

FEATURES = ["age", "income", "loyal", "risk", "region"]


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    logging: dict
    propensity: bool = True
    client_effect: float = 0.0
    visits: int = 1
    cluster: bool = False
    expect_flagged: bool = False
    estimator: str = "dr"
    seeds: tuple = (1, 2)
    #: False — сценарий показывает известное ограничение: цифры попадают в таблицу,
    #: но пороги приёмки к нему не применяются.
    enforce: bool = True


SCENARIOS = {
    "logged": Scenario(
        "logged", "ε-greedy логирование (ε=0.2), пропенсити в логах",
        dict(kind="epsilon_greedy", epsilon=0.2),
    ),
    "softmax": Scenario(
        "softmax", "softmax-логирование (T=2): умеренно тяжёлые веса",
        dict(kind="softmax", temperature=2.0),
    ),
    "weak_overlap": Scenario(
        "weak_overlap", "«острое» softmax-логирование (T=0.5): перекрытия почти нет",
        dict(kind="softmax", temperature=0.5), expect_flagged=True,
    ),
    "estimated": Scenario(
        "estimated", "ε-greedy логирование, пропенсити НЕ залогированы (bootstrap)",
        dict(kind="epsilon_greedy", epsilon=0.2), propensity=False,
    ),
    "estimated_pair2": Scenario(
        "estimated_pair2",
        "пропенсити НЕ залогированы, другая пара (A8 → B6): ограничение — DR смещён",
        dict(kind="epsilon_greedy", epsilon=0.2), propensity=False, seeds=(8, 6), enforce=False,
    ),
    "clustered": Scenario(
        "clustered", "5 визитов на клиента, клиентский эффект (кластерный ДИ)",
        dict(kind="epsilon_greedy", epsilon=0.2), client_effect=1.5, visits=5, cluster=True,
    ),
    "dm": Scenario(
        "dm", "основной DM (estimator='dm', bootstrap), ε-greedy 0.2, пропенсити в логах",
        dict(kind="epsilon_greedy", epsilon=0.2), estimator="dm",
    ),
    "dm_pair2": Scenario(
        "dm_pair2", "основной DM на другой паре политик (A3 → B5), где его выигрыш меньше",
        dict(kind="epsilon_greedy", epsilon=0.2), estimator="dm", seeds=(3, 5),
    ),
}


def run_scenario(s: Scenario, *, runs: int, clients: int, n_boot: int, seed: int) -> pd.DataFrame:
    rows = []
    for run in range(runs):
        sim = OfferSimulator(seed=seed + run, client_effect=s.client_effect)
        data = sim.sample_clients(max(clients // s.visits, 50), visits=s.visits)
        policy_a = sim.policy(seed=s.seeds[0], **s.logging)
        policy_b = sim.policy("softmax", temperature=1.0, seed=s.seeds[1])
        logs = sim.simulate(data, policy_a, log_propensity=s.propensity)
        report = ps.evaluate(
            logs, policy_b, action="offer", reward=["accepted", "revenue"],
            propensity="propensity" if s.propensity else None,
            cluster="client_id" if s.cluster else None,
            features=FEATURES, n_boot=n_boot, random_state=run, estimator=s.estimator,
        )
        for metric, m in report.metrics.items():
            truth = sim.true_value(logs, policy_b, metric) - sim.true_value(logs, policy_a, metric)
            for row in m.rows:
                d = row.delta
                rows.append({
                    "scenario": s.name, "metric": metric, "estimator": row.estimator, "run": run,
                    "error": d.value - truth,
                    "covered": (d.low <= truth <= d.high) if d.low is not None else np.nan,
                    "width": (d.high - d.low) if d.low is not None else np.nan,
                    "unreliable": m.trust == "unreliable",
                })
    return pd.DataFrame(rows)


def aggregate(raw: pd.DataFrame) -> pd.DataFrame:
    """Сводка по сценарию, метрике и оценщику.

    Смещение, RMSE и покрытие считаются по прогонам, которые библиотека **не** пометила
    как ненадёжные: именно в них пользователь сделает вывод. RMSE по всем прогонам
    показывает, что было бы, если игнорировать вердикт.
    """
    def summarize(g: pd.DataFrame) -> pd.Series:
        trusted = g[~g["unreliable"]]
        e = trusted["error"].to_numpy()
        n = len(e)
        return pd.Series({
            "runs": len(g),
            "unreliable_share": g["unreliable"].mean(),
            "bias": e.mean() if n else np.nan,
            "bias_mcse": e.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan,
            "rmse": np.sqrt(np.mean(e**2)) if n else np.nan,
            "rmse_all": np.sqrt(np.mean(g["error"].to_numpy() ** 2)),
            "coverage": trusted["covered"].mean() if n else np.nan,
            "trusted_runs": n,
            "ci_width": trusted["width"].mean() if n else np.nan,
        })

    order = {name: i for i, name in enumerate(["dr", "sndr", "ips", "snips", "dm"])}
    grouped = raw.groupby(["scenario", "metric", "estimator"])
    out = grouped.apply(summarize, include_groups=False).reset_index()
    out["_order"] = out["estimator"].map(order)
    return out.sort_values(["scenario", "metric", "_order"]).drop(columns="_order")


def check(table: pd.DataFrame) -> list[str]:
    """Пороги приёмки для основного оценщика сценария на прогонах, признанных пригодными."""
    problems = []
    primary = table["estimator"] == table["scenario"].map(lambda name: SCENARIOS[name].estimator)
    for _, r in table[primary].iterrows():
        label = f"{r['scenario']}/{r['metric']}"
        if SCENARIOS[r["scenario"]].expect_flagged and r["unreliable_share"] < 0.3:
            problems.append(f"{label}: перекрытия нет, но ненадёжными помечено лишь {r['unreliable_share']:.0%}")
        if r["trusted_runs"] < 5 or not SCENARIOS[r["scenario"]].enforce:
            continue
        if abs(r["bias"]) > 3 * r["bias_mcse"]:
            problems.append(f"{label}: смещение {r['bias']:.4g} больше 3 MCSE ({r['bias_mcse']:.4g})")
        floor = 0.90 - 2 * np.sqrt(0.09 / r["trusted_runs"])
        if r["coverage"] < floor:
            problems.append(f"{label}: покрытие {r['coverage']:.2f} ниже {floor:.2f}")
    return problems


def to_markdown(table: pd.DataFrame, settings: str) -> str:
    def fmt(x, spec):
        return "—" if pd.isna(x) else format(x, spec)

    lines = [
        settings, "",
        "| Сценарий | Метрика | Оценщик | Помечено «ненадёжно» | Смещение (±MCSE) | RMSE "
        "| Покрытие 95% ДИ (±MCSE) | RMSE по всем прогонам |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, r in table.iterrows():
        bias = "—" if pd.isna(r["bias"]) else f"{r['bias']:+.4g} (±{fmt(r['bias_mcse'], '.2g')})"
        if pd.isna(r["coverage"]):
            coverage = "—"
        else:
            mcse = np.sqrt(max(r["coverage"] * (1 - r["coverage"]), 0.0025) / max(r["trusted_runs"], 1))
            coverage = f"{r['coverage']:.2f} (±{mcse:.2f})"
        lines.append(
            f"| {r['scenario']} | {r['metric']} | {r['estimator']} | {r['unreliable_share']:.0%} | {bias} "
            f"| {fmt(r['rmse'], '.4g')} | {coverage} | {fmt(r['rmse_all'], '.4g')} |"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenarios", nargs="+", default=list(SCENARIOS), choices=list(SCENARIOS))
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--clients", type=int, default=3000, help="строк в одном прогоне")
    parser.add_argument("--n-boot", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=None, help="записать таблицу в markdown-файл")
    parser.add_argument("--strict", action="store_true", help="код возврата 1 при нарушении порогов")
    args = parser.parse_args()

    frames = []
    for name in args.scenarios:
        s = SCENARIOS[name]
        bootstrap = (not s.propensity) or s.estimator == "dm"
        runs = max(args.runs // 2, 10) if bootstrap else args.runs  # bootstrap дорог
        start = time.perf_counter()
        frames.append(run_scenario(s, runs=runs, clients=args.clients, n_boot=args.n_boot, seed=args.seed))
        print(f"{name}: {runs} прогонов за {time.perf_counter() - start:.0f} с — {s.description}", file=sys.stderr)
    table = aggregate(pd.concat(frames, ignore_index=True))
    settings = (f"Прогонов на сценарий: {args.runs} (сценарии с bootstrap — {max(args.runs // 2, 10)}), "
                f"строк в прогоне: {args.clients}, bootstrap-реплик: {args.n_boot}, seed: {args.seed}.")
    markdown = to_markdown(table, settings)
    print(markdown)
    if args.output is not None:
        args.output.write_text(markdown + "\n", encoding="utf-8")
    problems = check(table)
    for p in problems:
        print("НАРУШЕНИЕ:", p, file=sys.stderr)
    return 1 if (problems and args.strict) else 0


if __name__ == "__main__":
    sys.exit(main())
