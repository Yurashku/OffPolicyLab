"""Основной сценарий на открытых реальных данных с известной истиной.

Сценарий: логи политики A без пропенсити, сравнение двух политик, автоматический выбор оценщика,
bootstrap-интервалы для V(A), V(B) и Δ, вердикт доверия и согласие оценщиков.

Два набора данных:

- **Hillstrom** (MineThatData E-Mail Analytics, 64 000 клиентов). Рассылка была рандомизирована:
  мужское письмо, женское или ничего, по 1/3. Логи «прод-политики» A строятся отбором с отказом
  из половины клиентов: строка с действием a остаётся с вероятностью pi_A(a|x). Тогда признаки
  распределены как в исходных данных, а действия — как у A. Пропенсити в логи не попадают.
  Истина для любой политики — IPS по другой половине клиентов с известной вероятностью 1/3.
- **Open Bandit Dataset** (ZOZOTOWN, кампания men, CC BY 4.0). A — равномерно случайная политика
  (453 тыс. показов, 34 товара × 3 позиции), колонка пропенсити убирается. B — Bernoulli
  Thompson Sampling, работавшая в те же дни на другой половине пользователей. Истина V(B) — CTR в
  её логах (4.08 млн показов).

Запуск (данные скачиваются в ``--data-dir``; OBD — архив 413 МБ)::

    python examples/real_data.py hillstrom --seeds 0 1 2 3 4 5 6 7 8 9
    python examples/real_data.py obd --rows 100000
"""

from __future__ import annotations

import argparse
import time
import urllib.request
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

import policyscope as ps

HILLSTROM_URL = ("http://www.minethatdata.com/"
                 "Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv")
OBD_URL = "https://research.zozo.com/data_release/open_bandit_dataset.zip"
OFFERS = ["mens", "none", "womens"]


def download(url: str, path: Path) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Скачиваю {url} -> {path}")
        urllib.request.urlretrieve(url, path)
    return path


# ---------------------------------------------------------------------------
# Hillstrom
# ---------------------------------------------------------------------------


def _probs(score: np.ndarray, index: pd.Index) -> pd.DataFrame:
    p = np.exp(score - score.max(axis=1, keepdims=True))
    return pd.DataFrame(p / p.sum(axis=1, keepdims=True), index=index, columns=OFFERS)


def policy_prod(x: pd.DataFrame) -> pd.DataFrame:
    """A: правило с порогами (логистикой по сырым признакам точно не описывается) и 15% исследования."""
    score = np.zeros((len(x), 3))
    score[:, 0] = 1.0 * (x["mens"] == 1) + 0.4 * (x["history"] > 300)
    score[:, 1] = 0.8 * (x["recency"] > 9) - 0.6 * (x["history"] > 300)
    score[:, 2] = 1.0 * (x["womens"] == 1) + 0.2 * (x["channel"] == "Web")
    return 0.85 * _probs(score, x.index) + 0.15 / 3


def policy_rule(x: pd.DataFrame) -> pd.DataFrame:
    """«rule»: мужское письмо тем, кто покупал мужское, остальным — женское."""
    a = np.where(x["mens"].to_numpy() == 1, 0, 2)
    return pd.DataFrame(np.eye(3)[a], index=x.index, columns=OFFERS)


def policy_new(x: pd.DataFrame) -> pd.DataFrame:
    """«new»: чаще мужское; новичкам и покупателям женского — женское; давним — без письма."""
    score = np.zeros((len(x), 3))
    score[:, 0] = 1.5
    score[:, 1] = -1.0 + 1.2 * (x["recency"] > 10)
    score[:, 2] = 0.8 * (x["newbie"] == 1) + 0.6 * (x["womens"] == 1)
    return _probs(score, x.index)


def hillstrom_split(h: pd.DataFrame, seed: int, logs_share: float = 0.6):
    rng = np.random.default_rng(seed)
    in_logs = rng.random(len(h)) < logs_share
    pool, holdout = h[in_logs], h[~in_logs].reset_index(drop=True)
    p_a = policy_prod(pool).to_numpy()[np.arange(len(pool)), pd.Index(OFFERS).get_indexer(pool["offer"])]
    logs = pool[rng.random(len(pool)) < p_a].reset_index(drop=True)
    return logs, holdout


def truth_terms(holdout: pd.DataFrame, policy, metric: str) -> np.ndarray:
    """Вклады IPS по рандомизированной отложенной половине: среднее — несмещённая истина."""
    p = policy(holdout).to_numpy()[np.arange(len(holdout)), pd.Index(OFFERS).get_indexer(holdout["offer"])]
    return 3.0 * p * holdout[metric].to_numpy()


def run_hillstrom(data_dir: Path, seeds: list[int], n_boot: int) -> pd.DataFrame:
    h = pd.read_csv(download(HILLSTROM_URL, data_dir / "hillstrom.csv"))
    h["offer"] = h.pop("segment").map({"Mens E-Mail": "mens", "No E-Mail": "none", "Womens E-Mail": "womens"})
    rows = []
    for seed in seeds:
        logs, holdout = hillstrom_split(h, seed)
        start = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # прогноз длительности bootstrap
            ev = ps.Evaluator(logs.drop(columns="conversion"), action="offer", reward=["visit", "spend"],
                              random_state=seed)
            report = ev.evaluate(policy_new, baseline=policy_rule, name="new", baseline_name="rule",
                                 n_boot=n_boot)
        seconds = time.time() - start
        if seed == seeds[0]:
            print(report.to_markdown(), "\n")
        for metric, m in report.metrics.items():
            t_new, t_rule = truth_terms(holdout, policy_new, metric), truth_terms(holdout, policy_rule, metric)
            t_delta = t_new - t_rule
            se = {k: v.std() / np.sqrt(len(v)) for k, v in (("new", t_new), ("rule", t_rule), ("delta", t_delta))}
            rows.append(dict(
                seed=seed, metric=metric, n_logs=len(logs), seconds=round(seconds),
                trust=m.trust, decision=m.decision, agreement=m.agreement,
                v_rule=m.v_a.value, v_rule_low=m.v_a.low, v_rule_high=m.v_a.high, truth_rule=t_rule.mean(),
                v_new=m.v_b.value, v_new_low=m.v_b.low, v_new_high=m.v_b.high, truth_new=t_new.mean(),
                delta=m.delta.value, delta_low=m.delta.low, delta_high=m.delta.high, delta_se=m.delta.se,
                truth_delta=t_delta.mean(), truth_delta_se=se["delta"],
                truth_new_se=se["new"], truth_rule_se=se["rule"],
            ))
    return pd.DataFrame(rows)


def summarize_hillstrom(df: pd.DataFrame) -> pd.DataFrame:
    """z = (оценка − истина) / sqrt(SE оценки² + SE истины²): при честном интервале ~ N(0, 1)."""
    out = []
    for metric, g in df.groupby("metric", sort=False):
        for target, est, low, high in (("V(rule)", "v_rule", "v_rule_low", "v_rule_high"),
                                       ("V(new)", "v_new", "v_new_low", "v_new_high"),
                                       ("Δ", "delta", "delta_low", "delta_high")):
            truth = g[{"v_rule": "truth_rule", "v_new": "truth_new", "delta": "truth_delta"}[est]]
            t_se = g[{"v_rule": "truth_rule_se", "v_new": "truth_new_se", "delta": "truth_delta_se"}[est]]
            se = (g[high] - g[low]) / (2 * 1.96)
            z = (g[est] - truth) / np.sqrt(se ** 2 + t_se ** 2)
            out.append({
                "метрика": metric, "величина": target, "прогонов": len(g),
                "средняя оценка": g[est].mean(), "средняя истина": truth.mean(),
                "среднее z": z.mean(), "SD z": z.std(), "|z| < 1.96": (z.abs() < 1.96).mean(),
            })
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Open Bandit Dataset
# ---------------------------------------------------------------------------


def obd_files(data_dir: Path) -> Path:
    root = data_dir / "open_bandit_dataset"
    if not (root / "bts" / "men" / "men.csv").exists():
        archive = download(OBD_URL, data_dir / "open_bandit_dataset.zip")
        with zipfile.ZipFile(archive) as z:
            for name in z.namelist():
                if "/men/" in name and name.endswith(".csv"):
                    z.extract(name, data_dir)
    return root


def bts_policy(root: Path):
    """B: распределение товаров по (час, позиция), с которым BTS реально показывал их в своём плече."""
    bts = pd.read_csv(root / "bts" / "men" / "men.csv", usecols=["timestamp", "item_id", "position", "click"])
    bts["hour"] = pd.to_datetime(bts["timestamp"], format="ISO8601").dt.floor("h")
    counts = bts.groupby(["hour", "position", "item_id"]).size().unstack(fill_value=0)
    table = counts.div(counts.sum(axis=1), axis=0)
    overall = bts.groupby(["position", "item_id"]).size().unstack(fill_value=0)
    overall = overall.div(overall.sum(axis=1), axis=0)

    def policy(frame: pd.DataFrame) -> pd.DataFrame:
        hour = pd.to_datetime(frame["timestamp"], format="ISO8601").dt.floor("h")
        probs = table.reindex(pd.MultiIndex.from_arrays([hour, frame["position"]])).to_numpy()
        missing = np.isnan(probs).any(axis=1)
        if missing.any():  # час без показов BTS — средний профиль позиции
            probs[missing] = overall.reindex(frame["position"][missing]).to_numpy()
        return pd.DataFrame(probs, index=frame.index, columns=table.columns)

    truth = bts["click"].mean(), bts["click"].std() / np.sqrt(len(bts))
    return policy, truth


def run_obd(data_dir: Path, rows: int, n_boot: int, seed: int) -> None:
    root = obd_files(data_dir)
    random = pd.read_csv(root / "random" / "men" / "men.csv", index_col=0)
    truth_a = random["click"].mean()
    logs = random.sample(rows, random_state=seed).reset_index(drop=True) if rows < len(random) else random
    policy, (truth_b, truth_b_se) = bts_policy(root)
    print(f"Истина: V(Random) = {truth_a:.5f} (все {len(random)} показов), "
          f"V(BTS) = {truth_b:.5f} ± {truth_b_se:.5f}, Δ = {truth_b - truth_a:+.5f}\n")

    start = time.time()
    report = ps.evaluate(logs.drop(columns="propensity_score"), policy, action="item_id", reward="click",
                         name="BTS", baseline_name="Random", n_boot=n_boot, random_state=seed)
    print(f"Без пропенсити: {len(logs)} строк, {time.time() - start:.0f} с\n")
    print(report.to_markdown(), "\n")
    start = time.time()
    logged = ps.evaluate(logs, policy, action="item_id", reward="click", propensity="propensity_score",
                         name="BTS", baseline_name="Random", random_state=seed)
    print(f"Контроль с логированными пропенсити ({time.time() - start:.0f} с):\n")
    print(logged.summary().T.to_string())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dataset", choices=["hillstrom", "obd"])
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--rows", type=int, default=100_000, help="OBD: размер выборки логов Random")
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--output", type=Path, help="Hillstrom: CSV с результатами по прогонам")
    args = ap.parse_args()
    if args.dataset == "hillstrom":
        df = run_hillstrom(args.data_dir, args.seeds, args.n_boot)
        if args.output:
            df.to_csv(args.output, index=False)
        cols = ["seed", "metric", "trust", "decision", "agreement", "delta", "delta_low", "delta_high",
                "truth_delta", "truth_delta_se"]
        with pd.option_context("display.width", 200, "display.max_columns", 20):
            print(df[cols].round(4).to_string(index=False), "\n")
            print(summarize_hillstrom(df).round(4).to_string(index=False))
    else:
        run_obd(args.data_dir, args.rows, args.n_boot, args.seeds[0])


if __name__ == "__main__":
    main()
