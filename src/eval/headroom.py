"""Welch CI of (method - baseline) from summary statistics (mean, Student-t half-width, n)."""
from __future__ import annotations

import argparse
import csv
import math
from typing import Iterable

from scipy import stats  # KHÔNG CHẮC: scipy must be in requirements.txt


def se_from_half(half: float, n: int) -> float:
    """summary.csv 'half' = t_{0.975, n-1} * SE, so SE = half / t."""
    if n < 2:
        raise ValueError("need n >= 2")
    return half / stats.t.ppf(0.975, n - 1)


def welch_diff_ci(mean_a: float, half_a: float, n_a: int,
                  mean_b: float, half_b: float, n_b: int, conf: float = 0.95) -> dict:
    """CI of (mean_b - mean_a) with Welch-Satterthwaite df."""
    se_a, se_b = se_from_half(half_a, n_a), se_from_half(half_b, n_b)
    d = mean_b - mean_a
    se2 = se_a ** 2 + se_b ** 2
    if se2 == 0.0:
        return {"diff": d, "lo": d, "hi": d, "df": math.inf, "excludes_zero": d != 0.0}
    df = se2 ** 2 / (se_a ** 4 / (n_a - 1) + se_b ** 4 / (n_b - 1))
    t = stats.t.ppf(0.5 + conf / 2.0, df)
    lo, hi = d - t * math.sqrt(se2), d + t * math.sqrt(se2)
    return {"diff": d, "lo": lo, "hi": hi, "df": df, "excludes_zero": lo > 0.0 or hi < 0.0}


def load_summary(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def headroom_table(rows: Iterable[dict], baseline: str = "naive_bce",
                   metrics: tuple[str, ...] = ("test_pr_auc", "test_efl_ratio")) -> list[dict]:
    rows = list(rows)
    key = {(r["delay"], r["method"], r["metric"]): r for r in rows}
    out = []
    for (delay, method, metric), r in key.items():
        if method == baseline or metric not in metrics or (delay, baseline, metric) not in key:
            continue
        a = key[(delay, baseline, metric)]
        ci = welch_diff_ci(float(a["mean"]), float(a["half"]), int(a["n"]),
                           float(r["mean"]), float(r["half"]), int(r["n"]))
        out.append({"delay": delay, "metric": metric, "method": method, **ci})
    return sorted(out, key=lambda x: (x["delay"], x["metric"], x["method"]))


def has_headroom(row: dict) -> bool:
    """Pre-specified rule (my choice, no external source): oracle better than A0 on PR-AUC, CI excludes 0."""
    return row["metric"] == "test_pr_auc" and row["method"] == "oracle" and row["excludes_zero"] and row["diff"] > 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True, help="path to summary.csv")
    args = ap.parse_args(argv)
    for r in headroom_table(load_summary(args.summary)):
        print(f"{r['delay']:<11} {r['metric']:<15} {r['method']:<14} diff={r['diff']:+.4f} "
              f"CI=[{r['lo']:+.4f}, {r['hi']:+.4f}] df={r['df']:.2f} excludes_0={r['excludes_zero']}"
              f"{'  <-- HEADROOM' if has_headroom(r) else ''}")
    print("note: n small -> low power; 'CI includes 0' is NOT equivalence.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())