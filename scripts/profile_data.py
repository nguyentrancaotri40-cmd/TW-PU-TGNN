#!/usr/bin/env python
"""E0 pre-flight: profile the IEEE-CIS time axis, weekly fraud rate and amounts.

Usage (from the repo root)::

    python scripts/profile_data.py --data_dir data/ieee-cis
    python scripts/profile_data.py --data_dir data/ieee-cis --nrows 50000   # quick debug

Outputs
-------
* stdout: min/max day, weekly table, proposed 65/16/19 split, amount stats.
* ``results/eda/weekly_fraud_dist.png`` (folder is created if missing).
* ``results/eda/weekly_stats.csv``.

Only the four columns needed for profiling are read. The weekly view covers
the whole labelled timeline to expose drift; it is for *looking*, never for
tuning on Val/Test labels.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # make `src` importable when run as a script
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")  # headless (Kaggle / servers)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.loader import (
    AMOUNT_COL,
    ID_COL,
    SECONDS_PER_DAY,
    SECONDS_PER_WEEK,
    TARGET_COL,
    TIME_COL,
    DataLoader,
    TimeStats,
    get_time_stats,
)
from src.data.splitter import SplitBoundaries, compute_split_boundaries, summarize_splits, temporal_split

PROFILE_COLS = [ID_COL, TARGET_COL, TIME_COL, AMOUNT_COL]
DRIFT_Z = 3.0          # |z| above this flags a week (exploratory, not a test)
MIN_WEEK_TXN = 500     # ignore tiny weeks for the drift flag


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E0 data profiling for IEEE-CIS.")
    p.add_argument("--data_dir", type=str, default="data", help="Folder with train_transaction.csv.")
    p.add_argument("--out_dir", type=str, default=str(REPO_ROOT / "results" / "eda"),
                   help="Where to save plots/tables (default: <repo>/results/eda).")
    p.add_argument("--nrows", type=int, default=None, help="Read only the first N rows (debug).")
    return p.parse_args()


# --------------------------------------------------------------------------- #
def weekly_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Transactions, fraud count and fraud rate per (relative) 7-day bucket."""
    tmp = pd.DataFrame(
        {
            "week": (df[TIME_COL] // SECONDS_PER_WEEK).astype(int),
            "day": (df[TIME_COL] // SECONDS_PER_DAY).astype(int),
            "fraud": df[TARGET_COL].astype(int),
        }
    )
    agg = tmp.groupby("week").agg(
        n_txn=("fraud", "size"), n_fraud=("fraud", "sum"), days_observed=("day", "nunique")
    )
    agg["fraud_rate"] = agg["n_fraud"] / agg["n_txn"]
    agg["is_partial"] = agg["days_observed"] < 7
    p0 = float(df[TARGET_COL].mean())
    se = np.sqrt(p0 * (1.0 - p0) / agg["n_txn"])
    agg["z_vs_pooled"] = (agg["fraud_rate"] - p0) / se
    agg["drift_flag"] = (agg["z_vs_pooled"].abs() > DRIFT_Z) & (agg["n_txn"] >= MIN_WEEK_TXN) & ~agg["is_partial"]
    return agg


def print_time_stats(ts: TimeStats) -> None:
    print("=" * 78)
    print("1) TIME AXIS (TransactionDT is a relative offset in seconds, not a calendar date)")
    print("=" * 78)
    print(f"  min : {ts.min_seconds:>12,.0f} s = {ts.min_day:8.3f} days = {ts.min_day / 7:6.2f} weeks")
    print(f"  max : {ts.max_seconds:>12,.0f} s = {ts.max_day:8.3f} days = {ts.max_day / 7:6.2f} weeks")
    print(f"  span: {ts.span_days:.3f} days | day buckets touched: {ts.n_calendar_days} "
          f"| week buckets touched: {ts.n_weeks}")


def print_weekly(weekly: pd.DataFrame, pooled_rate: float) -> None:
    print("\n" + "=" * 78)
    print(f"2) WEEKLY DISTRIBUTION (pooled fraud rate = {100 * pooled_rate:.3f}%)")
    print("=" * 78)
    view = pd.DataFrame(
        {
            "days": [f"{w * 7}-{w * 7 + 6}" for w in weekly.index],
            "n_txn": weekly["n_txn"],
            "n_fraud": weekly["n_fraud"],
            "fraud_%": (100 * weekly["fraud_rate"]).round(3),
            "z": weekly["z_vs_pooled"].round(1),
            "note": np.where(weekly["is_partial"], "partial", "")
            + np.where(weekly["drift_flag"], " DRIFT?", ""),
        },
        index=weekly.index.rename("week"),
    )
    print(view.to_string())
    flagged = weekly.index[weekly["drift_flag"]].tolist()
    print(f"\n  Weeks with |z|>{DRIFT_Z:g} (n>={MIN_WEEK_TXN}, full weeks): {flagged or 'none'} "
          "(exploratory flag, not a drift test)")


def plot_weekly(weekly: pd.DataFrame, pooled_rate: float, b: SplitBoundaries, path: Path) -> None:
    """Two panels: weekly volume and weekly fraud rate, with split boundaries."""
    path.parent.mkdir(parents=True, exist_ok=True)
    x = weekly.index.to_numpy()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [1, 1.2]})

    colors = np.where(weekly["is_partial"], "#b0b7c3", "#4c78a8")
    ax1.bar(x, weekly["n_txn"], color=colors, edgecolor="none")
    ax1.set_ylabel("Transactions / week")

    ax2.plot(x, 100 * weekly["fraud_rate"], marker="o", ms=4, lw=1.4, color="#e45756", label="weekly fraud rate")
    ax2.axhline(100 * pooled_rate, ls="--", lw=1, color="black", label=f"pooled {100 * pooled_rate:.2f}%")
    flagged = weekly[weekly["drift_flag"]]
    if len(flagged):
        ax2.scatter(flagged.index, 100 * flagged["fraud_rate"], s=70, facecolors="none",
                    edgecolors="black", label=f"|z|>{DRIFT_Z:g}")
    ax2.set_ylabel("Fraud rate (%)")
    ax2.set_xlabel("Relative week index (7-day buckets of TransactionDT)")

    # bar for week w covers days [7w, 7w+7) and is centred at x=w -> day d sits at d/7 - 0.5
    for label, day in (("T_train", b.t_train_day), ("T_val", b.t_val_day)):
        xpos = day / 7.0 - 0.5
        for ax in (ax1, ax2):
            ax.axvline(xpos, color="#54a24b", ls=":", lw=1.8)
        ax1.text(xpos, 1.02, f"{label} (day {day:.0f})", color="#2f7d32", fontsize=8, ha="center",
                 va="bottom", transform=ax1.get_xaxis_transform())
    ax1.set_title("IEEE-CIS: weekly volume and fraud rate (grey = partial week)", pad=22)
    handles, labels = ax2.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def print_split(df: pd.DataFrame, b: SplitBoundaries) -> None:
    print("\n" + "=" * 78)
    print("3) PROPOSED TEMPORAL SPLIT 65/16/19 (ratios over the time span, day-aligned)")
    print("=" * 78)
    print(f"  {b.summary()}")
    train_df, val_df, test_df = temporal_split(df)
    summary = summarize_splits(train_df, val_df, test_df)
    print(summary.round(3).to_string())
    total_fraud_amt = summary["fraud_amount"].sum()
    print(f"\n  Total fraud amount = {total_fraud_amt:,.2f} (EFL of a model that flags nothing).")
    low = summary[summary["n_fraud"] < 1000]
    if len(low):
        print(f"  WARNING: split(s) with < 1000 frauds -> wide CIs: {low.index.tolist()}")


def print_amount_stats(df: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print(f"4) {AMOUNT_COL} STATISTICS (basis for the amount-dependent delay and EFL)")
    print("=" * 78)
    qs = [0.9, 0.95, 0.99]

    def describe(s: pd.Series) -> dict[str, float]:
        out = {"count": float(s.count()), "min": s.min(), "mean": s.mean(), "median": s.median(), "std": s.std()}
        out.update({f"p{int(q * 100)}": s.quantile(q) for q in qs})
        out["max"] = s.max()
        return out

    table = pd.DataFrame(
        {
            "all": describe(df[AMOUNT_COL]),
            "legit": describe(df.loc[df[TARGET_COL] == 0, AMOUNT_COL]),
            "fraud": describe(df.loc[df[TARGET_COL] == 1, AMOUNT_COL]),
        }
    )
    print(table.round(2).to_string())
    p99 = df[AMOUNT_COL].quantile(0.99)
    top = df[AMOUNT_COL] >= p99
    fraud_amt = df.loc[df[TARGET_COL] == 1, AMOUNT_COL]
    print(f"\n  Skewness: {df[AMOUNT_COL].skew():.2f} (heavy right tail -> consider log scale / amount_ref=median).")
    print(f"  Top 1% of transactions (>= {p99:,.2f}) hold {100 * df.loc[top, AMOUNT_COL].sum() / df[AMOUNT_COL].sum():.1f}% "
          f"of total amount and {100 * df.loc[top & (df[TARGET_COL] == 1), AMOUNT_COL].sum() / fraud_amt.sum():.1f}% of fraud amount.")
    print(f"  Suggested amount_ref (median) = {df[AMOUNT_COL].median():,.2f}")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    loader = DataLoader(args.data_dir)
    print(f"Loading {loader.transaction_path} (cols: {PROFILE_COLS}) ...")
    df = loader.load_transactions(usecols=PROFILE_COLS, nrows=args.nrows)
    print(f"  rows={len(df):,} | frauds={int(df[TARGET_COL].sum()):,} "
          f"({100 * df[TARGET_COL].mean():.3f}%)")

    ts = get_time_stats(df)
    print_time_stats(ts)

    weekly = weekly_stats(df)
    pooled = float(df[TARGET_COL].mean())
    print_weekly(weekly, pooled)
    weekly.to_csv(out_dir / "weekly_stats.csv")

    b = compute_split_boundaries(df)
    print_split(df, b)
    print_amount_stats(df)

    png = out_dir / "weekly_fraud_dist.png"
    plot_weekly(weekly, pooled, b, png)
    print(f"\nSaved: {png}\nSaved: {out_dir / 'weekly_stats.csv'}")


if __name__ == "__main__":
    main()
