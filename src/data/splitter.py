"""Temporal Train/Val/Test split (65/16/19) on ``TransactionDT``.

Boundaries are half-open on the time axis::

    train: [t_start, t_train)    val: [t_train, t_val)    test: [t_val, t_end)

so the three frames can never overlap in time. A transaction exactly on a
boundary goes to the *later* split.

By default the ratios apply to the **time span** (whole days), not to row
counts: delays are expressed in days, so day-aligned cut points keep the
"maturity" arithmetic clean. Use ``by="count"`` to split by row count instead.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from src.data.loader import (
    AMOUNT_COL,
    ID_COL,
    SECONDS_PER_DAY,
    TARGET_COL,
    TIME_COL,
)

SplitBy = Literal["time", "count"]


@dataclass(frozen=True)
class SplitBoundaries:
    """Cut points in seconds (same unit as ``TransactionDT``)."""

    t_start: float
    t_train: float  # first instant of Val  (= end of Train, exclusive)
    t_val: float    # first instant of Test (= end of Val, exclusive)
    t_end: float    # exclusive upper bound of the data

    @property
    def t_train_day(self) -> float:
        return self.t_train / SECONDS_PER_DAY

    @property
    def t_val_day(self) -> float:
        return self.t_val / SECONDS_PER_DAY

    def summary(self) -> str:
        return (
            f"train=[{self.t_start / SECONDS_PER_DAY:.2f}, {self.t_train_day:.2f}) d | "
            f"val=[{self.t_train_day:.2f}, {self.t_val_day:.2f}) d | "
            f"test=[{self.t_val_day:.2f}, {self.t_end / SECONDS_PER_DAY:.2f}) d  "
            f"(t_train={self.t_train:,.0f}s, t_val={self.t_val:,.0f}s)"
        )


def _validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    ratios = (train_ratio, val_ratio, test_ratio)
    if any(r <= 0 for r in ratios):
        raise ValueError(f"All ratios must be > 0, got {ratios}.")
    if not np.isclose(sum(ratios), 1.0, atol=1e-6):
        raise ValueError(f"Ratios must sum to 1, got {sum(ratios):.6f}.")


def _round_half_up(x: float) -> int:
    return int(np.floor(x + 0.5))


def compute_split_boundaries(
    df: pd.DataFrame,
    train_ratio: float = 0.65,
    val_ratio: float = 0.16,
    test_ratio: float = 0.19,
    time_col: str = TIME_COL,
    by: SplitBy = "time",
    snap_to_day: bool = True,
) -> SplitBoundaries:
    """Compute ``(t_train, t_val)`` without materialising the splits."""
    _validate_ratios(train_ratio, val_ratio, test_ratio)
    if time_col not in df.columns:
        raise KeyError(f"Column '{time_col}' not found.")
    if len(df) == 0:
        raise ValueError("Empty DataFrame.")

    t = df[time_col].to_numpy(dtype=np.float64)
    t_min, t_max = float(t.min()), float(t.max())

    if by == "time":
        if snap_to_day:
            day0 = np.floor(t_min / SECONDS_PER_DAY) * SECONDS_PER_DAY
            n_days = int(np.floor((t_max - day0) / SECONDS_PER_DAY)) + 1
            cut1 = _round_half_up(train_ratio * n_days)
            cut2 = _round_half_up((train_ratio + val_ratio) * n_days)
            if not (1 <= cut1 < cut2 < n_days):
                raise ValueError(
                    f"Only {n_days} day(s) of data: too short for a day-aligned "
                    f"{train_ratio}/{val_ratio}/{test_ratio} split."
                )
            t_start, t_end = day0, day0 + n_days * SECONDS_PER_DAY
            t_train, t_val = day0 + cut1 * SECONDS_PER_DAY, day0 + cut2 * SECONDS_PER_DAY
        else:
            t_start, t_end = t_min, t_max + 1.0
            span = t_end - t_start
            t_train = t_start + train_ratio * span
            t_val = t_start + (train_ratio + val_ratio) * span
    elif by == "count":
        ts = np.sort(t)
        n = len(ts)
        t_train = float(ts[min(int(train_ratio * n), n - 1)])
        t_val = float(ts[min(int((train_ratio + val_ratio) * n), n - 1)])
        if snap_to_day:
            t_train = np.floor(t_train / SECONDS_PER_DAY) * SECONDS_PER_DAY
            t_val = np.floor(t_val / SECONDS_PER_DAY) * SECONDS_PER_DAY
        t_start, t_end = t_min, t_max + 1.0
    else:
        raise ValueError(f"Unknown split mode '{by}' (use 'time' or 'count').")

    if not (t_start < t_train < t_val < t_end):
        raise ValueError(
            f"Degenerate boundaries: start={t_start}, train={t_train}, val={t_val}, end={t_end}."
        )
    return SplitBoundaries(float(t_start), float(t_train), float(t_val), float(t_end))


def temporal_split(
    df: pd.DataFrame,
    train_ratio: float = 0.65,
    val_ratio: float = 0.16,
    test_ratio: float = 0.19,
    time_col: str = TIME_COL,
    by: SplitBy = "time",
    snap_to_day: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split ``df`` chronologically into ``(train_df, val_df, test_df)``.

    Each returned frame is sorted by time (stable, ties broken by
    ``TransactionID`` when present) with a fresh RangeIndex. Raises if any
    split is empty or if the time ranges overlap.
    """
    b = compute_split_boundaries(
        df, train_ratio, val_ratio, test_ratio, time_col=time_col, by=by, snap_to_day=snap_to_day
    )
    sort_cols = [time_col] + ([ID_COL] if ID_COL in df.columns else [])
    ordered = df.sort_values(sort_cols, kind="stable").reset_index(drop=True)
    t = ordered[time_col].to_numpy(dtype=np.float64)

    train_df = ordered.loc[t < b.t_train].reset_index(drop=True)
    val_df = ordered.loc[(t >= b.t_train) & (t < b.t_val)].reset_index(drop=True)
    test_df = ordered.loc[t >= b.t_val].reset_index(drop=True)

    for name, part in (("train", train_df), ("val", val_df), ("test", test_df)):
        if len(part) == 0:
            raise ValueError(f"The {name} split is empty; check ratios / time range.")
    assert len(train_df) + len(val_df) + len(test_df) == len(df)
    assert train_df[time_col].max() < val_df[time_col].min()
    assert val_df[time_col].max() < test_df[time_col].min()
    return train_df, val_df, test_df


def summarize_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    time_col: str = TIME_COL,
    target_col: str = TARGET_COL,
    amount_col: str = AMOUNT_COL,
) -> pd.DataFrame:
    """One row per split: size, share, day range, fraud count/rate, fraud amount."""
    total = len(train_df) + len(val_df) + len(test_df)
    rows = []
    for name, part in (("train", train_df), ("val", val_df), ("test", test_df)):
        fraud = part[target_col] == 1
        rows.append(
            {
                "split": name,
                "n_txn": len(part),
                "share_%": 100.0 * len(part) / total,
                "day_min": part[time_col].min() / SECONDS_PER_DAY,
                "day_max": part[time_col].max() / SECONDS_PER_DAY,
                "n_fraud": int(fraud.sum()),
                "fraud_rate_%": 100.0 * float(fraud.mean()),
                "fraud_amount": float(part.loc[fraud, amount_col].sum()) if amount_col in part else np.nan,
            }
        )
    return pd.DataFrame(rows).set_index("split")


__all__ = ["SplitBoundaries", "compute_split_boundaries", "temporal_split", "summarize_splits"]
