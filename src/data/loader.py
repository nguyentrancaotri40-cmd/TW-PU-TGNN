"""Data loading for the IEEE-CIS Fraud Detection dataset.

Leakage-aware design rules
--------------------------
* **No global imputation.** Numeric NaN are kept as NaN. Filling them with a
  mean/median computed on the whole file would leak Val/Test statistics into
  Train; fit imputers on the training split only (later rounds). Categorical
  NaN can safely be mapped to a constant token (``fill_categorical_missing``)
  because no statistic is involved.
* **No label-derived columns are ever created here.** The column names that
  later modules add (``detection_time``, ``y_obs``, ``pnu_status``) are
  registered below so that ``get_feature_columns`` can exclude them.
* ``TransactionDT`` is a *relative* offset in seconds from an undisclosed
  reference timestamp, not a calendar date. Day/week indices derived from it
  are therefore relative as well.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import pandas as pd

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Constants shared by every module in src/data (single source of truth)
# --------------------------------------------------------------------------- #
SECONDS_PER_DAY: int = 86_400
SECONDS_PER_WEEK: int = 7 * SECONDS_PER_DAY

TRANSACTION_FILE = "train_transaction.csv"
IDENTITY_FILE = "train_identity.csv"

ID_COL = "TransactionID"
TIME_COL = "TransactionDT"
AMOUNT_COL = "TransactionAmt"
TARGET_COL = "isFraud"

# Columns created by delay_simulator.py (all of them depend on the label).
DETECTION_COL = "detection_time"
Y_OBS_COL = "y_obs"
PNU_COL = "pnu_status"
IN_TRAIN_VIEW_COL = "in_train_view"

LABEL_DERIVED_COLS: tuple[str, ...] = (TARGET_COL, DETECTION_COL, Y_OBS_COL, PNU_COL)
# Never fed to a model: ids, raw time (non-stationary proxy), labels and their
# derivatives, and the train-view flag.
NON_FEATURE_COLS: tuple[str, ...] = (ID_COL, TIME_COL, IN_TRAIN_VIEW_COL) + LABEL_DERIVED_COLS

# String-valued columns of train_transaction.csv (everything else is numeric).
_TXN_CATEGORICAL: tuple[str, ...] = (
    "ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain",
) + tuple(f"M{i}" for i in range(1, 10))

MISSING_TOKEN = "__missing__"


class LeakageError(ValueError):
    """Raised when a label-derived column is about to be used as a feature."""


# --------------------------------------------------------------------------- #
# Time statistics
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TimeStats:
    """Min/max of ``TransactionDT`` in seconds, with day/week conversions."""

    min_seconds: float
    max_seconds: float

    @property
    def min_day(self) -> float:
        return self.min_seconds / SECONDS_PER_DAY

    @property
    def max_day(self) -> float:
        return self.max_seconds / SECONDS_PER_DAY

    @property
    def span_days(self) -> float:
        """Elapsed days between the first and the last transaction."""
        return self.max_day - self.min_day

    @property
    def n_calendar_days(self) -> int:
        """Number of (relative) day buckets touched by the data."""
        return int(self.max_seconds // SECONDS_PER_DAY - self.min_seconds // SECONDS_PER_DAY + 1)

    @property
    def n_weeks(self) -> int:
        """Number of (relative) 7-day buckets touched by the data."""
        return int(self.max_seconds // SECONDS_PER_WEEK - self.min_seconds // SECONDS_PER_WEEK + 1)

    def summary(self) -> str:
        return (
            f"{TIME_COL}: min={self.min_seconds:,.0f}s ({self.min_day:.2f} d), "
            f"max={self.max_seconds:,.0f}s ({self.max_day:.2f} d), "
            f"span={self.span_days:.2f} d, day buckets={self.n_calendar_days}, "
            f"week buckets={self.n_weeks}"
        )


def get_time_stats(df: pd.DataFrame, time_col: str = TIME_COL) -> TimeStats:
    """Return min/max of ``time_col`` (seconds) with day conversions."""
    if time_col not in df.columns:
        raise KeyError(f"Column '{time_col}' not found.")
    if len(df) == 0:
        raise ValueError("Empty DataFrame: cannot compute time statistics.")
    t = df[time_col]
    if t.isna().any():
        raise ValueError(f"'{time_col}' contains NaN; time axis must be complete.")
    return TimeStats(min_seconds=float(t.min()), max_seconds=float(t.max()))


# --------------------------------------------------------------------------- #
# CSV reading helpers
# --------------------------------------------------------------------------- #
def _transaction_dtypes(columns: Sequence[str]) -> dict[str, str]:
    """Memory-friendly dtypes for train_transaction.csv (avoids a float64 peak)."""
    dtypes: dict[str, str] = {}
    for c in columns:
        if c == ID_COL:
            dtypes[c] = "int32"
        elif c == TARGET_COL:
            dtypes[c] = "int8"
        elif c == TIME_COL:
            dtypes[c] = "int64"
        elif c == AMOUNT_COL:
            dtypes[c] = "float64"  # keep full precision for financial sums (EFL)
        elif c in _TXN_CATEGORICAL:
            dtypes[c] = "category"
        else:
            dtypes[c] = "float32"
    return dtypes


def _downcast(df: pd.DataFrame) -> pd.DataFrame:
    """float64 -> float32 and text -> category (amount/time columns untouched)."""
    for c in list(df.columns):
        if c in (AMOUNT_COL, TIME_COL):
            continue
        s = df[c]
        if pd.api.types.is_float_dtype(s):
            df[c] = s.astype("float32")
        elif pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            df[c] = s.astype("category")
    return df


def _resolve_columns(path: Path, usecols: Iterable[str] | None) -> list[str]:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    if usecols is None:
        return header
    wanted = set(usecols)
    missing = sorted(wanted - set(header))
    if missing:
        raise KeyError(f"Columns not found in {path.name}: {missing[:10]}")
    return [c for c in header if c in wanted]  # keep file order


def _read_csv_typed(
    path: Path,
    usecols: Iterable[str] | None,
    nrows: int | None,
    dtype_fn: Callable[[Sequence[str]], dict[str, str]] | None,
) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    cols = _resolve_columns(path, usecols)
    if dtype_fn is not None:
        try:
            return pd.read_csv(path, usecols=cols, nrows=nrows, dtype=dtype_fn(cols))
        except (ValueError, TypeError) as exc:  # unexpected string in a "numeric" column
            logger.warning("Typed read of %s failed (%s); falling back to inference.", path.name, exc)
    return _downcast(pd.read_csv(path, usecols=cols, nrows=nrows))


# --------------------------------------------------------------------------- #
# Public loader
# --------------------------------------------------------------------------- #
class DataLoader:
    """Loads IEEE-CIS ``train_transaction.csv`` and ``train_identity.csv``.

    Only the *train* files carry ``isFraud``; the Kaggle test files have no
    labels and must not be used for evaluation.
    """

    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir)

    @property
    def transaction_path(self) -> Path:
        return self.data_dir / TRANSACTION_FILE

    @property
    def identity_path(self) -> Path:
        return self.data_dir / IDENTITY_FILE

    def load_transactions(
        self, usecols: Iterable[str] | None = None, nrows: int | None = None
    ) -> pd.DataFrame:
        """Load ``train_transaction.csv`` (optionally a column subset / first rows)."""
        df = _read_csv_typed(self.transaction_path, usecols, nrows, _transaction_dtypes)
        if ID_COL in df.columns and df[ID_COL].duplicated().any():
            raise ValueError(f"Duplicate {ID_COL} values in {self.transaction_path.name}.")
        if TIME_COL in df.columns and df[TIME_COL].isna().any():
            raise ValueError(f"NaN in {TIME_COL}.")
        return df

    def load_identity(
        self, usecols: Iterable[str] | None = None, nrows: int | None = None
    ) -> pd.DataFrame:
        """Load ``train_identity.csv`` (covers only a subset of transactions)."""
        return _read_csv_typed(self.identity_path, usecols, nrows, None)

    def load_merged(
        self,
        transaction_cols: Iterable[str] | None = None,
        identity_cols: Iterable[str] | None = None,
        nrows: int | None = None,
    ) -> pd.DataFrame:
        """Left-join identity onto transactions and add ``has_identity`` (0/1).

        ``has_identity`` is known at transaction time, so it is a legitimate
        feature (it is *not* derived from the label).
        """
        txn = self.load_transactions(usecols=transaction_cols, nrows=nrows)
        if identity_cols is not None:
            identity_cols = list(dict.fromkeys([ID_COL, *identity_cols]))
        ident = self.load_identity(usecols=identity_cols)
        if ident[ID_COL].duplicated().any():
            logger.warning("Duplicate %s in identity table; keeping first occurrence.", ID_COL)
            ident = ident.drop_duplicates(subset=ID_COL, keep="first")
        merged = txn.merge(ident, on=ID_COL, how="left", validate="many_to_one")
        merged["has_identity"] = merged[ID_COL].isin(ident[ID_COL]).astype("int8")
        return merged


# --------------------------------------------------------------------------- #
# Missing values (leakage-safe only)
# --------------------------------------------------------------------------- #
def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column missing counts/percentages, sorted from most to least missing."""
    n = len(df)
    rep = pd.DataFrame({"n_missing": df.isna().sum()})
    rep["pct_missing"] = 100.0 * rep["n_missing"] / max(n, 1)
    return rep.sort_values("n_missing", ascending=False)


def fill_categorical_missing(df: pd.DataFrame, token: str = MISSING_TOKEN) -> pd.DataFrame:
    """Replace NaN in text/categorical columns by a constant token (returns a copy).

    Numeric NaN are intentionally left untouched (see module docstring).
    """
    out = df.copy()
    for c in out.columns:
        s = out[c]
        if isinstance(s.dtype, pd.CategoricalDtype):
            if token not in s.cat.categories:
                s = s.cat.add_categories([token])
            out[c] = s.fillna(token)
        elif pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            out[c] = s.fillna(token)
    return out


# --------------------------------------------------------------------------- #
# Feature-column guards (used by tests/test_leakage.py and prepare_data.py)
# --------------------------------------------------------------------------- #
def get_feature_columns(df: pd.DataFrame, exclude: Iterable[str] = ()) -> list[str]:
    """Columns allowed in the feature matrix ``X``.

    Excludes ids, raw ``TransactionDT``, the target and every label-derived
    column (``detection_time``, ``y_obs``, ``pnu_status``, ``in_train_view``).
    """
    banned = set(NON_FEATURE_COLS) | set(exclude)
    return [c for c in df.columns if c not in banned]


def assert_no_label_columns(columns: Iterable[str], forbid_raw_time: bool = True) -> None:
    """Raise ``LeakageError`` if ``columns`` contains a label-derived name.

    This is a *name* check (exact names + any name containing ``isfraud``).
    It cannot detect a target-encoded column with an innocent name; that needs
    a value-based audit (planned for a later round).
    """
    bad: list[str] = []
    for c in columns:
        low = str(c).lower()
        if c in LABEL_DERIVED_COLS or "isfraud" in low:
            bad.append(str(c))
        elif forbid_raw_time and c == TIME_COL:
            bad.append(str(c))
    if bad:
        raise LeakageError(f"Label-derived / forbidden columns in features: {sorted(set(bad))}")


__all__ = [
    "SECONDS_PER_DAY", "SECONDS_PER_WEEK", "ID_COL", "TIME_COL", "AMOUNT_COL", "TARGET_COL",
    "DETECTION_COL", "Y_OBS_COL", "PNU_COL", "IN_TRAIN_VIEW_COL", "LABEL_DERIVED_COLS",
    "NON_FEATURE_COLS", "LeakageError", "TimeStats", "get_time_stats", "DataLoader",
    "missing_report", "fill_categorical_missing", "get_feature_columns", "assert_no_label_columns",
]