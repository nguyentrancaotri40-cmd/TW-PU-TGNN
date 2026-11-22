"""Label-delay simulation and P/N/U masking.

Two steps, deliberately separated so the canary tests can attack each one:

1. ``simulate_delay``  -> adds ``detection_time`` = TransactionDT + delay.
2. ``assign_pnu_masks``-> given the cut ``t_train``, derives the *observable*
   label ``y_obs`` and the status P / N / U.

Anti-leakage invariants (checked in ``tests/test_leakage.py``)
---------------------------------------------------------------
I1. The delay of a transaction is drawn for **every** row (fraud or not) from
    a hash of (TransactionID, seed) plus, optionally, the amount. It never
    reads ``isFraud``, so ``detection_time`` carries no label information and
    does not depend on row order.
I2. ``y_obs = 1`` iff the row is fraud, lies before ``t_train`` and was
    detected by ``t_train``. Hence ``detection_time > t_train`` => ``y_obs = 0``.
I3. ``pnu_status`` is a function of (time, ``y_obs``, ``t_train``, window)
    only; it never reads ``isFraud`` directly. Otherwise "U vs N" would tell
    the model which old rows are hidden fraud.

Status definition (rows with TransactionDT < t_train form the *train view*)::

    P : y_obs == 1                                  (confirmed fraud)
    N : y_obs == 0 and TransactionDT + W <= t_train (mature, assumed legit)
    U : everything else (immature rows, and every row at/after t_train)

``W`` is the maturity window. If every delay is capped (``cap_days``) and
``W >= cap``, no mature fraud can be missing, so N is clean. With uncapped
delays, a fraud detected after ``W`` stays in N as label noise (realistic).

The delay laws and the amount dependence are *simulation assumptions* chosen
for the benchmark, not estimates from IEEE-CIS (which has no detection time).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, fields
from typing import Any, Literal, Mapping

import numpy as np
import pandas as pd

from src.data.loader import (
    AMOUNT_COL,
    DETECTION_COL,
    ID_COL,
    IN_TRAIN_VIEW_COL,
    PNU_COL,
    SECONDS_PER_DAY,
    TARGET_COL,
    TIME_COL,
    Y_OBS_COL,
)

logger = logging.getLogger(__name__)

DelayMode = Literal["fixed", "exponential", "weibull"]
PNU_CATEGORIES = ["P", "N", "U"]


@dataclass(frozen=True)
class DelayParams:
    """Parameters of the delay law (all durations in days).

    fixed        : every delay equals ``tau_days``.
    exponential  : Weibull with shape 1 and scale ``scale_days`` (mean ~ scale).
    weibull      : shape ``shape``, scale ``scale_days``.

    Amount dependence (exponential/weibull): the scale of row *i* is
    ``scale_days * (amount_i / amount_ref) ** (-amount_exponent)``.
    ``amount_exponent = 0`` (default) means amount-independent delay; a
    positive value makes large amounts detected faster (an assumption to be
    justified in the paper, not a fact about IEEE-CIS). ``amount_ref`` defaults
    to the median amount of the rows before ``t_train``.

    ``cap_days`` truncates the law (inverse-CDF on [0, F(cap)], no probability
    mass piled at the cap). ``None`` keeps the full heavy tail.
    """

    tau_days: float = 7.0
    scale_days: float = 7.0
    shape: float = 1.5
    amount_exponent: float = 0.0
    amount_ref: float | None = None
    cap_days: float | None = 30.0
    min_delay_hours: float = 1.0
    seed: int = 42

    @classmethod
    def from_dict(cls, params: Mapping[str, Any] | "DelayParams" | None) -> "DelayParams":
        if params is None:
            return cls()
        if isinstance(params, DelayParams):
            return params
        unknown = set(params) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown delay parameter(s): {sorted(unknown)}")
        return cls(**params)

    def validate(self, mode: str) -> None:
        if mode not in ("fixed", "exponential", "weibull"):
            raise ValueError(f"Unknown delay mode '{mode}'.")
        if mode == "fixed" and self.tau_days <= 0:
            raise ValueError("tau_days must be > 0.")
        if mode != "fixed" and (self.scale_days <= 0 or self.shape <= 0):
            raise ValueError("scale_days and shape must be > 0.")
        if self.amount_exponent < 0:
            raise ValueError("amount_exponent must be >= 0.")
        if self.min_delay_hours <= 0:
            raise ValueError("min_delay_hours must be > 0 (delays must be strictly positive).")
        if self.cap_days is not None and self.cap_days * 24.0 < self.min_delay_hours:
            raise ValueError("cap_days is shorter than min_delay_hours.")


# --------------------------------------------------------------------------- #
# Deterministic, order-independent uniform draws
# --------------------------------------------------------------------------- #
def _hash_uniform(ids: np.ndarray, seed: int) -> np.ndarray:
    """splitmix64-style hash of (id, seed) -> float64 in [0, 1).

    The draw of a transaction depends only on its id and the seed, so shuffling
    or re-splitting the DataFrame never changes its delay.
    """
    gamma = 0x9E3779B97F4A7C15
    offset = np.uint64(((int(seed) + 1) * gamma) & ((1 << 64) - 1))
    x = ids.astype(np.uint64) + offset
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    x = x ^ (x >> np.uint64(31))
    return (x >> np.uint64(11)).astype(np.float64) * (1.0 / float(1 << 53))


def simulate_delay(
    df: pd.DataFrame,
    t_train: float,
    mode: DelayMode = "weibull",
    params: Mapping[str, Any] | DelayParams | None = None,
    *,
    id_col: str = ID_COL,
    time_col: str = TIME_COL,
    amount_col: str = AMOUNT_COL,
) -> pd.DataFrame:
    """Return a copy of ``df`` with ``detection_time`` (seconds, TransactionDT scale).

    ``detection_time = TransactionDT + delay`` is the instant at which the
    transaction *would* be confirmed as fraud; it is generated for all rows
    and is independent of ``isFraud`` (invariant I1). ``t_train`` is used only
    to compute the default ``amount_ref`` from pre-cut rows; with
    ``amount_exponent = 0`` the draws do not depend on it at all.
    """
    p = DelayParams.from_dict(params)
    p.validate(mode)
    if not np.isfinite(t_train):
        raise ValueError("t_train must be finite.")
    if time_col not in df.columns:
        raise KeyError(f"Column '{time_col}' not found.")

    n = len(df)
    t = df[time_col].to_numpy(dtype=np.float64)
    min_delay_s = p.min_delay_hours * 3600.0

    if mode == "fixed":
        delay_s = np.full(n, p.tau_days * SECONDS_PER_DAY, dtype=np.float64)
    else:
        if id_col in df.columns:
            ids = df[id_col].to_numpy()
        else:
            logger.warning("'%s' missing: hashing row positions (order-dependent).", id_col)
            ids = np.arange(n)
        u = _hash_uniform(ids, p.seed)
        k = 1.0 if mode == "exponential" else p.shape

        scale = np.full(n, p.scale_days, dtype=np.float64)
        if p.amount_exponent > 0:
            if amount_col not in df.columns:
                raise KeyError(f"Column '{amount_col}' needed for amount-dependent delay.")
            amt = df[amount_col].to_numpy(dtype=np.float64)
            ref = p.amount_ref
            if ref is None:
                pre_cut = amt[t < t_train]
                pre_cut = pre_cut[np.isfinite(pre_cut)]
                if pre_cut.size == 0:
                    raise ValueError("No amounts before t_train to derive amount_ref.")
                ref = float(np.median(pre_cut))
            amt = np.clip(np.where(np.isfinite(amt), amt, ref), 1e-6, None)
            scale = scale * (amt / ref) ** (-p.amount_exponent)

        if p.cap_days is not None:
            f_cap = -np.expm1(-((p.cap_days / scale) ** k))  # F(cap) per row
            u = u * f_cap
        delay_days = scale * (-np.log1p(-u)) ** (1.0 / k)  # inverse Weibull CDF
        delay_s = delay_days * SECONDS_PER_DAY
        if p.cap_days is not None:
            delay_s = np.minimum(delay_s, p.cap_days * SECONDS_PER_DAY)

    delay_s = np.maximum(delay_s, min_delay_s)  # strictly positive delays
    out = df.copy()
    out[DETECTION_COL] = t + delay_s
    return out


def assign_pnu_masks(
    df: pd.DataFrame,
    t_train: float,
    maturity_window_days: float | None = None,
    *,
    time_col: str = TIME_COL,
    target_col: str = TARGET_COL,
    detection_col: str = DETECTION_COL,
) -> pd.DataFrame:
    """Add ``y_obs`` (0/1), ``pnu_status`` (P/N/U) and ``in_train_view``.

    ``maturity_window_days`` (W) defaults to ``max(detection_time - TransactionDT)``,
    i.e. the longest simulated delay; this is label-independent. In a pipeline
    pass it explicitly (from the config) so it does not vary with the subset.
    Rows at/after ``t_train`` get ``y_obs = 0`` and status ``U``; losses must
    restrict themselves to ``in_train_view``.
    """
    for c in (time_col, target_col, detection_col):
        if c not in df.columns:
            raise KeyError(f"Column '{c}' not found (run simulate_delay first?).")

    t = df[time_col].to_numpy(dtype=np.float64)
    det = df[detection_col].to_numpy(dtype=np.float64)
    y = df[target_col].to_numpy()

    in_view = t < t_train
    y_obs = ((y == 1) & in_view & (det <= t_train)).astype(np.int8)

    if maturity_window_days is None:
        window_s = float(np.max(det - t)) if len(df) else 0.0
    else:
        if maturity_window_days <= 0:
            raise ValueError("maturity_window_days must be > 0.")
        window_s = maturity_window_days * SECONDS_PER_DAY
    mature = in_view & (t + window_s <= t_train)

    # Status uses y_obs + time only (invariant I3), never the raw label.
    status = np.where(y_obs == 1, "P", np.where(mature, "N", "U"))

    out = df.copy()
    out[Y_OBS_COL] = y_obs
    out[PNU_COL] = pd.Categorical(status, categories=PNU_CATEGORIES)
    out[IN_TRAIN_VIEW_COL] = in_view
    return out


def pnu_diagnostics(df: pd.DataFrame, target_col: str = TARGET_COL) -> dict[str, float]:
    """Evaluator-only summary (reads ``isFraud``; never feed it to a model)."""
    view = df[df[IN_TRAIN_VIEW_COL]]
    fraud = view[target_col] == 1
    status = view[PNU_COL]
    n = max(len(view), 1)
    n_fraud = int(fraud.sum())
    hidden = int((fraud & (view[Y_OBS_COL] == 0)).sum())
    return {
        "n_train_view": len(view),
        "n_P": int((status == "P").sum()),
        "n_N": int((status == "N").sum()),
        "n_U": int((status == "U").sum()),
        "share_U_%": 100.0 * float((status == "U").sum()) / n,
        "n_fraud_true": n_fraud,
        "n_fraud_hidden": hidden,  # fraud present in the view but unobserved
        "hidden_fraud_%": 100.0 * hidden / max(n_fraud, 1),
        "n_fraud_in_N": int((fraud & (status == "N")).sum()),  # label noise in N
    }


__all__ = [
    "DelayMode", "DelayParams", "simulate_delay", "assign_pnu_masks", "pnu_diagnostics",
]
