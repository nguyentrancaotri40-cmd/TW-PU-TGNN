"""Evaluation metrics, numpy only (no scikit-learn): PR-AUC, ROC-AUC, Recall@FPR, EFL, 95% CI, Recovery Ratio.

Conventions
-----------
* ``scores``: any monotone score (the trainer passes LOGITS, avoiding sigmoid saturation ties).
* PR-AUC = average precision (sum over distinct thresholds of (R_k - R_{k-1}) * P_k, scikit-learn's definition).
* ROC-AUC = Mann-Whitney U with average ranks for ties.
* ``threshold_at_fpr``: the lowest threshold with FP <= floor(fpr * n_neg); a row is flagged if ``score >= thr``.
  It is meant to be fitted on VALIDATION and applied to TEST (never choose a threshold on test labels).
* EFL (Expected Financial Loss) at a given threshold = sum(amount of missed frauds) + review_cost * #FP.
  KHÔNG CHẮC: the brief does not define EFL, this definition is mine and must be frozen before the benchmark.
  ``review_cost = 0`` means no explicit false-positive cost (the FPR budget plays that role).
  ``efl_ratio`` = EFL / total fraud amount (1.0 = the "flag nothing" model when review_cost = 0).
* ``recovery_ratio`` is reported ONLY when (oracle - naive) is outside the 95% CI of the naive baseline A0.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

# two-sided 95% Student-t critical values by degrees of freedom; df > 10 falls back to 1.96 (slightly
# anti-conservative, irrelevant for the 3 seeds of the benchmark)
_T975: dict[int, float] = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
                           6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


# --------------------------------------------------------------------------- #
# Validation and curve primitives
# --------------------------------------------------------------------------- #
def _prepare(y_true: Any, scores: Any) -> tuple[np.ndarray, np.ndarray, int, int]:
    y = np.asarray(y_true).reshape(-1)
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    if y.size == 0 or y.size != s.size:
        raise ValueError("y_true and scores must be non-empty and of equal length.")
    if not np.isfinite(s).all():
        raise ValueError("scores contain NaN/inf.")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("y_true must contain only 0/1.")
    y = y.astype(np.int64)
    n_pos = int(y.sum())
    n_neg = int(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        raise ValueError(f"Both classes are required (positives={n_pos}, negatives={n_neg}).")
    return y, s, n_pos, n_neg


def _cum_counts(y: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cumulative TP / FP and the threshold at each DISTINCT score, in descending score order."""
    order = np.argsort(-s, kind="mergesort")
    s_s, y_s = s[order], y[order]
    tp = np.cumsum(y_s)
    fp = np.cumsum(1 - y_s)
    last = np.r_[np.flatnonzero(np.diff(s_s) != 0), s_s.size - 1]  # last row of every tie group
    return tp[last].astype(np.float64), fp[last].astype(np.float64), s_s[last]


def _roc_auc(y: np.ndarray, s: np.ndarray, n_pos: int, n_neg: int) -> float:
    order = np.argsort(s, kind="mergesort")
    _, inv, counts = np.unique(s[order], return_inverse=True, return_counts=True)
    cum = np.cumsum(counts)
    avg_rank = (cum - counts + cum + 1) / 2.0  # mean 1-based rank of each tie group
    ranks = np.empty(s.size, dtype=np.float64)
    ranks[order] = avg_rank[inv]
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _average_precision(y: np.ndarray, s: np.ndarray, n_pos: int) -> float:
    tp, fp, _ = _cum_counts(y, s)
    precision = tp / (tp + fp)
    recall = tp / n_pos
    recall_prev = np.r_[0.0, recall[:-1]]
    return float(np.sum((recall - recall_prev) * precision))


def _threshold_at_fpr(y: np.ndarray, s: np.ndarray, n_neg: int, fpr: float) -> float:
    tp, fp, thr = _cum_counts(y, s)
    allowed = math.floor(fpr * n_neg + 1e-9)  # FP budget
    ok = np.flatnonzero(fp <= allowed)  # fp never decreases -> ok is a prefix
    return float(thr[ok[-1]]) if ok.size else float("inf")  # inf = flag nothing


# --------------------------------------------------------------------------- #
# Public metrics
# --------------------------------------------------------------------------- #
def roc_auc(y_true: Any, scores: Any) -> float:
    y, s, n_pos, n_neg = _prepare(y_true, scores)
    return _roc_auc(y, s, n_pos, n_neg)


def average_precision(y_true: Any, scores: Any) -> float:
    """PR-AUC as average precision."""
    y, s, n_pos, _ = _prepare(y_true, scores)
    return _average_precision(y, s, n_pos)


def threshold_at_fpr(y_true: Any, scores: Any, fpr: float = 0.01) -> float:
    """Lowest threshold whose false-positive rate is <= ``fpr`` (``inf`` if even the top score exceeds it)."""
    y, s, _, n_neg = _prepare(y_true, scores)
    return _threshold_at_fpr(y, s, n_neg, fpr)


def recall_at_fpr(y_true: Any, scores: Any, fpr: float = 0.01) -> float:
    """Recall at the threshold of :func:`threshold_at_fpr`, on the SAME data (own-curve recall)."""
    y, s, n_pos, n_neg = _prepare(y_true, scores)
    thr = _threshold_at_fpr(y, s, n_neg, fpr)
    return float(((s >= thr) & (y == 1)).sum() / n_pos)


def summarize_scores(y_true: Any, scores: Any, *, fpr: float = 0.01) -> dict[str, float]:
    """PR-AUC, ROC-AUC and Recall@FPR (+ the threshold that realises it) in one pass."""
    y, s, n_pos, n_neg = _prepare(y_true, scores)
    thr = _threshold_at_fpr(y, s, n_neg, fpr)
    return {
        "pr_auc": _average_precision(y, s, n_pos),
        "roc_auc": _roc_auc(y, s, n_pos, n_neg),
        "recall_at_fpr": float(((s >= thr) & (y == 1)).sum() / n_pos),
        "threshold_at_fpr": thr,
        "n_pos": float(n_pos),
        "n_neg": float(n_neg),
        "prevalence": n_pos / (n_pos + n_neg),
    }


def operating_point(
    y_true: Any, scores: Any, amount: Any, threshold: float, *, review_cost: float = 0.0
) -> dict[str, float]:
    """Confusion counts, recall/FPR and EFL when every row with ``score >= threshold`` is flagged."""
    y, s, n_pos, n_neg = _prepare(y_true, scores)
    a = np.asarray(amount, dtype=np.float64).reshape(-1)
    if a.size != y.size:
        raise ValueError("amount must have the same length as y_true.")
    if not np.isfinite(a).all() or (a < 0).any():
        raise ValueError("amount must be finite and non-negative.")
    flagged = s >= threshold
    tp = int((flagged & (y == 1)).sum())
    fp = int((flagged & (y == 0)).sum())
    total_fraud = float(a[y == 1].sum())
    missed = float(a[(y == 1) & ~flagged].sum())
    efl = missed + float(review_cost) * fp
    return {
        "threshold": float(threshold),
        "tp": float(tp), "fp": float(fp), "fn": float(n_pos - tp),
        "recall": tp / n_pos,
        "fpr": fp / n_neg,
        "precision": tp / (tp + fp) if (tp + fp) else float("nan"),
        "missed_fraud_amount": missed,
        "total_fraud_amount": total_fraud,
        "efl": efl,
        "efl_ratio": efl / total_fraud if total_fraud > 0 else float("nan"),
    }


# --------------------------------------------------------------------------- #
# Confidence interval and Recovery Ratio (pre-registered rules)
# --------------------------------------------------------------------------- #
def ci95(values: Sequence[float]) -> tuple[float, float, float]:
    """``(mean, low, high)`` of a Student-t 95% CI over seeds (needs >= 2 values)."""
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    if v.size < 2:
        raise ValueError("A CI needs at least 2 values (the benchmark uses 3 seeds).")
    mean, sd = float(v.mean()), float(v.std(ddof=1))
    half = _T975.get(v.size - 1, 1.96) * sd / math.sqrt(v.size)
    return mean, mean - half, mean + half


def ci_position(value: float, naive_values: Sequence[float]) -> str:
    """Where ``value`` falls relative to the naive baseline's 95% CI: ``above`` / ``below`` / ``inside``."""
    _, lo, hi = ci95(naive_values)
    return "above" if value > hi else "below" if value < lo else "inside"


def counts_as_improvement(
    method_values: Sequence[float], naive_values: Sequence[float], higher_is_better: bool = True
) -> bool:
    """Pre-registered rule: the method's seed-mean must lie outside A0's 95% CI, on the favourable side."""
    m = float(np.mean(method_values))
    pos = ci_position(m, naive_values)
    return pos == "above" if higher_is_better else pos == "below"


@dataclass(frozen=True)
class RecoveryResult:
    reported: bool
    value: float | None  # not clipped: can be < 0 (worse than naive) or > 1 (better than oracle)
    reason: str
    method_mean: float
    naive_mean: float
    oracle_mean: float
    naive_ci: tuple[float, float]


def recovery_ratio(
    method_values: Sequence[float],
    naive_values: Sequence[float],
    oracle_values: Sequence[float],
    higher_is_better: bool = True,
) -> RecoveryResult:
    """RR = (S_method - S_naive) / (S_oracle - S_naive), sign-flipped for lower-is-better scores (EFL).

    Reported only if the oracle is outside A0's 95% CI on the favourable side; otherwise the denominator
    is indistinguishable from seed noise and the ratio is unstable.
    """
    m, n, o = (float(np.mean(v)) for v in (method_values, naive_values, oracle_values))
    _, lo, hi = ci95(naive_values)
    if not counts_as_improvement(oracle_values, naive_values, higher_is_better):
        return RecoveryResult(False, None, "oracle - naive gap lies inside the 95% CI of A0", m, n, o, (lo, hi))
    sgn = 1.0 if higher_is_better else -1.0
    return RecoveryResult(True, sgn * (m - n) / (sgn * (o - n)), "ok", m, n, o, (lo, hi))


__all__ = [
    "roc_auc", "average_precision", "threshold_at_fpr", "recall_at_fpr", "summarize_scores",
    "operating_point", "ci95", "ci_position", "counts_as_improvement", "RecoveryResult", "recovery_ratio",
]