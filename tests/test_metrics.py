"""Round-6 tests: metrics with hand-computed values, CI and the Recovery Ratio gate."""
from __future__ import annotations

import math

import numpy as np
import pytest

from src.eval import metrics as M

Y = np.array([1, 0, 1, 0, 0])
S = np.array([0.9, 0.8, 0.7, 0.3, 0.1])
AMT = np.array([100.0, 5.0, 50.0, 1.0, 1.0])


def test_average_precision_and_roc_auc_by_hand():
    assert M.average_precision(Y, S) == pytest.approx(0.5 * 1.0 + 0.5 * (2.0 / 3.0))
    assert M.roc_auc(Y, S) == pytest.approx(5.0 / 6.0)


def test_ties_give_chance_level():
    y, s = np.array([1, 0]), np.array([0.5, 0.5])
    assert M.roc_auc(y, s) == pytest.approx(0.5)
    assert M.average_precision(y, s) == pytest.approx(0.5)


def test_threshold_and_recall_at_fpr():
    assert M.threshold_at_fpr(Y, S, 0.01) == pytest.approx(0.9)     # budget 0 FP
    assert M.recall_at_fpr(Y, S, 0.01) == pytest.approx(0.5)
    assert M.threshold_at_fpr(Y, S, 0.34) == pytest.approx(0.7)     # budget 1 FP of 3
    assert M.recall_at_fpr(Y, S, 0.34) == pytest.approx(1.0)
    assert M.threshold_at_fpr(np.array([0, 1]), np.array([0.9, 0.1]), 0.01) == math.inf
    assert M.recall_at_fpr(np.array([0, 1]), np.array([0.9, 0.1]), 0.01) == 0.0


def test_operating_point_and_efl():
    op = M.operating_point(Y, S, AMT, threshold=0.9)
    assert op["missed_fraud_amount"] == 50.0 and op["efl"] == 50.0
    assert op["efl_ratio"] == pytest.approx(50.0 / 150.0)
    op2 = M.operating_point(Y, S, AMT, threshold=0.8, review_cost=2.0)   # flags rows 0 and 1 -> 1 FP
    assert op2["fp"] == 1.0 and op2["efl"] == pytest.approx(50.0 + 2.0)
    assert M.operating_point(Y, S, AMT, threshold=math.inf)["efl_ratio"] == pytest.approx(1.0)  # flag nothing


def test_summarize_scores_matches_the_single_metrics():
    m = M.summarize_scores(Y, S, fpr=0.34)
    assert m["pr_auc"] == pytest.approx(M.average_precision(Y, S))
    assert m["roc_auc"] == pytest.approx(M.roc_auc(Y, S))
    assert m["recall_at_fpr"] == pytest.approx(1.0) and m["prevalence"] == pytest.approx(0.4)


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        M.roc_auc(np.ones(4), np.arange(4.0))                  # one class only
    with pytest.raises(ValueError):
        M.roc_auc(np.array([0, 1]), np.array([np.nan, 1.0]))   # NaN score


def test_ci95_three_seeds():
    mean, lo, hi = M.ci95([0.1, 0.2, 0.3])
    assert mean == pytest.approx(0.2)
    assert hi - mean == pytest.approx(4.303 * 0.1 / math.sqrt(3))
    assert lo == pytest.approx(mean - (hi - mean))
    with pytest.raises(ValueError):
        M.ci95([0.5])


def test_counts_as_improvement_needs_the_favourable_side():
    naive = [0.29, 0.30, 0.31]                                  # CI ~ 0.30 +- 0.0248
    assert M.counts_as_improvement([0.34, 0.35, 0.36], naive) is True
    assert M.counts_as_improvement([0.31, 0.31, 0.31], naive) is False    # inside the CI
    assert M.counts_as_improvement([0.20, 0.21, 0.22], naive) is False    # outside but WORSE
    assert M.counts_as_improvement([0.20, 0.21, 0.22], naive, higher_is_better=False) is True


def test_recovery_ratio_is_reported_only_if_the_oracle_gap_exceeds_the_naive_ci():
    naive, oracle, method = [0.29, 0.30, 0.31], [0.39, 0.40, 0.41], [0.34, 0.35, 0.36]
    r = M.recovery_ratio(method, naive, oracle)
    assert r.reported and r.value == pytest.approx(0.5)
    weak_oracle = [0.30, 0.31, 0.32]                                      # mean 0.31 is inside A0's CI
    r2 = M.recovery_ratio(method, naive, weak_oracle)
    assert (not r2.reported) and r2.value is None and "CI" in r2.reason


def test_recovery_ratio_for_a_lower_is_better_score():
    naive, oracle, method = [99.0, 100.0, 101.0], [89.0, 90.0, 91.0], [94.0, 95.0, 96.0]
    r = M.recovery_ratio(method, naive, oracle, higher_is_better=False)
    assert r.reported and r.value == pytest.approx((100 - 95) / (100 - 90))