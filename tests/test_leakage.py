"""Canary tests for data leakage (round 3: masking + feature columns).

Run from the repo root:  ``pytest -q``  (or ``python -m pytest -q``).

A canary test must be able to FAIL. Besides the happy path, several tests
below inject a deliberate leak and assert that the check catches it.

(a) Masking integrity    -> implemented
(b) Feature leakage      -> implemented
(c) Causal graph checks  -> empty skeleton, completed in round 4
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.delay_simulator import assign_pnu_masks, pnu_diagnostics, simulate_delay
from src.data.loader import (
    DETECTION_COL,
    ID_COL,
    IN_TRAIN_VIEW_COL,
    PNU_COL,
    SECONDS_PER_DAY,
    TARGET_COL,
    TIME_COL,
    Y_OBS_COL,
    LeakageError,
    assert_no_label_columns,
    get_feature_columns,
    get_time_stats,
)
from src.data.splitter import compute_split_boundaries, temporal_split

# --------------------------------------------------------------------------- #
# Dummy data (self-contained, mimics IEEE-CIS: relative seconds starting at day 1)
# --------------------------------------------------------------------------- #
N_DAYS = 60


def make_dummy_df(n_rows: int = 3000, n_days: int = N_DAYS, fraud_rate: float = 0.06, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.sort(rng.integers(SECONDS_PER_DAY, (n_days + 1) * SECONDS_PER_DAY, size=n_rows))
    df = pd.DataFrame(
        {
            ID_COL: np.arange(2_987_000, 2_987_000 + n_rows),
            TARGET_COL: (rng.random(n_rows) < fraud_rate).astype(np.int8),
            TIME_COL: t,
            "TransactionAmt": np.round(np.exp(rng.normal(4.0, 1.2, n_rows)), 2).clip(0.25),
            "ProductCD": rng.choice(list("WCRHS"), n_rows),
            "card1": rng.integers(1000, 2000, n_rows).astype(np.float32),
            "addr1": rng.integers(100, 500, n_rows).astype(np.float32),
            "V1": rng.normal(size=n_rows).astype(np.float32),
            "V2": rng.normal(size=n_rows).astype(np.float32),
            "V3": rng.normal(size=n_rows).astype(np.float32),
        }
    )
    df.loc[rng.random(n_rows) < 0.2, "V3"] = np.nan
    return df


@pytest.fixture(scope="module")
def dummy_df() -> pd.DataFrame:
    return make_dummy_df()


@pytest.fixture(scope="module")
def t_train(dummy_df: pd.DataFrame) -> float:
    return compute_split_boundaries(dummy_df).t_train


# name -> (mode, params, maturity_window_days passed to assign_pnu_masks)
DELAY_CONFIGS: dict[str, tuple[str, dict, float | None]] = {
    "fixed_7d": ("fixed", {"tau_days": 7}, None),
    "exp_7d_cap30": ("exponential", {"scale_days": 7, "cap_days": 30}, None),
    "weibull_amount": ("weibull", {"scale_days": 10, "shape": 1.3, "amount_exponent": 0.3, "cap_days": 30}, None),
    # Uncapped heavy tail with an explicit window W=30 d: some frauds are detected
    # AFTER the row is already "mature" (label noise inside N).
    "exp_uncapped_W30": ("exponential", {"scale_days": 10, "cap_days": None}, 30.0),
}


def build_masked(df: pd.DataFrame, t_train: float, name: str) -> pd.DataFrame:
    mode, params, window = DELAY_CONFIGS[name]
    with_delay = simulate_delay(df, t_train, mode=mode, params=params)
    return assign_pnu_masks(with_delay, t_train, maturity_window_days=window)


# --------------------------------------------------------------------------- #
# (a) Masking integrity
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", list(DELAY_CONFIGS))
def test_a_late_detection_means_y_obs_zero(dummy_df, t_train, name):
    """detection_time > t_train  =>  y_obs == 0 (fraud rows become U)."""
    m = build_masked(dummy_df, t_train, name)
    late = m[DETECTION_COL] > t_train
    late_fraud = late & (m[TARGET_COL] == 1)

    assert late_fraud.any(), "vacuous test: no late-detected fraud in dummy data"
    assert (m.loc[late, Y_OBS_COL] == 0).all()
    if DELAY_CONFIGS[name][1].get("cap_days", 1) is not None:  # capped laws: no mature hidden fraud
        assert (m.loc[late_fraud, PNU_COL] == "U").all()

    # converse: fraud detected in time, before the cut -> observed positive
    seen = (m[TARGET_COL] == 1) & m[IN_TRAIN_VIEW_COL] & (m[DETECTION_COL] <= t_train)
    assert seen.any()
    assert (m.loc[seen, Y_OBS_COL] == 1).all()
    assert (m.loc[seen, PNU_COL] == "P").all()

    # P <=> y_obs==1 ; N and U never carry a positive observed label
    assert ((m[PNU_COL] == "P") == (m[Y_OBS_COL] == 1)).all()
    assert (m.loc[m[PNU_COL] != "P", Y_OBS_COL] == 0).all()
    # rows at/after the cut are never observed
    assert (m.loc[~m[IN_TRAIN_VIEW_COL], Y_OBS_COL] == 0).all()
    assert (m.loc[~m[IN_TRAIN_VIEW_COL], PNU_COL] == "U").all()


def test_a_boundary_is_inclusive_at_t_train():
    t_cut = 10 * SECONDS_PER_DAY
    df = pd.DataFrame(
        {
            TIME_COL: [t_cut - 1000.0] * 3,
            TARGET_COL: [1, 1, 1],
            DETECTION_COL: [t_cut - 1.0, t_cut, t_cut + 1.0],
        }
    )
    m = assign_pnu_masks(df, t_cut, maturity_window_days=1.0)
    assert m[Y_OBS_COL].tolist() == [1, 1, 0]
    assert m[PNU_COL].astype(str).tolist() == ["P", "P", "U"]


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("name", list(DELAY_CONFIGS))
def test_a_permuting_hidden_labels_changes_nothing(dummy_df, t_train, name, seed):
    """Shuffle isFraud among all rows with detection_time > t_train (immature
    region + Val/Test): X, detection_time, y_obs and P/N/U must be identical."""
    base = build_masked(dummy_df, t_train, name)
    region = base[DETECTION_COL] > t_train

    permuted = dummy_df.copy()
    vals = permuted.loc[region, TARGET_COL].to_numpy().copy()
    np.random.default_rng(seed).shuffle(vals)
    permuted.loc[region, TARGET_COL] = vals
    assert (permuted[TARGET_COL] != dummy_df[TARGET_COL]).any(), "vacuous: permutation changed nothing"

    other = build_masked(permuted, t_train, name)
    cols = get_feature_columns(base)
    pd.testing.assert_frame_equal(base[cols], other[cols])
    pd.testing.assert_series_equal(base[DETECTION_COL], other[DETECTION_COL])
    pd.testing.assert_series_equal(base[Y_OBS_COL], other[Y_OBS_COL])
    pd.testing.assert_series_equal(base[PNU_COL], other[PNU_COL])


def test_a_status_is_a_function_of_time_not_of_label(dummy_df, t_train):
    """With uncapped delays some mature rows are undetected fraud. They must
    sit in N together with legit rows; if status read isFraud they would be U
    and the model could learn 'U-and-old = fraud'."""
    m = build_masked(dummy_df, t_train, "exp_uncapped_W30")
    window_s = 30.0 * SECONDS_PER_DAY
    expected_n = m[IN_TRAIN_VIEW_COL] & (m[TIME_COL] + window_s <= t_train) & (m[Y_OBS_COL] == 0)
    assert ((m[PNU_COL] == "N") == expected_n).all()
    hidden_in_n = (m[PNU_COL] == "N") & (m[DETECTION_COL] > t_train)
    assert hidden_in_n.any(), "vacuous: no long-delay rows in N"
    assert pnu_diagnostics(m)["n_fraud_in_N"] >= 0  # reported as label noise, not hidden


def test_a_checker_catches_a_naive_label_leak(dummy_df, t_train):
    """Negative control: y_obs = isFraud (no masking) must violate the invariant."""
    m = build_masked(dummy_df, t_train, "fixed_7d")
    leaky = m.copy()
    leaky[Y_OBS_COL] = leaky[TARGET_COL].astype(np.int8)
    late = leaky[DETECTION_COL] > t_train
    assert not (leaky.loc[late, Y_OBS_COL] == 0).all()


# -- delay generator sanity (supports (a)) ----------------------------------- #
def test_delay_is_label_independent_and_order_independent(dummy_df, t_train):
    params = {"scale_days": 8, "shape": 1.4, "amount_exponent": 0.2, "cap_days": 30}
    full = simulate_delay(dummy_df, t_train, mode="weibull", params=params).set_index(ID_COL)[DETECTION_COL]

    no_label = simulate_delay(dummy_df.drop(columns=[TARGET_COL]), t_train, mode="weibull", params=params)
    pd.testing.assert_series_equal(full, no_label.set_index(ID_COL)[DETECTION_COL])  # never reads isFraud

    shuffled = dummy_df.sample(frac=1.0, random_state=3)
    again = simulate_delay(shuffled, t_train, mode="weibull", params=params).set_index(ID_COL)[DETECTION_COL]
    pd.testing.assert_series_equal(full, again.loc[full.index])  # same delay per TransactionID


def test_delay_bounds_and_seed(dummy_df, t_train):
    d = simulate_delay(dummy_df, t_train, "exponential", {"scale_days": 7, "cap_days": 30, "min_delay_hours": 1})
    delay = d[DETECTION_COL] - d[TIME_COL]
    assert (delay >= 3600.0 - 1e-6).all() and (delay <= 30 * SECONDS_PER_DAY + 1e-6).all()

    fixed = simulate_delay(dummy_df, t_train, "fixed", {"tau_days": 7})
    assert np.allclose(fixed[DETECTION_COL] - fixed[TIME_COL], 7 * SECONDS_PER_DAY)

    other_seed = simulate_delay(dummy_df, t_train, "exponential", {"scale_days": 7, "cap_days": 30, "seed": 7})
    assert not np.allclose(d[DETECTION_COL], other_seed[DETECTION_COL])


def test_amount_exponent_makes_large_amounts_faster(dummy_df, t_train):
    d = simulate_delay(dummy_df, t_train, "weibull", {"scale_days": 10, "shape": 1.2, "amount_exponent": 0.5, "cap_days": 60})
    delay = d[DETECTION_COL] - d[TIME_COL]
    lo, hi = d["TransactionAmt"].quantile([1 / 3, 2 / 3])
    assert delay[d["TransactionAmt"] >= hi].median() < delay[d["TransactionAmt"] <= lo].median()


def test_delay_rejects_unknown_params(dummy_df, t_train):
    with pytest.raises(ValueError):
        simulate_delay(dummy_df, t_train, "weibull", {"scale": 3})


# --------------------------------------------------------------------------- #
# (b) Feature leakage
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def masked_df(dummy_df, t_train) -> pd.DataFrame:
    return build_masked(dummy_df, t_train, "weibull_amount")


def test_b_feature_matrix_has_no_isfraud(masked_df):
    cols = get_feature_columns(masked_df)
    X = masked_df[cols]
    assert TARGET_COL not in X.columns
    for banned in (Y_OBS_COL, PNU_COL, DETECTION_COL, IN_TRAIN_VIEW_COL, ID_COL, TIME_COL):
        assert banned not in X.columns, banned
    assert_no_label_columns(X.columns)  # must not raise
    assert {"TransactionAmt", "card1", "V1"} <= set(X.columns)  # real features survive

    # no feature is a verbatim copy of the label (numeric columns only)
    y = masked_df[TARGET_COL].to_numpy(dtype=float)
    for c in cols:
        if pd.api.types.is_numeric_dtype(X[c]):
            v = X[c].to_numpy(dtype=float)
            assert not np.array_equal(v, y, equal_nan=True), f"{c} copies isFraud"
            assert not np.array_equal(v, 1 - y, equal_nan=True), f"{c} copies 1-isFraud"


@pytest.mark.parametrize("bad", [TARGET_COL, "isFraud_lag1", Y_OBS_COL, PNU_COL, DETECTION_COL, TIME_COL])
def test_b_guard_detects_injected_leaks(masked_df, bad):
    cols = get_feature_columns(masked_df)
    with pytest.raises(LeakageError):
        assert_no_label_columns([*cols, bad])


# --------------------------------------------------------------------------- #
# (c) Causal graph integrity  --  EMPTY SKELETON (to be completed in round 4)
# --------------------------------------------------------------------------- #
def check_causal_graph_integrity(
    edge_index: np.ndarray,
    node_time: np.ndarray,
    node_split: np.ndarray,
) -> None:
    """Round 4. Planned checks (raise AssertionError on violation):

    1. Causality: for every edge (src -> dst), ``node_time[src] < node_time[dst]``
       (strict; PyG's temporal sampler allows equality, so ties need a rule).
    2. No Val/Test -> Train edge: ``node_split[src] <= node_split[dst]`` with
       order train < val < test, and the Train graph holds only Train nodes.
    3. (hook) sampler output: every sampled neighbour is no later than its seed.

    Args:
        edge_index: int array of shape (2, E) with rows (src, dst).
        node_time: float array (N,) with TransactionDT of each node.
        node_split: int array (N,) with 0=train, 1=val, 2=test.
    """
    raise NotImplementedError("Round 4: needs src/graph/builder.py and sampler.py")


@pytest.mark.skip(reason="Round 4: causal graph builder/sampler not written yet")
def test_c_causal_graph_integrity():
    check_causal_graph_integrity(
        edge_index=np.empty((2, 0), dtype=np.int64),
        node_time=np.empty(0),
        node_split=np.empty(0, dtype=np.int64),
    )


# --------------------------------------------------------------------------- #
# Supporting modules: splitter and loader time stats
# --------------------------------------------------------------------------- #
def test_temporal_split_is_disjoint_ordered_and_close_to_ratios(dummy_df):
    train, val, test = temporal_split(dummy_df)
    assert len(train) + len(val) + len(test) == len(dummy_df)
    assert train[TIME_COL].max() < val[TIME_COL].min()
    assert val[TIME_COL].max() < test[TIME_COL].min()
    assert set(train[ID_COL]).isdisjoint(val[ID_COL]) and set(val[ID_COL]).isdisjoint(test[ID_COL])

    b = compute_split_boundaries(dummy_df)
    span = b.t_end - b.t_start
    tol = 1.0 / N_DAYS + 1e-9  # one day of rounding
    assert abs((b.t_train - b.t_start) / span - 0.65) <= tol
    assert abs((b.t_val - b.t_train) / span - 0.16) <= tol
    assert (b.t_train % SECONDS_PER_DAY == 0) and (b.t_val % SECONDS_PER_DAY == 0)


def test_temporal_split_rejects_bad_ratios(dummy_df):
    with pytest.raises(ValueError):
        temporal_split(dummy_df, 0.7, 0.2, 0.2)


def test_time_stats_in_days(dummy_df):
    ts = get_time_stats(dummy_df)
    assert ts.min_day == pytest.approx(dummy_df[TIME_COL].min() / SECONDS_PER_DAY)
    assert ts.max_day == pytest.approx(dummy_df[TIME_COL].max() / SECONDS_PER_DAY)
    assert 1.0 <= ts.min_day < 1.5 and ts.max_day <= N_DAYS + 1


import torch
import torch.nn.functional as F

from src.losses import (
    DelayIPWLoss, MaturityMaskLoss, NaiveBCELoss, TimeWindowNNDFLoss, PNU_N, PNU_P, PNU_U,
)


def _toy():
    """6 rows: 2 P, 2 N, 2 U (U rows have y_obs = 0)."""
    pnu = torch.tensor([PNU_P, PNU_P, PNU_N, PNU_N, PNU_U, PNU_U])
    y = torch.tensor([1, 1, 0, 0, 0, 0])
    age = torch.tensor([40.0, 35.0, 50.0, 31.0, 2.0, 5.0])
    logits = torch.tensor([0.3, -0.2, 0.1, -0.5, 0.7, -1.0])
    return logits, y, pnu, age


def test_mask_loss_ignores_u_rows():
    logits, y, pnu, _ = _toy()
    base = MaturityMaskLoss()(logits, y, pnu)
    l2 = logits.clone()
    l2[pnu == PNU_U] += 5.0                                  # change ONLY the U logits
    assert torch.equal(base, MaturityMaskLoss()(l2, y, pnu))
    m = pnu != PNU_U
    assert torch.allclose(base, F.binary_cross_entropy_with_logits(logits[m], y[m].float()))


def test_mask_loss_gradient_is_zero_on_u_rows():
    logits, y, pnu, _ = _toy()
    logits.requires_grad_(True)
    MaturityMaskLoss()(logits, y, pnu).backward()
    assert (logits.grad[pnu == PNU_U] == 0).all()
    assert (logits.grad[pnu != PNU_U] != 0).all()


def test_naive_bce_counts_u_as_negative_and_respects_view():
    logits, y, pnu, _ = _toy()
    assert torch.allclose(NaiveBCELoss()(logits, y, pnu),
                          F.binary_cross_entropy_with_logits(logits, y.float()))
    view = torch.tensor([True] * 5 + [False])
    assert torch.allclose(NaiveBCELoss()(logits, y, pnu, in_train_view=view),
                          F.binary_cross_entropy_with_logits(logits[:5], y[:5].float()))


def test_ipw_weights_and_normalisation():
    logits, y, pnu, age = _toy()
    lam = 0.1
    loss = DelayIPWLoss(lam=lam)(logits, y, pnu, age_days=age)
    w = torch.where(pnu == PNU_P, torch.ones(6), 1 - torch.exp(-lam * age))
    bce = F.binary_cross_entropy_with_logits(logits, y.float(), reduction="none")
    assert torch.allclose(loss, (w * bce).sum() / w.sum())
    assert w[4] < w[2]                                       # young U row weighs less than old N row


def test_nndf_matches_formula():
    logits, y, pnu, _ = _toy()
    pi = 0.2
    loss = TimeWindowNNDFLoss(prior=pi)(logits, y, pnu)
    P, U = logits[pnu == PNU_P], logits[pnu == PNU_U]
    pos = pi * F.softplus(-P).mean()
    neg = F.softplus(U).mean() - pi * F.softplus(P).mean()
    assert torch.allclose(loss, pos + torch.clamp(neg, min=0))


def test_nndf_clamps_negative_risk_and_stops_u_gradient():
    z = torch.tensor([5.0, 5.0, -5.0, -5.0])
    y = torch.tensor([1, 1, 0, 0])
    pnu = torch.tensor([PNU_P, PNU_P, PNU_U, PNU_U])
    r = TimeWindowNNDFLoss(prior=0.5).risks(z, y, pnu)
    assert r["neg_risk_raw"] < 0 and r["neg_risk"] == 0
    z.requires_grad_(True)
    TimeWindowNNDFLoss(prior=0.5)(z, y, pnu).backward()
    assert (z.grad[2:] == 0).all()                           # U rows get no gradient once clamped


def test_nndf_without_positives_is_zero_but_differentiable():
    z = torch.tensor([0.1, 0.2], requires_grad=True)
    out = TimeWindowNNDFLoss(prior=0.1)(z, torch.zeros(2), torch.tensor([PNU_U, PNU_U]))
    assert out.item() == 0.0
    out.backward()                                           # must not raise