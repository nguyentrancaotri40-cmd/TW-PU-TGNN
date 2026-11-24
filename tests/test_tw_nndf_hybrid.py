import inspect
import math

import pytest
import torch
import torch.nn.functional as F

from src.losses import PNU_N, PNU_P, PNU_U
from src.losses.calibration import (
    MatureTemperatureTracker, SlidingWindowPrior, TemperatureScaler, TemperatureScaling,
)
from src.losses.maturity_mask import (
    MaturityMaskedBCELoss, MaturityMaskLoss, trusted_negative_mask,
)
from src.losses.tw_nndf import HybridTWnnDFLoss, TimeWindowNNDFLoss

KEYS = ("pos_risk", "neg_risk_raw", "clamped", "ascent")


def make(n_p=4, n_n=8, n_u=6, seed=0, p_logit=None, u_logit=None):
    g = torch.Generator().manual_seed(seed)
    pnu = torch.tensor([PNU_P] * n_p + [PNU_N] * n_n + [PNU_U] * n_u)
    f = torch.randn(len(pnu), generator=g)
    if p_logit is not None:
        f[pnu == PNU_P] = p_logit
        f[pnu == PNU_U] = u_logit
    y = (pnu == PNU_P).float()
    amount = torch.full((len(pnu),), 10.0)
    age = torch.where(pnu == PNU_U, torch.tensor(5.0), torch.tensor(40.0))
    view = torch.ones(len(pnu), dtype=torch.bool)
    return f, y, pnu, age, view, amount


def run(loss, f, y, pnu, age, view, amount):
    return loss(f, y, pnu, age_days=age, in_train_view=view, amount=amount)


# ------------------------------------------------------------------ BCE + nnPU structure ----------------------
def test_n_rows_get_bce_gradient_in_hybrid_but_not_in_pu_only():
    f, y, pnu, age, view, amt = make()
    f = f.clone().requires_grad_(True)
    run(HybridTWnnDFLoss(0.1), f, y, pnu, age, view, amt).backward()
    assert (f.grad[pnu == PNU_N].abs() > 0).all()
    f2 = make()[0].clone().requires_grad_(True)
    TimeWindowNNDFLoss(0.1, mode="clamp")(f2, y, pnu, age_days=age, in_train_view=view).backward()
    assert torch.all(f2.grad[pnu == PNU_N] == 0)


def test_trusted_mask_requires_amount_and_min_age():
    pnu = torch.tensor([PNU_U, PNU_U, PNU_U, PNU_N, PNU_P])
    amt = torch.tensor([500.0, 500.0, 5.0, 500.0, 500.0])
    age = torch.tensor([20.0, 1.0, 20.0, 20.0, 20.0])
    assert trusted_negative_mask(pnu, amt, age, 100.0, 10.0).tolist() == [True, False, False, False, False]
    assert trusted_negative_mask(pnu, amt, age, 100.0, 0.0).tolist() == [True, True, False, False, False]
    cleared = torch.tensor([False, True, True, True, True])
    assert trusted_negative_mask(pnu, amt, age, 100.0, 10.0, cleared).tolist()[0] is False
    with pytest.raises(ValueError):
        trusted_negative_mask(pnu, amt, None, 100.0, 10.0)


def test_trusted_u_rows_enter_bce_and_leave_u():
    f, y, pnu, age, view, amt = make(p_logit=5.0, u_logit=-5.0)  # neg_raw < 0 -> clamp kills the PU gradient on U
    u_idx = (pnu == PNU_U).nonzero().squeeze(1)
    amt = amt.clone()
    amt[u_idx] = 500.0
    age = age.clone()
    age[u_idx[:3]] = 20.0  # old and large -> trusted ; the other 3 stay young -> remain U
    f = f.clone().requires_grad_(True)
    loss = HybridTWnnDFLoss(0.1, mode="clamp", amount_threshold=100.0, amount_min_age_days=10.0)
    run(loss, f, y, pnu, age, view, amt).backward()
    assert (f.grad[u_idx[:3]].abs() > 0).all()  # trusted: BCE target 0
    assert torch.all(f.grad[u_idx[3:]] == 0)  # still U: clamped PU term, no BCE


def test_ascent_restores_u_gradient_and_clamp_does_not():
    f, y, pnu, age, view, amt = make(p_logit=5.0, u_logit=-5.0)
    for mode, expect in (("nnpu", True), ("clamp", False)):
        ff = f.clone().requires_grad_(True)
        loss = HybridTWnnDFLoss(0.1, mode=mode)
        run(loss, ff, y, pnu, age, view, amt).backward()
        assert float(loss.last_risks["clamped"]) == 1.0
        assert float(loss.last_risks["ascent"]) == (1.0 if expect else 0.0)
        assert bool((ff.grad[pnu == PNU_U].abs() > 0).all()) is expect


def test_lambda_pu_zero_equals_plain_bce_on_p_and_n():
    f, y, pnu, age, view, amt = make()
    out = run(HybridTWnnDFLoss(0.1, lambda_pu=0.0), f, y, pnu, age, view, amt)
    keep = pnu != PNU_U
    assert torch.allclose(out, F.binary_cross_entropy_with_logits(f[keep], y[keep]))


def test_cost_sensitive_weights_change_the_pu_term_only_when_enabled():
    f, y, pnu, age, view, amt = make()
    amt2 = amt.clone()
    amt2[pnu == PNU_U] = torch.tensor([1.0, 5.0, 20.0, 40.0, 3.0, 9.0])
    off = HybridTWnnDFLoss(0.1)
    assert torch.allclose(run(off, f, y, pnu, age, view, amt), run(off, f, y, pnu, age, view, amt2))
    on = HybridTWnnDFLoss(0.1, amount_weighted=True, amount_ref=10.0)
    assert not torch.allclose(run(on, f, y, pnu, age, view, amt), run(on, f, y, pnu, age, view, amt2))
    assert on.needs_amount and not off.needs_amount
    with pytest.raises(ValueError):
        run(on, f, y, pnu, age, view, None)


# ------------------------------------------------------------------ sliding_prior (FIX 1) ---------------------
def _table():
    return SlidingWindowPrior(torch.tensor([0.05, 0.2]), 0.1, 10.0)  # age 5 -> bin 0 ; age 15 -> bin 1


def test_hybrid_accepts_sliding_prior_and_uses_it_for_the_batch_prior():
    f, y, pnu, age, view, amt = make()  # remaining U rows have age 5.0 -> pi = 0.05
    dyn = HybridTWnnDFLoss(0.5, sliding_prior=_table())
    run(dyn, f, y, pnu, age, view, amt)
    assert float(dyn.last_risks["prior_used"]) == pytest.approx(0.05)
    fixed = HybridTWnnDFLoss(0.5)
    run(fixed, f, y, pnu, age, view, amt)
    assert float(fixed.last_risks["prior_used"]) == pytest.approx(0.5)


def test_sliding_prior_follows_the_age_of_the_u_rows():
    f, y, pnu, age, view, amt = make()
    old = age.clone()
    old[pnu == PNU_U] = 15.0  # bin 1 -> 0.2
    a, b = HybridTWnnDFLoss(0.5, sliding_prior=_table()), HybridTWnnDFLoss(0.5, sliding_prior=_table())
    run(a, f, y, pnu, age, view, amt)
    run(b, f, y, pnu, old, view, amt)
    assert float(b.last_risks["prior_used"]) == pytest.approx(0.2)
    assert float(a.last_risks["neg_risk_raw"]) != float(b.last_risks["neg_risk_raw"])


def test_sliding_prior_ignores_trusted_rows_and_needs_age_days():
    f, y, pnu, age, view, amt = make()
    u_idx = (pnu == PNU_U).nonzero().squeeze(1)
    amt = amt.clone()
    amt[u_idx[:3]] = 500.0
    age = age.clone()
    age[u_idx[:3]] = 15.0  # trusted rows (old+large) would pull pi to 0.2 if they stayed in U
    loss = HybridTWnnDFLoss(0.5, amount_threshold=100.0, amount_min_age_days=10.0, sliding_prior=_table())
    run(loss, f, y, pnu, age, view, amt)
    assert float(loss.last_risks["prior_used"]) == pytest.approx(0.05)  # only the 3 young U rows
    with pytest.raises(ValueError):
        loss(f, y, pnu, in_train_view=view, amount=amt)  # no age_days


def _grid():
    ages = torch.arange(100).float().repeat_interleave(200) + 0.5
    b, i = ages.floor().long(), torch.arange(20000) % 200
    k = torch.where(b >= 60, 20, torch.where(b >= 30, 4, 0))
    return ages, (i < k).float()


def test_sliding_window_prior_fit_follows_the_past_window_and_ignores_young_rows():
    ages, y = _grid()
    sp = SlidingWindowPrior.fit(y, ages, 30.0, span_days=14.0, shrink=200.0)
    assert float(sp.table[0]) < 0.04 and float(sp.table[40]) > 0.08  # windows [30,44) vs [70,84)
    y2 = y.clone()
    y2[ages < 30.0] = 1.0  # young labels must not change pi
    assert torch.allclose(sp.table, SlidingWindowPrior.fit(y2, ages, 30.0).table)
    assert sp.query(torch.tensor([0.5, 1e6])).shape == (2,)


# ------------------------------------------------------------------ maturity_mask amount_aware (FIX 2) --------
def test_maturity_masked_bce_alias_and_amount_aware_gradients():
    assert MaturityMaskedBCELoss is MaturityMaskLoss
    f, y, pnu, age, view, amt = make()
    u_idx = (pnu == PNU_U).nonzero().squeeze(1)
    amt = amt.clone()
    amt[u_idx] = 500.0
    age = age.clone()
    age[u_idx[:3]] = 20.0
    ff = f.clone().requires_grad_(True)
    loss = MaturityMaskedBCELoss(amount_threshold=100.0, amount_min_age_days=10.0)
    loss(ff, y, pnu, age_days=age, in_train_view=view, amount=amt).backward()
    assert (ff.grad[u_idx[:3]].abs() > 0).all() and torch.all(ff.grad[u_idx[3:]] == 0)
    assert loss.needs_amount and not MaturityMaskedBCELoss().needs_amount
    plain = MaturityMaskedBCELoss()(f, y, pnu, age_days=age, in_train_view=view)
    assert not torch.allclose(plain, loss(f, y, pnu, age_days=age, in_train_view=view, amount=amt))
    with pytest.raises(ValueError):
        loss(f, y, pnu, age_days=age, in_train_view=view)  # amount missing


# ------------------------------------------------------------------ calibration / anti-leakage ----------------
def test_temperature_is_fitted_in_the_right_direction():
    g = torch.Generator().manual_seed(0)
    z0 = torch.randn(20000, generator=g) * 2.0
    y = torch.bernoulli(torch.sigmoid(z0), generator=g)
    over, under = MatureTemperatureTracker(), MatureTemperatureTracker()
    over.collect(3.0 * z0, y)
    under.collect(0.3 * z0, y)
    assert over.refit() > 1.5 and under.refit() < 1.0
    assert TemperatureScaling is TemperatureScaler


def test_tracker_sees_only_mature_train_rows_not_u_rows():
    def collected(u_shift):
        f, y, pnu, age, view, amt = make(seed=1)
        f = f.clone()
        f[pnu == PNU_U] += u_shift  # only the U rows change
        t = MatureTemperatureTracker()
        run(HybridTWnnDFLoss(0.1, calibrator=t), f.requires_grad_(True), y, pnu, age, view, amt)
        return t

    a, b = collected(0.0), collected(5.0)
    za, ya = a.collected()
    zb, yb = b.collected()
    assert za.numel() == 12 and torch.equal(za, zb) and torch.equal(ya, yb)  # 4 P + 8 N, never U
    assert a.refit() == b.refit()


def test_validation_style_calls_cannot_change_T():
    f, y, pnu, age, view, amt = make()
    t = MatureTemperatureTracker()
    loss = HybridTWnnDFLoss(0.1, calibrator=t)
    with torch.no_grad():
        run(loss, f, y, pnu, age, view, amt)
    assert t.n_collected == 0
    loss.eval()
    run(loss, f.clone().requires_grad_(True), y, pnu, age, view, amt)
    assert t.n_collected == 0 and t.temperature == 1.0
    assert loss.end_epoch() == 1.0


def test_no_api_accepts_validation_or_test_data():
    for fn in (MatureTemperatureTracker.collect, MatureTemperatureTracker.refit, TemperatureScaler.fit,
               TemperatureScaler.fit_on_mature, SlidingWindowPrior.fit, SlidingWindowPrior.query):
        names = list(inspect.signature(fn).parameters)
        assert not any(n.startswith(("val", "test")) for n in names), (fn, names)


def test_fit_on_mature_ignores_young_rows():
    g = torch.Generator().manual_seed(3)
    z = torch.randn(4000, generator=g) * 2
    y = torch.bernoulli(torch.sigmoid(z), generator=g)
    age = torch.cat([torch.full((2000,), 50.0), torch.full((2000,), 1.0)])
    a = TemperatureScaler().fit_on_mature(z, y, age, 30.0)
    z2, y2 = z.clone(), y.clone()
    z2[2000:], y2[2000:] = -z[2000:] * 7, 1 - y[2000:]  # corrupt ONLY the young rows
    b = TemperatureScaler().fit_on_mature(z2, y2, age, 30.0)
    assert math.isclose(a.temperature, b.temperature, rel_tol=1e-5)


def test_refit_changes_the_pu_term_input():
    f, y, pnu, age, view, amt = make()
    t = MatureTemperatureTracker()
    loss = HybridTWnnDFLoss(0.1, calibrator=t)
    run(loss, f, y, pnu, age, view, amt)
    r1 = float(loss.last_risks["neg_risk_raw"])
    with torch.no_grad():
        t.scaler.log_t.fill_(math.log(2.0))
    run(loss, f, y, pnu, age, view, amt)
    assert float(loss.last_risks["neg_risk_raw"]) != r1 and t.temperature == pytest.approx(2.0)


def test_no_u_in_batch_gives_bce_only_and_complete_risk_keys():
    f, y, pnu, age, view, amt = make(n_u=0)
    loss = HybridTWnnDFLoss(0.1, sliding_prior=_table())
    out = run(loss, f.clone().requires_grad_(True), y, pnu, age, view, amt)
    assert torch.isfinite(out) and all(k in loss.last_risks for k in (*KEYS, "bce", "prior_used"))


def test_validation_and_amount_requirement():
    f, y, pnu, age, view, _ = make()
    with pytest.raises(ValueError):
        HybridTWnnDFLoss(0.1, unlabeled_statuses=("U", "N"))
    with pytest.raises(ValueError):
        run(HybridTWnnDFLoss(0.1, amount_threshold=100.0), f, y, pnu, age, view, None)


def test_tiny_training_loop_with_epoch_refits_is_finite():
    torch.manual_seed(0)
    f0, y, pnu, age, view, amt = make(n_p=6, n_n=20, n_u=20)
    x = torch.randn(len(pnu), 5)
    x[pnu == PNU_P] += 1.5
    lin = torch.nn.Linear(5, 1)
    opt = torch.optim.Adam(lin.parameters(), lr=1e-2)
    loss_fn = HybridTWnnDFLoss(0.1, amount_weighted=True, amount_ref=10.0, calibrator=MatureTemperatureTracker(),
                               sliding_prior=_table())
    for step in range(40):
        opt.zero_grad()
        loss = run(loss_fn, lin(x).squeeze(1), y, pnu, age, view, amt)
        loss.backward()
        assert torch.isfinite(loss)
        opt.step()
        if step % 10 == 9:
            assert 0.25 <= loss_fn.end_epoch() <= 4.0