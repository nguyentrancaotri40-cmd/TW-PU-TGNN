"""Round-5 tests: loss gradient/output invariants, model label-invariance, forward -> losses smoke.

Self-contained (toy batches); needs torch + torch_geometric, no pyg-lib.
"""
from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
import torch.nn.functional as F  # noqa: E402
from torch_geometric.data import HeteroData  # noqa: E402

from src.graph.builder import PNU_CODES, TXN  # noqa: E402
from src.losses import (  # noqa: E402
    DelayIPWLoss, MaturityMaskLoss, NaiveBCELoss, TimeWindowNNDFLoss, compute_age_days,
)
from src.losses.canary import (  # noqa: E402
    assert_loss_invariant_to_status, assert_nonzero_grad_on_status, assert_zero_grad_on_status,
    logits_grad, model_leakage_hook,
)
from src.models.classifier import FraudClassifier  # noqa: E402
from src.models.temporal_gnn import ModularTemporalGNN  # noqa: E402

P, N, U = PNU_CODES["P"], PNU_CODES["N"], PNU_CODES["U"]


# --------------------------------------------------------------------------- #
# Rows with known statuses
# --------------------------------------------------------------------------- #
def make_rows(n_p: int = 6, n_n: int = 20, n_u: int = 14, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    pnu = torch.cat([torch.full((n_p,), P), torch.full((n_n,), N), torch.full((n_u,), U)]).long()
    y_obs = (pnu == P).long()
    age = torch.cat([torch.linspace(0.5, 9.0, n_p), torch.linspace(8.0, 60.0, n_n), torch.linspace(0.2, 6.8, n_u)])
    preds = torch.randn(pnu.numel(), generator=g)
    return preds, y_obs, pnu, age


# ---- naive_bce --------------------------------------------------------------- #
def test_naive_bce_matches_torch_and_gives_every_u_row_a_gradient():
    preds, y, pnu, _ = make_rows()
    loss, grad = logits_grad(NaiveBCELoss(), preds, y, pnu)
    torch.testing.assert_close(loss, F.binary_cross_entropy_with_logits(preds, y.float()))
    torch.testing.assert_close(grad, (torch.sigmoid(preds) - y.float()) / preds.numel())
    assert_nonzero_grad_on_status(grad, pnu, U, "naive_bce")


def test_naive_bce_ignores_rows_outside_the_train_view():
    preds, y, pnu, _ = make_rows()
    view = torch.ones_like(pnu, dtype=torch.bool)
    view[-5:] = False
    _, grad = logits_grad(NaiveBCELoss(), preds, y, pnu, in_train_view=view)
    assert bool((grad[~view] == 0).all()) and bool((grad[view] != 0).all())


# ---- maturity_mask ----------------------------------------------------------- #
def test_maturity_mask_has_zero_u_gradient_and_u_invariant_output():
    preds, y, pnu, _ = make_rows()
    fn = MaturityMaskLoss()
    loss, grad = logits_grad(fn, preds, y, pnu)
    keep = pnu != U
    torch.testing.assert_close(loss, F.binary_cross_entropy_with_logits(preds[keep], y[keep].float()))
    assert_zero_grad_on_status(grad, pnu, U, "maturity_mask")
    assert_nonzero_grad_on_status(grad, pnu, P, "maturity_mask")
    assert_nonzero_grad_on_status(grad, pnu, N, "maturity_mask")
    assert_loss_invariant_to_status(fn, preds, y, pnu, U)


# ---- tw_nndf ----------------------------------------------------------------- #
def _clamped_rows(prior: float = 0.035):
    preds, y, pnu, age = make_rows()
    preds = torch.where(pnu == P, torch.full_like(preds, 4.0), preds)  # l(P,-1) large -> raw < 0
    preds = torch.where(pnu == U, torch.full_like(preds, -8.0), preds)  # l(U,-1) ~ 0
    return preds, y, pnu, age


def test_tw_nndf_u_rows_get_zero_gradient_when_the_clamp_is_active():
    fn = TimeWindowNNDFLoss(prior=0.035)
    preds, y, pnu, _ = _clamped_rows()
    loss, grad = logits_grad(fn, preds, y, pnu)
    assert float(fn.last_risks["neg_risk_raw"]) < 0, "test setup is vacuous: the clamp is not active"
    assert float(fn.last_risks["clamped"]) == 1.0
    assert_zero_grad_on_status(grad, pnu, U, "tw_nndf")
    assert_zero_grad_on_status(grad, pnu, N, "tw_nndf")  # N is not part of U by default
    assert_nonzero_grad_on_status(grad, pnu, P, "tw_nndf")
    assert_loss_invariant_to_status(fn, preds, y, pnu, U, scale=1.0)  # small noise: clamp stays active


def test_tw_nndf_u_rows_get_gradient_when_the_clamp_is_inactive():
    fn = TimeWindowNNDFLoss(prior=0.035)
    preds, y, pnu, _ = make_rows()
    preds = torch.where(pnu == P, torch.full_like(preds, -6.0), preds)
    preds = torch.where(pnu == U, torch.full_like(preds, 2.0), preds)
    _, grad = logits_grad(fn, preds, y, pnu)
    assert float(fn.last_risks["neg_risk_raw"]) > 0, "test setup is vacuous: the clamp is active"
    assert_nonzero_grad_on_status(grad, pnu, U, "tw_nndf")


def test_tw_nndf_matches_the_closed_form():
    preds, y, pnu, _ = make_rows(seed=3)
    pi = 0.1
    p, u = preds[pnu == P], preds[pnu == U]
    pos = pi * F.softplus(-p).mean()
    neg = F.softplus(u).mean() - pi * F.softplus(p).mean()
    expected = pos + torch.clamp(neg, min=0.0)
    torch.testing.assert_close(TimeWindowNNDFLoss(prior=pi)(preds, y, pnu), expected)


def test_tw_nndf_empty_p_or_u_gives_connected_zero():
    fn = TimeWindowNNDFLoss(prior=0.05)
    preds = torch.randn(8, requires_grad=True)
    only_n = torch.full((8,), N).long()
    loss = fn(preds, torch.zeros(8).long(), only_n)
    assert float(loss) == 0.0 and fn.last_risks is None
    loss.backward()  # must not raise


# ---- delay_ipw --------------------------------------------------------------- #
def test_delay_ipw_weights_follow_the_formula_and_young_rows_get_smaller_gradient():
    lam = 0.3
    fn = DelayIPWLoss(lam=lam)
    preds, y, pnu, age = make_rows()
    w = fn.weights(pnu, age)
    expected = torch.where(pnu == P, torch.ones_like(age), 1.0 - torch.exp(-lam * age))
    torch.testing.assert_close(w, expected)

    flat = torch.zeros_like(preds)  # equal logits: gradient is proportional to the weight
    _, grad = logits_grad(fn, flat, y, pnu, age_days=age)
    u_rows = pnu == U
    gu, au = grad[u_rows].abs(), age[u_rows]
    assert bool((gu > 0).all()) and bool((gu[au.argsort()].diff() >= 0).all())  # older -> larger


def test_delay_ipw_reduces_to_naive_bce_for_a_huge_rate():
    preds, y, pnu, age = make_rows()
    ipw = DelayIPWLoss(lam=1e6)(preds, y, pnu, age_days=age)  # every w == 1
    torch.testing.assert_close(ipw, NaiveBCELoss()(preds, y, pnu))


def test_delay_ipw_mature_only_scope_masks_u():
    preds, y, pnu, age = make_rows()
    fn = DelayIPWLoss(lam=0.3, negative_scope="mature_only")
    _, grad = logits_grad(fn, preds, y, pnu, age_days=age)
    assert_zero_grad_on_status(grad, pnu, U, "delay_ipw[mature_only]")


# ---- shared behaviour -------------------------------------------------------- #
@pytest.mark.parametrize("make", [NaiveBCELoss, MaturityMaskLoss])
def test_empty_selection_returns_connected_zero(make):
    preds = torch.randn(6, requires_grad=True)
    pnu = torch.full((6,), U).long()
    loss = make()(preds, torch.zeros(6).long(), pnu, in_train_view=torch.zeros(6, dtype=torch.bool))
    assert float(loss) == 0.0
    loss.backward()


def test_all_four_losses_equal_ln2_at_zero_logits():
    preds, y, pnu, age = make_rows()
    z = torch.zeros_like(preds)
    ln2 = math.log(2.0)
    vals = [
        NaiveBCELoss()(z, y, pnu),
        MaturityMaskLoss()(z, y, pnu),
        DelayIPWLoss(lam=0.3)(z, y, pnu, age_days=age),
        TimeWindowNNDFLoss(prior=0.035)(z, y, pnu),  # pi*ln2 + (1-pi)*ln2
    ]
    for v in vals:
        assert float(v) == pytest.approx(ln2, abs=1e-5)


# --------------------------------------------------------------------------- #
# Toy disjoint batch (same layout as NeighborLoader output)
# --------------------------------------------------------------------------- #
VOCAB = {"card": 40, "device": 25}


def make_toy_batch(bs: int = 12, hist: int = 3, in_dim: int = 16, seed: int = 0) -> HeteroData:
    g = torch.Generator().manual_seed(seed)
    n_txn = bs * (1 + hist)
    owner = torch.cat([torch.arange(bs), torch.arange(bs).repeat_interleave(hist)])
    seed_time = 10_000_000 + torch.arange(bs, dtype=torch.long) * 7_200
    j = torch.cat([torch.zeros(bs, dtype=torch.long), torch.arange(hist).repeat(bs)])
    pos = torch.arange(n_txn)
    time = torch.where(pos < bs, seed_time[owner], seed_time[owner] - (hist - j + 1) * 86_400)

    pnu = torch.full((n_txn,), N).long()
    pnu[:2] = P
    pnu[7:bs] = U  # seeds: 2 P, 5 N, rest U (bs >= 8)
    batch = HeteroData()
    t = batch[TXN]
    t.x = torch.randn(n_txn, in_dim, generator=g)
    t.time = time
    t.batch = owner
    t.batch_size = bs
    t.pnu = pnu
    t.y_obs = (pnu == P).long()
    t.y_true = t.y_obs.clone()
    t.in_train_view = torch.ones(n_txn, dtype=torch.bool)

    hist_idx = torch.arange(bs, n_txn)
    for e, size in VOCAB.items():
        ent = batch[e]
        ent.node_id = torch.randint(0, size, (bs,), generator=g)
        ent.batch = torch.arange(bs)
        ent.time = seed_time - 5 * 86_400
        rev, fwd = (e, f"rev_uses_{e}", TXN), (TXN, f"uses_{e}", e)
        batch[rev].edge_index = torch.stack([torch.arange(bs), torch.arange(bs)])
        batch[rev].timestamp = seed_time.clone()
        batch[fwd].edge_index = torch.stack([hist_idx, owner[hist_idx]])
        batch[fwd].timestamp = time[hist_idx]
    return batch


def make_model(conv: str, use_temporal: bool = True) -> FraudClassifier:
    backbone = ModularTemporalGNN(
        in_dim=16, entity_vocab_sizes=VOCAB, hidden_dim=32, num_layers=2, dropout=0.0,
        entity_embedding_dim=8, conv_type=conv, gat_heads=4, max_history=2, use_temporal=use_temporal,
    )
    return FraudClassifier(backbone, dropout=0.0)


@pytest.mark.parametrize("use_temporal", [True, False])
@pytest.mark.parametrize("conv", ["sage", "gat"])
def test_model_output_shape_and_backward(conv, use_temporal):
    torch.manual_seed(0)
    batch, model = make_toy_batch(), make_model(conv, use_temporal)
    logits = model(batch)
    assert logits.shape == (12,) and bool(torch.isfinite(logits).all())
    logits.sum().backward()
    assert model.backbone.ent_emb["card"].weight.grad is not None
    if use_temporal:
        assert float(model.backbone.gru.weight_ih_l0.grad.abs().sum()) > 0


@pytest.mark.parametrize("conv", ["sage", "gat"])
def test_model_output_does_not_depend_on_label_fields(conv):
    torch.manual_seed(0)
    model_leakage_hook(make_model(conv), make_toy_batch())  # raises on any label dependence


def test_gat_requires_divisible_hidden_dim():
    with pytest.raises(ValueError):
        ModularTemporalGNN(in_dim=4, entity_vocab_sizes=VOCAB, hidden_dim=30, conv_type="gat", gat_heads=4)


@pytest.mark.parametrize("conv", ["sage", "gat"])
def test_forward_then_four_losses_smoke(conv):
    torch.manual_seed(0)
    batch, model = make_toy_batch(), make_model(conv)
    st, bs = batch[TXN], 12
    age = compute_age_days(st.time[:bs], float(st.time[:bs].max()) + 30 * 86_400)
    losses = {
        "naive_bce": NaiveBCELoss(),
        "maturity_mask": MaturityMaskLoss(),
        "delay_ipw": DelayIPWLoss(lam=0.2),
        "tw_nndf": TimeWindowNNDFLoss(prior=0.05),
    }
    for name, fn in losses.items():
        model.zero_grad(set_to_none=True)
        loss = fn(model(batch), st.y_obs[:bs], st.pnu[:bs], age_days=age, in_train_view=st.in_train_view[:bs])
        assert loss.dim() == 0 and bool(torch.isfinite(loss)), name
        assert 0.1 < float(loss) < 3.0, f"{name}: implausible initial loss {float(loss)}"  # ~ ln 2 at init
        loss.backward()