"""Round-9b tests: Kiryo-style gradient-ascent branch of TimeWindowNNDFLoss (dummy scores, no real data)."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from src.graph.builder import PNU_CODES  # noqa: E402
from src.losses import TimeWindowNNDFLoss  # noqa: E402

P, U = PNU_CODES["P"], PNU_CODES["U"]
N_P, N_U, PI = 50, 500, 0.0338
SP4, SM4 = float(torch.sigmoid(torch.tensor(4.0))), float(torch.sigmoid(torch.tensor(-4.0)))


def batch(f_p: float = 4.0, f_u: float = -4.0):
    preds = torch.cat([torch.full((N_P,), f_p), torch.full((N_U,), f_u)]).requires_grad_(True)
    pnu = torch.cat([torch.full((N_P,), P), torch.full((N_U,), U)]).long()
    return preds, (pnu == P).long(), pnu, torch.ones_like(pnu, dtype=torch.bool)


def run(loss_fn, f_p: float = 4.0, f_u: float = -4.0):
    preds, y, pnu, view = batch(f_p, f_u)
    loss = loss_fn(preds, y, pnu, age_days=torch.zeros_like(preds), in_train_view=view)
    loss.backward()
    return loss.detach(), preds.grad[:N_P], preds.grad[N_P:], loss_fn.last_risks


def test_ascent_branch_gives_a_restoring_gradient_where_clamp_gives_none():
    _, gp_c, gu_c, r_c = run(TimeWindowNNDFLoss(PI, mode="clamp"))
    assert float(r_c["neg_risk_raw"]) < -0.05 and float(gu_c.abs().max()) == 0.0   # dead gradient on U (old behaviour)

    loss, gp, gu, r = run(TimeWindowNNDFLoss(PI, mode="nnpu"))
    assert float(r["ascent"]) == 1.0 and float(r["clamped"]) == 1.0
    assert float(loss) == pytest.approx(-float(r["neg_risk_raw"]))                # loss = -gamma * neg_raw, gamma = 1
    # gradient DESCENT on this loss: U scores go UP (grad < 0) and P scores go DOWN (grad > 0)
    torch.testing.assert_close(gu, torch.full((N_U,), -SM4 / N_U), rtol=1e-4, atol=1e-9)
    # no positive-risk term in this branch: only d(pi * mean softplus(f_P))/df = pi * sigmoid(f) / n_P
    torch.testing.assert_close(gp, torch.full((N_P,), PI * SP4 / N_P), rtol=1e-4, atol=1e-9)


def test_gamma_scales_the_ascent_gradient():
    _, _, gu, _ = run(TimeWindowNNDFLoss(PI, mode="nnpu", gamma=0.5))
    torch.testing.assert_close(gu, torch.full((N_U,), -0.5 * SM4 / N_U), rtol=1e-4, atol=1e-9)


def test_nnpu_equals_clamp_when_neg_risk_is_not_negative():
    a = run(TimeWindowNNDFLoss(0.001, mode="clamp"))                              # neg_raw ~ +0.014 > 0
    b = run(TimeWindowNNDFLoss(0.001, mode="nnpu"))
    assert float(b[3]["ascent"]) == 0.0 and float(b[3]["neg_risk_raw"]) > 0
    torch.testing.assert_close(a[0], b[0])
    torch.testing.assert_close(a[1], b[1])
    torch.testing.assert_close(a[2], b[2])


def test_beta_keeps_the_full_risk_branch_for_slightly_negative_neg_risk():
    # |neg_raw| ~ 0.118 < beta = 0.2 -> full unbiased risk, UNclamped: loss = pos_risk + neg_raw
    loss, gp, gu, r = run(TimeWindowNNDFLoss(PI, mode="nnpu", beta=0.2))
    assert float(r["ascent"]) == 0.0 and float(r["neg_risk_raw"]) < 0
    assert float(loss) == pytest.approx(float(r["pos_risk"] + r["neg_risk_raw"]), rel=1e-5)
    torch.testing.assert_close(gu, torch.full((N_U,), SM4 / N_U), rtol=1e-4, atol=1e-9)
    torch.testing.assert_close(gp, torch.full((N_P,), -PI / N_P), rtol=1e-4, atol=1e-9)   # -pi*(sig(-f)+sig(f))/nP


def test_validation_and_empty_batch():
    for kw in ({"mode": "bogus"}, {"beta": -0.1}, {"gamma": 0.0}, {"gamma": 1.5}):
        with pytest.raises(ValueError):
            TimeWindowNNDFLoss(PI, **kw)
    preds = torch.zeros(5, requires_grad=True)
    pnu = torch.full((5,), U).long()                                               # no P row
    loss = TimeWindowNNDFLoss(PI, mode="nnpu")(preds, (pnu == P).long(), pnu, age_days=torch.zeros(5))
    assert float(loss.detach()) == 0.0


def test_tiny_training_loop_stays_finite_in_both_modes(capsys):
    """Smoke only: no claim about the dynamics (unknown). Run with `-s` to see the printed clamp rates."""
    g = torch.Generator().manual_seed(0)
    n_p, n_u = 200, 4000
    hidden = torch.rand(n_u, generator=g) < 0.0138
    x = torch.randn(n_p + n_u, 3, generator=g)
    x[:n_p, 0] += 2.0
    x[n_p:][hidden, 0] += 2.0
    pnu = torch.cat([torch.full((n_p,), P), torch.full((n_u,), U)]).long()
    y, view = (pnu == P).long(), torch.ones(n_p + n_u, dtype=torch.bool)
    for mode in ("clamp", "nnpu"):
        w = torch.zeros(3, requires_grad=True)
        opt, loss_fn, rates = torch.optim.SGD([w], lr=0.5), TimeWindowNNDFLoss(0.0338, mode=mode), []
        for _ in range(200):
            opt.zero_grad()
            loss = loss_fn(x @ w, y, pnu, age_days=torch.zeros(len(y)), in_train_view=view)
            loss.backward()
            opt.step()
            rates.append(float(loss_fn.last_risks["clamped"]))
            assert torch.isfinite(loss) and torch.isfinite(w).all()
        print(f"[smoke] {mode}: clamped rate (last 50 steps) = {sum(rates[-50:]) / 50:.2f}, w = {w.detach().tolist()}")