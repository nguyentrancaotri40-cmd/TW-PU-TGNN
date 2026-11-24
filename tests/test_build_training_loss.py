from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
yaml = pytest.importorskip("yaml")
pytest.importorskip("torch_geometric")

from src.graph.builder import PNU_CODES, TXN  # noqa: E402
from src.losses.maturity_mask import MaturityMaskLoss  # noqa: E402
from src.losses.tw_nndf import HybridTWnnDFLoss, TimeWindowNNDFLoss  # noqa: E402
from src.train.trainer import TrainConfig, build_training_loss  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DAY, W = 86400.0, 30.0
P, N, U = PNU_CODES["P"], PNU_CODES["N"], PNU_CODES["U"]


class _Graph(dict):
    label_cut = 100 * DAY


def fake_train_ds():
    n = 4000
    idx = torch.arange(n)
    age = (idx % 100).float() + 0.5  # 0.5 .. 99.5 days before the cut
    y_obs = ((idx % 25) == 0).long()
    pnu = torch.where(y_obs == 1, torch.tensor(P), torch.where(age >= W, torch.tensor(N), torch.tensor(U)))
    store = SimpleNamespace(
        pnu=pnu.long(), in_train_view=torch.ones(n, dtype=torch.bool), y_obs=y_obs,
        time=(_Graph.label_cut - age.double() * DAY), amount=((idx % 100) + 1).float() * 10.0,
        y_true=(y_obs.bool() | ((idx % 40) == 1)).long(),
    )
    return SimpleNamespace(data=_Graph({TXN: store}))


META = {"config": {"delay": {"maturity_window_days": W}}}


def build(loss, params):
    return build_training_loss(TrainConfig(loss=loss, loss_params=params), fake_train_ds(), META)


def test_maturity_mask_amount_aware_keys_reach_the_loss():
    fn, info = build("maturity_mask", {"amount_aware": True, "amount_quantile": 0.9, "amount_min_age_days": None})
    ds = fake_train_ds().data[TXN]
    assert isinstance(fn, MaturityMaskLoss) and fn.needs_amount
    assert fn.amount_threshold == pytest.approx(float(torch.quantile(ds.amount.float(), 0.9)))
    assert fn.amount_min_age_days == pytest.approx(W / 2) and "n_U_trusted" in info


def test_maturity_mask_without_amount_keys_is_unchanged():
    fn, _ = build("maturity_mask", {})
    assert fn.amount_threshold is None and not fn.needs_amount
    fn2, _ = build("maturity_mask", {"amount_aware": False, "amount_quantile": 0.95, "amount_min_age_days": None})
    assert fn2.amount_threshold is None


def test_pu_only_ignores_hybrid_keys_without_crashing():
    fn, _ = build("tw_nndf", {"unlabeled_statuses": ["U"], "mode": "nnpu", "amount_aware": True, "lambda_pu": 0.3})
    assert type(fn) is TimeWindowNNDFLoss


@pytest.mark.parametrize("prior_mode,expect_dynamic", [("sliding_window", False), ("sliding_dynamic", True)])
def test_hybrid_builds_with_and_without_sliding_prior(prior_mode, expect_dynamic):
    params = {"variant": "hybrid", "unlabeled_statuses": ["U"], "mode": "nnpu", "lambda_pu": 0.3,
              "prior_mode": prior_mode, "amount_weighted": True, "amount_aware": True,
              "calibration": {"enabled": True, "max_samples": 1000}}
    fn, info = build("tw_nndf", params)
    assert isinstance(fn, HybridTWnnDFLoss) and fn.needs_amount and fn.lambda_pu == 0.3
    assert (fn.sliding_prior is not None) is expect_dynamic and fn.calibrator is not None
    ds = fake_train_ds().data[TXN]
    k = 256
    out = fn(torch.randn(k), ds.y_obs[:k], ds.pnu[:k],
             age_days=(_Graph.label_cut - ds.time[:k]).float() / DAY, in_train_view=ds.in_train_view[:k],
             amount=ds.amount[:k])
    assert torch.isfinite(out) and info["variant"] == "hybrid"


@pytest.mark.parametrize("loss", ["maturity_mask", "tw_nndf", "naive_bce", "delay_ipw"])
def test_every_loss_builds_from_base_nnpu_yaml(loss):
    cfg = yaml.safe_load((ROOT / "configs" / "base_nnpu.yaml").read_text(encoding="utf-8"))
    cfg["loss"]["delay_ipw"] = {"lam": 0.1}  # as run_benchmark.resolve_cfg does
    fn, _ = build_training_loss(TrainConfig.from_cfg(cfg, loss=loss), fake_train_ds(), META)
    assert fn is not None