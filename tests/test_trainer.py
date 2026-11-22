"""Round-6 tests: schedule helpers, config parsing and a tiny end-to-end fit on toy batches (CPU, no pyg-lib)."""
from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
from torch_geometric.data import HeteroData  # noqa: E402

from src.graph.builder import PNU_CODES, TXN  # noqa: E402
from src.losses import DelayIPWLoss, MaturityMaskLoss, NaiveBCELoss, TimeWindowNNDFLoss  # noqa: E402
from src.models.classifier import FraudClassifier  # noqa: E402
from src.models.temporal_gnn import ModularTemporalGNN  # noqa: E402
from src.train.trainer import TrainConfig, Trainer, compute_batch_loss  # noqa: E402
from src.train.utils import EarlyStopper, warmup_factor  # noqa: E402

P, N, U = PNU_CODES["P"], PNU_CODES["N"], PNU_CODES["U"]
VOCAB = {"card": 40, "device": 25}
BS, HIST, IN_DIM = 16, 3, 16


def test_warmup_factor():
    assert warmup_factor(0, 4) == pytest.approx(0.25)
    assert warmup_factor(3, 4) == 1.0 and warmup_factor(99, 4) == 1.0
    assert warmup_factor(0, 0) == 1.0


def test_early_stopper_counts_patience_only_after_warmup():
    s = EarlyStopper(patience=2, start_epoch=1)
    assert [s.update(v, e) for e, v in enumerate([0.5, 0.4, 0.4], 1)] == [True, False, False]
    assert s.should_stop and s.best_epoch == 1

    w = EarlyStopper(patience=2, start_epoch=3)            # epochs 1-2 are warm-up
    for e, v in enumerate([0.5, 0.4, 0.3, 0.2], 1):
        w.update(v, e)
        if e < 4:
            assert not w.should_stop
    assert w.should_stop and w.bad_epochs == 2

    nan = EarlyStopper(patience=1)
    assert nan.update(float("nan"), 1) is False and nan.best is None


def test_train_config_from_cfg():
    cfg = {"seed": 7, "train": {"epochs": 10, "warmup_epochs": 2, "patience": 3},
           "eval": {"fpr": 0.02}, "loss": {"name": "naive_bce", "delay_ipw": {"lam": 0.3}}}
    t = TrainConfig.from_cfg(cfg, loss="delay_ipw")
    assert (t.loss, t.seed, t.epochs, t.warmup_epochs, t.fpr) == ("delay_ipw", 7, 10, 2, 0.02)
    assert t.loss_params == {"lam": 0.3}
    with pytest.raises(ValueError):
        TrainConfig(epochs=2, warmup_epochs=2)               # warm-up must leave at least one epoch


# --------------------------------------------------------------------------- #
# Toy batches with the NeighborLoader layout
# --------------------------------------------------------------------------- #
def toy_batch(seed: int, with_y_true: bool = True) -> HeteroData:
    g = torch.Generator().manual_seed(seed)
    n = BS * (1 + HIST)
    owner = torch.cat([torch.arange(BS), torch.arange(BS).repeat_interleave(HIST)])
    seed_time = 10_000_000 + torch.arange(BS, dtype=torch.long) * 7_200
    j = torch.cat([torch.zeros(BS, dtype=torch.long), torch.arange(HIST).repeat(BS)])
    time = torch.where(torch.arange(n) < BS, seed_time[owner], seed_time[owner] - (HIST - j + 1) * 86_400)

    pnu = torch.full((n,), N).long()
    pnu[:2] = P
    pnu[9:BS] = U
    batch = HeteroData()
    t = batch[TXN]
    t.x = torch.randn(n, IN_DIM, generator=g)
    t.time, t.batch, t.batch_size = time, owner, BS
    t.pnu = pnu
    t.y_obs = (pnu == P).long()
    t.in_train_view = torch.ones(n, dtype=torch.bool)
    t.amount = torch.rand(n, generator=g) * 100 + 1
    if with_y_true:
        t.y_true = t.y_obs.clone()
        t.y_true[[14, 15]] = 1                               # two frauds hidden in U
    hist_idx = torch.arange(BS, n)
    for e, size in VOCAB.items():
        ent = batch[e]
        ent.node_id = torch.randint(0, size, (BS,), generator=g)
        ent.batch = torch.arange(BS)
        ent.time = seed_time - 5 * 86_400
        rev, fwd = (e, f"rev_uses_{e}", TXN), (TXN, f"uses_{e}", e)
        batch[rev].edge_index = torch.stack([torch.arange(BS), torch.arange(BS)])
        batch[rev].timestamp = seed_time.clone()
        batch[fwd].edge_index = torch.stack([hist_idx, owner[hist_idx]])
        batch[fwd].timestamp = time[hist_idx]
    return batch


class FakeDS:
    """Minimal stand-in for TemporalGraphDataset: iterable of batches + ``data.label_cut``."""

    def __init__(self, batches: list[HeteroData]) -> None:
        self.batches = batches
        self.data = SimpleNamespace(label_cut=10_000_000.0 + 40 * 86_400.0)

    def __iter__(self):
        return iter(self.batches)

    def __len__(self) -> int:
        return len(self.batches)


def toy_model(use_temporal: bool = True) -> FraudClassifier:
    bb = ModularTemporalGNN(in_dim=IN_DIM, entity_vocab_sizes=VOCAB, hidden_dim=16, num_layers=2, dropout=0.0,
                            entity_embedding_dim=8, max_history=2, use_temporal=use_temporal)
    return FraudClassifier(bb, dropout=0.0)


@pytest.mark.parametrize("use_temporal", [True, False])
@pytest.mark.parametrize("name", ["naive_bce", "maturity_mask", "delay_ipw", "tw_nndf", "oracle"])
def test_fit_runs_for_every_loss_and_the_oracle_switch(name, use_temporal):
    torch.manual_seed(0)
    losses = {"naive_bce": NaiveBCELoss(), "maturity_mask": MaturityMaskLoss(), "delay_ipw": DelayIPWLoss(lam=0.2),
              "tw_nndf": TimeWindowNNDFLoss(prior=0.1), "oracle": NaiveBCELoss()}
    train, val = FakeDS([toy_batch(s) for s in range(3)]), FakeDS([toy_batch(10)])
    cfg = TrainConfig(loss="naive_bce" if name == "oracle" else name, oracle=(name == "oracle"), epochs=2,
                      warmup_epochs=1, patience=2, device="cpu")
    res = Trainer(toy_model(use_temporal), losses[name], train, val, cfg, log=lambda s: None).fit()
    assert res["epochs_run"] == len(res["history"]) == 2
    assert math.isfinite(res["val"]["pr_auc"]) and math.isfinite(res["train_seconds"])
    assert 1 <= res["best_epoch"] <= 2
    if name == "tw_nndf":
        assert "nndf_clamped_rate" in res["history"][0] and "nndf_neg_risk_raw" in res["history"][0]


def test_oracle_needs_y_true_in_the_train_batches():
    batch = toy_batch(0, with_y_true=False)
    with pytest.raises(KeyError):
        compute_batch_loss(toy_model(), NaiveBCELoss(), batch, 1e7, oracle=True)
    loss, _ = compute_batch_loss(toy_model(), NaiveBCELoss(), batch, 1e7, oracle=False)   # y_obs path still works
    assert torch.isfinite(loss)