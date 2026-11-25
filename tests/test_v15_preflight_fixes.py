"""V1.5 preflight fixes: delay_very_high, benchmark matrix (72 + 18), sliding_unlabeled, T >= 0.5, train-only prior epsilon,
and the C6 check (variant=pu_only logs bce=0.0000 without crashing). CPU only, no real data.

Run the C6 part alone and see the log lines:  pytest -q -s tests/test_v15_preflight_fixes.py -k pu_only
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
yaml = pytest.importorskip("yaml")
pytest.importorskip("torch_geometric")
from torch_geometric.data import HeteroData  # noqa: E402

from src.graph.builder import PNU_CODES, TXN  # noqa: E402
from src.losses.calibration import (  # noqa: E402
    PRIOR_EPS, T_MIN_DEFAULT, MatureTemperatureTracker, unlabeled_prior, unlabeled_prior_raw,
)
from src.losses.tw_nndf import HybridTWnnDFLoss, TimeWindowNNDFLoss  # noqa: E402
from src.models.classifier import FraudClassifier  # noqa: E402
from src.models.temporal_gnn import ModularTemporalGNN  # noqa: E402
from src.train.trainer import TrainConfig, Trainer  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DELAYS = ["delay_low", "delay_high", "delay_very_high"]
P, N, U = PNU_CODES["P"], PNU_CODES["N"], PNU_CODES["U"]


def _yaml(rel: str) -> dict:
    return yaml.safe_load((ROOT / rel).read_text(encoding="utf-8"))


def _load_run_benchmark():
    spec = importlib.util.spec_from_file_location("run_benchmark_v15", ROOT / "scripts" / "run_benchmark.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_benchmark_v15"] = mod
    spec.loader.exec_module(mod)
    return mod


def _trunc_weibull_median(shape: float, scale: float, cap: float) -> float:
    """Median of a Weibull truncated (renormalised) to [0, cap], by bisection."""
    cdf = lambda x: 1.0 - math.exp(-((x / scale) ** shape))  # noqa: E731
    target = cdf(cap) / 2.0
    lo, hi = 0.0, cap
    for _ in range(200):
        mid = (lo + hi) / 2.0
        lo, hi = (mid, hi) if cdf(mid) < target else (lo, mid)
    return (lo + hi) / 2.0


# ------------------------------------------------------------------ item 1: delay regimes + matrix ------------
@pytest.mark.parametrize("delay", DELAYS)
def test_delay_yaml_w_covers_cap_and_lam_matches_the_median(delay):
    d = _yaml(f"configs/{delay}.yaml")["delay"]
    p = d["params"]
    assert d["maturity_window_days"] >= p["cap_days"], "W < cap: late frauds would hide inside N"
    lam = _yaml("configs/benchmark.yaml")["benchmark"]["lam_by_delay"][delay]
    median = _trunc_weibull_median(p["shape"], p["scale_days"], p["cap_days"])
    assert lam == pytest.approx(math.log(2.0) / median, rel=0.03)


def test_delay_very_high_values():
    d = _yaml("configs/delay_very_high.yaml")["delay"]
    assert d["mode"] == "weibull" and d["maturity_window_days"] == 45 and d["params"]["cap_days"] == 45


def test_benchmark_matrix_is_72_plus_18_and_every_delay_has_a_config():
    rb = _load_run_benchmark()
    bench = _yaml("configs/benchmark.yaml")["benchmark"]
    plan = rb.build_plan(bench)
    methods, oracle = [s for s in plan if not s.oracle], [s for s in plan if s.oracle]
    assert (len(methods), len(oracle), len(plan)) == (72, 18, 90)
    assert [s.oracle for s in plan] == [False] * 72 + [True] * 18          # oracle runs LAST
    assert {s.delay for s in plan} == set(DELAYS)
    assert {s.backbone for s in methods} == {"sage", "gat"}
    assert {s.loss for s in methods} == {"naive_bce", "maturity_mask", "delay_ipw", "tw_nndf"}
    assert {s.seed for s in plan} == {42, 43, 44}
    for d in DELAYS:
        assert (ROOT / "configs" / f"{d}.yaml").exists()
        assert bench["lam_by_delay"][d] is not None


# ------------------------------------------------------------------ item 3: configs ---------------------------
@pytest.mark.parametrize("name", ["base.yaml", "base_nnpu.yaml"])
def test_both_base_configs_are_hybrid_sliding_unlabeled_with_t_floor(name):
    loss = _yaml(f"configs/{name}")["loss"]
    t = loss["tw_nndf"]
    assert t["variant"] == "hybrid" and t["prior_mode"] == "sliding_unlabeled"
    assert t["calibration"]["t_min"] >= 0.5
    assert loss["maturity_mask"]["amount_aware"] is False                 # C12 locked for the main matrix


def test_unlabeled_prior_uses_only_numerical_epsilon():
    # An under-estimated pooled prior can yield a non-positive raw estimate.
    # It is clipped only to PRIOR_EPS, never to a floor informed by hidden fraud labels.
    assert unlabeled_prior_raw(0.01, 100, 1000) < 0
    assert unlabeled_prior(0.01, 100, 1000) == pytest.approx(PRIOR_EPS)
    # Normal case: (0.05 * 1040 - 40) / 1000 = 0.012 stays untouched.
    assert unlabeled_prior(0.05, 40, 1000) == pytest.approx(0.012)
    # An explicit zero floor still leaves the numerical epsilon safeguard.
    assert unlabeled_prior(0.01, 100, 1000, min_prior=0.0) == pytest.approx(PRIOR_EPS)


# ------------------------------------------------------------------ item 2: temperature floor -----------------
def test_temperature_floor_is_clamped_and_logged():
    assert MatureTemperatureTracker().t_min == T_MIN_DEFAULT == 0.5
    g = torch.Generator().manual_seed(0)
    z0 = torch.randn(20000, generator=g) * 2.0
    y = torch.bernoulli(torch.sigmoid(z0), generator=g)
    tracker = MatureTemperatureTracker()
    for _ in range(3):                                   # 3 "epochs" of under-confident logits (optimal T ~ 0.3)
        tracker.collect(0.3 * z0, y)
        t = tracker.refit()
        assert t >= 0.5 - 1e-6
    assert tracker.last_raw_temperature is not None and tracker.last_raw_temperature < 0.5
    assert tracker.temperature == pytest.approx(0.5, abs=1e-5) and tracker.floor_hits == 3


def test_temperature_above_the_floor_is_not_touched():
    g = torch.Generator().manual_seed(1)
    z0 = torch.randn(20000, generator=g) * 2.0
    y = torch.bernoulli(torch.sigmoid(z0), generator=g)
    tracker = MatureTemperatureTracker()
    tracker.collect(3.0 * z0, y)                         # over-confident: optimal T ~ 3
    assert 2.0 < tracker.refit() <= 4.0 and tracker.floor_hits == 0


# ------------------------------------------------------------------ item 5: C6 (pu_only) ----------------------
VOCAB = {"card": 40, "device": 25}
BS, HIST, IN_DIM = 16, 3, 16


def _toy_batch(seed: int) -> HeteroData:
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
    t.y_true = t.y_obs.clone()
    t.y_true[[14, 15]] = 1
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


class _FakeDS:
    def __init__(self, batches):
        self.batches = batches
        self.data = SimpleNamespace(label_cut=10_000_000.0 + 40 * 86_400.0)

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


def _fit(loss_fn):
    bb = ModularTemporalGNN(in_dim=IN_DIM, entity_vocab_sizes=VOCAB, hidden_dim=16, num_layers=2, dropout=0.0,
                            entity_embedding_dim=8, max_history=2, use_temporal=False)
    cfg = TrainConfig(loss="tw_nndf", epochs=2, warmup_epochs=1, patience=2, device="cpu")
    logs: list[str] = []
    res = Trainer(FraudClassifier(bb, dropout=0.0), loss_fn, _FakeDS([_toy_batch(s) for s in range(3)]),
                  _FakeDS([_toy_batch(10)]), cfg, log=logs.append).fit()
    return res, logs


def test_variant_pu_only_logs_bce_zero_and_does_not_crash():
    torch.manual_seed(0)
    res, logs = _fit(TimeWindowNNDFLoss(prior=0.1, mode="nnpu"))            # variant = pu_only
    nndf_lines = [ln for ln in logs if "nndf" in ln]
    for ln in nndf_lines:
        print(ln)
    assert len(nndf_lines) == 2 and all("bce=0.0000" in ln for ln in nndf_lines)
    assert all(h["nndf_bce"] == 0.0 for h in res["history"])


def test_variant_hybrid_has_a_positive_bce():
    torch.manual_seed(0)
    res, logs = _fit(HybridTWnnDFLoss(prior=0.1, mode="nnpu", lambda_pu=0.3))
    assert all(h["nndf_bce"] > 0.0 for h in res["history"])
    assert all("bce=0.0000" not in ln for ln in logs if "nndf" in ln)