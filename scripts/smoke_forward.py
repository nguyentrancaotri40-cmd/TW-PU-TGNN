#!/usr/bin/env python
"""Round-5 smoke test on ONE real batch: backbone(s) -> the four losses.

    python scripts/smoke_forward.py --processed_dir data/processed/delay_low --backbone both

Needs the processed graphs from scripts/prepare_data.py and pyg-lib. Prints shapes, P/N/U counts,
the estimated prior and each loss value (expect roughly ln 2 = 0.69 at initialisation), and exits
with status 1 if a check fails.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import yaml

from src.data.dataset import TemporalGraphDataset, read_meta, seed_labels
from src.graph.builder import PNU_CODES, TXN
from src.losses import (
    DelayIPWLoss, MaturityMaskLoss, NaiveBCELoss, TimeWindowNNDFLoss,
    compute_age_days, estimate_class_prior, lam_from_median_delay,
)
from src.models.classifier import build_classifier


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Forward + 4 losses on one real batch.")
    p.add_argument("--processed_dir", default="data/processed/delay_low")
    p.add_argument("--config", default=str(REPO_ROOT / "configs" / "base.yaml"))
    p.add_argument("--backbone", choices=["sage", "gat", "both"], default="both")
    p.add_argument("--lam", type=float, default=None, help="delay_ipw rate; default ln2 / (W / 2) (arbitrary, smoke only)")
    p.add_argument("--max_tries", type=int, default=30, help="batches to scan for one with both P and U seeds")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    pdir = Path(args.processed_dir)
    meta = read_meta(pdir)
    s = cfg["sampler"]
    torch.manual_seed(0)

    ds = TemporalGraphDataset.from_file(
        pdir / "train.pt", num_neighbors=list(s["num_neighbors"]), batch_size=int(s["batch_size"]),
        shuffle=True, strict_time=bool(s.get("strict_time", True)),
        temporal_strategy=str(s.get("temporal_strategy", "uniform")), check_causality=True,
    )
    assert ds.feature_dim == int(meta["feature_dim"]), "feature_dim mismatch between graph and meta.json"

    batch = None
    for i, b in enumerate(ds):
        pnu = b[TXN].pnu[: int(b[TXN].batch_size)]
        if bool((pnu == PNU_CODES["P"]).any()) and bool((pnu == PNU_CODES["U"]).any()):
            batch = b
            break
        if i + 1 >= args.max_tries:
            break
    if batch is None:
        print("FAIL: no batch with both P and U seeds found; raise --max_tries or check the delay config.")
        return 1

    labels = seed_labels(batch)
    bs = int(batch[TXN].batch_size)
    age = compute_age_days(labels["time"], ds.data.label_cut)
    pnu = labels["pnu"]
    print(f"batch: seeds={bs} | P={int((pnu == 0).sum())} N={int((pnu == 1).sum())} U={int((pnu == 2).sum())} "
          f"| x={tuple(batch[TXN].x.shape)} | node types={batch.node_types}")

    window = float(meta["config"]["delay"]["maturity_window_days"])
    full = ds.data[TXN]
    prior = estimate_class_prior(full.y_obs, compute_age_days(full.time, ds.data.label_cut), window, full.in_train_view)
    lam = args.lam if args.lam is not None else lam_from_median_delay(window / 2.0)
    print(f"prior (mature region, observed labels only) = {prior:.4f} | delay_ipw lam = {lam:.4f} | W = {window} d")

    losses = {
        "naive_bce": NaiveBCELoss(),
        "maturity_mask": MaturityMaskLoss(),
        "delay_ipw": DelayIPWLoss(lam=lam),
        "tw_nndf": TimeWindowNNDFLoss(prior=prior),
    }
    ok = True
    for bb in (["sage", "gat"] if args.backbone == "both" else [args.backbone]):
        model = build_classifier(cfg["model"], meta, conv_type=bb)
        model.train()
        logits = model(batch)
        good = logits.shape == (bs,) and bool(torch.isfinite(logits).all())
        ok &= good
        print(f"\n[{bb}] params={model.num_parameters():,} | logits shape={tuple(logits.shape)} "
              f"mean={float(logits.mean()):+.3f} std={float(logits.std()):.3f} {'OK' if good else 'FAIL'}")
        for name, fn in losses.items():
            model.zero_grad(set_to_none=True)
            loss = fn(model(batch), labels["y_obs"], pnu, age_days=age, in_train_view=labels["in_train_view"])
            loss.backward()
            has_grad = any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in model.parameters())
            sane = bool(torch.isfinite(loss)) and loss.dim() == 0 and 0.1 < float(loss) < 3.0
            extra = ""
            if name == "tw_nndf" and fn.last_risks is not None:
                r = fn.last_risks
                extra = f" | pos={float(r['pos_risk']):.4f} neg_raw={float(r['neg_risk_raw']):+.4f} clamped={bool(r['clamped'])}"
            ok &= sane and has_grad
            print(f"  {name:<14} loss={float(loss):.4f} grad={'yes' if has_grad else 'NO'} "
                  f"{'OK' if sane else 'CHECK'}{extra}")
    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())