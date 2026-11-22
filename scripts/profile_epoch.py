#!/usr/bin/env python
"""E0 profiling on the real graph: seconds per training/validation epoch and peak VRAM.

    python scripts/profile_epoch.py --processed_dir data/processed/delay_high --backbone sage --max_batches 60

It times a SAMPLE of ``--max_batches`` training batches after ``--warmup_batches`` untimed ones and
EXTRAPOLATES to a full epoch (mean time per batch x number of batches). Fetch time (temporal disjoint
sampling, includes pyg-lib and collation) is separated from step time (transfer + forward + backward + Adam).
Peak VRAM is the CUDA peak of the sampled training steps. On CPU there is no VRAM figure and the timings
are NOT Kaggle timings: run it on the Kaggle GPU for the real E0 numbers.

Planning table: runs x epochs x (train + val epoch). Early stopping makes real runs shorter, so it is an
upper bound; no weekly GPU quota is assumed (pass --weekly_gpu_hours if you know yours).
"""
from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import yaml

from src.data.dataset import load_phase_datasets, read_meta
from src.graph.builder import TXN
from src.models.classifier import FraudClassifier, entity_vocab_sizes_from_meta
from src.train.trainer import TrainConfig, build_training_loss, compute_batch_loss
from src.train.utils import get_device, peak_memory_mb, reset_peak_memory, set_seed, sync, write_json


def _bool(v: str) -> bool:
    if v.lower() in ("1", "true", "yes", "y"):
        return True
    if v.lower() in ("0", "false", "no", "n"):
        return False
    raise argparse.ArgumentTypeError("expected true/false")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Profile one epoch (time + VRAM) on the real graph.")
    p.add_argument("--processed_dir", default="data/processed/delay_high")
    p.add_argument("--config", default=str(REPO_ROOT / "configs" / "base.yaml"))
    p.add_argument("--backbone", choices=["sage", "gat"], default="sage")
    p.add_argument("--use_temporal", type=_bool, default=True, help="false = GRU ablation (SAGE/GAT + MLP)")
    p.add_argument("--lp", type=_bool, default=False, help="enable LP features (in_dim 457 -> 461)")
    p.add_argument("--loss", choices=["naive_bce", "maturity_mask", "delay_ipw", "tw_nndf"], default="naive_bce")
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--max_batches", type=int, default=60, help="timed training batches (<= 0: the whole epoch)")
    p.add_argument("--warmup_batches", type=int, default=5)
    p.add_argument("--eval_batches", type=int, default=10)
    p.add_argument("--epochs", type=int, default=None, help="epochs per run for the planning table (default: train.epochs)")
    p.add_argument("--runs", type=int, nargs="*", default=[72, 90], help="run counts for the planning table")
    p.add_argument("--weekly_gpu_hours", type=float, default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None)
    return p.parse_args()


def _stats(x: list[float]) -> dict[str, float]:
    a = np.asarray(x, dtype=np.float64)
    return {"mean": float(a.mean()), "median": float(np.median(a)), "p90": float(np.percentile(a, 90))}


def main() -> int:
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    pdir = Path(args.processed_dir)
    meta = read_meta(pdir)
    device = get_device(args.device)
    set_seed(args.seed)

    sampler_cfg = dict(cfg["sampler"])
    if args.batch_size:
        sampler_cfg["batch_size"] = args.batch_size
    features_cfg = copy.deepcopy(dict(cfg.get("features", {})))
    features_cfg["lp_features"] = {**(features_cfg.get("lp_features") or {}), "enabled": args.lp}
    dsets = load_phase_datasets(pdir, sampler_cfg, phases=("train", "val"), features_cfg=features_cfg)
    train_ds, val_ds = dsets["train"], dsets["val"]

    model_cfg = dict(cfg["model"])
    model_cfg["use_temporal"] = args.use_temporal
    model = FraudClassifier.from_config(model_cfg, in_dim=train_ds.model_in_dim,
                                        entity_vocab_sizes=entity_vocab_sizes_from_meta(meta),
                                        conv_type=args.backbone).to(device)
    tcfg = TrainConfig.from_cfg(cfg, loss=args.loss)
    loss_fn, info = build_training_loss(tcfg, train_ds, meta)
    loss_fn = loss_fn.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    label_cut = float(train_ds.data.label_cut)

    gpu = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    print(f"device={gpu} | backbone={args.backbone} temporal={args.use_temporal} lp={args.lp} loss={args.loss} | "
          f"in_dim={train_ds.model_in_dim} params={model.num_parameters():,} | "
          f"train batches/epoch={len(train_ds)} (batch_size={sampler_cfg['batch_size']}) val batches={len(val_ds)}")

    # ---- training steps -------------------------------------------------------------- #
    model.train()
    limit = len(train_ds) if args.max_batches <= 0 else args.warmup_batches + args.max_batches
    t_fetch, t_step, nodes = [], [], []
    reset_peak_memory(device)
    it = iter(train_ds)
    for i in range(limit):
        t0 = time.perf_counter()
        try:
            batch = next(it)
        except StopIteration:
            break
        t1 = time.perf_counter()
        batch = batch.to(device)
        loss, _ = compute_batch_loss(model, loss_fn, batch, label_cut, False)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sync(device)
        t2 = time.perf_counter()
        if i >= args.warmup_batches:
            t_fetch.append(t1 - t0)
            t_step.append(t2 - t1)
            nodes.append(int(batch[TXN].x.size(0)))
    if not t_step:
        print("FAIL: no timed batches (raise --max_batches or lower --warmup_batches).")
        return 1
    peak_train = peak_memory_mb(device)
    total = [a + b for a, b in zip(t_fetch, t_step)]
    train_epoch_s = float(np.mean(total)) * len(train_ds)

    # ---- validation steps -------------------------------------------------------------- #
    model.eval()
    t_eval = []
    reset_peak_memory(device)
    with torch.no_grad():
        it = iter(val_ds)
        for i in range(args.eval_batches):
            t0 = time.perf_counter()
            try:
                batch = next(it)
            except StopIteration:
                break
            batch = batch.to(device)
            model(batch)
            sync(device)
            t_eval.append(time.perf_counter() - t0)
    peak_eval = peak_memory_mb(device)
    val_epoch_s = float(np.mean(t_eval)) * len(val_ds) if t_eval else float("nan")

    epochs = args.epochs or int(cfg["train"]["epochs"])
    per_epoch = train_epoch_s + val_epoch_s
    plan = {str(r): {"hours_upper_bound": r * epochs * per_epoch / 3600.0} for r in args.runs}
    if args.weekly_gpu_hours:
        for r in args.runs:
            plan[str(r)]["weekly_quota_fraction"] = plan[str(r)]["hours_upper_bound"] / args.weekly_gpu_hours

    result = {
        "device": gpu, "backbone": args.backbone, "use_temporal": args.use_temporal, "lp": args.lp,
        "loss": args.loss, "in_dim": train_ds.model_in_dim, "n_params": model.num_parameters(),
        "batch_size": sampler_cfg["batch_size"], "n_train_batches": len(train_ds), "n_val_batches": len(val_ds),
        "timed_batches": len(t_step), "fetch_sec_per_batch": _stats(t_fetch), "step_sec_per_batch": _stats(t_step),
        "fetch_share": float(np.sum(t_fetch) / np.sum(total)),
        "nodes_per_batch": _stats([float(n) for n in nodes]),
        "train_epoch_sec_est": train_epoch_s, "val_epoch_sec_est": val_epoch_s,
        "peak_vram_train_mb": peak_train, "peak_vram_eval_mb": peak_eval,
        "epochs_assumed": epochs, "plan": plan, "loss_info": info,
        "note": "sampled batches extrapolated to a full epoch; upper bound (no early stopping); not Kaggle numbers on CPU",
    }
    out = Path(args.out) if args.out else REPO_ROOT / "results" / "profile" / (
        f"epoch_{pdir.name}_{args.backbone}_{'gru' if args.use_temporal else 'nogru'}_{'lp' if args.lp else 'nolp'}.json")
    write_json(out, result)

    print(f"\ntimed batches={len(t_step)} | fetch {np.mean(t_fetch) * 1e3:.0f} ms + step {np.mean(t_step) * 1e3:.0f} ms "
          f"per batch (fetch share {100 * result['fetch_share']:.0f}%) | nodes/batch ~{np.mean(nodes):,.0f}")
    print(f"train epoch ~ {train_epoch_s / 60:.1f} min | val epoch ~ {val_epoch_s / 60:.1f} min")
    print("peak VRAM (train): " + ("n/a (CPU)" if peak_train is None else
          f"{peak_train['allocated']:,.0f} MB allocated / {peak_train['reserved']:,.0f} MB reserved"))
    for r, v in plan.items():
        q = f" = {100 * v['weekly_quota_fraction']:.0f}% of the weekly quota" if "weekly_quota_fraction" in v else ""
        print(f"plan: {r} runs x {epochs} epochs ~ {v['hours_upper_bound']:.1f} GPU-hours (upper bound){q}")
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())