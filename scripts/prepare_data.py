#!/usr/bin/env python
"""One-time data preparation: CSV -> split -> label delay -> phase graphs (.pt).

Pipeline
--------
1. load train_transaction.csv (+ train_identity.csv)       src/data/loader.py
2. temporal split 65/16/19                                  src/data/splitter.py
3. simulate label delay + P/N/U masks at t_train            src/data/delay_simulator.py
4. fit features / super-nodes on TRAIN, build HeteroData    src/graph/builder.py
5. validate every graph, save ``train.pt`` ``val.pt`` ``test.pt`` + metadata

Usage (from the repo root)::

    python scripts/prepare_data.py --data_dir data --out_dir data/processed \\
        --config configs/base.yaml --delay_config configs/delay_low.yaml

Output folder: ``<out_dir>/<delay_config stem>/`` (e.g. ``data/processed/delay_low``)
holding ``train.pt``, ``val.pt``, ``test.pt``, ``builder_state.pt``, ``meta.json``.
Graph structure and features do not depend on the delay regime, only the labels do,
so run it once per regime (two regimes ~ 2x the disk, roughly rows x features x 4 bytes
x 2.5 per regime because the cumulative val/test graphs repeat earlier rows).

Refuses to overwrite an existing output folder unless ``--overwrite`` is given.
"""
from __future__ import annotations

import argparse
import copy
import json
import platform
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch
import torch_geometric
import yaml

from src.data.delay_simulator import DelayParams, assign_pnu_masks, pnu_diagnostics, simulate_delay
from src.data.loader import AMOUNT_COL, TARGET_COL, DataLoader, get_time_stats
from src.data.splitter import compute_split_boundaries, summarize_splits, temporal_split
from src.graph.builder import TXN, GraphBuilder, GraphConfig, validate_phase_graph


# --------------------------------------------------------------------------- #
# Config helpers
# --------------------------------------------------------------------------- #
def deep_update(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``overlay`` into a copy of ``base``."""
    out = copy.deepcopy(base)
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(base_path: Path, delay_path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(base_path.read_text(encoding="utf-8")) or {}
    delay = yaml.safe_load(delay_path.read_text(encoding="utf-8")) or {}
    if "delay" not in delay:
        raise KeyError(f"{delay_path} must contain a top-level 'delay:' section.")
    return deep_update(cfg, delay)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="One-time data preparation (CSV -> phase graphs).")
    p.add_argument("--data_dir", type=str, default=None, help="Folder with the IEEE-CIS CSVs (default: paths.data_dir).")
    p.add_argument("--out_dir", type=str, default=None, help="Root output folder (default: paths.processed_dir).")
    p.add_argument("--config", type=str, default=str(REPO_ROOT / "configs" / "base.yaml"))
    p.add_argument("--delay_config", type=str, default=str(REPO_ROOT / "configs" / "delay_low.yaml"))
    p.add_argument("--nrows", type=int, default=None, help="Read only the first N transaction rows (debug).")
    p.add_argument("--no_identity", action="store_true", help="Skip train_identity.csv (the device entity disappears).")
    p.add_argument("--overwrite", action="store_true", help="Replace an existing output folder.")
    return p.parse_args()


def _resolve(path: str | None, fallback: str) -> Path:
    p = Path(path if path is not None else fallback)
    return p if p.is_absolute() else Path.cwd() / p


# --------------------------------------------------------------------------- #
def apply_delay(frame: pd.DataFrame, t_train: float, delay_cfg: dict[str, Any], amount_ref: float | None) -> pd.DataFrame:
    """Delay + P/N/U masks for one split frame (draws depend on TransactionID only)."""
    params = dict(delay_cfg.get("params", {}))
    if float(params.get("amount_exponent", 0.0)) > 0 and params.get("amount_ref") is None:
        params["amount_ref"] = amount_ref  # median TRAIN amount, shared by all frames
    with_delay = simulate_delay(frame, t_train, mode=delay_cfg["mode"], params=params)
    return assign_pnu_masks(with_delay, t_train, maturity_window_days=delay_cfg.get("maturity_window_days"))


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config), Path(args.delay_config))
    paths = cfg.get("paths", {})
    data_dir = _resolve(args.data_dir, paths.get("data_dir", "data"))
    out_root = _resolve(args.out_dir, paths.get("processed_dir", "data/processed"))
    out_dir = out_root / Path(args.delay_config).stem
    if out_dir.exists() and any(out_dir.iterdir()):
        if not args.overwrite:
            sys.exit(f"Output folder {out_dir} is not empty. Use --overwrite to replace it.")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # 1) load --------------------------------------------------------------
    loader = DataLoader(data_dir)
    print(f"[1/5] Loading {loader.transaction_path} ...")
    df = loader.load_transactions(nrows=args.nrows) if args.no_identity else loader.load_merged(nrows=args.nrows)
    print(f"      rows={len(df):,} cols={df.shape[1]} frauds={int(df[TARGET_COL].sum()):,} "
          f"({100 * df[TARGET_COL].mean():.3f}%)")
    print(f"      {get_time_stats(df).summary()}")

    # 2) split -------------------------------------------------------------
    s = cfg["split"]
    split_kw = dict(train_ratio=s["train_ratio"], val_ratio=s["val_ratio"], test_ratio=s["test_ratio"],
                    by=s.get("by", "time"), snap_to_day=s.get("snap_to_day", True))
    bounds = compute_split_boundaries(df, **split_kw)
    train_df, val_df, test_df = temporal_split(df, **split_kw)
    del df
    print(f"[2/5] Split: {bounds.summary()}")
    print(summarize_splits(train_df, val_df, test_df).round(3).to_string())

    # 3) delay + P/N/U -----------------------------------------------------
    delay_cfg = cfg["delay"]
    DelayParams.from_dict(delay_cfg.get("params")).validate(delay_cfg["mode"])  # fail early
    amount_ref = float(train_df[AMOUNT_COL].median())
    frames = [apply_delay(f, bounds.t_train, delay_cfg, amount_ref) for f in (train_df, val_df, test_df)]
    del train_df, val_df, test_df
    diag = pnu_diagnostics(frames[0])
    print(f"[3/5] Delay mode={delay_cfg['mode']} W={delay_cfg.get('maturity_window_days')} d | "
          f"train view: P={diag['n_P']:,} N={diag['n_N']:,} U={diag['n_U']:,} "
          f"({diag['share_U_%']:.1f}% U) | frauds hidden: {diag['n_fraud_hidden']:,} "
          f"({diag['hidden_fraud_%']:.1f}%) | frauds in N (label noise): {diag['n_fraud_in_N']:,}")

    # 4) fit + build ---------------------------------------------------------
    print("[4/5] Fitting encoder / super-nodes on TRAIN and building graphs ...")
    builder = GraphBuilder(GraphConfig.from_dict(cfg.get("graph"))).fit(*frames, bounds)
    for name, st in builder.entity_stats.items():
        print(f"      entity {name:<7} train entities={st['n_entities_train']:>8,.0f} "
              f"deg p50/p90/p99/max={st['deg_p50']:.0f}/{st['deg_p90']:.0f}/{st['deg_p99']:.0f}/{st['deg_max']:.0f} "
              f"cap={st['max_degree_cap']:.0f} -> removed {st['n_super_removed']:.0f} super-nodes "
              f"({st['train_edges_dropped_%']:.1f}% of train edges)")

    graph_summary: dict[str, Any] = {}
    cumulative: list[pd.DataFrame] = []
    for phase, frame in zip(("train", "val", "test"), frames):
        cumulative.append(frame)
        data = builder.build(pd.concat(cumulative, ignore_index=True), phase)
        validate_phase_graph(data)  # fail fast: nothing leaky is ever written to disk
        path = out_dir / f"{phase}.pt"
        torch.save(data, path)
        size_mb = path.stat().st_size / 1e6
        graph_summary[phase] = {
            "n_transactions": int(data[TXN].num_nodes),
            "n_seeds": int(data[TXN].seed_mask.sum()),
            "entity_nodes": {nt: int(data[nt].num_nodes) for nt in data.node_types if nt != TXN},
            "edges": {"/".join(et): int(data[et].edge_index.size(1)) for et in data.edge_types},
            "file_mb": round(size_mb, 1),
        }
        print(f"      saved {path.name}: txn={graph_summary[phase]['n_transactions']:,} "
              f"seeds={graph_summary[phase]['n_seeds']:,} entities={graph_summary[phase]['entity_nodes']} "
              f"({size_mb:,.0f} MB)")
        del data

    # 5) metadata ------------------------------------------------------------
    torch.save(builder.state_dict(), out_dir / "builder_state.pt")
    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": cfg,
        "boundaries": {"t_start": bounds.t_start, "t_train": bounds.t_train, "t_val": bounds.t_val, "t_end": bounds.t_end,
                       "t_train_day": bounds.t_train_day, "t_val_day": bounds.t_val_day},
        "amount_ref_train_median": amount_ref,
        "pnu_train_diagnostics": diag,
        "feature_dim": len(builder.encoder.feature_names),
        "dropped_missing_cols": getattr(builder.encoder, "dropped_missing_cols", {}),
        "entity_stats": builder.entity_stats,
        "graphs": graph_summary,
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "torch_geometric": torch_geometric.__version__, "pandas": pd.__version__, "numpy": np.__version__},
    }
    (out_dir / "meta.json").write_text(json.dumps(_jsonable(meta), indent=2), encoding="utf-8")
    print(f"[5/5] Wrote builder_state.pt and meta.json to {out_dir}  (total {time.time() - t0:,.0f}s)")


if __name__ == "__main__":
    main()