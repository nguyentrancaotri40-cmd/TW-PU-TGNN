#!/usr/bin/env python
"""Benchmark runner with auto-resume.

    python scripts/run_benchmark.py --dry_run                       # plan + counts, nothing is trained
    python scripts/run_benchmark.py --max_hours 10 --device auto    # real run; re-run the same command to resume

Design
------
* The matrix comes from configs/benchmark.yaml (groups with ``enabled`` flags). It is NOT frozen: every result row
  carries ``prereg_status`` (src.train.utils.PREREG_STATUS).
* One CSV row is appended (flush + fsync) after each finished run. Resume = skip every run_id whose row has
  status == "ok". A run killed mid-way leaves no row and is simply re-run.
* Failed runs are written with status == "failed" and are NOT retried unless --retry_failed (a deterministic bug
  could otherwise burn hours on every restart).
* ``cfg_hash`` (sampler/model/train/eval/loss/features + lam_by_delay) is stored in every row; resuming with a
  different hash is refused unless --allow_config_change, so two protocols never mix in one CSV.
* --max_hours stops BEFORE a run that would not fit (estimate = slowest finished run of this invocation, or
  --assume_run_minutes for the first one). Exit code 0, re-run to continue.
* Seeds [42, 43, 44]: GPU runs are not bit-reproducible and the pyg-lib sampler seeding is unverified (KHÔNG CHẮC).
"""
from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import json
import os
import shutil
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml

from src.train.utils import PREREG_STATUS, deep_update, jsonable, load_config, make_run_id, write_json

LOSSES = ("naive_bce", "maturity_mask", "delay_ipw", "tw_nndf")
HASHED_SECTIONS = ("sampler", "model", "train", "eval", "loss", "features")

RESULT_COLUMNS: list[str] = [
    "run_id", "status", "group", "delay", "loss", "oracle", "backbone", "use_temporal", "lp_features", "seed",
    "prior_override", "in_dim", "n_params", "best_epoch", "epochs_run",
    "val_pr_auc", "val_roc_auc", "test_pr_auc", "test_roc_auc", "test_recall_at_fpr",
    "test_recall_at_valthr", "test_fpr_at_valthr", "test_efl", "test_efl_ratio",
    "lam", "nndf_prior", "nndf_prior_estimated", "diag_true_fraud_rate_in_U",
    "nndf_clamped_best", "nndf_clamped_mean", "nndf_neg_risk_raw_best", "nndf_pos_risk_best", "n_empty_batches_total",
    "train_seconds", "wall_seconds", "peak_vram_alloc_mb", "prereg_status", "cfg_hash", "meta_sha",
    "finished_utc", "error",
]


# --------------------------------------------------------------------------- #
# Plan
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RunSpec:
    group: str
    delay: str          # directory name under data/processed/
    loss: str
    oracle: bool
    backbone: str
    use_temporal: bool
    lp: bool
    seed: int
    prior: float | None = None

    @property
    def label(self) -> str:
        return "oracle" if self.oracle else self.loss

    @property
    def run_id(self) -> str:
        rid = make_run_id(self.delay, self.label, self.backbone, self.use_temporal, self.lp, self.seed)
        return rid if self.prior is None else f"{rid}_pi{self.prior:g}"


def build_plan(bench: Mapping[str, Any]) -> list[RunSpec]:
    """Expand the benchmark spec into an ordered list of runs; duplicate run_ids are an error."""
    seeds = [int(s) for s in bench["seeds"]]
    default_delays = list(bench["delays"])
    default_backbone = str(bench.get("backbone", "sage"))
    plan: list[RunSpec] = []
    seen: set[str] = set()
    for gname, g in bench["groups"].items():
        if not g.get("enabled", True):
            continue
        oracle = bool(g.get("oracle", False))
        losses = ["naive_bce"] if oracle else list(g["losses"])
        bad = [x for x in losses if x not in LOSSES]
        if bad:
            raise ValueError(f"group '{gname}': unknown loss {bad}; allowed {LOSSES}")
        prior_map = g.get("priors_by_delay")
        for delay in g.get("delays", default_delays):
            priors: list[float | None] = [None] if prior_map is None else list(prior_map.get(delay, []))
            for loss in losses:
                for prior in priors:
                    if prior is not None and (loss != "tw_nndf" or oracle):
                        raise ValueError(f"group '{gname}': priors_by_delay only makes sense for tw_nndf")
                    for seed in seeds:
                        spec = RunSpec(group=gname, delay=delay, loss=loss, oracle=oracle,
                                       backbone=str(g.get("backbone", default_backbone)),
                                       use_temporal=bool(g.get("use_temporal", True)), lp=bool(g.get("lp", False)),
                                       seed=seed, prior=None if prior is None else float(prior))
                        if spec.run_id in seen:
                            raise ValueError(f"duplicate run_id {spec.run_id} (group '{gname}')")
                        seen.add(spec.run_id)
                        plan.append(spec)
    return plan


def resolve_cfg(base_cfg: Mapping[str, Any], bench: Mapping[str, Any], spec: RunSpec, allow_default_lam: bool) -> dict:
    """Per-run config. delay_ipw gets its lambda from ``lam_by_delay`` ONLY (base.yaml's lam is ignored)."""
    cfg = copy.deepcopy(dict(base_cfg))
    if spec.loss == "delay_ipw" and not spec.oracle:
        lam = (bench.get("lam_by_delay") or {}).get(spec.delay)
        if lam is None and not allow_default_lam:
            raise KeyError(f"lam_by_delay['{spec.delay}'] is not set in benchmark.yaml (decision C). "
                           f"Set lambda = ln2 / median_delay_days of that delay, or pass --allow_default_lam (smoke only).")
        loss_cfg = cfg.setdefault("loss", {})
        d = dict(loss_cfg.get("delay_ipw") or {})
        d.pop("lam", None)
        if lam is not None:
            d["lam"] = float(lam)
        loss_cfg["delay_ipw"] = d
    return cfg


def hash_config(base_cfg: Mapping[str, Any], bench: Mapping[str, Any]) -> str:
    payload = {k: base_cfg.get(k) for k in HASHED_SECTIONS}
    payload["lam_by_delay"] = bench.get("lam_by_delay")
    payload["backbone"] = bench.get("backbone")
    return hashlib.sha256(json.dumps(jsonable(payload), sort_keys=True).encode()).hexdigest()[:12]


# --------------------------------------------------------------------------- #
# Results file
# --------------------------------------------------------------------------- #
def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != RESULT_COLUMNS:
            raise RuntimeError(f"{path}: header differs from RESULT_COLUMNS (older version?). Move it aside or migrate it.")
        return list(reader)


def append_row(path: Path, row: Mapping[str, Any]) -> None:
    """Append one row and fsync. A truncated last line (killed session) is terminated first."""
    path.parent.mkdir(parents=True, exist_ok=True)
    new = (not path.exists()) or path.stat().st_size == 0
    needs_newline = False
    if not new:
        with path.open(newline="", encoding="utf-8") as f:
            header = next(csv.reader(f), None)
        if header != RESULT_COLUMNS:
            raise RuntimeError(f"{path}: header differs from RESULT_COLUMNS; refusing to append.")
        with path.open("rb") as fb:
            fb.seek(-1, os.SEEK_END)
            needs_newline = fb.read(1) != b"\n"
    with path.open("a", newline="", encoding="utf-8") as f:
        if needs_newline:
            f.write("\n")
        w = csv.DictWriter(f, fieldnames=RESULT_COLUMNS, extrasaction="ignore", restval="")
        if new:
            w.writeheader()
        w.writerow(jsonable(dict(row)))
        f.flush()
        os.fsync(f.fileno())


def _best_history_row(history: Sequence[Mapping[str, Any]], best_epoch: int) -> Mapping[str, Any]:
    for h in history:
        if int(h.get("epoch", -1)) == int(best_epoch):
            return h
    return {}


def _meta_sha(pdir: Path) -> str:
    p = pdir / "meta.json"
    return hashlib.sha256(p.read_bytes()).hexdigest()[:12] if p.exists() else ""


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def finalize_row(spec: RunSpec, res: Mapping[str, Any], *, wall: float, cfg_hash: str, meta_sha: str) -> dict[str, Any]:
    """run_one() row + benchmark fields + tw_nndf diagnostics at the best epoch (the model that is evaluated)."""
    row = dict(res["row"])
    det = res["details"]
    info, hist = det.get("info", {}), det.get("fit", {}).get("history", [])
    row.update(run_id=spec.run_id, status="ok", group=spec.group, prior_override=spec.prior, wall_seconds=wall,
               cfg_hash=cfg_hash, meta_sha=meta_sha, finished_utc=_now_utc(), error="",
               lam=info.get("lam"), nndf_prior=info.get("prior"), nndf_prior_estimated=info.get("prior_estimated"),
               diag_true_fraud_rate_in_U=info.get("diag_true_fraud_rate_in_U"))
    if hist and "nndf_clamped_rate" in hist[0]:
        h = _best_history_row(hist, int(row["best_epoch"]))
        row["nndf_clamped_best"] = h.get("nndf_clamped_rate")
        row["nndf_neg_risk_raw_best"] = h.get("nndf_neg_risk_raw")
        row["nndf_pos_risk_best"] = h.get("nndf_pos_risk")
        row["nndf_clamped_mean"] = sum(float(x["nndf_clamped_rate"]) for x in hist) / len(hist)
        row["n_empty_batches_total"] = sum(float(x.get("n_empty_batches", 0.0)) for x in hist)
    return row


def failed_row(spec: RunSpec, exc: BaseException, *, wall: float, cfg_hash: str) -> dict[str, Any]:
    msg = f"{type(exc).__name__}: {exc}".replace("\n", " ")[:300]
    return {"run_id": spec.run_id, "status": "failed", "group": spec.group, "delay": spec.delay, "loss": spec.label,
            "oracle": spec.oracle, "backbone": spec.backbone, "use_temporal": spec.use_temporal,
            "lp_features": spec.lp, "seed": spec.seed, "prior_override": spec.prior, "wall_seconds": wall,
            "prereg_status": PREREG_STATUS, "cfg_hash": cfg_hash, "finished_utc": _now_utc(), "error": msg}


def _free_memory() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 - cleanup must never kill the benchmark loop
        pass


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #
def execute(
    plan: Sequence[RunSpec], base_cfg: Mapping[str, Any], bench: Mapping[str, Any], *,
    processed_root: Path, results_path: Path, details_dir: Path, run_fn: Callable[..., dict[str, Any]],
    device: str = "auto", max_hours: float | None = None, assume_run_seconds: float | None = None,
    retry_failed: bool = False, allow_default_lam: bool = False, allow_config_change: bool = False,
    max_train_batches: int | None = None, clock: Callable[[], float] = time.monotonic,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    cfg_hash = hash_config(base_cfg, bench)
    rows = read_rows(results_path)
    stale = {r["cfg_hash"] for r in rows if r.get("status") == "ok" and r.get("cfg_hash")} - {cfg_hash}
    if stale and not allow_config_change:
        raise RuntimeError(f"{results_path} holds runs with config hash {sorted(stale)} but the current one is "
                           f"{cfg_hash}. Use another --results file, or --allow_config_change if you accept mixing.")
    ok = {r["run_id"] for r in rows if r.get("status") == "ok"}
    failed = {r["run_id"] for r in rows if r.get("status") == "failed"} - ok
    todo = [s for s in plan if s.run_id not in ok and (retry_failed or s.run_id not in failed)]
    n_skipped_failed = sum(1 for s in plan if s.run_id in failed and not retry_failed)
    cfgs = {s.run_id: resolve_cfg(base_cfg, bench, s, allow_default_lam) for s in todo}  # fail NOW, not after hours

    log(f"[benchmark] plan={len(plan)} done={len([s for s in plan if s.run_id in ok])} "
        f"to_run={len(todo)} skipped_failed={n_skipped_failed} cfg_hash={cfg_hash} prereg={PREREG_STATUS}")
    t_start = clock()
    walls: list[float] = [float(assume_run_seconds)] if assume_run_seconds else []
    n_ok = n_fail = 0
    stopped = False
    for i, spec in enumerate(todo, 1):
        if max_hours is not None and walls and (clock() - t_start) + max(walls) > max_hours * 3600.0:
            stopped = True
            log(f"[benchmark] time budget: the next run (~{max(walls) / 60:.0f} min) would not fit in {max_hours} h. "
                f"Stopping; re-run the same command to resume.")
            break
        log(f"[benchmark] ({i}/{len(todo)}) {spec.run_id}")
        pdir = processed_root / spec.delay
        extra = {} if spec.prior is None else {"prior": spec.prior}
        t0 = clock()
        try:
            res = run_fn(cfgs[spec.run_id], pdir, loss=spec.loss, oracle=spec.oracle, backbone=spec.backbone,
                         use_temporal=spec.use_temporal, lp=spec.lp, seed=spec.seed, device=device,
                         save_dir=None, max_train_batches=max_train_batches, **extra)
            wall = clock() - t0
            row = finalize_row(spec, res, wall=wall, cfg_hash=cfg_hash, meta_sha=_meta_sha(pdir))
            write_json(details_dir / f"{spec.run_id}.json", res["details"])
            walls.append(wall)
            n_ok += 1
        except KeyboardInterrupt:
            raise
        except Exception as exc:  # noqa: BLE001 - includes CUDA OOM; one bad run must not stop the matrix
            wall = clock() - t0
            log(traceback.format_exc())
            row = failed_row(spec, exc, wall=wall, cfg_hash=cfg_hash)
            n_fail += 1
        append_row(results_path, row)
        _free_memory()
    return {"n_todo": len(todo), "n_ok": n_ok, "n_failed": n_fail, "n_skipped_failed": n_skipped_failed,
            "n_remaining": len(todo) - n_ok - n_fail, "stopped_for_budget": stopped, "cfg_hash": cfg_hash}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def summarize_plan(plan: Sequence[RunSpec]) -> str:
    by_group: dict[str, int] = {}
    by_delay: dict[str, int] = {}
    for s in plan:
        by_group[s.group] = by_group.get(s.group, 0) + 1
        by_delay[s.delay] = by_delay.get(s.delay, 0) + 1
    return (f"{len(plan)} runs | by group: {by_group} | by delay: {by_delay} | "
            f"seeds: {sorted({s.seed for s in plan})}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the DelayGraphBench matrix with auto-resume.")
    p.add_argument("--config", default=str(REPO_ROOT / "configs" / "base.yaml"))
    p.add_argument("--benchmark", default=str(REPO_ROOT / "configs" / "benchmark.yaml"))
    p.add_argument("--processed_root", default=str(REPO_ROOT / "data" / "processed"))
    p.add_argument("--results", default=str(REPO_ROOT / "results" / "benchmark_results.csv"))
    p.add_argument("--details_dir", default=str(REPO_ROOT / "results" / "runs"))
    p.add_argument("--seed_results", default=None,
                   help="CSV from a previous session (e.g. a Kaggle dataset); copied to --results if that file is absent")
    p.add_argument("--device", default="auto")
    p.add_argument("--max_hours", type=float, default=None)
    p.add_argument("--assume_run_minutes", type=float, default=None, help="duration guess for the first run of a session")
    p.add_argument("--only_group", nargs="*", default=None)
    p.add_argument("--seeds", type=int, nargs="*", default=None, help="override benchmark.seeds (smoke tests)")
    p.add_argument("--delays", nargs="*", default=None, help="override benchmark.delays (smoke tests)")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--retry_failed", action="store_true")
    p.add_argument("--allow_default_lam", action="store_true", help="use the W/2 default lambda (smoke only)")
    p.add_argument("--allow_config_change", action="store_true")
    p.add_argument("--smoke", action="store_true", help="2 epochs x 20 batches, separate results file")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--est_epoch_minutes", type=float, default=None, help="(train+val) epoch minutes for an hours bound")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None, run_fn: Callable[..., dict[str, Any]] | None = None) -> int:
    args = parse_args(argv)
    base_cfg = load_config(args.config)
    bench = copy.deepcopy(yaml.safe_load(Path(args.benchmark).read_text(encoding="utf-8"))["benchmark"])
    if args.seeds:
        bench["seeds"] = args.seeds
    if args.delays:
        bench["delays"] = args.delays
    plan = build_plan(bench)
    if args.only_group:
        plan = [s for s in plan if s.group in args.only_group]
    if args.limit:
        plan = plan[: args.limit]

    results, details = Path(args.results), Path(args.details_dir)
    max_batches, allow_lam = None, args.allow_default_lam
    if args.smoke:
        base_cfg = deep_update(base_cfg, {"train": {"epochs": 2, "warmup_epochs": 1}})
        results = REPO_ROOT / "results" / "smoke" / "benchmark_results_smoke.csv"
        details = REPO_ROOT / "results" / "smoke" / "runs"
        max_batches, allow_lam = 20, True
    root = Path(args.processed_root)

    print(f"[benchmark] pre-registration status: {PREREG_STATUS.upper()} (the run matrix is not frozen)")
    print(f"[benchmark] {summarize_plan(plan)}")
    missing_dirs = sorted({s.delay for s in plan if not (root / s.delay / "meta.json").exists()})
    missing_lam = sorted({s.delay for s in plan if s.loss == "delay_ipw" and not s.oracle
                          and (bench.get("lam_by_delay") or {}).get(s.delay) is None})
    if args.dry_run:
        if args.est_epoch_minutes:
            epochs = int(base_cfg["train"]["epochs"])
            print(f"[benchmark] upper bound (no early stopping): {len(plan)} x {epochs} epochs x "
                  f"{args.est_epoch_minutes} min = {len(plan) * epochs * args.est_epoch_minutes / 60:.1f} h")
        for s in plan[:5]:
            print(f"  e.g. {s.run_id}")
        if missing_dirs:
            print(f"[benchmark] WARNING missing processed dirs / meta.json: {missing_dirs}")
        if missing_lam:
            print(f"[benchmark] WARNING lam_by_delay not set for {missing_lam}: a real run would refuse (decision C)")
        return 0
    if missing_dirs:
        print(f"[benchmark] ERROR missing processed dirs / meta.json under {root}: {missing_dirs}")
        return 2
    if args.seed_results and not results.exists() and Path(args.seed_results).exists():
        results.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.seed_results, results)
        print(f"[benchmark] seeded {results} from {args.seed_results}")
    if run_fn is None:
        from src.train.trainer import run_one   # lazy: keep --dry_run and planning tests free of torch work
        run_fn = run_one

    out = execute(plan, base_cfg, bench, processed_root=root, results_path=results, details_dir=details,
                  run_fn=run_fn, device=args.device, max_hours=args.max_hours,
                  assume_run_seconds=None if args.assume_run_minutes is None else args.assume_run_minutes * 60.0,
                  retry_failed=args.retry_failed, allow_default_lam=allow_lam,
                  allow_config_change=args.allow_config_change, max_train_batches=max_batches)
    print(f"[benchmark] finished: ok={out['n_ok']} failed={out['n_failed']} remaining={out['n_remaining']} "
          f"skipped_failed={out['n_skipped_failed']} budget_stop={out['stopped_for_budget']} -> {results}")
    return 1 if out["n_failed"] else 0


if __name__ == "__main__":
    sys.exit(main())