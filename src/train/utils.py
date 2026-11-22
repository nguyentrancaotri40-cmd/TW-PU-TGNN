"""Training helpers: config merge, seeding, device/timing/VRAM, JSON, LR warm-up, early stopping."""
from __future__ import annotations

import copy
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml

# The benchmark matrix (72 vs 90 runs) is NOT frozen yet: every result row carries this tag.
# Change it to "frozen" only after the matrix, thresholds and sweeps are fixed and committed.
PREREG_STATUS: str = "unfrozen"


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def deep_update(base: dict[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``overlay`` into a copy of ``base``."""
    out = copy.deepcopy(base)
    for k, v in overlay.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(base_path: str | Path, delay_path: str | Path | None = None) -> dict[str, Any]:
    """``base.yaml`` (+ an optional ``delay_*.yaml`` overlay that must contain a ``delay:`` section)."""
    cfg = yaml.safe_load(Path(base_path).read_text(encoding="utf-8")) or {}
    if delay_path is not None:
        overlay = yaml.safe_load(Path(delay_path).read_text(encoding="utf-8")) or {}
        if "delay" not in overlay:
            raise KeyError(f"{delay_path} must contain a top-level 'delay:' section.")
        cfg = deep_update(cfg, overlay)
    return cfg


# --------------------------------------------------------------------------- #
# Reproducibility / device / timing
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    """Seed python, numpy and torch. GPU scatter ops are NOT bit-reproducible, and whether the pyg-lib
    sampler honours ``torch.manual_seed`` was not verified (KHÔNG CHẮC): expect seed-to-seed variance."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(name)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return dev


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory_mb(device: torch.device) -> dict[str, float] | None:
    """Peak allocated / reserved CUDA memory in MB since the last reset (``None`` on CPU)."""
    if device.type != "cuda":
        return None
    return {
        "allocated": torch.cuda.max_memory_allocated(device) / 2**20,
        "reserved": torch.cuda.max_memory_reserved(device) / 2**20,
    }


class Stopwatch:
    """Accumulating wall-clock timer that synchronises CUDA before reading the clock."""

    def __init__(self, device: torch.device | None = None) -> None:
        self.device = device
        self.total = 0.0
        self._t0: float | None = None

    def start(self) -> None:
        if self.device is not None:
            sync(self.device)
        self._t0 = time.perf_counter()

    def stop(self) -> float:
        if self._t0 is None:
            raise RuntimeError("Stopwatch.stop() before start().")
        if self.device is not None:
            sync(self.device)
        dt = time.perf_counter() - self._t0
        self.total += dt
        self._t0 = None
        return dt


# --------------------------------------------------------------------------- #
# Schedules
# --------------------------------------------------------------------------- #
def warmup_factor(step: int, warmup_steps: int) -> float:
    """Linear LR multiplier in (0, 1]: ``(step + 1) / warmup_steps`` until warm-up ends, then 1."""
    if warmup_steps <= 0:
        return 1.0
    return min(1.0, (step + 1) / float(warmup_steps))


class EarlyStopper:
    """Early stopping on a metric to MAXIMISE (PR-AUC, ROC-AUC, recall).

    Epochs before ``start_epoch`` (the warm-up) can still become the best epoch, but a non-improving
    epoch only counts towards ``patience`` from ``start_epoch`` on. NaN/inf never counts as improvement.
    """

    def __init__(self, patience: int, min_delta: float = 0.0, start_epoch: int = 1) -> None:
        if patience < 1:
            raise ValueError("patience must be >= 1.")
        self.patience = patience
        self.min_delta = float(min_delta)
        self.start_epoch = start_epoch
        self.best: float | None = None
        self.best_epoch = 0
        self.bad_epochs = 0

    def update(self, value: float | None, epoch: int) -> bool:
        """Record the epoch's metric; return True if it is a new best."""
        finite = value is not None and math.isfinite(value)
        improved = finite and (self.best is None or value > self.best + self.min_delta)
        if improved:
            self.best, self.best_epoch, self.bad_epochs = float(value), epoch, 0
        elif epoch >= self.start_epoch:
            self.bad_epochs += 1
        return bool(improved)

    @property
    def should_stop(self) -> bool:
        return self.bad_epochs >= self.patience


# --------------------------------------------------------------------------- #
# Output helpers
# --------------------------------------------------------------------------- #
def jsonable(obj: Any) -> Any:
    """Make numpy / torch / Path values JSON-safe; non-finite floats become ``null``."""
    if isinstance(obj, Mapping):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    if isinstance(obj, torch.Tensor):
        return jsonable(obj.detach().cpu().tolist())
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, Path):
        return str(obj)
    return obj if (obj is None or isinstance(obj, str)) else str(obj)


def write_json(path: str | Path, obj: Any) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(jsonable(obj), indent=2), encoding="utf-8")
    return p


def make_run_id(delay: str, loss: str, backbone: str, use_temporal: bool, lp: bool, seed: int) -> str:
    """Stable id, e.g. ``delay_high_tw_nndf_sage_gru_nolp_s42`` (also the auto-resume key in round 7)."""
    return "_".join([delay, loss, backbone, "gru" if use_temporal else "nogru", "lp" if lp else "nolp", f"s{seed}"])