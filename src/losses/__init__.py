"""Shared helpers for the four label-delay losses (all take LOGITS and return a scalar).

Common signature::

    loss(preds, y_obs, pnu_status, *, age_days=None, in_train_view=None)

preds         [B] (or [B, 1]) logits of the seed transactions
y_obs         [B] observed label at the label cut (0/1)
pnu_status    [B] integer code 0=P, 1=N, 2=U (``src.graph.builder.PNU_CODES``)
age_days      [B] days between the transaction and the label cut (``compute_age_days``)
in_train_view [B] bool, True before the label cut (None -> all True)

``y_true`` is never an argument: no loss can see the hidden labels. A batch with nothing to learn
from returns a zero that stays connected to the autograd graph.

The loss classes are imported lazily (module ``__getattr__``) so that each loss module can import
these helpers from the package without an import cycle.
"""
from __future__ import annotations

import importlib
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from src.data.loader import SECONDS_PER_DAY
from src.graph.builder import PNU_CODES

PNU_P: int = PNU_CODES["P"]
PNU_N: int = PNU_CODES["N"]
PNU_U: int = PNU_CODES["U"]


def prepare_inputs(
    preds: Tensor,
    y_obs: Tensor,
    pnu_status: Tensor,
    in_train_view: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Validate shapes; return ``(logits [B] float, y [B] float, pnu [B] long, view [B] bool)``."""
    if preds.dim() == 2 and preds.size(1) == 1:
        preds = preds.squeeze(1)
    if preds.dim() != 1:
        raise ValueError(f"preds must have shape [B] or [B, 1], got {tuple(preds.shape)}.")
    n = preds.numel()
    y = y_obs.reshape(-1)
    pnu = pnu_status.reshape(-1)
    if y.numel() != n or pnu.numel() != n:
        raise ValueError("preds, y_obs and pnu_status must have the same length.")
    if in_train_view is None:
        view = torch.ones(n, dtype=torch.bool, device=preds.device)
    else:
        view = in_train_view.reshape(-1).bool()
        if view.numel() != n:
            raise ValueError("in_train_view must have the same length as preds.")
    return preds, y.to(preds.dtype), pnu.long(), view


def zero_loss(preds: Tensor) -> Tensor:
    """Scalar 0 that keeps the autograd graph (an empty selection must not break ``backward``)."""
    return preds.sum() * 0.0


def compute_age_days(time: Tensor, label_cut: float) -> Tensor:
    """Age in days of each transaction at the label cut (clamped at 0)."""
    return ((float(label_cut) - time.to(torch.float64)).clamp_min(0.0) / float(SECONDS_PER_DAY)).to(torch.float32)


_LAZY: dict[str, tuple[str, str]] = {
    "NaiveBCELoss": ("src.losses.naive_bce", "NaiveBCELoss"),
    "naive_bce_loss": ("src.losses.naive_bce", "naive_bce_loss"),
    "MaturityMaskLoss": ("src.losses.maturity_mask", "MaturityMaskLoss"),
    "maturity_mask_loss": ("src.losses.maturity_mask", "maturity_mask_loss"),
    "DelayIPWLoss": ("src.losses.delay_ipw", "DelayIPWLoss"),
    "delay_ipw_loss": ("src.losses.delay_ipw", "delay_ipw_loss"),
    "lam_from_median_delay": ("src.losses.delay_ipw", "lam_from_median_delay"),
    "TimeWindowNNDFLoss": ("src.losses.tw_nndf", "TimeWindowNNDFLoss"),
    "tw_nndf_loss": ("src.losses.tw_nndf", "tw_nndf_loss"),
    "estimate_class_prior": ("src.losses.tw_nndf", "estimate_class_prior"),
}
LOSS_CLASSES: dict[str, str] = {
    "naive_bce": "NaiveBCELoss",
    "maturity_mask": "MaturityMaskLoss",
    "delay_ipw": "DelayIPWLoss",
    "tw_nndf": "TimeWindowNNDFLoss",
}


def __getattr__(name: str) -> Any:  # lazy re-exports (PEP 562)
    if name in _LAZY:
        module, attr = _LAZY[name]
        return getattr(importlib.import_module(module), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def build_loss(name: str, **kwargs: Any) -> nn.Module:
    """Instantiate a loss by name, e.g. ``build_loss('delay_ipw', lam=0.23)``."""
    if name not in LOSS_CLASSES:
        raise KeyError(f"Unknown loss '{name}'. Available: {sorted(LOSS_CLASSES)}")
    return __getattr__(LOSS_CLASSES[name])(**kwargs)


__all__ = [
    "PNU_P", "PNU_N", "PNU_U", "prepare_inputs", "zero_loss", "compute_age_days", "build_loss",
    "LOSS_CLASSES", *sorted(_LAZY),
]