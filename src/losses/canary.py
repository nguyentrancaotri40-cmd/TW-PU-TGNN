"""Canary hooks for the loss layer and the model (reusable by tests and, in debug mode, by the trainer).

Invariants checked
  * naive_bce     : every U row has a NON-zero gradient (U is read as negative).
  * maturity_mask : every U row has exactly ZERO gradient and the loss value does not change when U
                    logits change.
  * tw_nndf       : U rows have zero gradient ONLY while the negative-risk clamp is active
                    (``neg_risk_raw < 0``); outside that regime the U gradient is non-zero by design.
  * model         : logits do not depend on label fields (y_obs, y_true, pnu) of the batch.
"""
from __future__ import annotations

import copy
from typing import Callable, Iterable

import torch
from torch import Tensor

LossFn = Callable[..., Tensor]


def logits_grad(loss_fn: LossFn, preds: Tensor, y_obs: Tensor, pnu_status: Tensor, **kwargs) -> tuple[Tensor, Tensor]:
    """Return ``(loss value, d loss / d logits)``; the gradient is all zeros if the loss is detached."""
    p = preds.detach().clone().float().requires_grad_(True)
    loss = loss_fn(p, y_obs, pnu_status, **kwargs)
    if not loss.requires_grad:
        return loss.detach(), torch.zeros_like(p)
    (grad,) = torch.autograd.grad(loss, p, allow_unused=True)
    return loss.detach(), torch.zeros_like(p) if grad is None else grad


def assert_zero_grad_on_status(grad: Tensor, pnu_status: Tensor, status: int, name: str = "loss") -> None:
    rows = pnu_status.reshape(-1) == status
    if not bool(rows.any()):
        raise AssertionError(f"{name}: vacuous check, no row with status {status}.")
    if bool((grad[rows] != 0).any()):
        raise AssertionError(f"{name}: rows with status {status} received a non-zero gradient.")


def assert_nonzero_grad_on_status(
    grad: Tensor, pnu_status: Tensor, status: int, name: str = "loss", require_all: bool = True
) -> None:
    rows = pnu_status.reshape(-1) == status
    if not bool(rows.any()):
        raise AssertionError(f"{name}: vacuous check, no row with status {status}.")
    nz = grad[rows] != 0
    ok = bool(nz.all()) if require_all else bool(nz.any())
    if not ok:
        raise AssertionError(f"{name}: rows with status {status} have a zero gradient where non-zero is expected.")


def assert_loss_invariant_to_status(
    loss_fn: LossFn,
    preds: Tensor,
    y_obs: Tensor,
    pnu_status: Tensor,
    status: int,
    *,
    scale: float = 5.0,
    seed: int = 0,
    **kwargs,
) -> None:
    """The loss value must be bit-identical after perturbing the logits of every row with ``status``."""
    rows = pnu_status.reshape(-1) == status
    if not bool(rows.any()):
        raise AssertionError(f"vacuous check, no row with status {status}.")
    g = torch.Generator().manual_seed(seed)
    noise = torch.randn(int(rows.sum()), generator=g).to(preds.device, preds.dtype) * scale
    other = preds.detach().clone()
    other[rows] = other[rows] + noise
    base = loss_fn(preds.detach(), y_obs, pnu_status, **kwargs)
    changed = loss_fn(other, y_obs, pnu_status, **kwargs)
    if not torch.equal(base, changed):
        raise AssertionError(f"loss changed ({float(base)!r} -> {float(changed)!r}) when status-{status} logits changed.")


def model_leakage_hook(
    model: torch.nn.Module,
    batch,
    label_fields: Iterable[str] = ("y_obs", "y_true", "pnu"),
    node_type: str = "transaction",
    seed: int = 0,
) -> None:
    """Scramble the label fields of the batch and require identical logits (eval mode, no dropout).

    Catches any path through which a label could reach the forward pass (e.g. a future label feature).
    """
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            base = model(batch)
            scrambled = copy.deepcopy(batch)
            store = scrambled[node_type]
            g = torch.Generator().manual_seed(seed)
            changed = False
            for f in label_fields:
                if f in store:
                    v = store[f]
                    perm = torch.randperm(v.numel(), generator=g).to(v.device)
                    new = v[perm]
                    changed = changed or not torch.equal(new, v)
                    store[f] = new
            if not changed:
                raise AssertionError("vacuous hook: no label field found, or scrambling changed nothing.")
            other = model(scrambled)
    finally:
        model.train(was_training)
    if not torch.equal(base, other):
        raise AssertionError("model output depends on label fields: a label reaches the forward pass.")