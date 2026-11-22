"""Delay-weighted BCE: observed zeros are weighted by how old they are."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.losses import PNU_P, PNU_U, prepare_inputs, zero_loss


def lam_from_median_delay(median_days: float) -> float:
    """Rate ``lambda`` (1/day) of an exponential delay with the given median: ``ln 2 / median``.

    Only an approximation for Weibull delays (KHÔNG CHẮC, CẦN VERIFY): choose ``lam`` by a sweep.
    """
    if median_days <= 0:
        raise ValueError("median_days must be > 0.")
    return math.log(2.0) / median_days


class DelayIPWLoss(nn.Module):
    """Weighted BCE with ``w = 1 - exp(-lam * age_days)`` on observed zeros; P rows weigh 1.

    ``w(age)`` is the probability that a fraud of that age has already been reported when the delay is
    exponential with rate ``lam``: a fresh zero (age ~ 0) is almost surely "not reported yet" and barely
    counts; an old zero counts fully. KHÔNG CHẮC, CẦN VERIFY: this is the heuristic of the brief, NOT a
    proven unbiased estimator.

    Ambiguity in the brief ("BCE on P/N, weight N by age"), resolved by ``negative_scope``:
      * ``"all_zeros"`` (default): every observed zero (N and U) is weighted by age. Rows in N are all
        older than the maturity window, so their weight is ~1 and the age effect lives in U.
      * ``"mature_only"``: literal reading, U is masked and only N is weighted. Then the loss is
        nearly identical to ``MaturityMaskLoss`` (N weights are close to 1). Kept as an ablation.

    Args:
        lam: rate in 1/day (see ``lam_from_median_delay``).
        normalize: ``"weights"`` divides by the sum of weights (scale comparable to BCE);
            ``"count"`` divides by the number of rows in scope.
        min_weight: floor of the zero-row weights (0 -> a row at age 0 has no gradient).
        negative_scope: ``"all_zeros"`` or ``"mature_only"``.
    """

    def __init__(
        self,
        lam: float,
        normalize: str = "weights",
        min_weight: float = 0.0,
        negative_scope: str = "all_zeros",
    ) -> None:
        super().__init__()
        if lam <= 0:
            raise ValueError("lam must be > 0.")
        if normalize not in ("weights", "count"):
            raise ValueError("normalize must be 'weights' or 'count'.")
        if not 0.0 <= min_weight <= 1.0:
            raise ValueError("min_weight must be in [0, 1].")
        if negative_scope not in ("all_zeros", "mature_only"):
            raise ValueError("negative_scope must be 'all_zeros' or 'mature_only'.")
        self.lam = float(lam)
        self.normalize = normalize
        self.min_weight = float(min_weight)
        self.negative_scope = negative_scope

    def weights(self, pnu: Tensor, age_days: Tensor) -> Tensor:
        """Per-row weight before masking by ``in_train_view``."""
        age = age_days.clamp(min=0.0)
        w_zero = (1.0 - torch.exp(-self.lam * age)).clamp(min=self.min_weight)
        w = torch.where(pnu == PNU_P, torch.ones_like(w_zero), w_zero)
        if self.negative_scope == "mature_only":
            w = torch.where(pnu == PNU_U, torch.zeros_like(w), w)
        return w

    def forward(
        self,
        preds: Tensor,
        y_obs: Tensor,
        pnu_status: Tensor,
        *,
        age_days: Tensor | None = None,
        in_train_view: Tensor | None = None,
    ) -> Tensor:
        if age_days is None:
            raise ValueError("DelayIPWLoss needs age_days (see compute_age_days).")
        preds, y, pnu, view = prepare_inputs(preds, y_obs, pnu_status, in_train_view)
        age = age_days.reshape(-1).to(preds.dtype)
        if age.numel() != preds.numel():
            raise ValueError("age_days must have the same length as preds.")
        scope = view & (pnu != PNU_U) if self.negative_scope == "mature_only" else view
        if not bool(scope.any()):
            return zero_loss(preds)

        w = self.weights(pnu, age) * scope.to(preds.dtype)  # rows out of scope weigh 0
        bce = F.binary_cross_entropy_with_logits(preds, y, reduction="none")
        denom = w.sum() if self.normalize == "weights" else scope.sum().to(preds.dtype)
        return (w * bce).sum() / denom.clamp_min(1e-8)


def delay_ipw_loss(
    preds: Tensor,
    y_obs: Tensor,
    pnu_status: Tensor,
    *,
    age_days: Tensor,
    lam: float,
    in_train_view: Tensor | None = None,
    **kwargs,
) -> Tensor:
    """Functional form of :class:`DelayIPWLoss` (extra kwargs go to the constructor)."""
    return DelayIPWLoss(lam, **kwargs)(
        preds, y_obs, pnu_status, age_days=age_days, in_train_view=in_train_view
    )