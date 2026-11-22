"""Maturity masking: BCE on P and N only; the unlabeled set U contributes exactly 0."""
from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.losses import PNU_U, prepare_inputs, zero_loss


class MaturityMaskLoss(nn.Module):
    """Mean BCE over rows with status P or N.

    The mean runs over the USED rows, so the scale is comparable with ``NaiveBCELoss``. Boolean
    indexing makes the gradient of every U row exactly 0 and the loss value independent of U logits.

    Observation (my reasoning, not from a source): with the default setting the young region keeps
    its positives (P, detected early) but loses all of its negatives (U), so it is positive-heavy and
    may push predictions up. Ablation switch (not part of the 72 runs): ``mature_only_window_days=W``
    also drops every row younger than W days (needs ``age_days``), including young positives.
    """

    def __init__(self, mature_only_window_days: float | None = None) -> None:
        super().__init__()
        if mature_only_window_days is not None and mature_only_window_days <= 0:
            raise ValueError("mature_only_window_days must be > 0.")
        self.mature_only_window_days = mature_only_window_days

    def forward(
        self,
        preds: Tensor,
        y_obs: Tensor,
        pnu_status: Tensor,
        *,
        age_days: Tensor | None = None,
        in_train_view: Tensor | None = None,
    ) -> Tensor:
        preds, y, pnu, view = prepare_inputs(preds, y_obs, pnu_status, in_train_view)
        mask = view & (pnu != PNU_U)
        if self.mature_only_window_days is not None:
            if age_days is None:
                raise ValueError("age_days is required when mature_only_window_days is set.")
            mask = mask & (age_days.reshape(-1).to(preds.dtype) >= self.mature_only_window_days)
        if not bool(mask.any()):
            return zero_loss(preds)
        return F.binary_cross_entropy_with_logits(preds[mask], y[mask])


def maturity_mask_loss(
    preds: Tensor,
    y_obs: Tensor,
    pnu_status: Tensor,
    *,
    age_days: Tensor | None = None,
    in_train_view: Tensor | None = None,
    mature_only_window_days: float | None = None,
) -> Tensor:
    """Functional form of :class:`MaturityMaskLoss`."""
    return MaturityMaskLoss(mature_only_window_days)(
        preds, y_obs, pnu_status, age_days=age_days, in_train_view=in_train_view
    )