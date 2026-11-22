"""Baseline A0: plain BCE where every unlabeled row counts as a negative."""
from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.losses import prepare_inputs, zero_loss


class NaiveBCELoss(nn.Module):
    """``BCEWithLogits`` over all train-view rows with ``y_obs`` as target.

    Immature rows (status U, ``y_obs = 0``) are treated as legitimate: this is the bias the other
    three losses try to remove. ``pnu_status`` and ``age_days`` are accepted only to share one
    signature. Gradient on a U row is ``sigmoid(f) / n``, never 0.

    Oracle runs reuse this class: pass ``y_true`` as ``y_obs`` (trainer switch, round 6).
    """

    def forward(
        self,
        preds: Tensor,
        y_obs: Tensor,
        pnu_status: Tensor,
        *,
        age_days: Tensor | None = None,
        in_train_view: Tensor | None = None,
    ) -> Tensor:
        preds, y, _, view = prepare_inputs(preds, y_obs, pnu_status, in_train_view)
        if not bool(view.any()):
            return zero_loss(preds)
        return F.binary_cross_entropy_with_logits(preds[view], y[view])


def naive_bce_loss(
    preds: Tensor,
    y_obs: Tensor,
    pnu_status: Tensor,
    *,
    age_days: Tensor | None = None,
    in_train_view: Tensor | None = None,
) -> Tensor:
    """Functional form of :class:`NaiveBCELoss`."""
    return NaiveBCELoss()(preds, y_obs, pnu_status, age_days=age_days, in_train_view=in_train_view)