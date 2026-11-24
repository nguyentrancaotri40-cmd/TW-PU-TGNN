"""Maturity masking: BCE on P and N only; U contributes 0, except amount-aware TRUSTED rows.

Window + Amount-Aware masking
-----------------------------
Rule: mature negative  <=>  age >= W   OR   (amount >= p95(TRAIN amounts)  AND  age >= amount_min_age_days).
N rows already encode ``age >= W``; the second branch promotes U rows to "trusted negative".
Business rationale: a large transaction (amount >= p95) enters MANUAL REVIEW immediately (real time). If the review
does not flag it and no complaint arrives, it is treated as mature even before W days have passed.
ASSUMPTION (KHÔNG CHẮC): there is no ``is_manual_review_cleared`` column, so it is assumed True for every large
transaction; pass ``manual_review_cleared`` to ``trusted_negative_mask`` to override it when real data exist.

IMPROVEMENT OVER THE ORIGINAL FORMULA (record it in Threats to Validity): the literal rule would also label a
brand-new, large, unreported transaction as negative. A trusted row must ALSO satisfy
``age >= amount_min_age_days`` (the trainer defaults it to W/2; 0 reproduces the literal rule). W/2 is a design
choice, not an estimate. Trusted rows can still hide frauds: the trainer logs ``diag_true_fraud_rate_in_trusted_U``
(reads y_true for DIAGNOSTICS only). The p95 is computed by the trainer on TRAIN amounts only.

Only ``y_obs``, ``pnu``, ``age_days`` and ``amount`` are read; ``y_true`` is never an input.
"""
from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.losses import PNU_U, prepare_inputs, zero_loss


def trusted_negative_mask(
    pnu_status: Tensor,
    amount: Tensor,
    age_days: Tensor | None,
    amount_threshold: float,
    amount_min_age_days: float = 0.0,
    manual_review_cleared: Tensor | None = None,
) -> Tensor:
    """Bool mask of U rows that amount-aware masking promotes to "mature negative".

    ``U  &  amount >= amount_threshold  &  age >= amount_min_age_days  [&  manual_review_cleared]``.
    Rows outside the train view must be removed by the caller (``& in_train_view``).
    """
    pnu = pnu_status.reshape(-1)
    amt = amount.reshape(-1)
    if amt.numel() != pnu.numel():
        raise ValueError("amount must have the same length as pnu_status.")
    mask = (pnu == PNU_U) & (amt >= amount_threshold)
    if amount_min_age_days > 0:
        if age_days is None:
            raise ValueError("age_days is required when amount_min_age_days > 0.")
        age = age_days.reshape(-1)
        if age.numel() != pnu.numel():
            raise ValueError("age_days must have the same length as pnu_status.")
        mask = mask & (age.to(amt.dtype) >= amount_min_age_days)
    if manual_review_cleared is not None:
        mask = mask & manual_review_cleared.reshape(-1).bool()
    return mask


class MaturityMaskLoss(nn.Module):
    """Mean BCE over the rows kept by the mask.

    Default mask = rows with status P or N. With ``amount_threshold`` set (the trainer passes the p95 of the TRAIN
    amounts) trusted U rows are also kept, with target ``y_obs = 0``. ``mature_only_window_days=W`` is an ablation
    that also drops every row younger than W (needs ``age_days``).
    """

    def __init__(
        self,
        mature_only_window_days: float | None = None,
        amount_threshold: float | None = None,
        amount_min_age_days: float = 0.0,
    ) -> None:
        super().__init__()
        if mature_only_window_days is not None and mature_only_window_days <= 0:
            raise ValueError("mature_only_window_days must be > 0.")
        if amount_threshold is not None and amount_threshold <= 0:
            raise ValueError("amount_threshold must be > 0.")
        if amount_min_age_days < 0:
            raise ValueError("amount_min_age_days must be >= 0.")
        self.mature_only_window_days = mature_only_window_days
        self.amount_threshold = None if amount_threshold is None else float(amount_threshold)
        self.amount_min_age_days = float(amount_min_age_days)

    @property
    def needs_amount(self) -> bool:
        """The trainer hands ``amount`` only to losses that declare they need it."""
        return self.amount_threshold is not None

    def forward(
        self,
        preds: Tensor,
        y_obs: Tensor,
        pnu_status: Tensor,
        *,
        age_days: Tensor | None = None,
        in_train_view: Tensor | None = None,
        amount: Tensor | None = None,
    ) -> Tensor:
        preds, y, pnu, view = prepare_inputs(preds, y_obs, pnu_status, in_train_view)
        keep = pnu != PNU_U
        if self.amount_threshold is not None:
            if amount is None:
                raise ValueError("amount is required when amount_threshold is set.")
            keep = keep | trusted_negative_mask(
                pnu, amount.to(preds.device), age_days, self.amount_threshold, self.amount_min_age_days
            )
        mask = view & keep
        if self.mature_only_window_days is not None:
            if age_days is None:
                raise ValueError("age_days is required when mature_only_window_days is set.")
            mask = mask & (age_days.reshape(-1).to(preds.dtype) >= self.mature_only_window_days)
        if not bool(mask.any()):
            return zero_loss(preds)
        return F.binary_cross_entropy_with_logits(preds[mask], y[mask])


MaturityMaskedBCELoss = MaturityMaskLoss  # name used in the project brief


def maturity_mask_loss(
    preds: Tensor,
    y_obs: Tensor,
    pnu_status: Tensor,
    *,
    age_days: Tensor | None = None,
    in_train_view: Tensor | None = None,
    amount: Tensor | None = None,
    mature_only_window_days: float | None = None,
    amount_threshold: float | None = None,
    amount_min_age_days: float = 0.0,
) -> Tensor:
    """Functional form of :class:`MaturityMaskLoss`."""
    return MaturityMaskLoss(mature_only_window_days, amount_threshold, amount_min_age_days)(
        preds, y_obs, pnu_status, age_days=age_days, in_train_view=in_train_view, amount=amount
    )


__all__ = ["trusted_negative_mask", "MaturityMaskLoss", "MaturityMaskedBCELoss", "maturity_mask_loss"]