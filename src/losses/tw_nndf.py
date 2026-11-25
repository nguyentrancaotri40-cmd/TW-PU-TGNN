"""TW-nnDF = Maturity-Masked BCE (mature region) + Cost-Sensitive nnPU (immature region).

Two classes
-----------
``TimeWindowNNDFLoss`` (variant ``pu_only``): the V1 / V1.5 PU risk of P versus U only; N rows are NOT used.
``HybridTWnnDFLoss`` (variant ``hybrid``, the specification of the project brief)::

    loss = BCE(P=1, N=0, trusted-U=0)  +  lambda_pu * PU_risk(P vs U_remaining)

* BCE region ("mature"): P rows, N rows, and (amount-aware masking) U rows with ``amount >= p95_train`` and
  ``age >= amount_min_age_days`` (``maturity_mask.trusted_negative_mask``). Trusted rows leave U.
* PU region: P versus the remaining U. P appears in BOTH terms (design choice, KHÔNG CHẮC).
* Cost-sensitive: ``amount_weighted=True`` multiplies every P / U row's loss by ``min(amount / amount_ref, max_weight)``
  inside the same means, with the same prior. The identity behind it needs the hidden positives in U to share P's
  amount distribution (SCAR); the amount-dependent delay of the benchmark violates it (threat to validity).
* Prior: a fixed ``prior`` (scalar from the trainer), or, if ``sliding_prior`` is given, the batch prior
  ``mean(sliding_prior.query(age_days[remaining U rows]))``: a per-batch scalar computed from TRAIN-only tables.
* Calibration: the PU term sees ``logits / T`` with ``T`` from ``MatureTemperatureTracker`` (fitted on the
  maturity-masked TRAIN rows with ``y_obs``, never on validation). The BCE term sees the raw logits.
* Only ``y_obs``, ``pnu``, ``age_days`` and ``amount`` are read; ``y_true`` is never an input.

PU modes: ``clamp`` = ``pos + max(0, neg_raw)``; ``nnpu`` = Kiryo et al. 2017 style: ``neg_raw >= -beta`` ->
``pos + neg_raw``, else ``-gamma * neg_raw`` (Algorithm 1 as recalled, KHÔNG CHẮC, CẦN VERIFY).
"""
from __future__ import annotations

from typing import Any, Literal, Protocol, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.graph.builder import PNU_CODES
from src.losses import PNU_N, PNU_P, PNU_U, prepare_inputs, zero_loss
from src.losses.maturity_mask import trusted_negative_mask

NNDFMode = Literal["clamp", "nnpu"]
_PRIOR_EPS = 1e-4


class AgePrior(Protocol):
    """Anything with ``query(age_days) -> pi`` (e.g. ``calibration.SlidingWindowPrior``)."""

    def query(self, age_days: Tensor) -> Tensor: ...


def estimate_class_prior(
    y_obs: Tensor,
    age_days: Tensor,
    window_days: float,
    in_train_view: Tensor | None = None,
    eps: float = 1e-4,
) -> float:
    """Positive prior = fraction of observed positives among MATURE train-view rows (observed labels only)."""
    y = y_obs.reshape(-1).float()
    age = age_days.reshape(-1)
    view = torch.ones_like(y, dtype=torch.bool) if in_train_view is None else in_train_view.reshape(-1).bool()
    mature = view & (age >= window_days)
    if int(mature.sum()) == 0:
        raise ValueError("No mature rows: cannot estimate the class prior.")
    return float(min(max(float(y[mature].mean()), eps), 1.0 - eps))


class TimeWindowNNDFLoss(nn.Module):
    """PU risk (logistic loss) of P versus U; see the module docstring.

    Args:
        prior: positive prior ``pi`` in (0, 1).
        unlabeled_statuses: statuses forming U, a non-empty subset of ``("U", "N")``.
        mode: ``"clamp"`` or ``"nnpu"``.
        beta / gamma: ``nnpu`` only; tolerance >= 0 and ascent discount in (0, 1].
        amount_weighted / amount_ref / max_weight: cost-sensitive weights ``min(amount / amount_ref, max_weight)``.
    After every ``forward`` the detached components are in ``self.last_risks`` (``pos_risk``, ``neg_risk_raw``,
    ``neg_risk``, ``clamped``, ``ascent``). ``clamped`` = (neg_raw < 0) in both modes.
    """

    def __init__(
        self,
        prior: float,
        unlabeled_statuses: Sequence[str] = ("U",),
        mode: NNDFMode = "clamp",
        beta: float = 0.0,
        gamma: float = 1.0,
        amount_weighted: bool = False,
        amount_ref: float | None = None,
        max_weight: float = 10.0,
    ) -> None:
        super().__init__()
        if not 0.0 < prior < 1.0:
            raise ValueError("prior must be in (0, 1).")
        statuses = tuple(unlabeled_statuses)
        if not statuses or any(s not in ("U", "N") for s in statuses):
            raise ValueError("unlabeled_statuses must be a non-empty subset of ('U', 'N').")
        if mode not in ("clamp", "nnpu"):
            raise ValueError("mode must be 'clamp' or 'nnpu'.")
        if beta < 0.0:
            raise ValueError("beta must be >= 0.")
        if not 0.0 < gamma <= 1.0:
            raise ValueError("gamma must be in (0, 1].")
        if amount_weighted and (amount_ref is None or amount_ref <= 0):
            raise ValueError("amount_weighted=True needs amount_ref > 0.")
        if max_weight <= 0:
            raise ValueError("max_weight must be > 0.")
        self.prior = float(prior)
        self.unlabeled_statuses = statuses
        self.mode = mode
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.amount_weighted = bool(amount_weighted)
        self.amount_ref = None if amount_ref is None else float(amount_ref)
        self.max_weight = float(max_weight)
        self.register_buffer(
            "_unlabeled_codes", torch.tensor([PNU_CODES[s] for s in statuses], dtype=torch.long), persistent=False
        )
        self.last_risks: dict[str, Tensor] | None = None

    @property
    def needs_amount(self) -> bool:
        """The trainer hands ``amount`` only to losses that declare they need it."""
        return self.amount_weighted

    def _row_weights(self, amount: Tensor | None, n: int, like: Tensor) -> Tensor:
        if amount is None:
            raise ValueError("amount_weighted=True needs the `amount` argument.")
        a = amount.reshape(-1).to(device=like.device, dtype=like.dtype)
        if a.numel() != n:
            raise ValueError("amount must have the same length as preds.")
        return (a / self.amount_ref).clamp(min=0.0, max=self.max_weight)

    def risks(
        self,
        preds: Tensor,
        y_obs: Tensor,
        pnu_status: Tensor,
        in_train_view: Tensor | None = None,
        amount: Tensor | None = None,
        prior: Tensor | float | None = None,
    ) -> dict[str, Tensor] | None:
        """Risk components, or ``None`` if P or U is empty. ``prior`` overrides ``self.prior`` for this call."""
        preds, _, pnu, view = prepare_inputs(preds, y_obs, pnu_status, in_train_view)
        pos = view & (pnu == PNU_P)
        unl = view & torch.isin(pnu, self._unlabeled_codes.to(pnu.device))
        if not bool(pos.any()) or not bool(unl.any()):
            return None
        if self.amount_weighted:
            w = self._row_weights(amount, preds.numel(), preds)
            wp, wu = w[pos], w[unl]
        else:
            wp = wu = torch.ones((), dtype=preds.dtype, device=preds.device)
        pi = self.prior if prior is None else prior
        pos_as_pos = (wp * F.softplus(-preds[pos])).mean()  # l(f, +1) on P
        pos_as_neg = (wp * F.softplus(preds[pos])).mean()  # l(f, -1) on P
        unl_as_neg = (wu * F.softplus(preds[unl])).mean()  # l(f, -1) on U
        neg_raw = unl_as_neg - pi * pos_as_neg  # estimates (1 - pi) * E_N[l(f, -1)], should be >= 0
        return {
            "pos_risk": pi * pos_as_pos,
            "neg_risk_raw": neg_raw,
            "neg_risk": torch.clamp(neg_raw, min=0.0),  # logging / clamp mode only
            "clamped": (neg_raw < 0).to(preds.dtype),
        }

    def _pu_loss(self, parts: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """``(PU loss, ascent flag)`` from the risk components."""
        raw = parts["neg_risk_raw"]
        if self.mode == "clamp":
            return parts["pos_risk"] + parts["neg_risk"], torch.zeros_like(raw)
        use_ascent = raw < -self.beta
        loss = torch.where(use_ascent, -self.gamma * raw, parts["pos_risk"] + raw)
        return loss, use_ascent.to(raw.dtype)

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
        parts = self.risks(preds, y_obs, pnu_status, in_train_view, amount)
        if parts is None:
            self.last_risks = None
            return zero_loss(preds.reshape(-1))
        loss, ascent = self._pu_loss(parts)
        self.last_risks = {k: v.detach() for k, v in parts.items()}
        self.last_risks["ascent"] = ascent.detach()
        return loss


class HybridTWnnDFLoss(TimeWindowNNDFLoss):
    """Maturity-masked BCE + lambda_pu * nnPU.

    V2 additions:
      - bce_pos_weight: positive-class weight in the BCE term only.
      - u_conf_filter: retain only confident U rows in the PU term.
      - u_conf_hi / u_conf_lo: confidence thresholds; 0 < lo < hi < 1.

    Defaults preserve the V1.5 loss behavior.
    """

    def __init__(
        self,
        prior: float,
        unlabeled_statuses: Sequence[str] = ("U",),
        mode: NNDFMode = "nnpu",
        beta: float = 0.0,
        gamma: float = 1.0,
        amount_weighted: bool = False,
        amount_ref: float | None = None,
        max_weight: float = 10.0,
        *,
        lambda_pu: float = 1.0,
        amount_threshold: float | None = None,
        amount_min_age_days: float = 0.0,
        calibrator: Any | None = None,
        sliding_prior: AgePrior | None = None,
        bce_pos_weight: float = 1.0,
        u_conf_filter: bool = False,
        u_conf_hi: float = 0.80,
        u_conf_lo: float = 0.10,
    ) -> None:
        if tuple(unlabeled_statuses) != ("U",):
            raise ValueError(
                "hybrid variant: unlabeled_statuses must be ('U',) (N feeds the BCE term)."
            )

        super().__init__(
            prior, unlabeled_statuses, mode, beta, gamma,
            amount_weighted, amount_ref, max_weight,
        )

        if lambda_pu < 0:
            raise ValueError("lambda_pu must be >= 0.")
        if amount_threshold is not None and amount_threshold <= 0:
            raise ValueError("amount_threshold must be > 0.")
        if amount_min_age_days < 0:
            raise ValueError("amount_min_age_days must be >= 0.")
        if bce_pos_weight <= 0:
            raise ValueError("bce_pos_weight must be > 0.")
        if not 0.0 < u_conf_lo < u_conf_hi < 1.0:
            raise ValueError("need 0 < u_conf_lo < u_conf_hi < 1.")

        self.lambda_pu = float(lambda_pu)
        self.amount_threshold = (
            None if amount_threshold is None else float(amount_threshold)
        )
        self.amount_min_age_days = float(amount_min_age_days)
        self.calibrator = calibrator
        self.sliding_prior = sliding_prior

        self.bce_pos_weight = float(bce_pos_weight)
        self.u_conf_filter = bool(u_conf_filter)
        self.u_conf_hi = float(u_conf_hi)
        self.u_conf_lo = float(u_conf_lo)

    @property
    def needs_amount(self) -> bool:
        return self.amount_weighted or self.amount_threshold is not None

    def end_epoch(self) -> float | None:
        """Refit temperature using mature TRAIN rows collected during this epoch."""
        return None if self.calibrator is None else self.calibrator.refit()

    def _bce(self, logits: Tensor, target: Tensor) -> Tensor:
        # Preserve the original V1.5 numerical path when pos_weight == 1.
        if self.bce_pos_weight == 1.0:
            return F.binary_cross_entropy_with_logits(logits, target)

        return nn.BCEWithLogitsLoss(
            pos_weight=logits.new_tensor(self.bce_pos_weight)
        )(logits, target)

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
        preds, _, pnu, view = prepare_inputs(
            preds, y_obs, pnu_status, in_train_view
        )
        amt = None if amount is None else amount.reshape(-1).to(preds.device)

        # ---- trusted negatives ------------------------------------------------
        trusted = torch.zeros_like(view)
        if self.amount_threshold is not None:
            if amt is None:
                raise ValueError("amount_threshold is set: `amount` is required.")
            trusted = view & trusted_negative_mask(
                pnu, amt, age_days,
                self.amount_threshold,
                self.amount_min_age_days,
            )

        # ---- BCE: P=1, N=0, trusted-U=0 --------------------------------------
        is_pos = view & (pnu == PNU_P)
        keep_bce = is_pos | (view & (pnu == PNU_N)) | trusted

        if bool(keep_bce.any()):
            bce = self._bce(
                preds[keep_bce],
                is_pos[keep_bce].to(preds.dtype),
            )
        else:
            bce = zero_loss(preds)

        if (
            self.calibrator is not None
            and bool(keep_bce.any())
            and self.training
            and torch.is_grad_enabled()
        ):
            self.calibrator.collect(
                preds[keep_bce].detach(),
                is_pos[keep_bce],
            )

        # ---- PU: P versus remaining U, optionally calibrated -----------------
        z = (
            preds
            if self.calibrator is None
            else preds / self.calibrator.temperature
        )

        # Trusted U rows are treated as N for BCE and removed from PU.
        pnu_pu = torch.where(
            trusted,
            torch.full_like(pnu, PNU_N),
            pnu,
        )
        view_pu = view

        u_raw = view & (pnu == PNU_U)
        u_remaining = view & (pnu_pu == PNU_U)
        u_kept = u_remaining

        if self.u_conf_filter:
            # Detach the selection decision: no gradient through the mask.
            confidence = z.detach().sigmoid()
            confident = (
                (confidence > self.u_conf_hi)
                | (confidence < self.u_conf_lo)
            )
            u_kept = u_remaining & confident

            # Mask only U rows in the grey zone. P rows remain in PU.
            view_pu = view & ~(u_remaining & ~confident)

        n_u_remaining = u_remaining.sum().to(preds.dtype)
        n_u_kept = u_kept.sum().to(preds.dtype)
        kept_frac = torch.where(
            n_u_remaining > 0,
            n_u_kept / n_u_remaining.clamp_min(1.0),
            preds.new_zeros(()),
        )

        # ---- Sliding prior on the actual U rows entering this PU batch -------
        pi: Tensor | None = None
        if self.sliding_prior is not None:
            if age_days is None:
                raise ValueError("sliding_prior is set: `age_days` is required.")

            if bool(u_kept.any()):
                age_u = age_days.reshape(-1).to(preds.device)[u_kept]
                pi = (
                    self.sliding_prior.query(age_u)
                    .to(preds.dtype)
                    .mean()
                    .clamp(_PRIOR_EPS, 1.0 - _PRIOR_EPS)
                )

        parts = self.risks(
            z, y_obs, pnu_pu, view_pu, amt, prior=pi
        )

        # Connected zero is essential when confidence filtering removes all U
        # (or when P/U is otherwise absent in this batch).
        if parts is None:
            pu = zero_loss(preds)
            zero = preds.new_zeros(())
            risks = {
                "pos_risk": zero,
                "neg_risk_raw": zero,
                "neg_risk": zero,
                "clamped": zero,
                "ascent": zero,
                "prior_used": (
                    preds.new_tensor(self.prior) if pi is None else pi
                ).detach(),
            }
        else:
            pu, ascent = self._pu_loss(parts)
            risks = {k: v.detach() for k, v in parts.items()}
            risks["ascent"] = ascent.detach()
            risks["prior_used"] = (
                preds.new_tensor(self.prior) if pi is None else pi
            ).detach()

        risks.update({
            "bce": bce.detach(),
            "pu_loss": pu.detach(),
            "u_raw_count": u_raw.sum().to(preds.dtype).detach(),
            "u_pu_count": u_remaining.sum().to(preds.dtype).detach(),
            "u_kept_count": n_u_kept.detach(),
            "u_kept_frac": kept_frac.detach(),
        })
        self.last_risks = risks

        return bce + self.lambda_pu * pu


def tw_nndf_loss(
    preds: Tensor,
    y_obs: Tensor,
    pnu_status: Tensor,
    *,
    prior: float,
    age_days: Tensor | None = None,
    in_train_view: Tensor | None = None,
    amount: Tensor | None = None,
    unlabeled_statuses: Sequence[str] = ("U",),
    mode: NNDFMode = "clamp",
    beta: float = 0.0,
    gamma: float = 1.0,
    amount_weighted: bool = False,
    amount_ref: float | None = None,
    max_weight: float = 10.0,
) -> Tensor:
    """Functional form of :class:`TimeWindowNNDFLoss` (pu_only)."""
    return TimeWindowNNDFLoss(
        prior, unlabeled_statuses, mode, beta, gamma, amount_weighted, amount_ref, max_weight
    )(preds, y_obs, pnu_status, age_days=age_days, in_train_view=in_train_view, amount=amount)