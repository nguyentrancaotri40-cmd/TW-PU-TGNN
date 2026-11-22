"""Time-window nnDF risk, implemented from the formula in the brief.

    Loss = pi * mean l(P, +1) + max(0, mean l(U, -1) - pi * mean l(P, -1)),   l = logistic loss.

KHÔNG CHẮC, CẦN VERIFY (against Yasui & Kato, arXiv 2009.13092) before reporting results:
1. The brief writes ONE ``loss``. A PU risk needs the label-specific logistic loss:
   ``l(f, +1) = softplus(-f)`` on P in the first term, ``l(f, -1) = softplus(f)`` on U and on P in the
   second term. That is what is implemented.
2. Which rows form U is a choice. Default: U = status U (immature rows with y_obs = 0); N rows are not
   used (their gradient is 0). ``unlabeled_statuses=("U", "N")`` gives the classic PU reading.
3. Unbiasedness needs P to be a random sample of the positives and ``pi`` to be the positive prior of the
   unlabeled distribution. Observed positives are the EARLY-DETECTED ones, so the first assumption is
   violated here.

Known prior mismatch (from the simulation): the fraud rate hidden in U (~1.4-1.8%) is lower than the
prior estimated on the mature region (~3.5%). With ``pi`` too large, ``neg_risk_raw`` can go negative,
the ``max(0, .)`` clamp then removes the gradient of every U row, and the loss degenerates to the
positive term alone. Log ``last_risks['neg_risk_raw']`` and the clamp rate in round 6.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.graph.builder import PNU_CODES
from src.losses import PNU_P, prepare_inputs, zero_loss


def estimate_class_prior(
    y_obs: Tensor,
    age_days: Tensor,
    window_days: float,
    in_train_view: Tensor | None = None,
    eps: float = 1e-4,
) -> float:
    """Positive prior = fraction of observed positives among MATURE train-view rows.

    Uses observed labels only (never ``y_true``); compute it once on the whole train graph, not per
    batch. Assumes the mature region is fully reported (``window_days`` >= delay cap) and a stable
    prior over time; with uncapped delays it under-estimates the prior.
    """
    y = y_obs.reshape(-1).float()
    age = age_days.reshape(-1)
    view = torch.ones_like(y, dtype=torch.bool) if in_train_view is None else in_train_view.reshape(-1).bool()
    mature = view & (age >= window_days)
    if int(mature.sum()) == 0:
        raise ValueError("No mature rows: cannot estimate the class prior.")
    return float(min(max(float(y[mature].mean()), eps), 1.0 - eps))


class TimeWindowNNDFLoss(nn.Module):
    """nnDF risk with the logistic loss; see the module docstring for the open points.

    Args:
        prior: positive prior ``pi`` in (0, 1), e.g. from ``estimate_class_prior``.
        unlabeled_statuses: statuses forming U, a non-empty subset of ``("U", "N")``.

    A batch with no P row or no U row returns a zero loss (batch-level estimate: use batches large
    enough to contain observed positives). After every ``forward`` the detached components are kept in
    ``self.last_risks`` (``pos_risk``, ``neg_risk_raw``, ``neg_risk``, ``clamped``), or ``None`` for an
    empty batch; read them with ``.item()`` only at logging time to avoid a GPU sync per step.
    """

    def __init__(self, prior: float, unlabeled_statuses: Sequence[str] = ("U",)) -> None:
        super().__init__()
        if not 0.0 < prior < 1.0:
            raise ValueError("prior must be in (0, 1).")
        statuses = tuple(unlabeled_statuses)
        if not statuses or any(s not in ("U", "N") for s in statuses):
            raise ValueError("unlabeled_statuses must be a non-empty subset of ('U', 'N').")
        self.prior = float(prior)
        self.unlabeled_statuses = statuses
        self.register_buffer(
            "_unlabeled_codes", torch.tensor([PNU_CODES[s] for s in statuses], dtype=torch.long), persistent=False
        )
        self.last_risks: dict[str, Tensor] | None = None

    def risks(
        self,
        preds: Tensor,
        y_obs: Tensor,
        pnu_status: Tensor,
        in_train_view: Tensor | None = None,
    ) -> dict[str, Tensor] | None:
        """Risk components, or ``None`` if P or U is empty in the batch."""
        preds, _, pnu, view = prepare_inputs(preds, y_obs, pnu_status, in_train_view)
        pos = view & (pnu == PNU_P)
        unl = view & torch.isin(pnu, self._unlabeled_codes.to(pnu.device))
        if not bool(pos.any()) or not bool(unl.any()):
            return None
        pos_as_pos = F.softplus(-preds[pos]).mean()  # l(f, +1) on P
        pos_as_neg = F.softplus(preds[pos]).mean()  # l(f, -1) on P
        unl_as_neg = F.softplus(preds[unl]).mean()  # l(f, -1) on U
        neg_raw = unl_as_neg - self.prior * pos_as_neg  # estimates (1 - pi) * E_N[l(f, -1)], should be >= 0
        return {
            "pos_risk": self.prior * pos_as_pos,
            "neg_risk_raw": neg_raw,
            "neg_risk": torch.clamp(neg_raw, min=0.0),
            "clamped": (neg_raw < 0).to(preds.dtype),
        }

    def forward(
        self,
        preds: Tensor,
        y_obs: Tensor,
        pnu_status: Tensor,
        *,
        age_days: Tensor | None = None,
        in_train_view: Tensor | None = None,
    ) -> Tensor:
        parts = self.risks(preds, y_obs, pnu_status, in_train_view)
        if parts is None:
            self.last_risks = None
            return zero_loss(preds.reshape(-1))
        self.last_risks = {k: v.detach() for k, v in parts.items()}
        return parts["pos_risk"] + parts["neg_risk"]


def tw_nndf_loss(
    preds: Tensor,
    y_obs: Tensor,
    pnu_status: Tensor,
    *,
    prior: float,
    age_days: Tensor | None = None,
    in_train_view: Tensor | None = None,
    unlabeled_statuses: Sequence[str] = ("U",),
) -> Tensor:
    """Functional form of :class:`TimeWindowNNDFLoss`."""
    return TimeWindowNNDFLoss(prior, unlabeled_statuses)(
        preds, y_obs, pnu_status, age_days=age_days, in_train_view=in_train_view
    )