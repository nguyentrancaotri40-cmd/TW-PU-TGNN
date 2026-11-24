"""Train-only calibration helpers for TW-nnDF (anti-leakage by construction).

Anti-leakage contract
---------------------
* Every function receives tensors of the TRAIN graph only (observed labels ``y_obs``, ages, P/N/U codes, logits of
  TRAIN rows). No argument can carry validation / test rows and ``y_true`` is never used.
* ``MatureTemperatureTracker`` is fed ONLY by the loss, from the mature train rows of the training step; the
  validation pass never goes through the loss, and the tracker ignores calls made without autograd.
* Temperature scaling only rescales logits; it cannot repair a biased prior by itself (KHÔNG CHẮC how much it
  helps the nnPU term: an empirical question).

Safeguards added in V1.5 (preflight)
------------------------------------
* Temperature floor ``T >= T_MIN_DEFAULT = 0.5``. The PU term sees ``logits / T``; a refit with T < 0.5 (0.307 was
  observed after 3 epochs, and at T = 0.25 every U row lost its gradient) is CLAMPED up to 0.5. Trade-off: a little
  calibration sharpness is given up to keep the gradient signal on U. The raw (pre-clamp) value is kept in
  ``last_raw_temperature`` and every clamp increments ``floor_hits`` (report them in Threats to Validity).
* Prior floor ``MIN_UNLABELED_PRIOR = 0.002`` for the unlabeled prior: ``(pi*(n_P+n_U) - n_P) / n_U`` can be <= 0
  when the pooled prior is under-estimated, and a prior ~0 turns nnPU into "every U row is a negative".
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.losses import PNU_P, PNU_U

PRIOR_EPS: float = 1e-4
MIN_UNLABELED_PRIOR: float = 0.002   # floor of the unlabeled prior (true hidden rate in U is ~0.014-0.018)
T_MIN_DEFAULT: float = 0.5           # temperature floor (see module docstring)
T_MAX_DEFAULT: float = 4.0


@dataclass(frozen=True)
class PriorEstimate:
    """Pooled prior plus the per-window rates / counts it was built from (logged for diagnostics)."""

    prior: float
    window_rates: tuple[float, ...]
    window_counts: tuple[int, ...]


def _view_mask(ref: Tensor, in_train_view: Tensor | None) -> Tensor:
    if in_train_view is None:
        return torch.ones_like(ref, dtype=torch.bool)
    view = in_train_view.reshape(-1).bool()
    if view.numel() != ref.numel():
        raise ValueError("in_train_view must have the same length as the other inputs.")
    return view


def _clip(p: float, eps: float) -> float:
    return float(min(max(p, eps), 1.0 - eps))


def sliding_window_prior(
    y_obs: Tensor,
    age_days: Tensor,
    window_days: float,
    *,
    in_train_view: Tensor | None = None,
    span_days: float = 14.0,
    n_windows: int = 3,
    eps: float = PRIOR_EPS,
) -> PriorEstimate:
    """Pooled positive rate of the ``n_windows`` most recent MATURE windows of ``span_days`` each.

    Window k covers ages ``[W + k*span, W + (k+1)*span)``; k = 0 is the most recent fully-reported window.
    Young rows (age < W) and rows outside the train view never enter.
    """
    if window_days <= 0 or span_days <= 0 or n_windows < 1:
        raise ValueError("window_days and span_days must be > 0 and n_windows >= 1.")
    y = y_obs.reshape(-1).float()
    age = age_days.reshape(-1).float()
    if y.numel() != age.numel():
        raise ValueError("y_obs and age_days must have the same length.")
    view = _view_mask(y, in_train_view)
    rates: list[float] = []
    counts: list[int] = []
    for k in range(n_windows):
        lo = window_days + k * span_days
        m = view & (age >= lo) & (age < lo + span_days)
        n = int(m.sum())
        if n == 0:
            continue
        rates.append(float(y[m].mean()))
        counts.append(n)
    if not counts:
        raise ValueError("No mature rows in the requested windows: cannot estimate the prior.")
    pooled = sum(r * c for r, c in zip(rates, counts)) / sum(counts)
    return PriorEstimate(_clip(pooled, eps), tuple(rates), tuple(counts))


def young_counts(
    pnu_status: Tensor, age_days: Tensor, window_days: float, in_train_view: Tensor | None = None
) -> tuple[int, int]:
    """``(n_P, n_U)`` among train-view rows younger than W (the immature region)."""
    pnu = pnu_status.reshape(-1)
    age = age_days.reshape(-1)
    if pnu.numel() != age.numel():
        raise ValueError("pnu_status and age_days must have the same length.")
    young = _view_mask(age, in_train_view) & (age < window_days)
    return int((young & (pnu == PNU_P)).sum()), int((young & (pnu == PNU_U)).sum())


def unlabeled_prior_raw(pi_population: float, n_pos_young: int, n_unl_young: int) -> float:
    """Unclipped ``(pi * (n_P + n_U) - n_P) / n_U``; can be <= 0 (diagnostic: how far the pooled prior is off)."""
    if n_unl_young <= 0:
        raise ValueError("No unlabeled young rows: cannot convert the prior.")
    return (pi_population * (n_pos_young + n_unl_young) - n_pos_young) / n_unl_young


def unlabeled_prior(
    pi_population: float,
    n_pos_young: int,
    n_unl_young: int,
    eps: float = PRIOR_EPS,
    min_prior: float = MIN_UNLABELED_PRIOR,
) -> float:
    """Positive prior inside the young unlabeled set, floored at ``min_prior`` and clipped to (eps, 1 - eps).

    The floor protects nnPU from a prior ~0 (which would read every U row as a negative). ``min_prior = 0``
    reproduces the pre-V1.5 behaviour (only the ``eps`` clip).
    """
    if not 0.0 <= min_prior < 1.0:
        raise ValueError("min_prior must be in [0, 1).")
    raw = unlabeled_prior_raw(pi_population, n_pos_young, n_unl_young)
    return _clip(max(raw, min_prior), eps)


class SlidingWindowPrior:
    """pi as a function of a row's age (days before the label cut), from MATURE TRAIN rows only (observed labels).

    For a row of age ``a`` the window is the mature rows with age in ``[a + W, a + W + span)``: strictly older,
    fully reported, so only labels that were already knowable are used. ``shrink`` pulls a thin window towards the
    global mature prior (additive smoothing). Table is built once on the TRAIN graph; no val/test input exists.
    """

    def __init__(self, table: Tensor, global_prior: float, bin_days: float) -> None:
        self.table = table.detach().float().cpu()
        self.global_prior = float(global_prior)
        self.bin_days = float(bin_days)

    @classmethod
    def fit(
        cls,
        y_obs: Tensor,
        age_days: Tensor,
        window_days: float,
        in_train_view: Tensor | None = None,
        *,
        span_days: float = 14.0,
        shrink: float = 200.0,
        bin_days: float = 1.0,
        eps: float = PRIOR_EPS,
    ) -> "SlidingWindowPrior":
        if window_days <= 0 or span_days <= 0 or bin_days <= 0 or shrink < 0:
            raise ValueError("window_days, span_days, bin_days must be > 0 and shrink >= 0.")
        y = y_obs.reshape(-1).float().cpu()
        age = age_days.reshape(-1).float().cpu()
        if y.numel() != age.numel():
            raise ValueError("y_obs and age_days must have the same length.")
        view = _view_mask(y, None if in_train_view is None else in_train_view.cpu())
        mature = view & (age >= window_days)
        if int(mature.sum()) == 0:
            raise ValueError("No mature rows: cannot fit the sliding prior.")
        g = _clip(float(y[mature].mean()), eps)
        idx = (age[mature] / bin_days).floor().long()
        n_bins = int(idx.max()) + 1
        cnt = torch.bincount(idx, minlength=n_bins).double()
        pos = torch.bincount(idx, weights=y[mature].double(), minlength=n_bins)
        zero = torch.zeros(1, dtype=torch.float64)
        ccnt, cpos = torch.cat([zero, cnt.cumsum(0)]), torch.cat([zero, pos.cumsum(0)])
        off = int(math.ceil(window_days / bin_days))
        span = max(1, int(round(span_days / bin_days)))
        q = torch.arange(n_bins)
        lo = (q + off).clamp(max=n_bins)
        hi = (q + off + span).clamp(max=n_bins)
        n, k = ccnt[hi] - ccnt[lo], cpos[hi] - cpos[lo]
        num, den = k + shrink * g, n + shrink
        table = torch.where(den > 0, num / den.clamp_min(1e-12), torch.full_like(num, g))
        return cls(table.clamp(eps, 1.0 - eps).float(), g, bin_days)

    def query(self, age_days: Tensor) -> Tensor:
        """pi per age; ages beyond the table reuse the last bin (empty window -> global prior)."""
        idx = (age_days.reshape(-1).float() / self.bin_days).floor().long().clamp(0, self.table.numel() - 1)
        return self.table.to(age_days.device)[idx]


class TemperatureScaler(nn.Module):
    """One positive scalar ``T``: calibrated logit = logit / T. ``T`` is stored as ``exp(log_t)`` so it stays > 0."""

    def __init__(self, init_temperature: float = 1.0) -> None:
        super().__init__()
        if init_temperature <= 0:
            raise ValueError("init_temperature must be > 0.")
        self.log_t = nn.Parameter(torch.tensor(math.log(init_temperature)))

    @property
    def temperature(self) -> float:
        return float(self.log_t.detach().exp())

    def forward(self, logits: Tensor) -> Tensor:
        return logits / self.log_t.exp()

    def fit(self, logits: Tensor, labels: Tensor, max_iter: int = 100) -> "TemperatureScaler":
        """Minimise the BCE of ``logits / T`` against ``labels`` (0/1). Both classes are required."""
        z = logits.detach().reshape(-1).float()
        y = labels.detach().reshape(-1).float()
        if z.numel() == 0 or z.numel() != y.numel():
            raise ValueError("logits and labels must be non-empty and of equal length.")
        n_pos = int((y > 0.5).sum())
        if n_pos == 0 or n_pos == y.numel():
            raise ValueError("Both classes are required to fit the temperature.")
        opt = torch.optim.LBFGS([self.log_t], lr=0.5, max_iter=max_iter, line_search_fn="strong_wolfe")

        def closure() -> Tensor:
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(z / self.log_t.exp(), y)
            loss.backward()
            return loss

        opt.step(closure)
        with torch.no_grad():
            self.log_t.clamp_(-4.0, 4.0)
        return self

    def fit_on_mature(
        self,
        logits: Tensor,
        labels: Tensor,
        age_days: Tensor,
        window_days: float,
        *,
        in_train_view: Tensor | None = None,
        max_iter: int = 100,
    ) -> "TemperatureScaler":
        """Fit on the MATURE train rows only (``age >= W``); every other row is ignored by construction."""
        age = age_days.reshape(-1)
        mask = _view_mask(age, in_train_view) & (age >= window_days)
        return self.fit(logits.reshape(-1)[mask], labels.reshape(-1)[mask], max_iter)


TemperatureScaling = TemperatureScaler  # name used in the project brief


class MatureTemperatureTracker:
    """Fits ``T`` on the maturity-masked subset of the TRAIN set (labels ``y_obs``), once per epoch.

    The loss calls ``collect`` with the detached logits and 0/1 labels of the mature rows (P, N and trusted-U) of
    each TRAINING step; ``refit`` (epoch end) fits ``T`` on what was collected and clears the buffer. Epoch 1 uses
    ``T = init_temperature``. Logits come from the training pass (dropout on), so ``T`` is slightly stale and
    noisy (limitation, KHÔNG CHẮC). Validation data cannot reach this object: nothing but the loss feeds it.
    The tracker is a plain object (not an ``nn.Module``), so it stays on the CPU and is never in an optimizer.

    ``T`` is clamped to ``[t_min, t_max]`` after every refit (default ``[0.5, 4.0]``). When the fitted value is
    below ``t_min`` it is raised to ``t_min``: ``last_raw_temperature`` keeps the fitted value and ``floor_hits``
    counts how often this happened.
    """

    def __init__(
        self,
        max_samples: int = 200_000,
        t_min: float = T_MIN_DEFAULT,
        t_max: float = T_MAX_DEFAULT,
        init_temperature: float = 1.0,
    ) -> None:
        if max_samples < 2 or not 0 < t_min <= init_temperature <= t_max:
            raise ValueError("need max_samples >= 2 and 0 < t_min <= init_temperature <= t_max.")
        self.max_samples = int(max_samples)
        self.t_min, self.t_max = float(t_min), float(t_max)
        self.scaler = TemperatureScaler(init_temperature)
        self.last_raw_temperature: float | None = None  # fitted T before the clamp (diagnostic)
        self.floor_hits: int = 0                        # number of refits clamped up to t_min
        self._z: list[Tensor] = []
        self._y: list[Tensor] = []
        self._n = 0

    @property
    def temperature(self) -> float:
        return self.scaler.temperature

    @property
    def n_collected(self) -> int:
        return self._n

    def collect(self, logits: Tensor, labels: Tensor) -> None:
        z = logits.detach().reshape(-1).float()
        y = labels.detach().reshape(-1).float()
        if z.numel() != y.numel():
            raise ValueError("logits and labels must have the same length.")
        room = self.max_samples - self._n
        if room <= 0 or z.numel() == 0:
            return
        z, y = z[:room], y[:room]  # train batches are shuffled, so the first rows are a random sample
        self._z.append(z.cpu())
        self._y.append(y.cpu())
        self._n += int(z.numel())

    def collected(self) -> tuple[Tensor, Tensor]:
        if not self._z:
            return torch.empty(0), torch.empty(0)
        return torch.cat(self._z), torch.cat(self._y)

    def refit(self) -> float:
        """Fit ``T`` on the collected rows (needs both classes, else keep the old ``T``); clear the buffer."""
        z, y = self.collected()
        self._z, self._y, self._n = [], [], 0
        n_pos = int((y > 0.5).sum())
        if z.numel() == 0 or n_pos == 0 or n_pos == z.numel():
            return self.temperature
        self.scaler.fit(z, y)
        raw = self.scaler.temperature
        self.last_raw_temperature = raw
        if raw < self.t_min:
            self.floor_hits += 1
        with torch.no_grad():
            self.scaler.log_t.clamp_(math.log(self.t_min), math.log(self.t_max))
        return self.temperature


@torch.no_grad()
def calibrated_prior(logits_unlabeled: Tensor, scaler: TemperatureScaler, eps: float = PRIOR_EPS) -> float:
    """Mean calibrated probability over unlabeled rows (model-based prior estimate; not wired into the trainer)."""
    p = torch.sigmoid(scaler(logits_unlabeled.reshape(-1).float()))
    return _clip(float(p.mean()), eps)


__all__ = [
    "PriorEstimate", "sliding_window_prior", "young_counts", "unlabeled_prior", "unlabeled_prior_raw",
    "TemperatureScaler", "TemperatureScaling", "MatureTemperatureTracker", "SlidingWindowPrior",
    "calibrated_prior", "PRIOR_EPS", "MIN_UNLABELED_PRIOR", "T_MIN_DEFAULT", "T_MAX_DEFAULT",
]