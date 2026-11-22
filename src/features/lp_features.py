"""ATLAS-style LAG-SAFE label-propagation (LP) features for the seed transactions.

For every seed ``i`` scored at time ``t_i`` we count what the label system would ALREADY have told us
about the transactions linked to it through shared entities (card / device):

    lp_n_labeled   number of neighbours whose label is known at t_i
    lp_n_fraud     number of those that are confirmed fraud
    lp_fraud_rate  n_fraud / n_labeled   (0 when n_labeled = 0, never NaN)
    lp_has_fraud   1 if n_fraud > 0 else 0

Lag-safe vs the NAIVE LP feature
--------------------------------
The naive LP feature reads every neighbour label (``y_true``), including labels that arrive AFTER
``t_i``: it leaks the future. The lag-safe version only reads what is knowable at ``t_i``. This is the
feature-level arm of DelayGraphBench (ATLAS-style, per the project brief; the exact ATLAS definition
was NOT verified against its full text: KHÔNG CHẮC, CẦN VERIFY). Training still uses an ordinary loss.

Knowledge of neighbour ``j`` at the scoring time ``t_i`` (j must be STRICTLY earlier than i):

    P (confirmed fraud) : y_true_j == 1  AND  detection_time_j <= t_i
    N (matured negative): not P          AND  t_i - t_j >= W          (maturity window W, label-independent)
    U (unknown)         : everything else, never counted

This is exactly ``assign_pnu_masks`` evaluated with ``t_cut = t_i`` instead of ``t_train``.

How leakage is prevented
------------------------
1. ``y_true`` is read in ONE expression, ``(y == 1) & (detection_time <= t_seed)``. Wherever the gate is
   False the value of ``y_true`` has no effect: the labels of neighbours reported after ``t_i`` are
   invisible, whatever they are.
2. ``y_obs`` / ``pnu`` are NOT used: they are a snapshot at ``t_train``. For a train seed they would
   already contain labels that arrive between ``t_i`` and ``t_train`` (a leak); for a val/test seed they
   would hide labels that arrived between ``t_train`` and ``t_i``.
3. Negatives are matured by AGE (``t_i - t_j >= W``), not by the hidden delay of a legitimate row, so the
   N/U split never depends on the label.
4. The seed's own label is never read: only non-seed nodes are neighbours, they must be strictly earlier
   than the seed, and a node with the seed's own id is dropped (when ``n_id`` is available). Also, a seed
   always has ``detection_time > t_i``, so its own label can never pass the gate.
5. The transform must run BEFORE ``DropAttributes`` (it needs ``y_true``). ``build_neighbor_loader`` appends
   ``DropAttributes`` last; the transform raises if ``y_true`` is already gone.

Limits (read before interpreting results)
-----------------------------------------
* The counts cover the SAMPLED neighbours (<= ``num_neighbors[1]`` per entity under ``uniform``), not
  every neighbour in the graph: ``n_labeled`` saturates and the rate is a sample estimate.
* Only seeds get LP values. History nodes get zeros in the 4 columns (their own neighbourhood is not
  sampled), so the model sees 4 always-zero inputs on them.
* A neighbour reached through two entities is counted once (PyG disjoint sampling keeps one copy of a node
  per subgraph; KHÔNG CHẮC, CẦN VERIFY on a real batch).
* With uncapped delays a fraud that is already "mature" but still unreported counts as a negative
  (label noise, as in ``assign_pnu_masks``).
* ``negative_rule="detection_time"`` is the literal "detection_time <= t_i for both classes" reading: it
  uses the simulated delay of legitimate rows, which a real system does not have. Oracle-like ablation only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

import torch
from torch import Tensor
from torch_geometric.data import HeteroData

from src.data.loader import SECONDS_PER_DAY
from src.graph.builder import TXN

LP_FEATURE_NAMES: tuple[str, ...] = ("lp_n_labeled", "lp_n_fraud", "lp_fraud_rate", "lp_has_fraud")
N_LP_FEATURES: int = len(LP_FEATURE_NAMES)
NegativeRule = Literal["maturity_window", "detection_time"]
NEGATIVE_RULES: tuple[str, ...] = ("maturity_window", "detection_time")
_REQUIRED_ATTRS: tuple[str, ...] = ("x", "time", "detection_time", "y_true", "batch", "batch_size")


def compute_lp_features(
    *,
    time: Tensor,
    detection_time: Tensor,
    y_true: Tensor,
    subgraph: Tensor,
    batch_size: int,
    maturity_window_days: float,
    negative_rule: NegativeRule = "maturity_window",
    node_id: Tensor | None = None,
) -> Tensor:
    """Raw LP features ``[batch_size, 4]`` (float32) for the first ``batch_size`` nodes (the seeds).

    Args:
        time: ``[n]`` int64 transaction time in seconds (seeds first, then history nodes).
        detection_time: ``[n]`` float seconds (``delay_simulator``); seed ``i`` has detection_time > time.
        y_true: ``[n]`` 0/1 final label. Read ONLY through the ``detection_time <= t_seed`` gate.
        subgraph: ``[n]`` subgraph id of each node (``batch['transaction'].batch``).
        batch_size: number of seeds = the first ``batch_size`` nodes.
        maturity_window_days: W, an unreported neighbour older than W counts as a negative.
        negative_rule: ``"maturity_window"`` (default) or ``"detection_time"`` (oracle-like ablation).
        node_id: optional ``[n]`` global ids; a history node with the seed's own id is ignored.
    """
    n = time.numel()
    bs = int(batch_size)
    if not (detection_time.numel() == n and y_true.numel() == n and subgraph.numel() == n):
        raise ValueError("time, detection_time, y_true and subgraph must have the same length.")
    if not 0 <= bs <= n:
        raise ValueError("batch_size must be between 0 and the number of transaction nodes.")
    if negative_rule not in NEGATIVE_RULES:
        raise ValueError(f"negative_rule must be one of {NEGATIVE_RULES}.")
    if maturity_window_days <= 0:
        raise ValueError("maturity_window_days must be > 0.")

    device = time.device
    out = torch.zeros(bs, N_LP_FEATURES, dtype=torch.float32, device=device)
    if bs == 0 or n == bs:  # no seeds or no history
        return out

    sub = subgraph.long()
    row_of_sub = torch.full((int(sub.max()) + 1,), -1, dtype=torch.long, device=device)
    row_of_sub[sub[:bs]] = torch.arange(bs, device=device)
    row = row_of_sub[sub[bs:]]  # owning seed row of every history node
    if bool((row < 0).any()):
        raise ValueError("A history node belongs to a subgraph without a seed.")

    t = time.to(torch.float64)
    det = detection_time.to(torch.float64)
    t_h, det_h, y_h = t[bs:], det[bs:], y_true[bs:]
    t_seed = t[:bs][row]

    usable = t_h < t_seed  # STRICTLY earlier than the seed (also removes the seed itself)
    if node_id is not None:
        usable = usable & (node_id[bs:] != node_id[:bs][row])

    # The only place where y_true is read: a label counts only if its report arrived by t_seed.
    reported = (y_h == 1) & (det_h <= t_seed)
    if negative_rule == "maturity_window":
        matured = (t_seed - t_h) >= float(maturity_window_days) * SECONDS_PER_DAY  # age only
    else:
        matured = det_h <= t_seed
    negative = (~reported) & matured

    labeled = usable & (reported | negative)
    fraud = usable & reported
    n_labeled = torch.bincount(row[labeled], minlength=bs).to(torch.float32)
    n_fraud = torch.bincount(row[fraud], minlength=bs).to(torch.float32)
    out[:, 0] = n_labeled
    out[:, 1] = n_fraud
    out[:, 2] = n_fraud / n_labeled.clamp(min=1.0)  # 0 when n_labeled == 0, never NaN
    out[:, 3] = (n_fraud > 0).to(torch.float32)
    return out


class LPFeatureTransform:
    """Batch transform: append the 4 LP columns to ``batch['transaction'].x`` (F -> F + 4).

    Must run BEFORE ``DropAttributes`` (it needs ``y_true``). Rows of history nodes get zeros; seed rows
    get the lag-safe features. ``log1p_counts`` applies ``log1p`` to the two count columns so they live on
    the scale of the z-scored features (the rate and the flag are already in [0, 1]).
    """

    n_features: int = N_LP_FEATURES
    names: tuple[str, ...] = LP_FEATURE_NAMES

    def __init__(
        self,
        maturity_window_days: float,
        negative_rule: NegativeRule = "maturity_window",
        log1p_counts: bool = True,
        node_type: str = TXN,
    ) -> None:
        if maturity_window_days <= 0:
            raise ValueError("maturity_window_days must be > 0.")
        if negative_rule not in NEGATIVE_RULES:
            raise ValueError(f"negative_rule must be one of {NEGATIVE_RULES}.")
        self.maturity_window_days = float(maturity_window_days)
        self.negative_rule = negative_rule
        self.log1p_counts = log1p_counts
        self.node_type = node_type

    def __call__(self, batch: HeteroData) -> HeteroData:
        store = batch[self.node_type]
        if "lp_dim" in store:
            raise RuntimeError("LP features were already appended to this batch.")
        missing = [k for k in _REQUIRED_ATTRS if k not in store]
        if missing:
            raise KeyError(
                f"LP transform needs {missing} on '{self.node_type}'. It must run BEFORE DropAttributes "
                "(y_true is removed afterwards) and on a disjoint temporal batch."
            )
        bs = int(store.batch_size)
        lp = compute_lp_features(
            time=store.time,
            detection_time=store.detection_time,
            y_true=store.y_true,
            subgraph=store.batch,
            batch_size=bs,
            maturity_window_days=self.maturity_window_days,
            negative_rule=self.negative_rule,
            node_id=store.get("n_id"),
        )
        if self.log1p_counts:
            lp[:, :2] = torch.log1p(lp[:, :2])
        full = store.x.new_zeros(store.x.size(0), N_LP_FEATURES)
        full[:bs] = lp.to(store.x.dtype)  # history rows stay zero
        store.x = torch.cat([store.x, full], dim=1)
        store.lp_dim = N_LP_FEATURES
        return batch


@dataclass(frozen=True)
class LPFeatureConfig:
    """The ``features.lp_features`` section of configs/base.yaml."""

    enabled: bool = False
    maturity_window_days: float | None = None  # None -> meta['config']['delay']['maturity_window_days']
    negative_rule: str = "maturity_window"
    log1p_counts: bool = True

    @classmethod
    def from_dict(cls, cfg: Mapping[str, Any] | None) -> "LPFeatureConfig":
        if not cfg:
            return cls()
        w = cfg.get("maturity_window_days")
        return cls(
            enabled=bool(cfg.get("enabled", False)),
            maturity_window_days=None if w is None else float(w),
            negative_rule=str(cfg.get("negative_rule", "maturity_window")),
            log1p_counts=bool(cfg.get("log1p_counts", True)),
        )


def lp_transform_from_config(
    features_cfg: Mapping[str, Any] | None, meta: Mapping[str, Any] | None = None
) -> LPFeatureTransform | None:
    """Return the transform, or ``None`` when ``features.lp_features.enabled`` is false (the default)."""
    cfg = LPFeatureConfig.from_dict((features_cfg or {}).get("lp_features"))
    if not cfg.enabled:
        return None
    window = cfg.maturity_window_days
    if window is None:
        window = None if meta is None else meta.get("config", {}).get("delay", {}).get("maturity_window_days")
    if window is None:
        raise ValueError("LP features need maturity_window_days (set it in the config or pass meta.json).")
    return LPFeatureTransform(float(window), cfg.negative_rule, cfg.log1p_counts)


__all__ = [
    "LP_FEATURE_NAMES", "N_LP_FEATURES", "compute_lp_features", "LPFeatureTransform",
    "LPFeatureConfig", "lp_transform_from_config",
]