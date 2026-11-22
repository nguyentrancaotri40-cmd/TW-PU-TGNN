"""Causal ``NeighborLoader`` for the phase graphs (no peeking at the future).

How causality is enforced
-------------------------
* Every node type has an int64 ``time`` (transaction time / entity first-seen
  time) and edges carry ``timestamp`` instead. ``time_attr='time'`` therefore
  selects *node-level* temporal sampling, where a sampled neighbour must have
  ``time <= seed_time``.
* ``strict_time=True`` (default) feeds ``input_time = time - 1`` for every
  seed. With integer seconds this makes neighbours STRICTLY earlier than the
  seed, so same-second transactions are never visible (PyG alone would allow
  equality). Consequence: an entity whose first-seen time equals the seed's own
  time (a brand-new card/device) is not sampled; the seed simply has no history
  through it, which is the causal truth.
* Temporal sampling is always *disjoint*: each seed owns its subgraph (more
  memory than the shared-subgraph mode) and the batch holds ``batch[node_type].batch``.

Backend requirement
-------------------
Temporal sampling needs ``pyg-lib``. With ``torch-sparse`` only, PyG refuses
disjoint sampling. We raise instead of silently falling back to a
non-temporal sampler, which would leak the future.

``y_true`` is stripped from every mini-batch unless ``expose_true_labels=True``.
"""
from __future__ import annotations

from typing import Callable, Mapping, Sequence

import torch
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader

from src.graph.builder import TXN

TIME_ATTR = "time"
TEMPORAL_STRATEGIES = ("uniform", "last")


class CausalityError(AssertionError):
    """A sampled mini-batch contains a node or edge from the seed's future."""


def has_pyg_lib() -> bool:
    import torch_geometric.typing as pyg_typing

    return bool(getattr(pyg_typing, "WITH_PYG_LIB", False))


def require_pyg_lib() -> None:
    if not has_pyg_lib():
        raise ImportError(
            "Temporal NeighborLoader needs 'pyg-lib' (torch-sparse cannot do disjoint/temporal "
            "sampling). Install the wheel matching your torch/CUDA build, e.g. "
            "`pip install pyg_lib -f https://data.pyg.org/whl/torch-<TORCH>+<CUDA>.html`. "
            "Refusing to fall back to a non-temporal sampler (it would leak the future)."
        )


class DropAttributes:
    """Batch transform removing attributes (default: ``y_true``) from the seed type."""

    def __init__(self, names: Sequence[str], node_type: str = TXN) -> None:
        self.names = tuple(names)
        self.node_type = node_type

    def __call__(self, batch: HeteroData) -> HeteroData:
        store = batch[self.node_type]
        for name in self.names:
            if name in store:
                del store[name]
        return batch


class ComposeTransforms:
    """Apply batch transforms in order (LP features first, ``DropAttributes`` always last)."""

    def __init__(self, steps: Sequence[Callable[[HeteroData], HeteroData]]) -> None:
        self.steps = list(steps)

    def __call__(self, batch: HeteroData) -> HeteroData:
        for step in self.steps:
            batch = step(batch)
        return batch


def _validate_graph_for_sampling(data: HeteroData, input_type: str) -> None:
    if input_type not in data.node_types:
        raise KeyError(f"Seed node type '{input_type}' not in graph.")
    for nt in data.node_types:
        t = data[nt].get(TIME_ATTR)
        if t is None:
            raise ValueError(f"Node type '{nt}' has no '{TIME_ATTR}'; temporal sampling needs it on every type.")
        if t.dtype != torch.long:
            raise ValueError(f"'{nt}.{TIME_ATTR}' must be int64 (got {t.dtype}); strict_time relies on integers.")
    if any(TIME_ATTR in s for s in data.edge_stores):
        raise ValueError(f"Edges must not carry '{TIME_ATTR}' (use 'timestamp'): PyG forbids node+edge time.")


def make_seed_time(data: HeteroData, seed_idx: Tensor, strict_time: bool, input_type: str = TXN) -> Tensor:
    """Time handed to the sampler for each seed (``time - 1`` when strict)."""
    t = data[input_type][TIME_ATTR][seed_idx]
    return t - 1 if strict_time else t.clone()


def build_neighbor_loader(
    data: HeteroData,
    num_neighbors: Sequence[int] | Mapping[object, Sequence[int]],
    batch_size: int,
    *,
    seed_mask: Tensor | None = None,
    shuffle: bool = False,
    strict_time: bool = True,
    temporal_strategy: str = "uniform",
    num_workers: int = 0,
    drop_last: bool = False,
    expose_true_labels: bool = False,
    input_type: str = TXN,
    batch_transforms: Sequence[Callable[[HeteroData], HeteroData]] = (),
    check_backend: bool = True,
    **loader_kwargs,
) -> NeighborLoader:
    """Create a temporal, disjoint ``NeighborLoader`` over the seeds of ``data``.

    Args:
        data: a phase graph from ``GraphBuilder.build`` (node ``time``, edge ``timestamp``).
        num_neighbors: fan-out per hop, e.g. ``[15, 10]`` (applied to every edge type), or a
            dict keyed by edge type. A transaction has at most one neighbour per entity type in
            hop 1, so the first number is rarely binding.
        batch_size: number of seed transactions per mini-batch.
        seed_mask: bool mask over ``input_type`` nodes; defaults to ``data[input_type].seed_mask``.
        strict_time: neighbours strictly earlier than the seed (``input_time = time - 1``).
        temporal_strategy: ``uniform`` or ``last`` (most recent neighbours).
        expose_true_labels: keep ``y_true`` in batches (evaluation / oracle runs only).
        batch_transforms: callables run on every batch BEFORE ``DropAttributes`` (e.g. LPFeatureTransform,
            which needs ``y_true``). ``DropAttributes`` is always appended last unless
            ``expose_true_labels=True``.
        check_backend: raise if ``pyg-lib`` is missing. Only set False to build the object
            for inspection; iterating it without the backend fails anyway.
    """
    if temporal_strategy not in TEMPORAL_STRATEGIES:
        raise ValueError(f"temporal_strategy must be one of {TEMPORAL_STRATEGIES}.")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0.")
    if isinstance(num_neighbors, Sequence) and (len(num_neighbors) == 0 or any(int(k) <= 0 for k in num_neighbors)):
        raise ValueError("num_neighbors must be a non-empty list of positive ints (e.g. [15, 10]).")
    if check_backend:
        require_pyg_lib()
    _validate_graph_for_sampling(data, input_type)

    if seed_mask is None:
        seed_mask = data[input_type].get("seed_mask")
        if seed_mask is None:
            raise ValueError("No seed_mask given and the graph has none.")
    seed_idx = torch.nonzero(seed_mask, as_tuple=False).view(-1)
    if seed_idx.numel() == 0:
        raise ValueError("No seed nodes selected.")
    input_time = make_seed_time(data, seed_idx, strict_time, input_type)

    steps = list(batch_transforms)
    if not expose_true_labels:
        steps.append(DropAttributes(("y_true",), node_type=input_type))  # ALWAYS last
    transform = None if not steps else (steps[0] if len(steps) == 1 else ComposeTransforms(steps))

    return NeighborLoader(
        data,
        num_neighbors=list(num_neighbors) if not isinstance(num_neighbors, Mapping) else dict(num_neighbors),
        input_nodes=(input_type, seed_idx),
        input_time=input_time,
        time_attr=TIME_ATTR,
        temporal_strategy=temporal_strategy,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        transform=transform,
        **loader_kwargs,
    )


def verify_batch_causality(batch: HeteroData, *, strict_time: bool = True, input_type: str = TXN) -> None:
    """Assert that no sampled node/edge is later than its seed (use in tests / debugging).

    Relies on the disjoint layout: ``batch[nt].batch`` maps each node to its seed's
    subgraph and ``batch[input_type].seed_time`` holds the time given to the sampler.
    The first ``batch_size`` nodes of ``input_type`` are the seeds themselves and are skipped.
    """
    store = batch[input_type]
    bs = int(store.batch_size)
    seed_time = store.seed_time
    if seed_time.numel() != bs:
        raise CausalityError("seed_time and batch_size disagree.")

    for nt in batch.node_types:
        ns = batch[nt]
        if "batch" not in ns:
            raise CausalityError(f"'{nt}' has no subgraph ids: sampling was not disjoint/temporal.")
        sub, times = ns.batch, ns[TIME_ATTR]
        later = times > seed_time[sub]
        if nt == input_type:
            later[:bs] = False  # the seeds themselves (strict mode puts them 1 s after seed_time)
        if later.any():
            raise CausalityError(f"{int(later.sum())} '{nt}' node(s) later than their seed.")

    slack = 1 if strict_time else 0  # a seed's own edges carry the seed's time
    for et in batch.edge_types:
        es = batch[et]
        if "timestamp" not in es:
            continue
        sub = batch[et[2]].batch[es.edge_index[1]]
        if (es.timestamp > seed_time[sub] + slack).any():
            raise CausalityError(f"edge type {et}: edge later than the seed.")


__all__ = [
    "CausalityError", "DropAttributes", "ComposeTransforms", "has_pyg_lib", "require_pyg_lib",
    "make_seed_time", "build_neighbor_loader", "verify_batch_causality",
]