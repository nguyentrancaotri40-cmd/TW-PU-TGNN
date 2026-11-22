"""Mini-batch management on top of the phase graphs.

``TemporalGraphDataset`` owns ONE phase graph (train / val / test) and yields
standard PyG ``HeteroData`` mini-batches built by ``src.graph.sampler``:

* the first ``batch_size`` nodes of ``batch['transaction']`` are the seeds;
* node attributes: ``x`` (features), ``time``, ``y_obs``, ``pnu`` (0=P, 1=N, 2=U),
  ``amount`` (raw, for EFL), ``in_train_view``, ``detection_time``, ``tid``;
* ``y_true`` is present only when ``expose_true_labels=True`` (val/test evaluation
  and oracle runs). Never give it to a training loss on delayed labels.

Typical use::

    ds = load_phase_datasets("data/processed/delay_low", cfg["sampler"])
    for batch in ds["train"]:                 # temporal, causal mini-batches
        y_obs, pnu = seed_labels(batch)["y_obs"], seed_labels(batch)["pnu"]
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterator, Mapping

import torch
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader

from src.graph.builder import TXN, validate_phase_graph
from src.graph.sampler import build_neighbor_loader, verify_batch_causality
from src.features.lp_features import N_LP_FEATURES, LPFeatureTransform, lp_transform_from_config

PHASES = ("train", "val", "test")
SEED_ATTRS = ("y_obs", "pnu", "amount", "tid", "time", "detection_time", "in_train_view", "y_true")


def load_phase_graph(path: str | Path, validate: bool = True) -> HeteroData:
    """Load a ``.pt`` phase graph written by ``scripts/prepare_data.py``.

    ``weights_only=False`` is required because the file stores a PyG object; only
    load files you produced yourself.
    """
    data = torch.load(Path(path), weights_only=False)
    if not isinstance(data, HeteroData):
        raise TypeError(f"{path} does not contain a HeteroData object.")
    if validate:
        validate_phase_graph(data)
    return data


class TemporalGraphDataset:
    """One phase graph + its causal mini-batch loader."""

    def __init__(
        self,
        data: HeteroData,
        *,
        num_neighbors: list[int],
        batch_size: int,
        shuffle: bool = False,
        strict_time: bool = True,
        temporal_strategy: str = "uniform",
        num_workers: int = 0,
        drop_last: bool = False,
        expose_true_labels: bool = False,
        check_causality: bool = False,
        lp_features: LPFeatureTransform | None = None,
    ) -> None:
        self.data = data
        self.phase: str = data.phase
        self.num_neighbors = list(num_neighbors)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.strict_time = strict_time
        self.temporal_strategy = temporal_strategy
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.expose_true_labels = expose_true_labels
        self.check_causality = check_causality
        self.feature_dim = int(self.data[TXN].x.size(1))
        self.lp_features = lp_features
        self._loader: NeighborLoader | None = None

    # -- construction ------------------------------------------------------ #
    @classmethod
    def from_file(cls, path: str | Path, **kwargs: Any) -> "TemporalGraphDataset":
        return cls(load_phase_graph(path), **kwargs)

    # -- introspection ------------------------------------------------------ #
    @property
    def num_seeds(self) -> int:
        return int(self.data[TXN].seed_mask.sum())

    @property
    def model_in_dim(self) -> int:
        """Width of ``x`` as the model sees it: graph features (+4 when LP features are enabled)."""
        return self.feature_dim + (N_LP_FEATURES if self.lp_features is not None else 0)

    @property
    def entity_sizes(self) -> dict[str, int]:
        """Number of entity nodes per type in this graph (embedding table sizes)."""
        return {nt: int(self.data[nt].num_nodes) for nt in self.data.node_types if nt != TXN}

    @property
    def metadata(self) -> tuple[list[str], list[tuple[str, str, str]]]:
        return self.data.metadata()

    def __len__(self) -> int:
        n = self.num_seeds
        return n // self.batch_size if self.drop_last else math.ceil(n / self.batch_size)

    # -- batching ----------------------------------------------------------- #
    @property
    def loader(self) -> NeighborLoader:
        """Built lazily so the dataset can be inspected without ``pyg-lib``."""
        if self._loader is None:
            self._loader = build_neighbor_loader(
                self.data,
                self.num_neighbors,
                self.batch_size,
                shuffle=self.shuffle,
                strict_time=self.strict_time,
                temporal_strategy=self.temporal_strategy,
                num_workers=self.num_workers,
                drop_last=self.drop_last,
                expose_true_labels=self.expose_true_labels,
                batch_transforms=[self.lp_features] if self.lp_features is not None else (),
            )
        return self._loader

    def __iter__(self) -> Iterator[HeteroData]:
        for batch in self.loader:
            if self.check_causality:
                verify_batch_causality(batch, strict_time=self.strict_time)
            yield batch


def seed_labels(batch: HeteroData) -> dict[str, Tensor]:
    """Per-seed tensors (the first ``batch_size`` transaction nodes of a mini-batch)."""
    store = batch[TXN]
    bs = int(store.batch_size)
    return {k: store[k][:bs] for k in SEED_ATTRS if k in store}


def load_phase_datasets(
    processed_dir: str | Path,
    sampler_cfg: Mapping[str, Any],
    *,
    phases: tuple[str, ...] = PHASES,
    check_causality: bool = False,
    features_cfg: Mapping[str, Any] | None = None,
    expose_train_true_labels: bool = False,
) -> dict[str, TemporalGraphDataset]:
    """Load ``train.pt`` / ``val.pt`` / ``test.pt`` with the ``sampler:`` config section.

    Train batches are shuffled and hide ``y_true``; val/test batches are ordered and
    expose ``y_true`` for evaluation.
    """
    root = Path(processed_dir)
    lp = lp_transform_from_config(features_cfg, read_meta(root))  # None unless enabled
    common = dict(
        num_neighbors=list(sampler_cfg["num_neighbors"]),
        strict_time=bool(sampler_cfg.get("strict_time", True)),
        temporal_strategy=str(sampler_cfg.get("temporal_strategy", "uniform")),
        num_workers=int(sampler_cfg.get("num_workers", 0)),
        check_causality=check_causality,
        lp_features=lp,
    )
    out: dict[str, TemporalGraphDataset] = {}
    for phase in phases:
        is_train = phase == "train"
        out[phase] = TemporalGraphDataset.from_file(
            root / f"{phase}.pt",
            batch_size=int(sampler_cfg["batch_size" if is_train else "eval_batch_size"]),
            shuffle=is_train,
            expose_true_labels=(not is_train) or expose_train_true_labels,  # True only for the ORACLE run
            **common,
        )
    return out


def read_meta(processed_dir: str | Path) -> dict[str, Any]:
    """Read ``meta.json`` written by ``prepare_data.py`` (boundaries, delay, stats)."""
    return json.loads((Path(processed_dir) / "meta.json").read_text(encoding="utf-8"))


__all__ = [
    "PHASES", "load_phase_graph", "TemporalGraphDataset", "seed_labels", "load_phase_datasets", "read_meta",
]