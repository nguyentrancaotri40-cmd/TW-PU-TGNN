"""Build PyG ``HeteroData`` phase graphs from the P/N/U-masked DataFrames.

Schema
------
Node types : ``transaction`` plus entity types ``card`` / ``device``
             (the email domain is a TRANSACTION FEATURE block, not a node type: see A5).
Edge types : ``(transaction, uses_<e>, <e>)`` and the reverse
             ``(<e>, rev_uses_<e>, transaction)``; both carry ``timestamp``
             (= ``TransactionDT`` of the transaction endpoint). The reverse
             edges are what lets a seed transaction reach its entities when
             PyG samples along incoming edges.

Phase graphs (cumulative, physically cut)
-----------------------------------------
``train`` = rows before ``t_train``; ``val`` = rows before ``t_val`` (train+val);
``test`` = every row. A graph never holds a node later than its own phase end,
so no Val/Test node can exist in the Train graph. Seeds of a phase graph are
the nodes of that phase (``seed_mask``); earlier nodes are history only.

Time semantics (node-level ``time``, needed by ``NeighborLoader(time_attr='time')``)
-----------------------------------------------------------------------------
* transaction ``time`` = ``TransactionDT`` (int64 seconds);
* entity ``time`` = FIRST-SEEN time (earliest linked transaction);
* the edge attribute is called ``timestamp``, never ``time``: PyG refuses a
  ``time_attr`` that exists on both nodes and edges.
Entity nodes carry NO statistic computed over the phase (no degree, no counts):
such a value would summarise transactions that are in a seed's future.

Assumptions (also in configs/base.yaml)
---------------------------------------
A1. No true card/device id exists in IEEE-CIS: entities are PROXY keys built from
    several columns, so graph quality depends on this construction.
A2. Super-nodes (entities with more than ``max_degree`` TRAIN transactions, e.g.
    a very common device string) are removed. The set is fitted on the train period only and
    reused for Val/Test, so no future information decides it. An entity that
    first appears after the train period is never removed (neighbour fan-out
    bounds the cost of sampling it).
A3. Entity ids are assigned in first-seen order over the whole timeline, so the
    id of an entity is identical in the train/val/test graphs and each phase
    graph holds exactly ids ``0..K_phase-1``. Ids are labels only.
A4. Numeric NaN are imputed with TRAIN medians and z-scored with TRAIN
    statistics; categories come from the train period (unseen -> ``__other__``).
A5. Numeric columns whose TRAIN missing ratio exceeds ``features.max_missing_ratio`` are dropped
    before any statistic is computed (validation/test never decide which columns survive).
    The email domain is encoded as transaction features (``EmailFeatureBlock``): train
    frequency + one-hot of the top-k TRAIN domains + other + missing. ``train_freq`` is a static
    statistic of the train period (no label, no val/test information), like the z-score mean/std.
"""
from __future__ import annotations

import logging
from collections import Counter
import warnings
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Mapping

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

from src.data.loader import (
    AMOUNT_COL,
    DETECTION_COL,
    ID_COL,
    IN_TRAIN_VIEW_COL,
    MISSING_TOKEN,
    NON_FEATURE_COLS,
    PNU_COL,
    TARGET_COL,
    TIME_COL,
    Y_OBS_COL,
    assert_no_label_columns,
)
from src.data.splitter import SplitBoundaries

logger = logging.getLogger(__name__)

TXN = "transaction"
Phase = Literal["train", "val", "test"]
SPLIT_CODES: dict[str, int] = {"train": 0, "val": 1, "test": 2}
PNU_CODES: dict[str, int] = {"P": 0, "N": 1, "U": 2}
MISSING_RATIO_NAME = "__missing_ratio"


class GraphIntegrityError(ValueError):
    """A phase graph violates a causality / masking invariant."""


def _tensor(a: np.ndarray) -> torch.Tensor:
    """numpy -> torch with an owned, writable copy (pandas 3 hands out read-only views)."""
    return torch.from_numpy(np.array(a, copy=True))


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class EntitySpec:
    """One entity type. ``key_cols`` form the (proxy) key; a row gets an edge only
    if every column in ``required_cols`` is non-null. ``max_degree=None`` keeps all."""

    name: str
    key_cols: tuple[str, ...]
    required_cols: tuple[str, ...]
    max_degree: int | None = 500


@dataclass(frozen=True)
class EmailSpec:
    """``graph.features.email``: email domain -> transaction features.

    An empty ``cols`` disables the block.
    """

    cols: tuple[str, ...] = ("P_emaildomain",)
    top_k: int = 10
    add_frequency: bool = True

    @classmethod
    def from_dict(cls, d: Mapping[str, Any] | None) -> "EmailSpec":
        d = dict(d or {})
        cols = d.get("cols", cls.cols)
        cols = (cols,) if isinstance(cols, str) else tuple(cols)
        return cls(
            cols=cols,
            top_k=int(d.get("top_k", cls.top_k)),
            add_frequency=bool(d.get("add_frequency", cls.add_frequency)),
        )


@dataclass(frozen=True)
class FeatureSpec:
    onehot_cols: tuple[str, ...] = (
        "ProductCD", "card4", "card6", "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",
    )
    numeric_exclude: tuple[str, ...] = ("card1", "addr1", "addr2")
    max_categories: int = 30
    clip_z: float | None = 10.0
    add_missing_ratio: bool = True
    min_std: float = 1e-6
    max_missing_ratio: float | None = 0.9   # drop numeric cols with TRAIN missing ratio > this; None = off
    email: EmailSpec = field(default_factory=EmailSpec)


DEFAULT_ENTITIES: tuple[EntitySpec, ...] = (
    EntitySpec("card", ("card1", "card2", "card3", "card4", "card5", "card6"), ("card1",), 500),
    EntitySpec("device", ("DeviceInfo", "id_30", "id_31"), ("DeviceInfo",), 500),
)


@dataclass(frozen=True)
class GraphConfig:
    entities: tuple[EntitySpec, ...] = DEFAULT_ENTITIES
    features: FeatureSpec = field(default_factory=FeatureSpec)

    @classmethod
    def from_dict(cls, cfg: Mapping[str, Any] | None) -> "GraphConfig":
        """Build from the ``graph:`` section of configs/base.yaml."""
        if not cfg:
            return cls()
        ents = tuple(
            EntitySpec(
                name=name,
                key_cols=tuple(e["key_cols"]),
                required_cols=tuple(e.get("required_cols", e["key_cols"][:1])),
                max_degree=e.get("max_degree", 500),
            )
            for name, e in cfg.get("entities", {}).items()
        ) or DEFAULT_ENTITIES
        f = dict(cfg.get("features", {}))
        for k in ("onehot_cols", "numeric_exclude"):
            if k in f:
                f[k] = tuple(f[k])
        f["email"] = EmailSpec.from_dict(f.get("email"))  # dict | None -> EmailSpec
        return cls(entities=ents, features=FeatureSpec(**f))


# --------------------------------------------------------------------------- #
# Train-only column filter + email feature block
# --------------------------------------------------------------------------- #
def select_columns_by_missing_ratio(
    train_df: pd.DataFrame,
    columns: list[str],
    max_missing_ratio: float | None,
) -> tuple[list[str], dict[str, float]]:
    """Drop columns whose missing ratio on the TRAIN frame is > ``max_missing_ratio``.

    Only ``train_df`` is inspected. ``None`` disables the filter; 1.0 keeps everything.
    Column-wise loop on purpose: ``train_df[cols].isna()`` would allocate a full boolean frame.

    Returns ``(kept columns in original order, {dropped column: train missing ratio})``.
    """
    cols = list(columns)
    if max_missing_ratio is None or not cols:
        return cols, {}
    if not 0.0 < float(max_missing_ratio) <= 1.0:
        raise ValueError("max_missing_ratio must be in (0, 1] or None.")
    n = len(train_df)
    if n == 0:
        raise ValueError("Empty train frame: cannot compute missing ratios.")
    dropped: dict[str, float] = {}
    for c in cols:
        r = float(train_df[c].isna().sum()) / n
        if r > max_missing_ratio:
            dropped[c] = r
    kept = [c for c in cols if c not in dropped]
    if not kept:
        raise ValueError("max_missing_ratio removed every numeric column.")
    return kept, dropped


class EmailFeatureBlock:
    """Email domain -> numeric features of the transaction node (no email node / edges).

    Per column in ``spec.cols``: ``[train_freq] + one-hot(top_k TRAIN domains) + is_other + is_missing``.
    Top-k list and frequencies come from the frame given to :meth:`fit` (TRAIN) only; a domain
    unseen in train gets ``train_freq = 0`` and ``is_other = 1``. Values are lower-cased and
    stripped; NaN maps to ``MISSING_TOKEN`` (the sentinel is never a "top domain").
    """

    def __init__(self, spec: EmailSpec) -> None:
        if spec.cols and spec.top_k < 1:
            raise ValueError("email.top_k must be >= 1.")
        self.spec = spec
        self.top_domains: dict[str, list[str]] = {}
        self.train_freq: dict[str, dict[str, float]] = {}

    @staticmethod
    def _values(df: pd.DataFrame, col: str) -> np.ndarray:
        """Cleaned domains as an object array (a missing column counts as all-missing)."""
        if col not in df.columns:
            return np.full(len(df), MISSING_TOKEN, dtype=object)
        s = df[col].astype("string").fillna(MISSING_TOKEN).str.strip().str.lower()
        return s.to_numpy(dtype=object)

    def fit(self, train_df: pd.DataFrame) -> "EmailFeatureBlock":
        if len(train_df) == 0:
            raise ValueError("Empty train frame: cannot fit email features.")
        for col in self.spec.cols:
            if col not in train_df.columns:
                raise KeyError(f"Email column '{col}' not found in the train frame.")
            vals = self._values(train_df, col)
            counts = Counter(vals.tolist())
            n = float(len(vals))
            self.train_freq[col] = {str(d): c / n for d, c in counts.items()}
            ranked = sorted(((c, str(d)) for d, c in counts.items() if d != MISSING_TOKEN),
                            key=lambda t: (-t[0], t[1]))  # count desc, name asc: deterministic
            self.top_domains[col] = [d for _, d in ranked[: self.spec.top_k]]
        return self

    @property
    def feature_names(self) -> list[str]:
        names: list[str] = []
        for col in self.spec.cols:
            if self.spec.add_frequency:
                names.append(f"email:{col}:train_freq")
            names += [f"email:{col}:is:{d}" for d in self.top_domains[col]]
            names += [f"email:{col}:is_other", f"email:{col}:is_missing"]
        return names

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """float32 ``[len(df), len(feature_names)]``, no NaN."""
        if not self.top_domains:
            raise RuntimeError("EmailFeatureBlock.fit must be called first.")
        blocks: list[np.ndarray] = []
        for col in self.spec.cols:
            vals = self._values(df, col)
            n = len(vals)
            if self.spec.add_frequency:
                fmap = self.train_freq[col]
                freq = np.fromiter((fmap.get(v, 0.0) for v in vals), dtype=np.float32, count=n)
                blocks.append(freq.reshape(-1, 1))
            top = self.top_domains[col]
            onehot = (np.stack([vals == d for d in top], axis=1).astype(np.float32)
                      if top else np.zeros((n, 0), dtype=np.float32))
            is_missing = vals == MISSING_TOKEN
            is_other = ~((onehot.sum(axis=1) > 0) | is_missing)
            blocks += [onehot,
                       is_other.reshape(-1, 1).astype(np.float32),
                       is_missing.reshape(-1, 1).astype(np.float32)]
        return np.concatenate(blocks, axis=1)

    def state_dict(self) -> dict[str, Any]:
        return {"spec": asdict(self.spec), "top_domains": self.top_domains, "train_freq": self.train_freq}


# --------------------------------------------------------------------------- #
# Transaction features (fitted on TRAIN only)
# --------------------------------------------------------------------------- #
class TxnFeatureEncoder:
    """Median-impute + z-score numeric columns, one-hot a few categoricals.

    Every statistic comes from the frame given to :meth:`fit` (the train period).
    ``TransactionAmt`` is log1p-transformed first. Constant columns are dropped.
    """

    def __init__(self, spec: FeatureSpec | None = None) -> None:
        self.spec = spec or FeatureSpec()
        self.numeric_cols: list[str] = []
        self.median = np.empty(0, dtype=np.float32)
        self.mean = np.empty(0, dtype=np.float32)
        self.std = np.empty(0, dtype=np.float32)
        self.onehot: dict[str, list[str]] = {}
        self.fitted = False
        self.dropped_missing_cols: dict[str, float] = {}  # {col: TRAIN missing ratio}, filled by fit
        self.email_block: EmailFeatureBlock | None = None

    # -- helpers ----------------------------------------------------------- #
    def _numeric_matrix(self, df: pd.DataFrame) -> np.ndarray:
        missing = [c for c in self.numeric_cols if c not in df.columns]
        if missing:
            logger.warning("%d numeric feature column(s) absent (imputed): %s...", len(missing), missing[:5])
        X = df.reindex(columns=self.numeric_cols).to_numpy(dtype=np.float32, copy=True)
        if AMOUNT_COL in self.numeric_cols:
            j = self.numeric_cols.index(AMOUNT_COL)
            X[:, j] = np.log1p(np.maximum(X[:, j], 0.0))
        return X

    # -- fit / transform --------------------------------------------------- #
    def fit(self, train_df: pd.DataFrame) -> "TxnFeatureEncoder":
        sp = self.spec
        banned = set(NON_FEATURE_COLS) | set(sp.numeric_exclude) | set(sp.onehot_cols)
        cand = [
            c for c in train_df.columns
            if c not in banned
            and pd.api.types.is_numeric_dtype(train_df[c])
            and not pd.api.types.is_bool_dtype(train_df[c])
        ]
        assert_no_label_columns(cand + [c for c in sp.onehot_cols if c in train_df.columns])
        # Drop mostly-missing numeric columns using TRAIN only, BEFORE any statistic is computed.
        cand, self.dropped_missing_cols = select_columns_by_missing_ratio(
            train_df, cand, sp.max_missing_ratio
        )
        self.numeric_cols = cand
        X = self._numeric_matrix(train_df)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN columns
            med = np.nanmedian(X, axis=0)
        ok = np.isfinite(med)
        X = np.where(np.isnan(X), med, X)
        mean = X.mean(axis=0, dtype=np.float64)
        std = X.std(axis=0, dtype=np.float64)
        ok &= std > sp.min_std
        dropped = int((~ok).sum())
        if dropped:
            logger.info("Dropping %d constant / all-missing numeric columns (train).", dropped)
        keep = np.flatnonzero(ok)
        self.numeric_cols = [self.numeric_cols[i] for i in keep]
        self.median = med[keep].astype(np.float32)
        self.mean = mean[keep].astype(np.float32)
        self.std = std[keep].astype(np.float32)

        self.onehot = {}
        for c in sp.onehot_cols:
            if c not in train_df.columns:
                continue
            counts = train_df[c].astype("string").value_counts()
            self.onehot[c] = [str(v) for v in counts.index[: sp.max_categories]]
        self.email_block = EmailFeatureBlock(sp.email).fit(train_df) if sp.email.cols else None
        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("TxnFeatureEncoder.fit must be called first.")
        X = self._numeric_matrix(df)
        miss = np.isnan(X)
        ratio = miss.mean(axis=1, dtype=np.float32) if X.shape[1] else np.zeros(len(df), np.float32)
        X = np.where(miss, self.median, X)
        X = (X - self.mean) / self.std
        if self.spec.clip_z is not None:
            np.clip(X, -self.spec.clip_z, self.spec.clip_z, out=X)
        parts: list[np.ndarray] = [X.astype(np.float32, copy=False)]
        for c, cats in self.onehot.items():
            s = df[c].astype("string") if c in df.columns else pd.Series(pd.NA, index=df.index, dtype="string")
            isna = s.isna().to_numpy()
            cols = [(s == cat).to_numpy(dtype=bool, na_value=False) for cat in cats]
            known = np.logical_or.reduce(cols) if cols else np.zeros(len(df), dtype=bool)
            cols += [isna, ~isna & ~known]  # <c>=__nan__, <c>=__other__
            parts.append(np.stack(cols, axis=1).astype(np.float32))
        if self.email_block is not None:  # NEW: must sit at the same position as in feature_names
            parts.append(self.email_block.transform(df))
        if self.spec.add_missing_ratio:
            parts.append(ratio.reshape(-1, 1))
        return np.concatenate(parts, axis=1)

    @property
    def feature_names(self) -> list[str]:
        names = list(self.numeric_cols)
        for c, cats in self.onehot.items():
            names += [f"{c}={v}" for v in cats] + [f"{c}=__nan__", f"{c}=__other__"]
        if self.email_block is not None:  # same order as transform
            names += self.email_block.feature_names
        if self.spec.add_missing_ratio:
            names.append(MISSING_RATIO_NAME)
        return names

    def state_dict(self) -> dict[str, Any]:
        return {
            "spec": asdict(self.spec),
            "numeric_cols": list(self.numeric_cols),
            "median": self.median.tolist(),
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "onehot": {k: list(v) for k, v in self.onehot.items()},
            "dropped_missing_cols": dict(self.dropped_missing_cols),
            "email": self.email_block.state_dict() if self.email_block is not None else None,
        }


# --------------------------------------------------------------------------- #
# Entity keys
# --------------------------------------------------------------------------- #
def make_entity_keys(df: pd.DataFrame, spec: EntitySpec) -> pd.Series:
    """String key per row (``<NA>`` where the row has no valid entity).

    NaN inside a *non-required* key column becomes the token ``NA`` so that the
    key stays comparable across phases.
    """
    cols = [c for c in spec.key_cols if c in df.columns]
    req = [c for c in spec.required_cols if c in df.columns]
    if not cols or len(req) < len(spec.required_cols):
        return pd.Series(pd.NA, index=df.index, dtype="string")
    valid = df[req].notna().all(axis=1)
    parts = df[cols].astype("string").fillna("NA")
    keys = parts.iloc[:, 0]
    for c in parts.columns[1:]:
        keys = keys + "|" + parts[c]
    return keys.where(valid)


# --------------------------------------------------------------------------- #
# Builder
# --------------------------------------------------------------------------- #
class GraphBuilder:
    """Fit once on the train period, then ``build`` one graph per phase."""

    def __init__(self, config: GraphConfig | None = None) -> None:
        self.config = config or GraphConfig()
        self.encoder = TxnFeatureEncoder(self.config.features)
        self.vocab: dict[str, pd.Index] = {}
        self.entity_stats: dict[str, dict[str, float]] = {}
        self.boundaries: SplitBoundaries | None = None
        self.fitted = False

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        boundaries: SplitBoundaries,
    ) -> "GraphBuilder":
        """Fit encoder + super-node set on TRAIN; build entity id vocabularies.

        ``val_df``/``test_df`` are used ONLY to assign stable entity ids (first-seen
        order); no statistic of theirs reaches the features or the super-node set.
        """
        self.encoder.fit(train_df)
        self.boundaries = boundaries
        key_cols = sorted({c for s in self.config.entities for c in (*s.key_cols, *s.required_cols)})
        keep_cols = [c for c in key_cols if c in train_df.columns]
        full = pd.concat([f[keep_cols] for f in (train_df, val_df, test_df)], ignore_index=True)
        n_train = len(train_df)

        for spec in self.config.entities:
            keys = make_entity_keys(full, spec)
            tr_keys = keys.iloc[:n_train].dropna()
            deg = tr_keys.value_counts()
            capped = spec.max_degree is not None
            super_keys = deg.index[deg > spec.max_degree] if capped else deg.index[:0]
            kept = keys[~keys.isin(super_keys)].dropna()
            self.vocab[spec.name] = pd.Index(pd.unique(kept))  # first-seen order (time-sorted input)
            n_edges = int(len(tr_keys))
            dropped = int(deg[deg.index.isin(super_keys)].sum()) if len(super_keys) else 0
            q = deg.quantile([0.5, 0.9, 0.99]) if len(deg) else pd.Series([0, 0, 0], index=[0.5, 0.9, 0.99])
            self.entity_stats[spec.name] = {
                "n_entities_train": int(len(deg)),
                "n_super_removed": int(len(super_keys)),
                "train_edges": n_edges,
                "train_edges_dropped": dropped,
                "train_edges_dropped_%": 100.0 * dropped / max(n_edges, 1),
                "deg_p50": float(q.iloc[0]), "deg_p90": float(q.iloc[1]), "deg_p99": float(q.iloc[2]),
                "deg_max": float(deg.max()) if len(deg) else 0.0,
                "max_degree_cap": float(spec.max_degree) if capped else float("inf"),
                "vocab_size_all_phases": int(len(self.vocab[spec.name])),
            }
        self.fitted = True
        return self

    # ------------------------------------------------------------------ #
    def build(self, df: pd.DataFrame, phase: Phase) -> HeteroData:
        """Build the graph of ``phase`` from the cumulative, time-sorted frame ``df``.

        ``df`` must be train (phase=train), train+val (val) or train+val+test (test),
        already passed through ``simulate_delay`` and ``assign_pnu_masks``.
        """
        if not self.fitted or self.boundaries is None:
            raise RuntimeError("GraphBuilder.fit must be called first.")
        b = self.boundaries
        t = df[TIME_COL].to_numpy(dtype=np.int64)
        if len(t) > 1 and (np.diff(t) < 0).any():
            raise GraphIntegrityError("Input frame must be sorted by TransactionDT.")
        t_end = {"train": b.t_train, "val": b.t_val, "test": b.t_end}[phase]
        if len(t) == 0 or (t >= t_end).any():
            raise GraphIntegrityError(f"Rows at/after the end of the '{phase}' phase (t_end={t_end}).")

        split = ((t >= b.t_train).astype(np.int8) + (t >= b.t_val).astype(np.int8)).astype(np.int8)
        data = HeteroData()
        txn = data[TXN]
        x = self.encoder.transform(df)
        txn.x = torch.from_numpy(x)
        txn.time = _tensor(t)
        txn.detection_time = _tensor(df[DETECTION_COL].to_numpy(dtype=np.float64))
        txn.amount = _tensor(df[AMOUNT_COL].to_numpy(dtype=np.float32))
        txn.y_obs = _tensor(df[Y_OBS_COL].to_numpy(dtype=np.int8))
        txn.pnu = _tensor(df[PNU_COL].astype(str).map(PNU_CODES).to_numpy(dtype=np.int8))
        txn.in_train_view = _tensor(df[IN_TRAIN_VIEW_COL].to_numpy(dtype=bool))
        # Evaluation / oracle ONLY. The dataset layer strips it unless explicitly allowed.
        txn.y_true = _tensor(df[TARGET_COL].to_numpy(dtype=np.int8))
        txn.split = _tensor(split)
        txn.tid = _tensor(df[ID_COL].to_numpy(dtype=np.int64))
        txn.seed_mask = _tensor(split == SPLIT_CODES[phase])

        for spec in self.config.entities:
            keys = make_entity_keys(df, spec)
            valid = keys.notna().to_numpy()
            codes = np.full(len(df), -1, dtype=np.int64)
            if valid.any():
                codes[valid] = self.vocab[spec.name].get_indexer(keys[valid])
            src = np.flatnonzero(codes >= 0)
            if src.size == 0:
                logger.warning("Entity '%s' has no edges in the %s graph; type dropped.", spec.name, phase)
                continue
            dst = codes[src]
            n_ent = int(dst.max()) + 1
            uniq, first_pos = np.unique(dst, return_index=True)  # df is time-sorted: first = earliest
            if uniq.size != n_ent:  # ids must be contiguous (assumption A3)
                raise GraphIntegrityError(f"Entity ids of '{spec.name}' are not contiguous in the {phase} graph.")
            ts = t[src]
            ent = data[spec.name]
            ent.node_id = torch.arange(n_ent, dtype=torch.long)
            ent.time = _tensor(ts[first_pos].astype(np.int64))  # first-seen time
            ent.num_nodes = n_ent
            fwd = _tensor(np.stack([src, dst]))
            data[(TXN, f"uses_{spec.name}", spec.name)].edge_index = fwd
            data[(TXN, f"uses_{spec.name}", spec.name)].timestamp = _tensor(ts)
            data[(spec.name, f"rev_uses_{spec.name}", TXN)].edge_index = fwd.flip(0).contiguous()
            data[(spec.name, f"rev_uses_{spec.name}", TXN)].timestamp = _tensor(ts)

        data.phase = phase
        data.t_end = float(t_end)
        data.t_train = float(b.t_train)
        data.t_val = float(b.t_val)
        data.label_cut = float(b.t_train)  # y_obs / pnu are defined at this instant
        data.feature_names = self.encoder.feature_names
        return data

    def state_dict(self) -> dict[str, Any]:
        return {
            "config": {"entities": [asdict(e) for e in self.config.entities], "features": asdict(self.config.features)},
            "encoder": self.encoder.state_dict(),
            "vocab_sizes": {k: int(len(v)) for k, v in self.vocab.items()},
            "entity_stats": self.entity_stats,
            "boundaries": asdict(self.boundaries) if self.boundaries else None,
        }


# --------------------------------------------------------------------------- #
# Static integrity checks (used by prepare_data and tests/test_leakage.py)
# --------------------------------------------------------------------------- #
def validate_phase_graph(data: HeteroData) -> None:
    """Raise ``GraphIntegrityError`` listing every violated invariant.

    Checks: no node/edge later than the phase end; no Val/Test node in an earlier
    graph; seeds = the phase's own nodes; edge ``timestamp`` = time of its
    transaction endpoint; forward/reverse edge consistency; entity ``time`` =
    first-seen time; ``time`` only on nodes; label masking at the label cut;
    no label-derived feature name.
    """
    errs: list[str] = []
    phase = data.phase
    code = SPLIT_CODES[phase]
    txn = data[TXN]
    t = txn.time.numpy()

    if (t >= data.t_end).any():
        errs.append("transaction node at/after the end of the phase (future node in graph)")
    if len(t) > 1 and (np.diff(t) < 0).any():
        errs.append("transaction nodes are not sorted by time")
    split = txn.split.numpy()
    if (split > code).any():
        errs.append("node from a LATER phase present (e.g. Val/Test node in the Train graph)")
    expected = np.where(t >= data.t_val, 2, np.where(t >= data.t_train, 1, 0))
    if not np.array_equal(split, expected):
        errs.append("split codes disagree with the time boundaries")
    if not np.array_equal(txn.seed_mask.numpy(), split == code):
        errs.append("seed_mask differs from the phase's own nodes")

    for store in data.edge_stores:
        if "time" in store:
            errs.append("an edge store has a 'time' attribute (PyG forbids node+edge time)")
    for nt in data.node_types:
        if "time" not in data[nt]:
            errs.append(f"node type '{nt}' has no 'time'")

    for ent in [n for n in data.node_types if n != TXN]:
        fkey, rkey = (TXN, f"uses_{ent}", ent), (ent, f"rev_uses_{ent}", TXN)
        if fkey not in data.edge_types or rkey not in data.edge_types:
            errs.append(f"entity '{ent}': forward or reverse edge type missing")
            continue
        fe, re = data[fkey], data[rkey]
        src, dst = fe.edge_index[0].numpy(), fe.edge_index[1].numpy()
        n_ent = data[ent].num_nodes
        if src.size and (src.max() >= len(t) or dst.max() >= n_ent or src.min() < 0 or dst.min() < 0):
            errs.append(f"entity '{ent}': edge index out of range")
            continue
        if not torch.equal(re.edge_index, fe.edge_index.flip(0)):
            errs.append(f"entity '{ent}': reverse edges do not mirror forward edges")
        if not torch.equal(re.timestamp, fe.timestamp):
            errs.append(f"entity '{ent}': reverse timestamps differ")
        ts = fe.timestamp.numpy()
        if not np.array_equal(ts, t[src]):
            errs.append(f"entity '{ent}': edge timestamp != TransactionDT of its transaction")
        first = np.full(n_ent, np.iinfo(np.int64).max, dtype=np.int64)
        np.minimum.at(first, dst, ts)
        if not np.array_equal(first, data[ent].time.numpy()):
            errs.append(f"entity '{ent}': node time is not the first-seen time")

    y_obs, det = txn.y_obs.numpy(), txn.detection_time.numpy()
    late = det > data.label_cut
    if (y_obs[late] != 0).any():
        errs.append("y_obs == 1 for a transaction detected after the label cut")
    if (y_obs[~txn.in_train_view.numpy()] != 0).any():
        errs.append("y_obs == 1 outside the train view")
    if not np.array_equal(txn.pnu.numpy() == PNU_CODES["P"], y_obs == 1):
        errs.append("pnu==P is not equivalent to y_obs==1")
    if (txn.y_true.numpy()[y_obs == 1] != 1).any():
        errs.append("observed positive that is not a true fraud")
    try:
        assert_no_label_columns(data.feature_names)
    except ValueError as exc:
        errs.append(str(exc))

    if errs:
        raise GraphIntegrityError(f"{phase} graph violates {len(errs)} invariant(s): " + "; ".join(errs))


__all__ = [
    "TXN", "SPLIT_CODES", "PNU_CODES", "GraphIntegrityError", "EntitySpec", "EmailSpec",
    "FeatureSpec", "GraphConfig", "TxnFeatureEncoder", "make_entity_keys", "GraphBuilder",
    "validate_phase_graph", "EmailFeatureBlock", "select_columns_by_missing_ratio",
]