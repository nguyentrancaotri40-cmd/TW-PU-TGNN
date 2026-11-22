"""ModularTemporalGNN: spatial encoder (HeteroConv of SAGEConv or GATConv) + GRU temporal encoder.

Input: a mini-batch from ``src.graph.sampler.build_neighbor_loader`` (disjoint + temporal):
  * ``batch['transaction']``: ``x`` [N, F], ``time`` [N] int64 s, ``batch`` [N] subgraph id,
    ``batch_size``. The FIRST ``batch_size`` transaction nodes are the seeds.
  * entity nodes (``card`` / ``device``): ``node_id`` [K] (global id, identical in train/val/test).
  * edges ``(transaction, uses_<e>, <e>)`` and ``(<e>, rev_uses_<e>, transaction)``.
No label field (``y_obs``, ``y_true``, ``pnu``) is ever read here.

Output: seed embeddings ``[batch_size, hidden_dim]`` (the logit comes from ``FraudClassifier``).

Information flow with num_layers=2 and 2-hop sampling
  layer 1: entity <- history transactions (uses_*)   |  seed <- entity (rev_uses_*)
  layer 2: seed <- entity (now carrying history information)
A history transaction is reached only through an entity; its incoming edges would need a 3rd hop,
so its state stays equal to its input projection.

Backbones (``conv_type``)
  "sage": one ``SAGEConv`` per edge type (main backbone, the 72 benchmark runs).
  "gat" : one ``GATConv`` per edge type (secondary check). ``add_self_loops=False`` is required for
          bipartite edge types; ``hidden_dim`` must be divisible by ``gat_heads``.

Temporal encoder (DESIGN CHOICE of this project, not a reproduction of a paper; whether it helps is
KHÔNG CHẮC, CẦN VERIFY by the ablation ``use_temporal=False``)
  Each seed owns its own subgraph (disjoint batch). For every seed we take the other TRANSACTION nodes
  of its subgraph (its sampled history), sort them by time, keep the ``max_history`` most recent, and
  feed ``[projected features, log1p(days before the seed)]`` to a GRU: oldest ... most recent, then the
  seed itself. The last GRU state is fused with the seed's spatial embedding. The "sequence" is the set
  of transactions linked to the seed through shared entities, ordered by time; it is NOT a per-entity
  sequence.

Entity ids: ``nn.Embedding`` per entity type, sized by ``vocab_size_all_phases`` in meta.json. Ids that
never occur in the train period are never updated and keep a near-zero vector (std=0.01), which acts as
"unknown entity" at validation / test time.
"""
from __future__ import annotations

from typing import Literal, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATConv, HeteroConv, SAGEConv

from src.data.loader import SECONDS_PER_DAY
from src.graph.builder import TXN

ConvType = Literal["sage", "gat"]


class ModularTemporalGNN(nn.Module):
    """Spatial HeteroConv encoder + optional GRU temporal encoder.

    Args:
        in_dim: transaction feature dimension (meta['feature_dim'], 457 here).
        entity_vocab_sizes: ``{entity_type: number_of_ids}`` over ALL phases.
        hidden_dim: width of every node state and of the output embedding.
        num_layers: HeteroConv layers (2 reaches the history transactions).
        dropout: dropout in the encoder (and GAT attention dropout).
        entity_embedding_dim: width of the ID embeddings (projected to ``hidden_dim``).
        conv_type: "sage" or "gat".
        gat_heads: attention heads for "gat".
        max_history: most recent history transactions fed to the GRU per seed.
        gru_layers: stacked GRU layers.
        use_temporal: False -> spatial encoder only (SAGEConv/GATConv + MLP ablation).
    """

    def __init__(
        self,
        in_dim: int,
        entity_vocab_sizes: Mapping[str, int],
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        entity_embedding_dim: int = 32,
        conv_type: ConvType = "sage",
        gat_heads: int = 4,
        max_history: int = 20,
        gru_layers: int = 1,
        use_temporal: bool = True,
    ) -> None:
        super().__init__()
        if conv_type not in ("sage", "gat"):
            raise ValueError("conv_type must be 'sage' or 'gat'.")
        if conv_type == "gat" and hidden_dim % gat_heads != 0:
            raise ValueError("hidden_dim must be divisible by gat_heads for GAT.")
        if num_layers < 1 or max_history < 1:
            raise ValueError("num_layers and max_history must be >= 1.")
        if not entity_vocab_sizes:
            raise ValueError("At least one entity type is required.")
        self.conv_type = conv_type
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_history = max_history
        self.use_temporal = use_temporal
        self.entity_names: list[str] = list(entity_vocab_sizes)
        self.node_types: list[str] = [TXN, *self.entity_names]
        # edge-type names must match src/graph/builder.py
        self.edge_types: list[tuple[str, str, str]] = [
            et for e in self.entity_names for et in ((TXN, f"uses_{e}", e), (e, f"rev_uses_{e}", TXN))
        ]

        # --- input projections -------------------------------------------------------------
        self.txn_in = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.ent_emb = nn.ModuleDict(
            {e: nn.Embedding(max(int(n), 1), entity_embedding_dim) for e, n in entity_vocab_sizes.items()}
        )
        for emb in self.ent_emb.values():
            nn.init.normal_(emb.weight, std=0.01)  # never-trained ids stay ~0 ("unknown entity")
        self.ent_proj = nn.ModuleDict({e: nn.Linear(entity_embedding_dim, hidden_dim) for e in self.entity_names})

        # --- spatial encoder ---------------------------------------------------------------
        self.convs = nn.ModuleList(
            [
                HeteroConv(
                    {et: self._make_conv(conv_type, hidden_dim, gat_heads, dropout) for et in self.edge_types},
                    aggr="sum",
                )
                for _ in range(num_layers)
            ]
        )
        self.norms = nn.ModuleList(
            [nn.ModuleDict({nt: nn.LayerNorm(hidden_dim) for nt in self.node_types}) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)

        # --- temporal encoder --------------------------------------------------------------
        if use_temporal:
            self.gru = nn.GRU(
                input_size=hidden_dim + 1,  # +1: log1p(days between the history transaction and its seed)
                hidden_size=hidden_dim,
                num_layers=gru_layers,
                batch_first=True,
                dropout=dropout if gru_layers > 1 else 0.0,
            )
            self.fuse = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU())
        self.out_dim = hidden_dim

    @staticmethod
    def _make_conv(conv_type: str, hidden: int, heads: int, dropout: float) -> nn.Module:
        if conv_type == "sage":
            return SAGEConv((hidden, hidden), hidden)
        # bipartite GAT: self-loops would mix source and destination index spaces -> disabled
        return GATConv((hidden, hidden), hidden // heads, heads=heads, concat=True,
                       dropout=dropout, add_self_loops=False)

    # ----------------------------------------------------------------------------------------
    def forward(self, batch: HeteroData) -> Tensor:
        """Return seed embeddings ``[batch_size, out_dim]``."""
        store = batch[TXN]
        bs = int(store.batch_size)

        h: dict[str, Tensor] = {TXN: self.txn_in(store.x)}
        for e in self.entity_names:
            if e in batch.node_types:  # a type can be absent if it had no edges in this phase
                h[e] = self.ent_proj[e](self.ent_emb[e](batch[e].node_id))
        h_txn0 = h[TXN]  # input projection of every sampled transaction (history + seeds)

        present = set(batch.edge_types)
        edge_index_dict = {et: batch[et].edge_index for et in self.edge_types if et in present}
        for conv, norms in zip(self.convs, self.norms):
            out = conv(h, edge_index_dict)  # only destination types that received a message
            new_h: dict[str, Tensor] = {}
            for nt, x in h.items():
                if nt in out:
                    x = norms[nt](x + self.dropout(F.relu(out[nt])))
                new_h[nt] = x
            h = new_h

        spatial = h[TXN][:bs]  # seeds are the first ``batch_size`` transaction nodes
        if not self.use_temporal:
            return spatial
        temporal = self._temporal_encode(store, h_txn0, bs)
        return self.fuse(torch.cat([spatial, temporal], dim=-1))

    # ----------------------------------------------------------------------------------------
    def _temporal_encode(self, store, h0: Tensor, bs: int) -> Tensor:
        """GRU over [oldest history ... most recent history, seed] for every seed."""
        if "batch" not in store:
            raise ValueError("Temporal encoder needs disjoint batches (build_neighbor_loader provides them).")
        device = h0.device
        n = h0.size(0)
        sub = store.batch  # [n] subgraph id of each transaction node
        t = store.time  # [n] int64 seconds

        # row index of the seed that owns each node (does not assume subgraph id == seed position)
        row_of_sub = torch.zeros(bs, dtype=torch.long, device=device)
        row_of_sub[sub[:bs]] = torch.arange(bs, device=device)
        row = row_of_sub[sub]  # [n]; row[:bs] == arange(bs)

        seed_time = torch.zeros(bs, dtype=t.dtype, device=device)
        seed_time[row[:bs]] = t[:bs]
        dt_days = (seed_time[row] - t).clamp(min=0).to(h0.dtype) / float(SECONDS_PER_DAY)  # 0 for seeds
        feat = torch.cat([h0, torch.log1p(dt_days).unsqueeze(-1)], dim=-1)  # [n, hidden + 1]

        k = self.max_history
        seq = feat.new_zeros(bs, k + 1, feat.size(-1))
        kept = torch.zeros(bs, dtype=torch.long, device=device)  # history length kept per seed

        if n > bs:
            hist = torch.arange(bs, n, device=device)  # non-seed transaction nodes = history
            order = hist[torch.argsort(t[hist], stable=True)]  # by time ...
            order = order[torch.argsort(row[order], stable=True)]  # ... then grouped by seed row
            r = row[order]
            counts = torch.bincount(r, minlength=bs)  # history size per seed
            start = torch.cumsum(counts, dim=0) - counts  # first index of each group
            pos = torch.arange(order.numel(), device=device) - start[r]  # 0 = oldest of the group
            from_end = counts[r] - 1 - pos  # 0 = most recent of the group
            keep = from_end < k  # keep the k most recent
            order, r, from_end = order[keep], r[keep], from_end[keep]
            kept = counts.clamp(max=k)
            col = kept[r] - 1 - from_end  # oldest kept -> column 0, most recent -> column kept-1
            seq[r, col] = feat[order]

        seq[torch.arange(bs, device=device), kept] = feat[:bs]  # the seed closes its own sequence
        lengths = (kept + 1).cpu()  # >= 1
        packed = pack_padded_sequence(seq, lengths, batch_first=True, enforce_sorted=False)
        _, h_n = self.gru(packed)  # [gru_layers, bs, hidden], returned in the original seed order
        return h_n[-1]