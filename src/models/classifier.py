"""FraudClassifier: backbone embeddings -> MLP head -> one LOGIT per seed transaction."""
from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import HeteroData

from src.models.temporal_gnn import ModularTemporalGNN


def entity_vocab_sizes_from_meta(meta: Mapping[str, Any]) -> dict[str, int]:
    """Embedding-table sizes from ``meta.json`` (``vocab_size_all_phases`` covers val/test ids)."""
    return {
        name: int(st["vocab_size_all_phases"])
        for name, st in meta["entity_stats"].items()
        if int(st["vocab_size_all_phases"]) > 0
    }


class FraudClassifier(nn.Module):
    """``backbone -> Linear -> ReLU -> Dropout -> Linear`` returning logits ``[batch_size]``.

    No sigmoid inside: every loss in ``src/losses`` expects logits. Use ``predict_proba`` for
    probabilities (call ``model.eval()`` yourself).
    """

    def __init__(self, backbone: ModularTemporalGNN, head_hidden_dim: int | None = None, dropout: float = 0.2) -> None:
        super().__init__()
        self.backbone = backbone
        hidden = head_hidden_dim or backbone.out_dim
        self.head = nn.Sequential(
            nn.Linear(backbone.out_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, batch: HeteroData) -> Tensor:
        return self.head(self.backbone(batch)).squeeze(-1)

    @torch.no_grad()
    def predict_proba(self, batch: HeteroData) -> Tensor:
        return torch.sigmoid(self.forward(batch))

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @classmethod
    def from_config(
        cls,
        model_cfg: Mapping[str, Any],
        in_dim: int,
        entity_vocab_sizes: Mapping[str, int],
        conv_type: str | None = None,
    ) -> "FraudClassifier":
        """Build from the ``model:`` section of configs/base.yaml.

        ``conv_type`` (or ``model.backbone``): "sage" (main, default) or "gat" (secondary).
        Optional keys: max_history (20), gru_layers (1), use_temporal (True), gat_heads (4),
        head_hidden_dim (= hidden_dim).
        """
        backbone = ModularTemporalGNN(
            in_dim=in_dim,
            entity_vocab_sizes=entity_vocab_sizes,
            hidden_dim=int(model_cfg.get("hidden_dim", 128)),
            num_layers=int(model_cfg.get("num_layers", 2)),
            dropout=float(model_cfg.get("dropout", 0.2)),
            entity_embedding_dim=int(model_cfg.get("entity_embedding_dim", 32)),
            conv_type=conv_type or str(model_cfg.get("backbone", "sage")),
            gat_heads=int(model_cfg.get("gat_heads", 4)),
            max_history=int(model_cfg.get("max_history", 20)),
            gru_layers=int(model_cfg.get("gru_layers", 1)),
            use_temporal=bool(model_cfg.get("use_temporal", True)),
        )
        head_hidden = model_cfg.get("head_hidden_dim")
        return cls(
            backbone,
            head_hidden_dim=int(head_hidden) if head_hidden is not None else None,
            dropout=float(model_cfg.get("dropout", 0.2)),
        )


def build_classifier(
    model_cfg: Mapping[str, Any], meta: Mapping[str, Any], conv_type: str | None = None, extra_in_dim: int = 0
) -> FraudClassifier:
    """Convenience: ``in_dim`` and entity tables come from ``meta.json`` (written by prepare_data)."""
    return FraudClassifier.from_config(
        model_cfg,
        in_dim=int(meta["feature_dim"]) + int(extra_in_dim),  # +4 with LP features (ds.model_in_dim - ds.feature_dim)
        entity_vocab_sizes=entity_vocab_sizes_from_meta(meta),
        conv_type=conv_type,
    )