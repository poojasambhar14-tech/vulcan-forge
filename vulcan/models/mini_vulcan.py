"""
Mini-Vulcan backbone: our own research-scale payment event Transformer.

NOT a reproduction of Razorpay Vulcan's (undisclosed) architecture. Design
principles are borrowed broadly from PRAGMA / TransactionGPT / TabFormer
(see docs/research_basis.md) rather than copying any single system:

    FIELD TOKENIZER
          |
    EVENT ENCODER            (per-field categorical embeddings + continuous
          |                    projection, combined per event)
    EVENT REPRESENTATION
          |
    HISTORY TRANSFORMER       (bidirectional encoder over a window of K
          |                    events, for masked-event-modeling pretraining)
    SHARED PAYMENT STATE z_t
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn as nn

from vulcan.tokenization.tokenizer import FieldTokenizer


@dataclass
class MiniVulcanConfig:
    d_model: int = 96
    n_heads: int = 4
    n_layers: int = 3
    ffn_mult: int = 4
    dropout: float = 0.1
    history_len: int = 32

    @staticmethod
    def from_dict(d: dict) -> "MiniVulcanConfig":
        return MiniVulcanConfig(
            d_model=d["d_model"],
            n_heads=d["n_heads"],
            n_layers=d["n_layers"],
            ffn_mult=d.get("ffn_mult", 4),
            dropout=d.get("dropout", 0.1),
            history_len=d["history_len"],
        )


class EventEncoder(nn.Module):
    """Embeds one payment event (categorical fields + continuous vector)
    into a single d_model-dimensional representation."""

    def __init__(self, tokenizer: FieldTokenizer, d_model: int):
        super().__init__()
        self.field_names: List[str] = list(tokenizer.categorical_fields.keys())
        vocab_sizes = tokenizer.vocab_sizes()

        # small per-field embedding dim, concatenated then projected
        field_dim = max(8, d_model // max(1, len(self.field_names)))
        self.field_dim = field_dim
        self.embeddings = nn.ModuleDict(
            {name: nn.Embedding(vocab_sizes[name], field_dim) for name in self.field_names}
        )
        self.categorical_proj = nn.Linear(field_dim * len(self.field_names), d_model)
        # continuous input carries [value, is_masked_flag] per continuous field
        self.continuous_proj = nn.Linear(tokenizer.n_continuous * 2, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, categorical_ids: Dict[str, torch.Tensor], continuous: torch.Tensor) -> torch.Tensor:
        """categorical_ids: dict[name] -> LongTensor [B, K]
        continuous: FloatTensor [B, K, 2 * n_continuous] (value, is_masked flag interleaved as blocks)
        returns: FloatTensor [B, K, d_model]
        """
        embs = [self.embeddings[name](categorical_ids[name]) for name in self.field_names]
        cat_emb = torch.cat(embs, dim=-1)  # [B, K, field_dim * n_fields]
        cat_proj = self.categorical_proj(cat_emb)
        cont_proj = self.continuous_proj(continuous)
        event_repr = self.norm(cat_proj + cont_proj)
        return self.dropout(event_repr)


class HistoryTransformer(nn.Module):
    """Bidirectional Transformer encoder over a window of event
    representations, used for masked-event-modeling pretraining."""

    def __init__(self, cfg: MiniVulcanConfig):
        super().__init__()
        self.pos_emb = nn.Embedding(cfg.history_len, cfg.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_model * cfg.ffn_mult,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)
        self.d_model = cfg.d_model

    def forward(self, event_reprs: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """event_reprs: [B, K, d_model] -> returns contextualized [B, K, d_model]"""
        B, K, _ = event_reprs.shape
        positions = torch.arange(K, device=event_reprs.device).unsqueeze(0).expand(B, K)
        x = event_reprs + self.pos_emb(positions)
        return self.encoder(x, src_key_padding_mask=key_padding_mask)


class MiniVulcanBackbone(nn.Module):
    """Full backbone producing the shared payment state z_t (per-position
    contextualized representations; the "current" state is the last
    position of a window)."""

    def __init__(self, tokenizer: FieldTokenizer, cfg: MiniVulcanConfig):
        super().__init__()
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.event_encoder = EventEncoder(tokenizer, cfg.d_model)
        self.history_transformer = HistoryTransformer(cfg)

    def forward(self, categorical_ids: Dict[str, torch.Tensor], continuous: torch.Tensor) -> torch.Tensor:
        event_reprs = self.event_encoder(categorical_ids, continuous)
        z = self.history_transformer(event_reprs)
        return z  # [B, K, d_model]

    def current_state(self, categorical_ids: Dict[str, torch.Tensor], continuous: torch.Tensor) -> torch.Tensor:
        """Shared payment state z_t at the last position of the window."""
        z = self.forward(categorical_ids, continuous)
        return z[:, -1, :]

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
