from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

DEFAULT_GENERATION_CANDIDATES = 20


@dataclass(frozen=True)
class ModelConfig:
    src_vocab_size: int
    tgt_vocab_size: int
    d_model: int = 256
    n_heads: int = 8
    num_encoder_layers: int = 3
    num_decoder_layers: int = 3
    dim_feedforward: int = 1024
    dropout: float = 0.1
    max_len: int = 512


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float, max_len: int = 512) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class RouteTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.src_emb = nn.Embedding(cfg.src_vocab_size, cfg.d_model, padding_idx=0)
        self.tgt_emb = nn.Embedding(cfg.tgt_vocab_size, cfg.d_model, padding_idx=0)
        self.src_pos = PositionalEncoding(cfg.d_model, cfg.dropout, cfg.max_len)
        self.tgt_pos = PositionalEncoding(cfg.d_model, cfg.dropout, cfg.max_len)
        self.transformer = nn.Transformer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            num_encoder_layers=cfg.num_encoder_layers,
            num_decoder_layers=cfg.num_decoder_layers,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.generator = nn.Linear(cfg.d_model, cfg.tgt_vocab_size)

    def _causal_mask(self, length: int, device: torch.device) -> torch.Tensor:
        return torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=device),
            diagonal=1,
        )

    def encode(self, src_ids: torch.Tensor, src_pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        src = self.src_pos(self.src_emb(src_ids) * math.sqrt(self.cfg.d_model))
        return self.transformer.encoder(src, src_key_padding_mask=src_pad_mask)

    def decode(
        self,
        tgt_in_ids: torch.Tensor,
        memory: torch.Tensor,
        *,
        tgt_pad_mask: Optional[torch.Tensor] = None,
        memory_pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        tgt = self.tgt_pos(self.tgt_emb(tgt_in_ids) * math.sqrt(self.cfg.d_model))
        tgt_mask = self._causal_mask(tgt_in_ids.size(1), tgt_in_ids.device)
        hidden = self.transformer.decoder(
            tgt,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=memory_pad_mask,
        )
        return self.generator(hidden)

    def forward(
        self,
        src_ids: torch.Tensor,
        tgt_in_ids: torch.Tensor,
        *,
        src_pad_mask: Optional[torch.Tensor] = None,
        tgt_pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        memory = self.encode(src_ids, src_pad_mask=src_pad_mask)
        return self.decode(
            tgt_in_ids,
            memory,
            tgt_pad_mask=tgt_pad_mask,
            memory_pad_mask=src_pad_mask,
        )
