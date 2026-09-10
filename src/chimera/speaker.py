"""Speaker encoder: reference mel -> fixed voice embedding.

Conv downsampling + small Transformer encoder + attentive pooling.
This is what makes zero-shot cloning work: at inference you feed ANY
3-10s reference clip and get a voice vector that conditions the backbone.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpeakerEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg.spk_d_model
        self.conv = nn.Sequential(
            nn.Conv1d(cfg.n_mel, d, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(d, d, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(d, d, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=cfg.spk_heads, dim_feedforward=d * 4,
            batch_first=True, norm_first=True,
        )
        self.tr = nn.TransformerEncoder(layer, num_layers=cfg.spk_layers,
                                        enable_nested_tensor=False)
        self.pool = nn.Linear(d, 1)
        self.proj = nn.Linear(d, cfg.d_model)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """mel: [B, n_mel, T] -> voice vector [B, d_model], L2-normalized."""
        h = self.conv(mel).transpose(1, 2)  # [B, T/4, d]
        h = self.tr(h)
        w = self.pool(h).softmax(dim=1)     # attentive pooling
        v = (w * h).sum(dim=1)
        return F.normalize(self.proj(v), dim=-1)
