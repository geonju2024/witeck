"""Lightweight Siamese 1D-CNN for WITECK user authentication."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LiteStatsSiamese1DCNN(nn.Module):
    """Encode a 32-frame WITECK sequence into a cosine-ready embedding.

    WITECK stores 168 motion features followed by one valid-mask channel.
    The mask is excluded because it represents capture quality rather than
    identity. Three local convolutions retain all 32 frames, while mean/std
    pooling summarizes the performer's typical motion and variability.
    Normalized clip duration restores absolute tempo information that temporal
    resampling can weaken.
    """

    def __init__(
        self,
        input_dim: int = 169,
        embedding_dim: int = 64,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()

        if input_dim < 2:
            raise ValueError("input_dim must include features and valid mask")

        self.feature_dim = input_dim - 1
        self.features = nn.Sequential(
            nn.Conv1d(
                self.feature_dim,
                32,
                kernel_size=5,
                padding=2,
                bias=False,
            ),
            nn.GroupNorm(4, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(
                32,
                48,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(4, 48),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(
                48,
                64,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(8, 64),
            nn.GELU(),
        )
        self.embedding_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(64 * 2 + 1, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        duration: torch.Tensor,
    ) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [B,T,D], got shape={tuple(x.shape)}")
        if x.shape[-1] != self.feature_dim + 1:
            raise ValueError(
                f"Expected D={self.feature_dim + 1}, got D={x.shape[-1]}"
            )

        # [hand xyz, hand velocity, pose xyz, pose velocity]; ignore mask.
        h = self.features(x[:, :, : self.feature_dim].transpose(1, 2))
        mean = h.mean(dim=2)
        variance = h.var(dim=2, correction=0)
        std = torch.sqrt(variance.clamp_min(1e-6))
        pooled = torch.cat([mean, std, duration.unsqueeze(1)], dim=1)
        embedding = self.embedding_head(pooled)
        return F.normalize(embedding, p=2, dim=1)

