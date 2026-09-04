from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .common import AttentiveStatisticsPooling, SqueezeExcitation1D, valid_group_count


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, 5, padding=2 * dilation, dilation=dilation, bias=False),
            nn.GroupNorm(valid_group_count(channels), channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.GroupNorm(valid_group_count(channels), channels),
        )
        self.se = SqueezeExcitation1D(channels)

    def forward(self, x: Tensor) -> Tensor:
        return F.gelu(x + self.se(self.net(x)))


class Prototypical1DCNN(nn.Module):
    """Episodic 1D CNN with cosine prototypes for user enrollment."""

    def __init__(
        self,
        input_dim: int = 169,
        embedding_dim: int = 128,
        channels: int = 96,
        dropout: float = 0.2,
        temperature: float = 0.07,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.stem = nn.Sequential(
            nn.Conv1d(input_dim, channels, 1, bias=False),
            nn.GroupNorm(valid_group_count(channels), channels),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            ResidualConvBlock(channels, 1, dropout),
            ResidualConvBlock(channels, 2, dropout),
            ResidualConvBlock(channels, 4, dropout),
        )
        self.pool = AttentiveStatisticsPooling(channels)
        self.head = nn.Sequential(
            nn.Linear(channels * 2, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

    def encode(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError(f"expected [B,T,D], received {tuple(x.shape)}")
        x = self.blocks(self.stem(x.transpose(1, 2)))
        return F.normalize(self.head(self.pool(x)), dim=1)

    def compute_prototypes(self, support_x: Tensor, support_y: Tensor) -> tuple[Tensor, Tensor]:
        embeddings = self.encode(support_x)
        classes = torch.unique(support_y, sorted=True)
        prototypes = torch.stack(
            [embeddings[support_y == label].mean(dim=0) for label in classes]
        )
        return F.normalize(prototypes, dim=1), classes

    def forward(
        self, support_x: Tensor, support_y: Tensor, query_x: Tensor
    ) -> dict[str, Tensor]:
        prototypes, classes = self.compute_prototypes(support_x, support_y)
        query_embedding = self.encode(query_x)
        logits = query_embedding @ prototypes.T / self.temperature
        return {
            "logits": logits,
            "query_embedding": query_embedding,
            "prototypes": prototypes,
            "classes": classes,
        }

    @torch.no_grad()
    def authenticate(self, query_x: Tensor, owner_prototype: Tensor) -> Tensor:
        owner_prototype = F.normalize(owner_prototype.reshape(1, -1), dim=1)
        return self.encode(query_x) @ owner_prototype.T
