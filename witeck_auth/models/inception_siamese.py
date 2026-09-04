from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .common import (
    AttentiveStatisticsPooling,
    SiameseModel,
    SqueezeExcitation1D,
    valid_group_count,
)


class InceptionBlock1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        branch_channels: int = 32,
        kernels: tuple[int, ...] = (3, 5, 9),
        bottleneck_channels: int = 32,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.bottleneck = nn.Conv1d(in_channels, bottleneck_channels, 1, bias=False)
        self.branches = nn.ModuleList(
            nn.Conv1d(
                bottleneck_channels,
                branch_channels,
                kernel_size=kernel,
                padding=kernel // 2,
                bias=False,
            )
            for kernel in kernels
        )
        self.pool_branch = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, branch_channels, 1, bias=False),
        )
        out_channels = branch_channels * (len(kernels) + 1)
        self.norm = nn.GroupNorm(valid_group_count(out_channels), out_channels)
        self.project = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv1d(in_channels, out_channels, 1, bias=False)
        )
        self.se = SqueezeExcitation1D(out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        reduced = self.bottleneck(x)
        merged = torch.cat(
            [*[branch(reduced) for branch in self.branches], self.pool_branch(x)], dim=1
        )
        merged = self.dropout(F.gelu(self.norm(merged)))
        return self.se(merged + self.project(x))


class InceptionTimeEncoder(nn.Module):
    """Compact InceptionTime encoder for [B, T, D] WITECK sequences."""

    def __init__(
        self,
        input_dim: int = 169,
        embedding_dim: int = 128,
        branch_channels: int = 32,
        blocks: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        channels = branch_channels * 4
        self.stem = nn.Sequential(
            nn.Conv1d(input_dim, channels, 1, bias=False),
            nn.GroupNorm(valid_group_count(channels), channels),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            *[
                InceptionBlock1D(
                    channels,
                    branch_channels=branch_channels,
                    dropout=dropout,
                )
                for _ in range(blocks)
            ]
        )
        self.pool = AttentiveStatisticsPooling(channels)
        self.head = nn.Sequential(
            nn.Linear(channels * 2, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError(f"expected [B,T,D], received {tuple(x.shape)}")
        x = x.transpose(1, 2)
        x = self.blocks(self.stem(x))
        return F.normalize(self.head(self.pool(x)), dim=1)


class InceptionTimeSiamese(SiameseModel):
    def __init__(self, input_dim: int = 169, embedding_dim: int = 128, **kwargs) -> None:
        encoder = InceptionTimeEncoder(
            input_dim=input_dim, embedding_dim=embedding_dim, **kwargs
        )
        super().__init__(encoder)
