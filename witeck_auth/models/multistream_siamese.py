from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .common import AttentiveStatisticsPooling, SiameseModel, valid_group_count


WITECK_INPUT_DIM = 169
FEATURE_STREAMS = (
    ("hand_position", 0, 63),
    ("hand_velocity", 63, 126),
    ("pose_position", 126, 147),
    ("pose_velocity", 147, 168),
)
VALID_MASK_INDEX = 168


class MultiKernelDilatedBlock(nn.Module):
    """Residual temporal block with compact multi-scale dilated kernels."""

    def __init__(
        self,
        channels: int,
        dilation: int,
        kernels: tuple[int, ...] = (3, 5),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if not kernels or any(kernel <= 0 or kernel % 2 == 0 for kernel in kernels):
            raise ValueError("kernels must contain odd positive values")
        self.branches = nn.ModuleList(
            nn.Conv1d(
                channels,
                channels,
                kernel_size=kernel,
                padding=dilation * (kernel // 2),
                dilation=dilation,
                groups=channels,
                bias=False,
            )
            for kernel in kernels
        )
        self.mix = nn.Conv1d(channels * len(kernels), channels, 1, bias=False)
        self.norm = nn.GroupNorm(valid_group_count(channels), channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        multi_scale = torch.cat([branch(x) for branch in self.branches], dim=1)
        update = self.dropout(F.gelu(self.norm(self.mix(multi_scale))))
        return F.gelu(x + update)


class SemanticStream(nn.Module):
    """Encode one homogeneous WITECK feature group along the time axis."""

    def __init__(
        self,
        input_dim: int,
        channels: int,
        dilations: tuple[int, ...],
        kernels: tuple[int, ...],
        dropout: float,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(input_dim, channels, 1, bias=False),
            nn.GroupNorm(valid_group_count(channels), channels),
            nn.GELU(),
        )
        self.temporal = nn.Sequential(
            *[
                MultiKernelDilatedBlock(
                    channels,
                    dilation=dilation,
                    kernels=kernels,
                    dropout=dropout,
                )
                for dilation in dilations
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.temporal(self.stem(x.transpose(1, 2)))


class MultiStreamDilatedEncoder(nn.Module):
    """Feature-aware encoder for the WITECK [B, 32, 169] representation."""

    def __init__(
        self,
        input_dim: int = WITECK_INPUT_DIM,
        embedding_dim: int = 128,
        stream_channels: int = 48,
        fusion_channels: int = 128,
        dilations: tuple[int, ...] = (1, 2, 4),
        kernels: tuple[int, ...] = (3, 5),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if input_dim != WITECK_INPUT_DIM:
            raise ValueError(
                f"MultiStreamDilatedEncoder requires the WITECK 169-feature schema, "
                f"received input_dim={input_dim}"
            )
        self.streams = nn.ModuleDict(
            {
                name: SemanticStream(
                    end - start,
                    channels=stream_channels,
                    dilations=dilations,
                    kernels=kernels,
                    dropout=dropout,
                )
                for name, start, end in FEATURE_STREAMS
            }
        )
        fused_input = stream_channels * len(FEATURE_STREAMS) + 1
        self.fusion = nn.Sequential(
            nn.Conv1d(fused_input, fusion_channels, 1, bias=False),
            nn.GroupNorm(valid_group_count(fusion_channels), fusion_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pool = AttentiveStatisticsPooling(fusion_channels)
        self.head = nn.Sequential(
            nn.Linear(fusion_channels * 2, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3 or x.shape[-1] != WITECK_INPUT_DIM:
            raise ValueError(
                f"expected [B,T,{WITECK_INPUT_DIM}], received {tuple(x.shape)}"
            )
        encoded = [
            self.streams[name](x[:, :, start:end])
            for name, start, end in FEATURE_STREAMS
        ]
        valid_mask = x[:, :, VALID_MASK_INDEX : VALID_MASK_INDEX + 1].transpose(1, 2)
        fused = self.fusion(torch.cat([*encoded, valid_mask], dim=1))
        return F.normalize(self.head(self.pool(fused)), dim=1)


class MultiStreamDilatedSiamese(SiameseModel):
    def __init__(self, input_dim: int = 169, embedding_dim: int = 128, **kwargs) -> None:
        encoder = MultiStreamDilatedEncoder(
            input_dim=input_dim,
            embedding_dim=embedding_dim,
            **kwargs,
        )
        super().__init__(encoder)
