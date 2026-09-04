from __future__ import annotations

from torch import Tensor, nn
import torch.nn.functional as F

from .common import AttentiveStatisticsPooling, SiameseModel, valid_group_count


class TemporalResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float = 0.2) -> None:
        super().__init__()
        padding = dilation
        self.net = nn.Sequential(
            nn.Conv1d(
                channels, channels, kernel_size=3, padding=padding,
                dilation=dilation, groups=channels, bias=False,
            ),
            nn.Conv1d(channels, channels, kernel_size=1, bias=False),
            nn.GroupNorm(valid_group_count(channels), channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(
                channels, channels, kernel_size=3, padding=padding,
                dilation=dilation, groups=channels, bias=False,
            ),
            nn.Conv1d(channels, channels, kernel_size=1, bias=False),
            nn.GroupNorm(valid_group_count(channels), channels),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return F.gelu(x + self.net(x))


class TCNEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 169,
        embedding_dim: int = 128,
        channels: int = 96,
        dilations: tuple[int, ...] = (1, 2, 4, 8),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(input_dim, channels, 1, bias=False),
            nn.GroupNorm(valid_group_count(channels), channels),
            nn.GELU(),
        )
        self.tcn = nn.Sequential(
            *[TemporalResidualBlock(channels, d, dropout) for d in dilations]
        )
        self.pool = AttentiveStatisticsPooling(channels)
        self.head = nn.Sequential(
            nn.Linear(channels * 2, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError(f"expected [B,T,D], received {tuple(x.shape)}")
        x = self.tcn(self.stem(x.transpose(1, 2)))
        return F.normalize(self.head(self.pool(x)), dim=1)


class TCNSiamese(SiameseModel):
    def __init__(self, input_dim: int = 169, embedding_dim: int = 128, **kwargs) -> None:
        encoder = TCNEncoder(input_dim=input_dim, embedding_dim=embedding_dim, **kwargs)
        super().__init__(encoder)
