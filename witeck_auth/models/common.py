from __future__ import annotations

import math
import torch
from torch import Tensor, nn
import torch.nn.functional as F


def valid_group_count(channels: int, preferred: int = 8) -> int:
    """Return the largest useful GroupNorm group count dividing channels."""
    for groups in range(min(preferred, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class AttentiveStatisticsPooling(nn.Module):
    """Weighted temporal mean and standard deviation pooling."""

    def __init__(self, channels: int, hidden: int | None = None) -> None:
        super().__init__()
        hidden = hidden or max(32, channels // 2)
        self.attention = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(hidden, 1, kernel_size=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        weights = torch.softmax(self.attention(x), dim=-1)
        mean = torch.sum(weights * x, dim=-1)
        second = torch.sum(weights * x.square(), dim=-1)
        std = torch.sqrt(torch.clamp(second - mean.square(), min=1e-5))
        return torch.cat([mean, std], dim=1)


class SqueezeExcitation1D(nn.Module):
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(8, channels // reduction)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, hidden, 1),
            nn.GELU(),
            nn.Conv1d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x * self.net(x)


class SiameseModel(nn.Module):
    """Shared encoder and cosine verification head."""

    def __init__(self, encoder: nn.Module, temperature: float = 0.07) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.encoder = encoder
        self.log_temperature = nn.Parameter(torch.tensor(math.log(temperature)))

    @property
    def temperature(self) -> Tensor:
        return self.log_temperature.exp().clamp(0.02, 1.0)

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def forward(self, left: Tensor, right: Tensor) -> dict[str, Tensor]:
        left_embedding = self.encode(left)
        right_embedding = self.encode(right)
        similarity = F.cosine_similarity(left_embedding, right_embedding)
        logits = similarity / self.temperature
        return {
            "left_embedding": left_embedding,
            "right_embedding": right_embedding,
            "similarity": similarity,
            "logits": logits,
        }
