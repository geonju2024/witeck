"""Model definition for WITECK mobile Shared Dual Head v1.0.0."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedDualHead1DCNN(nn.Module):
    """One hand-only backbone with gesture and user embedding heads."""

    def __init__(
        self,
        input_dim: int = 127,
        num_user_classes: int = 7,
        embedding_dim: int = 128,
    ) -> None:
        super().__init__()
        if int(input_dim) != 127:
            raise ValueError(f"Expected hand-only D=127, got {input_dim}")

        def branch(dropout: float) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv1d(63, 48, kernel_size=5, padding=2, bias=False),
                nn.BatchNorm1d(48),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv1d(
                    48,
                    64,
                    kernel_size=3,
                    padding=2,
                    dilation=2,
                    bias=False,
                ),
                nn.BatchNorm1d(64),
                nn.ReLU(),
            )

        self.position_branch = branch(0.15)
        self.velocity_branch = branch(0.20)
        self.shared_projection = nn.Sequential(
            nn.Linear(64 * 4 + 1, 192),
            nn.LayerNorm(192),
            nn.ReLU(),
            nn.Dropout(0.20),
        )
        self.gesture_head = nn.Linear(192, embedding_dim)
        self.user_head = nn.Linear(192, embedding_dim)
        # Training-only auxiliary classifier. It is never used for enrollment.
        self.user_classifier = nn.Linear(embedding_dim, num_user_classes)

    @staticmethod
    def _statistics_pool(features: torch.Tensor) -> torch.Tensor:
        mean = features.mean(dim=2)
        variance = features.var(dim=2, unbiased=False)
        std = torch.sqrt(variance.clamp_min(1e-6))
        return torch.cat([mean, std], dim=1)

    def forward(
        self,
        x: torch.Tensor,
        duration: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.ndim != 3 or x.shape[1:] != (32, 127):
            raise ValueError(f"Expected [B,32,127], got {tuple(x.shape)}")

        position = x[:, :, :63].transpose(1, 2)
        velocity = x[:, :, 63:126].transpose(1, 2)
        position_stats = self._statistics_pool(self.position_branch(position))
        velocity_stats = self._statistics_pool(self.velocity_branch(velocity))
        fused = torch.cat(
            [position_stats, velocity_stats, duration.unsqueeze(1)],
            dim=1,
        )
        shared = self.shared_projection(fused)
        gesture_embedding = F.normalize(self.gesture_head(shared), p=2, dim=1)
        user_embedding = F.normalize(self.user_head(shared), p=2, dim=1)
        user_logits = self.user_classifier(user_embedding)
        return gesture_embedding, user_embedding, user_logits
