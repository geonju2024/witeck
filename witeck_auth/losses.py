from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class SiameseVerificationLoss(nn.Module):
    def forward(self, output: dict[str, Tensor], targets: Tensor) -> Tensor:
        return F.binary_cross_entropy_with_logits(output["logits"], targets.float())


class BatchHardTripletLoss(nn.Module):
    """Cosine-distance batch-hard triplet loss for batches with user labels."""

    def __init__(self, margin: float = 0.2) -> None:
        super().__init__()
        self.margin = margin

    def forward(self, embeddings: Tensor, labels: Tensor) -> Tensor:
        distance = 1.0 - embeddings @ embeddings.T
        same = labels[:, None].eq(labels[None, :])
        same.fill_diagonal_(False)
        different = ~labels[:, None].eq(labels[None, :])
        valid = same.any(dim=1) & different.any(dim=1)
        if not valid.any():
            return embeddings.sum() * 0.0
        hardest_positive = distance.masked_fill(~same, float("-inf")).max(dim=1).values
        hardest_negative = distance.masked_fill(~different, float("inf")).min(dim=1).values
        return F.relu(hardest_positive[valid] - hardest_negative[valid] + self.margin).mean()


def remap_episode_targets(query_y: Tensor, classes: Tensor) -> Tensor:
    matches = query_y[:, None].eq(classes[None, :])
    if not matches.any(dim=1).all():
        raise ValueError("every query class must be present in the support set")
    return matches.float().argmax(dim=1)
