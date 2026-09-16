"""Hand-only SupCon 1D-CNN tailored to the D=127 feature layout.

Unlike the generic D=169 encoder, this model treats hand position and
real-time hand velocity as two distinct signals.  Each compact temporal branch
uses mean+standard-deviation pooling, preserving both average hand formation
and the variability of a person's movement.  The detection-mask channel is
not used as an identity feature.  Duration is fused only after temporal
pooling.  Session/date metadata remains split-only information.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class HandOnlySupCon1DCNN(nn.Module):
    """Two-stream hand position/velocity encoder with statistics pooling."""

    def __init__(self, input_dim, num_classes, embedding_dim=128):
        super().__init__()
        if int(input_dim) != 127:
            raise ValueError(f"Hand-only model expects D=127, got {input_dim}")

        def branch(dropout):
            return nn.Sequential(
                nn.Conv1d(63, 48, kernel_size=5, padding=2, bias=False),
                nn.BatchNorm1d(48),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv1d(48, 64, kernel_size=3, padding=2, dilation=2, bias=False),
                nn.BatchNorm1d(64),
                nn.ReLU(),
            )

        self.position_branch = branch(0.15)
        self.velocity_branch = branch(0.20)
        # mean+std for each 64-channel branch, plus absolute duration.
        self.embedding_head = nn.Sequential(
            nn.Linear(64 * 4 + 1, 192),
            nn.LayerNorm(192),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(192, embedding_dim),
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)

    @staticmethod
    def statistics_pool(features):
        mean = features.mean(dim=2)
        variance = features.var(dim=2, unbiased=False)
        std = torch.sqrt(variance.clamp_min(1e-6))
        return torch.cat([mean, std], dim=1)

    def forward(self, x, duration):
        if x.ndim != 3 or x.shape[-1] != 127:
            raise ValueError(f"Expected [B,32,127], got {tuple(x.shape)}")
        position = x[:, :, :63].transpose(1, 2)
        velocity = x[:, :, 63:126].transpose(1, 2)
        position_stats = self.statistics_pool(self.position_branch(position))
        velocity_stats = self.statistics_pool(self.velocity_branch(velocity))
        fused = torch.cat(
            [position_stats, velocity_stats, duration.unsqueeze(1)], dim=1
        )
        embedding = F.normalize(self.embedding_head(fused), p=2, dim=1)
        return embedding, self.classifier(embedding)


def build_model(
    input_dim,
    num_classes,
    embedding_dim,
    sequence_mean=None,
    sequence_std=None,
):
    del sequence_mean, sequence_std
    return HandOnlySupCon1DCNN(input_dim, num_classes, embedding_dim)


def argument_value(name: str, default: str) -> str:
    if name in sys.argv:
        position = sys.argv.index(name)
        if position + 1 < len(sys.argv):
            return sys.argv[position + 1]
    return default


def main() -> None:
    root = Path(__file__).resolve().parent
    default_data = str(
        root / "dataset" / "dataset_1955_recent8_updated_20260905_hand_only.npz"
    )
    default_output = str(root / "output" / "hand_only_tailored_supcon")
    if "--data" not in sys.argv:
        sys.argv.extend(["--data", default_data])
    if "--output-dir" not in sys.argv:
        sys.argv.extend(["--output-dir", default_output])

    supcon = importlib.import_module("16_train_supcon_embedding")
    supcon.build_model = build_model
    supcon.main()

    output_dir = Path(argument_value("--output-dir", default_output))
    checkpoint_path = output_dir / "embedding_1dcnn_supcon.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    parameter_count = sum(
        value.numel() for value in checkpoint["model_state_dict"].values()
    )
    checkpoint.update(
        {
            "architecture": "hand-only-two-stream-stats-supcon",
            "feature_layout": "hand_xyz_63+hand_velocity_63+valid_mask_1",
            "streams": ["hand_position", "hand_velocity"],
            "pooling": "per_stream_mean+std",
            "valid_mask_usage": "excluded_from_identity_input",
            "parameter_count": parameter_count,
            "date_usage": "split_only",
        }
    )
    torch.save(checkpoint, checkpoint_path)

    summary_path = output_dir / "summary.txt"
    original = summary_path.read_text(encoding="utf-8")
    first_newline = original.find("\n")
    remainder = original[first_newline + 1 :] if first_newline >= 0 else original
    header = (
        "Hand-only two-stream statistics SupCon 1D-CNN authentication\n"
        "architecture=hand-only-two-stream-stats-supcon\n"
        "feature_layout=hand_xyz_63+hand_velocity_63+valid_mask_1\n"
        "streams=hand_position,hand_velocity\n"
        "pooling=per_stream_mean+std\n"
        "valid_mask_usage=excluded_from_identity_input\n"
        f"parameter_count={parameter_count}\n"
    )
    summary_path.write_text(header + remainder, encoding="utf-8")


if __name__ == "__main__":
    main()
