"""Skeletal-geometry augmented 1D-CNN for user authentication."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


HAND_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
)
FINGER_CHAINS = (
    (0, 1, 2, 3, 4),
    (0, 5, 6, 7, 8),
    (0, 9, 10, 11, 12),
    (0, 13, 14, 15, 16),
    (0, 17, 18, 19, 20),
)
FINGERTIPS = (4, 8, 12, 16, 20)


class GeometryAugmentedEmbedding1DCNN(nn.Module):
    """Append 68 explicit skeletal descriptors before temporal convolution."""

    def __init__(
        self,
        input_dim,
        num_classes,
        embedding_dim=128,
        sequence_mean=None,
        sequence_std=None,
    ):
        super().__init__()
        if int(input_dim) != 169:
            raise ValueError(f"Skeletal layout expects D=169, got {input_dim}")
        if sequence_mean is None or sequence_std is None:
            sequence_mean = np.zeros(input_dim, dtype=np.float32)
            sequence_std = np.ones(input_dim, dtype=np.float32)
        mean = torch.as_tensor(sequence_mean, dtype=torch.float32).reshape(1, 1, -1)
        std = torch.as_tensor(sequence_std, dtype=torch.float32).reshape(1, 1, -1)
        self.register_buffer("sequence_mean", mean)
        self.register_buffer("sequence_std", std)

        geometry_dim = 20 + 15 + 5 + 21 + 7
        self.geometry_norm = nn.LayerNorm(geometry_dim)
        self.features = nn.Sequential(
            nn.Conv1d(input_dim + geometry_dim, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 96, kernel_size=5, padding=2),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Conv1d(96, 128, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.AdaptiveAvgPool1d(1),
        )
        self.embedding_head = nn.Linear(128 + 1, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def skeletal_geometry(self, normalized_x):
        raw = normalized_x * self.sequence_std + self.sequence_mean
        hand = raw[:, :, :63].reshape(raw.shape[0], raw.shape[1], 21, 3)
        hand_velocity = raw[:, :, 63:126].reshape(
            raw.shape[0], raw.shape[1], 21, 3
        )
        pose_velocity = raw[:, :, 147:168].reshape(
            raw.shape[0], raw.shape[1], 7, 3
        )

        bone_lengths = torch.stack(
            [torch.linalg.vector_norm(hand[:, :, b] - hand[:, :, a], dim=-1)
             for a, b in HAND_EDGES],
            dim=-1,
        )
        angles = []
        for chain in FINGER_CHAINS:
            for center in range(1, 4):
                incoming = hand[:, :, chain[center - 1]] - hand[:, :, chain[center]]
                outgoing = hand[:, :, chain[center + 1]] - hand[:, :, chain[center]]
                angles.append(F.cosine_similarity(incoming, outgoing, dim=-1, eps=1e-6))
        joint_angles = torch.stack(angles, dim=-1)
        fingertip_distances = torch.stack(
            [torch.linalg.vector_norm(hand[:, :, tip] - hand[:, :, 0], dim=-1)
             for tip in FINGERTIPS],
            dim=-1,
        )
        hand_speed = torch.linalg.vector_norm(hand_velocity, dim=-1)
        pose_speed = torch.linalg.vector_norm(pose_velocity, dim=-1)
        geometry = torch.cat(
            [bone_lengths, joint_angles, fingertip_distances, hand_speed, pose_speed],
            dim=-1,
        )
        return self.geometry_norm(geometry)

    def forward(self, x, duration):
        geometry = self.skeletal_geometry(x)
        augmented = torch.cat([x, geometry], dim=2)
        pooled = self.features(augmented.transpose(1, 2)).squeeze(-1)
        embedding = self.embedding_head(
            torch.cat([pooled, duration.unsqueeze(1)], dim=1)
        )
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding, self.classifier(embedding)


def build_geometry_model(
    input_dim,
    num_classes,
    embedding_dim,
    sequence_mean=None,
    sequence_std=None,
):
    return GeometryAugmentedEmbedding1DCNN(
        input_dim,
        num_classes,
        embedding_dim,
        sequence_mean,
        sequence_std,
    )


def argument_value(name: str, default: str) -> str:
    if name in sys.argv:
        position = sys.argv.index(name)
        if position + 1 < len(sys.argv):
            return sys.argv[position + 1]
    return default


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    default_output = str(project_dir / "output" / "skeletal_geometry_supcon_auth")
    if "--output-dir" not in sys.argv:
        sys.argv.extend(["--output-dir", default_output])

    supcon = importlib.import_module("16_train_supcon_embedding")
    supcon.build_model = build_geometry_model
    supcon.main()

    output_dir = Path(argument_value("--output-dir", default_output))
    checkpoint_path = output_dir / "embedding_1dcnn_supcon.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    parameter_count = sum(
        value.numel() for value in checkpoint["model_state_dict"].values()
    )
    checkpoint.update(
        {
            "architecture": "skeletal-geometry-supcon",
            "geometry_features": (
                "20_bone_lengths+15_joint_angles+5_fingertip_distances+"
                "21_hand_speeds+7_pose_speeds"
            ),
            "geometry_dim": 68,
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
        "Skeletal-geometry augmented SupCon 1D-CNN authentication\n"
        "architecture=skeletal-geometry-supcon\n"
        "geometry_dim=68\n"
        "geometry=bone_lengths+joint_angles+fingertip_distances+joint_speeds\n"
        f"parameter_count={parameter_count}\n"
    )
    summary_path.write_text(header + remainder, encoding="utf-8")


if __name__ == "__main__":
    main()
