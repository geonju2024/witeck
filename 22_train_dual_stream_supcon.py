"""Position/velocity dual-stream 1D-CNN for user authentication."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class DualStreamEmbedding1DCNN(nn.Module):
    """Encode normalized positions and velocities in separate light branches."""

    def __init__(self, input_dim, num_classes, embedding_dim=128):
        super().__init__()
        if int(input_dim) != 169:
            raise ValueError(f"Dual-stream feature layout expects D=169, got {input_dim}")

        def branch():
            return nn.Sequential(
                nn.Conv1d(84, 48, kernel_size=5, padding=2),
                nn.BatchNorm1d(48),
                nn.ReLU(),
                nn.Dropout(0.15),
                nn.Conv1d(48, 64, kernel_size=3, padding=2, dilation=2),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )

        self.position_stream = branch()
        self.velocity_stream = branch()
        self.embedding_head = nn.Sequential(
            nn.Linear(64 + 64 + 1, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x, duration):
        # [hand xyz(63), hand velocity(63), pose(21), pose velocity(21), mask(1)]
        position = torch.cat([x[:, :, :63], x[:, :, 126:147]], dim=2)
        velocity = torch.cat([x[:, :, 63:126], x[:, :, 147:168]], dim=2)
        position_h = self.position_stream(position.transpose(1, 2)).squeeze(-1)
        velocity_h = self.velocity_stream(velocity.transpose(1, 2)).squeeze(-1)
        fused = torch.cat([position_h, velocity_h, duration.unsqueeze(1)], dim=1)
        embedding = F.normalize(self.embedding_head(fused), p=2, dim=1)
        return embedding, self.classifier(embedding)


def argument_value(name: str, default: str) -> str:
    if name in sys.argv:
        position = sys.argv.index(name)
        if position + 1 < len(sys.argv):
            return sys.argv[position + 1]
    return default


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    default_output = str(project_dir / "output" / "dual_stream_supcon_auth")
    if "--output-dir" not in sys.argv:
        sys.argv.extend(["--output-dir", default_output])

    supcon = importlib.import_module("16_train_supcon_embedding")
    supcon.Embedding1DCNN = DualStreamEmbedding1DCNN
    supcon.main()

    output_dir = Path(argument_value("--output-dir", default_output))
    checkpoint_path = output_dir / "embedding_1dcnn_supcon.pt"
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    checkpoint.update(
        {
            "architecture": "dual-stream-supcon",
            "training_loss": "cross_entropy+gesture_conditioned_supcon",
            "feature_streams": {
                "position": "hand_xyz+pose",
                "velocity": "hand_velocity+pose_velocity",
                "valid_mask": "excluded",
            },
            "parameter_count": sum(
                value.numel() for value in checkpoint["model_state_dict"].values()
            ),
            "date_usage": "split_only",
        }
    )
    torch.save(checkpoint, checkpoint_path)

    summary_path = output_dir / "summary.txt"
    original = summary_path.read_text(encoding="utf-8")
    first_newline = original.find("\n")
    remainder = original[first_newline + 1 :] if first_newline >= 0 else original
    parameter_count = sum(p.numel() for p in DualStreamEmbedding1DCNN(169, 7).parameters())
    header = (
        "Position/velocity dual-stream SupCon 1D-CNN authentication\n"
        "architecture=dual-stream-supcon\n"
        "position_stream=hand_xyz+pose\n"
        "velocity_stream=hand_velocity+pose_velocity\n"
        "valid_mask_usage=excluded\n"
        f"parameter_count={parameter_count}\n"
    )
    summary_path.write_text(header + remainder, encoding="utf-8")


if __name__ == "__main__":
    main()
