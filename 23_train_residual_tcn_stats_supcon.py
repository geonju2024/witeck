"""Light residual TCN with statistics pooling for user authentication."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualTCNBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float = 0.15):
        super().__init__()
        self.temporal = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.norm = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.temporal(x)
        x = self.norm(x)
        x = F.relu(x)
        x = self.dropout(x)
        return F.relu(x + residual)


class ResidualTCNStatsEmbedding(nn.Module):
    """Dilation 1/2/4 TCN followed by mean, std, and max pooling."""

    def __init__(self, input_dim, num_classes, embedding_dim=128):
        super().__init__()
        channels = 32
        self.input_projection = nn.Sequential(
            nn.Conv1d(input_dim, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
        )
        self.temporal_blocks = nn.Sequential(
            ResidualTCNBlock(channels, dilation=1),
            ResidualTCNBlock(channels, dilation=2),
            ResidualTCNBlock(channels, dilation=4),
        )
        self.embedding_head = nn.Sequential(
            nn.Linear(channels * 3 + 1, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x, duration):
        h = self.input_projection(x.transpose(1, 2))
        h = self.temporal_blocks(h)
        mean = h.mean(dim=2)
        std = torch.sqrt(h.var(dim=2, unbiased=False).clamp_min(1e-5))
        maximum = h.amax(dim=2)
        pooled = torch.cat([mean, std, maximum, duration.unsqueeze(1)], dim=1)
        embedding = F.normalize(self.embedding_head(pooled), p=2, dim=1)
        return embedding, self.classifier(embedding)


def argument_value(name: str, default: str) -> str:
    if name in sys.argv:
        position = sys.argv.index(name)
        if position + 1 < len(sys.argv):
            return sys.argv[position + 1]
    return default


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    default_output = str(project_dir / "output" / "residual_tcn_stats_supcon_auth")
    if "--output-dir" not in sys.argv:
        sys.argv.extend(["--output-dir", default_output])

    supcon = importlib.import_module("16_train_supcon_embedding")
    supcon.Embedding1DCNN = ResidualTCNStatsEmbedding
    supcon.main()

    output_dir = Path(argument_value("--output-dir", default_output))
    checkpoint_path = output_dir / "embedding_1dcnn_supcon.pt"
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    parameter_count = sum(
        value.numel() for value in checkpoint["model_state_dict"].values()
    )
    checkpoint.update(
        {
            "architecture": "residual-tcn-stats-supcon",
            "training_loss": "cross_entropy+gesture_conditioned_supcon",
            "dilations": [1, 2, 4],
            "pooling": "mean+std+max",
            "convolution": "standard_conv1d_32_channels",
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
        "Residual TCN statistics-pooling SupCon authentication\n"
        "architecture=residual-tcn-stats-supcon\n"
        "dilations=1,2,4\n"
        "pooling=mean+std+max\n"
        "convolution=standard_conv1d_32_channels\n"
        f"parameter_count={parameter_count}\n"
    )
    summary_path.write_text(header + remainder, encoding="utf-8")


if __name__ == "__main__":
    main()
