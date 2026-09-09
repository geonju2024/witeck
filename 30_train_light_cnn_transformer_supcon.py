"""Light CNN-Transformer SupCon model for unseen-user authentication.

The model is intentionally small for the 1,955-sample, 32-frame dataset:
a local Conv1D stem followed by one 64-dimensional Transformer encoder layer.
Dates/sessions remain split-only metadata.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class LightCNNTransformerEmbedding(nn.Module):
    """Local temporal CNN plus one global self-attention layer."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        embedding_dim: int = 128,
        seq_len: int = 32,
        model_dim: int = 64,
    ):
        super().__init__()
        self.seq_len = int(seq_len)
        self.local_stem = nn.Sequential(
            nn.Conv1d(input_dim, model_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(model_dim),
            nn.GELU(),
            nn.Dropout(0.10),
        )
        self.position = nn.Parameter(torch.zeros(1, self.seq_len, model_dim))
        nn.init.trunc_normal_(self.position, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=4,
            dim_feedforward=128,
            dropout=0.15,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=1,
            enable_nested_tensor=False,
        )
        self.final_norm = nn.LayerNorm(model_dim)
        self.embedding_head = nn.Sequential(
            nn.Linear(model_dim * 2 + 1, embedding_dim),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x, duration):
        if x.shape[1] > self.seq_len:
            raise ValueError(
                f"sequence length {x.shape[1]} exceeds configured {self.seq_len}"
            )
        h = self.local_stem(x.transpose(1, 2)).transpose(1, 2)
        h = h + self.position[:, : h.shape[1]]
        h = self.final_norm(self.encoder(h))
        mean = h.mean(dim=1)
        std = torch.sqrt(h.var(dim=1, unbiased=False).clamp_min(1e-5))
        pooled = torch.cat([mean, std, duration.unsqueeze(1)], dim=1)
        embedding = F.normalize(self.embedding_head(pooled), p=2, dim=1)
        return embedding, self.classifier(embedding)


def build_model(
    input_dim,
    num_classes,
    embedding_dim,
    sequence_mean=None,
    sequence_std=None,
):
    return LightCNNTransformerEmbedding(
        input_dim=input_dim,
        num_classes=num_classes,
        embedding_dim=embedding_dim,
    )


def argument_value(name: str, default: str) -> str:
    if name in sys.argv:
        position = sys.argv.index(name)
        if position + 1 < len(sys.argv):
            return sys.argv[position + 1]
    return default


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    default_output = str(project_dir / "output" / "light_cnn_transformer_supcon_auth")
    if "--output-dir" not in sys.argv:
        sys.argv.extend(["--output-dir", default_output])

    trainer = importlib.import_module("16_train_supcon_embedding")
    trainer.build_model = build_model
    trainer.main()

    output_dir = Path(argument_value("--output-dir", default_output))
    old_path = output_dir / "embedding_1dcnn_supcon.pt"
    new_path = output_dir / "embedding_light_cnn_transformer_supcon.pt"
    checkpoint = torch.load(old_path, map_location="cpu", weights_only=False)
    parameter_count = sum(
        value.numel() for value in checkpoint["model_state_dict"].values()
    )
    checkpoint.update(
        {
            "architecture": "light-cnn-transformer-supcon",
            "training_loss": "cross_entropy+gesture_conditioned_supcon",
            "transformer_model_dim": 64,
            "transformer_heads": 4,
            "transformer_layers": 1,
            "transformer_feedforward_dim": 128,
            "pooling": "mean+std",
            "parameter_count": parameter_count,
            "date_usage": "split_only",
        }
    )
    torch.save(checkpoint, new_path)
    old_path.unlink()

    summary_path = output_dir / "summary.txt"
    original = summary_path.read_text(encoding="utf-8")
    first_newline = original.find("\n")
    remainder = original[first_newline + 1 :] if first_newline >= 0 else original
    header = (
        "Light CNN-Transformer gesture-conditioned SupCon authentication\n"
        "architecture=light-cnn-transformer-supcon\n"
        "cnn_stem=kernel3_channels64\n"
        "transformer=model_dim64_heads4_layers1_ff128\n"
        "pooling=mean+std\n"
        f"parameter_count={parameter_count}\n"
    )
    summary_path.write_text(header + remainder, encoding="utf-8")
    print(f"Light CNN-Transformer checkpoint: {new_path}")


if __name__ == "__main__":
    main()
