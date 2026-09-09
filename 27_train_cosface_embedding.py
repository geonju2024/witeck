"""CosFace training with the same basic 1D-CNN authentication backbone."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class CosFaceHead(nn.Module):
    """Additive cosine-margin classifier used only during training."""

    def __init__(self, embedding_dim, num_classes, scale=16.0, margin=0.20):
        super().__init__()
        self.scale = float(scale)
        self.margin = float(margin)
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

    def cosine(self, embedding):
        return F.linear(
            F.normalize(embedding, p=2, dim=1),
            F.normalize(self.weight, p=2, dim=1),
        ).clamp(-1.0, 1.0)

    def cosine_logits(self, embedding):
        return self.cosine(embedding) * self.scale

    def forward(self, embedding, labels):
        cosine = self.cosine(embedding)
        one_hot = F.one_hot(labels, num_classes=cosine.shape[1]).to(cosine.dtype)
        return (cosine - one_hot * self.margin) * self.scale


def argument_value(name, default):
    if name in sys.argv:
        position = sys.argv.index(name)
        if position + 1 < len(sys.argv):
            return sys.argv[position + 1]
    return default


def main():
    project_dir = Path(__file__).resolve().parent
    default_output = str(project_dir / "output" / "cosface_embedding_auth")
    if "--output-dir" not in sys.argv:
        sys.argv.extend(["--output-dir", default_output])

    arcface = importlib.import_module("15_train_arcface_embedding")
    arcface.ArcFaceHead = CosFaceHead
    arcface.main()

    output_dir = Path(argument_value("--output-dir", default_output))
    old_path = output_dir / "embedding_1dcnn_arcface.pt"
    new_path = output_dir / "embedding_1dcnn_cosface.pt"
    checkpoint = torch.load(old_path, map_location="cpu", weights_only=False)
    checkpoint["training_head"] = "cosface"
    checkpoint["cosface_scale"] = checkpoint.pop("arcface_scale")
    checkpoint["cosface_margin"] = checkpoint.pop("arcface_margin")
    checkpoint["cosface_state_dict"] = checkpoint.pop("arcface_state_dict")
    torch.save(checkpoint, new_path)
    old_path.unlink()

    summary_path = output_dir / "summary.txt"
    summary = summary_path.read_text(encoding="utf-8")
    summary = summary.replace("ArcFace", "CosFace")
    summary = summary.replace("arcface_scale", "cosface_scale")
    summary = summary.replace("arcface_margin", "cosface_margin")
    summary_path.write_text(summary, encoding="utf-8")
    print(f"CosFace checkpoint: {new_path}")


if __name__ == "__main__":
    main()
