"""SupCon v2: separate projection head and verification-based selection."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


base = importlib.import_module("08_train_embedding")


class ProjectedSupConEmbedding1DCNN(base.Embedding1DCNN):
    """Keep the deployable embedding separate from the SupCon projection."""

    def __init__(self, input_dim, num_classes, embedding_dim=128):
        super().__init__(input_dim, num_classes, embedding_dim)
        self.contrastive_projector = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 64),
        )

    def project_for_contrastive(self, embedding):
        return F.normalize(self.contrastive_projector(embedding), p=2, dim=1)


def build_model(
    input_dim,
    num_classes,
    embedding_dim,
    sequence_mean=None,
    sequence_std=None,
):
    return ProjectedSupConEmbedding1DCNN(
        input_dim, num_classes, embedding_dim
    )


def run_projected_epoch(
    model,
    loader,
    ce_criterion,
    device,
    contrastive_weight,
    temperature,
    optimizer=None,
):
    supcon = importlib.import_module("16_train_supcon_embedding")
    training = optimizer is not None
    model.train(training)
    total_loss = total_ce = total_supcon = 0.0
    total_correct = total_n = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for xb, db, user_y, gesture_y in loader:
            xb, db = xb.to(device), db.to(device)
            user_y, gesture_y = user_y.to(device), gesture_y.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            embedding, logits = model(xb, db)
            projection = model.project_for_contrastive(embedding)
            ce_loss = ce_criterion(logits, user_y)
            metric_loss = supcon.gesture_conditioned_supcon_loss(
                projection, user_y, gesture_y, temperature
            )
            loss = ce_loss + contrastive_weight * metric_loss
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            n = len(user_y)
            total_loss += float(loss.item()) * n
            total_ce += float(ce_loss.item()) * n
            total_supcon += float(metric_loss.item()) * n
            total_correct += int((logits.argmax(dim=1) == user_y).sum().item())
            total_n += n
    n = max(total_n, 1)
    return total_loss / n, total_ce / n, total_supcon / n, total_correct / n


def argument_value(name, default):
    if name in sys.argv:
        position = sys.argv.index(name)
        if position + 1 < len(sys.argv):
            return sys.argv[position + 1]
    return default


def main():
    project_dir = Path(__file__).resolve().parent
    default_output = str(project_dir / "output" / "projected_supcon_v2_auth")
    if "--output-dir" not in sys.argv:
        sys.argv.extend(["--output-dir", default_output])
    if "--selection-metric" not in sys.argv:
        sys.argv.extend(["--selection-metric", "auth-eer"])

    supcon = importlib.import_module("16_train_supcon_embedding")
    supcon.build_model = build_model
    supcon.run_epoch = run_projected_epoch
    supcon.main()

    output_dir = Path(argument_value("--output-dir", default_output))
    checkpoint_path = output_dir / "embedding_1dcnn_supcon.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    parameter_count = sum(
        value.numel() for value in checkpoint["model_state_dict"].values()
    )
    checkpoint.update(
        {
            "architecture": "projected-supcon-v2",
            "contrastive_projection_dim": 64,
            "deployable_embedding_dim": checkpoint["embedding_dim"],
            "parameter_count": parameter_count,
            "date_usage": "split_and_validation_only",
        }
    )
    torch.save(checkpoint, checkpoint_path)

    summary_path = output_dir / "summary.txt"
    original = summary_path.read_text(encoding="utf-8")
    first_newline = original.find("\n")
    remainder = original[first_newline + 1 :] if first_newline >= 0 else original
    header = (
        "Projected SupCon v2 1D-CNN authentication\n"
        "architecture=projected-supcon-v2\n"
        "deployable_embedding_dim=128\n"
        "contrastive_projection_dim=64\n"
        "checkpoint_selection=validation_auth_eer\n"
        f"parameter_count={parameter_count}\n"
    )
    summary_path.write_text(header + remainder, encoding="utf-8")


if __name__ == "__main__":
    main()
