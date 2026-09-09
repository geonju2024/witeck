"""Center-loss training with the same basic 1D-CNN authentication backbone."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


base = importlib.import_module("08_train_embedding")


class CenterLossEmbedding1DCNN(base.Embedding1DCNN):
    def __init__(self, input_dim, num_classes, embedding_dim=128):
        super().__init__(input_dim, num_classes, embedding_dim)
        self.identity_centers = torch.nn.Parameter(
            torch.randn(num_classes, embedding_dim) * 0.02
        )

    def center_loss(self, embedding, labels):
        centers = F.normalize(self.identity_centers, p=2, dim=1)
        target_centers = centers[labels]
        return (1.0 - (embedding * target_centers).sum(dim=1)).mean()


def build_model(
    input_dim,
    num_classes,
    embedding_dim,
    sequence_mean=None,
    sequence_std=None,
):
    return CenterLossEmbedding1DCNN(input_dim, num_classes, embedding_dim)


def run_center_epoch(
    model,
    loader,
    ce_criterion,
    device,
    center_weight,
    temperature,
    optimizer=None,
):
    training = optimizer is not None
    model.train(training)
    total_loss = total_ce = total_center = 0.0
    total_correct = total_n = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for xb, db, user_y, _ in loader:
            xb, db, user_y = xb.to(device), db.to(device), user_y.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            embedding, logits = model(xb, db)
            ce_loss = ce_criterion(logits, user_y)
            metric_loss = model.center_loss(embedding, user_y)
            loss = ce_loss + center_weight * metric_loss
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            n = len(user_y)
            total_loss += float(loss.item()) * n
            total_ce += float(ce_loss.item()) * n
            total_center += float(metric_loss.item()) * n
            total_correct += int((logits.argmax(dim=1) == user_y).sum().item())
            total_n += n
    n = max(total_n, 1)
    return total_loss / n, total_ce / n, total_center / n, total_correct / n


def argument_value(name, default):
    if name in sys.argv:
        position = sys.argv.index(name)
        if position + 1 < len(sys.argv):
            return sys.argv[position + 1]
    return default


def main():
    project_dir = Path(__file__).resolve().parent
    default_output = str(project_dir / "output" / "center_loss_embedding_auth")
    if "--output-dir" not in sys.argv:
        sys.argv.extend(["--output-dir", default_output])
    if "--contrastive-weight" not in sys.argv:
        sys.argv.extend(["--contrastive-weight", "0.05"])

    trainer = importlib.import_module("16_train_supcon_embedding")
    trainer.build_model = build_model
    trainer.run_epoch = run_center_epoch
    trainer.main()

    output_dir = Path(argument_value("--output-dir", default_output))
    checkpoint_path = output_dir / "embedding_1dcnn_supcon.pt"
    new_path = output_dir / "embedding_1dcnn_center_loss.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint.update(
        {
            "architecture": "center-loss-1dcnn",
            "training_loss": "cross_entropy+center_loss",
            "center_loss_weight": float(
                argument_value("--contrastive-weight", "0.05")
            ),
        }
    )
    checkpoint.pop("positive_rule", None)
    checkpoint.pop("eligible_comparison_rule", None)
    torch.save(checkpoint, new_path)
    checkpoint_path.unlink()

    summary_path = output_dir / "summary.txt"
    lines = summary_path.read_text(encoding="utf-8").splitlines()
    lines = [
        line for line in lines
        if not line.startswith((
            "positive_rule=", "eligible_comparison_rule=", "contrastive_weight="
        ))
    ]
    lines[0] = "Center-loss basic 1D-CNN unseen-user authentication"
    lines.insert(1, "architecture=center-loss-1dcnn")
    lines.insert(2, f"center_loss_weight={checkpoint['center_loss_weight']}")
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Center-loss checkpoint: {new_path}")


if __name__ == "__main__":
    main()
