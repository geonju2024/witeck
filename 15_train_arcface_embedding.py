"""Train a lightweight ArcFace user embedding model.

Dates/sessions are used only to create train/validation/enrollment/test splits.
They are never passed to the model and never used by the training loss.
"""

from __future__ import annotations

import argparse
import copy
import csv
import importlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from two_stage_common import (
    apply_duration_stats,
    apply_sequence_stats,
    class_weights,
    file_sha256,
    fit_duration_stats,
    fit_sequence_stats,
    load_dataset,
)


base = importlib.import_module("08_train_embedding")
Embedding1DCNN = base.Embedding1DCNN
TRAIN_USERS = base.TRAIN_USERS
UNSEEN_USERS = base.UNSEEN_USERS


class ArcFaceHead(nn.Module):
    """Additive angular-margin classifier used only during training."""

    def __init__(self, embedding_dim, num_classes, scale=16.0, margin=0.20):
        super().__init__()
        self.scale = float(scale)
        self.margin = float(margin)
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m = float(np.cos(self.margin))
        self.sin_m = float(np.sin(self.margin))
        self.threshold = float(np.cos(np.pi - self.margin))
        self.mm = float(np.sin(np.pi - self.margin) * self.margin)

    def cosine_logits(self, embedding):
        cosine = F.linear(
            F.normalize(embedding, p=2, dim=1),
            F.normalize(self.weight, p=2, dim=1),
        )
        return cosine.clamp(-1.0, 1.0) * self.scale

    def forward(self, embedding, labels):
        cosine = F.linear(
            F.normalize(embedding, p=2, dim=1),
            F.normalize(self.weight, p=2, dim=1),
        ).clamp(-1.0, 1.0)
        sine = torch.sqrt(torch.clamp(1.0 - cosine.square(), min=1e-7))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.threshold, phi, cosine - self.mm)
        one_hot = F.one_hot(labels, num_classes=cosine.shape[1]).to(cosine.dtype)
        return (one_hot * phi + (1.0 - one_hot) * cosine) * self.scale


def run_epoch(model, arcface, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    arcface.train(training)
    total_loss = 0.0
    total_correct = 0
    total_n = 0
    context = torch.enable_grad() if training else torch.no_grad()

    with context:
        for xb, db, yb in loader:
            xb = xb.to(device)
            db = db.to(device)
            yb = yb.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            embedding, _ = model(xb, db)
            margin_logits = arcface(embedding, yb)
            loss = criterion(margin_logits, yb)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            # Prediction does not apply a label-dependent margin.
            pred = arcface.cosine_logits(embedding).argmax(dim=1)
            total_loss += float(loss.item()) * len(yb)
            total_correct += int((pred == yb).sum().item())
            total_n += len(yb)
    return total_loss / max(total_n, 1), total_correct / max(total_n, 1)


def main():
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default=str(project_dir / "dataset" / "dataset_1955_recent8_updated_20260905.npz"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(project_dir / "output" / "arcface_embedding_auth"),
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--scale", type=float, default=16.0)
    parser.add_argument("--margin", type=float, default=0.20)
    parser.add_argument("--enroll", type=int, default=3)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    base.seed_everything(args.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    _, X_seq, duration, meta, T, D = load_dataset(args.data)
    train_idx, val_idx = base.chronological_user_split(meta, TRAIN_USERS)
    seq_mean, seq_std = fit_sequence_stats(X_seq[train_idx])
    dur_mean, dur_std = fit_duration_stats(duration[train_idx])
    X_norm = apply_sequence_stats(X_seq, seq_mean, seq_std)
    duration_norm = apply_duration_stats(duration, dur_mean, dur_std)

    user_to_label = {user: i for i, user in enumerate(TRAIN_USERS)}
    y_all = np.full(len(X_seq), -1, dtype=np.int64)
    for user, label in user_to_label.items():
        y_all[meta["performer"] == user] = label

    train_loader = base.make_loader(
        X_norm[train_idx], duration_norm[train_idx], y_all[train_idx],
        args.batch_size, shuffle=True,
    )
    val_loader = base.make_loader(
        X_norm[val_idx], duration_norm[val_idx], y_all[val_idx],
        args.batch_size, shuffle=False,
    )

    model = Embedding1DCNN(D, len(TRAIN_USERS), args.embedding_dim).to(device)
    # This original linear classifier is intentionally unused. Keeping it in the
    # checkpoint preserves compatibility with the existing evaluator.
    model.classifier.requires_grad_(False)
    arcface = ArcFaceHead(
        args.embedding_dim, len(TRAIN_USERS), args.scale, args.margin
    ).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights(y_all[train_idx], len(TRAIN_USERS), device)
    )
    optimizer = torch.optim.Adam(
        [p for p in list(model.parameters()) + list(arcface.parameters()) if p.requires_grad],
        lr=args.lr,
        weight_decay=1e-4,
    )

    print("=" * 76)
    print("ArcFace basic 1D-CNN user embedding")
    print("=" * 76)
    print(f"device={device} N={len(X_seq)} T={T} D={D}")
    print(f"train={len(train_idx)} val={len(val_idx)} scale={args.scale} margin={args.margin}")
    print("date/session usage: split only; not model input or loss")

    best_val_acc = -1.0
    best_epoch = 0
    best_model_state = None
    best_arcface_state = None
    stale = 0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(
            model, arcface, train_loader, criterion, device, optimizer
        )
        val_loss, val_acc = run_epoch(
            model, arcface, val_loader, criterion, device, optimizer=None
        )
        print(
            f"Epoch {epoch:02d} | train loss={train_loss:.4f} acc={train_acc:.3f} | "
            f"val loss={val_loss:.4f} acc={val_acc:.3f}"
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())
            best_arcface_state = copy.deepcopy(arcface.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break

    if best_model_state is None:
        raise RuntimeError("No ArcFace checkpoint was created")
    model.load_state_dict(best_model_state)
    arcface.load_state_dict(best_arcface_state)
    model.eval()

    known_idx = np.where(np.isin(meta["performer"], TRAIN_USERS))[0]
    known_embeddings = base.extract_embeddings(
        model, X_norm[known_idx], duration_norm[known_idx], device
    )
    cal_labels, cal_scores = base.build_known_validation_scores(
        known_embeddings, known_idx, meta, enrollment_per_gesture=args.enroll
    )
    threshold, val_eer, val_eer_metrics = base.find_eer_threshold(
        cal_labels, cal_scores
    )
    rows, global_metrics, test_eer, test_eer_threshold = base.evaluate_unseen_users(
        model, X_norm, duration_norm, meta, device, threshold, args.enroll
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_hash = file_sha256(args.data)
    model_path = output_dir / "embedding_1dcnn_arcface.pt"
    torch.save(
        {
            "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "arcface_state_dict": {k: v.detach().cpu() for k, v in arcface.state_dict().items()},
            "input_dim": D,
            "seq_len": T,
            "embedding_dim": args.embedding_dim,
            "num_classes": len(TRAIN_USERS),
            "train_users": TRAIN_USERS,
            "unseen_users": UNSEEN_USERS,
            "threshold": threshold,
            "sequence_mean": seq_mean,
            "sequence_std": seq_std,
            "duration_mean": dur_mean,
            "duration_std": dur_std,
            "dataset_path": str(Path(args.data).resolve()),
            "dataset_sha256": dataset_hash,
            "enrollment_per_gesture": args.enroll,
            "seed": args.seed,
            "training_head": "arcface",
            "arcface_scale": args.scale,
            "arcface_margin": args.margin,
            "date_usage": "split_only",
            "best_epoch": best_epoch,
            "best_val_id_accuracy": best_val_acc,
            "validation_eer": val_eer,
            "unseen_metrics": global_metrics,
            "unseen_test_eer_analysis": test_eer,
        },
        model_path,
    )

    csv_path = output_dir / "unseen_user_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_path = output_dir / "summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("ArcFace basic 1D-CNN unseen-user authentication\n")
        f.write("date_usage=split_only\n")
        f.write(f"dataset={Path(args.data).resolve()}\n")
        f.write(f"dataset_sha256={dataset_hash}\n")
        f.write(f"seed={args.seed}\n")
        f.write(f"arcface_scale={args.scale}\narcface_margin={args.margin}\n")
        f.write(f"embedding_dim={args.embedding_dim}\n")
        f.write(f"best_epoch={best_epoch}\nbest_val_id_accuracy={best_val_acc:.6f}\n")
        f.write(f"validation_threshold={threshold:.6f}\nvalidation_eer={val_eer:.6f}\n")
        f.write(f"validation_far={val_eer_metrics['far']:.6f}\n")
        f.write(f"validation_frr={val_eer_metrics['frr']:.6f}\n")
        f.write(f"unseen_accuracy={global_metrics['accuracy']:.6f}\n")
        f.write(f"unseen_balanced_accuracy={global_metrics['balanced_accuracy']:.6f}\n")
        f.write(f"unseen_far={global_metrics['far']:.6f}\n")
        f.write(f"unseen_frr={global_metrics['frr']:.6f}\n")
        f.write(f"unseen_test_eer_analysis={test_eer:.6f}\n")
        f.write(f"unseen_test_eer_threshold_analysis={test_eer_threshold:.6f}\n")

    print("=" * 76)
    print(f"best val identity accuracy={best_val_acc:.4f} at epoch {best_epoch}")
    print(f"validation threshold={threshold:.6f} EER={val_eer:.4f}")
    print(
        f"unseen Acc={global_metrics['accuracy']:.4f} "
        f"BalAcc={global_metrics['balanced_accuracy']:.4f} "
        f"FAR={global_metrics['far']:.4f} FRR={global_metrics['frr']:.4f} "
        f"test EER={test_eer:.4f}"
    )
    print(f"Saved: {model_path}")


if __name__ == "__main__":
    main()
