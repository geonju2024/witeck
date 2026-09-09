"""Train predefined lightweight authentication variants.

Variants
--------
hard-triplet:
    Basic 1D-CNN + identity CE + same-gesture batch-hard triplet loss.
multiscale-supcon:
    Multi-kernel (3/5/7) 1D-CNN + identity CE + same-gesture SupCon.

Date/session metadata is used only for dataset splitting and evaluation.
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
    file_sha256,
    fit_duration_stats,
    fit_sequence_stats,
    load_dataset,
)


base = importlib.import_module("08_train_embedding")
supcon = importlib.import_module("16_train_supcon_embedding")
Embedding1DCNN = base.Embedding1DCNN
TRAIN_USERS = base.TRAIN_USERS
UNSEEN_USERS = base.UNSEEN_USERS


class MultiScaleEmbedding1DCNN(nn.Module):
    """Lightweight parallel temporal kernels followed by normalized embedding."""

    def __init__(self, input_dim, num_classes, embedding_dim=128):
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.branches = nn.ModuleList([
            nn.Conv1d(64, 32, kernel_size=kernel, padding=kernel // 2)
            for kernel in (3, 5, 7)
        ])
        self.fusion = nn.Sequential(
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Conv1d(96, 128, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.AdaptiveAvgPool1d(1),
        )
        self.embedding_head = nn.Linear(128 + 1, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x, duration):
        h = self.input_projection(x.transpose(1, 2))
        h = torch.cat([branch(h) for branch in self.branches], dim=1)
        h = self.fusion(h).squeeze(-1)
        h = torch.cat([h, duration.unsqueeze(1)], dim=1)
        embedding = F.normalize(self.embedding_head(h), p=2, dim=1)
        return embedding, self.classifier(embedding)


def same_gesture_batch_hard_triplet(
    embedding, user_labels, gesture_labels, margin=0.20
):
    n = embedding.shape[0]
    if n < 2:
        return embedding.sum() * 0.0
    distance = 1.0 - embedding @ embedding.T
    eye = torch.eye(n, dtype=torch.bool, device=embedding.device)
    same_user = user_labels[:, None].eq(user_labels[None, :])
    same_gesture = gesture_labels[:, None].eq(gesture_labels[None, :])
    positive = same_user & same_gesture & ~eye
    negative = ~same_user & same_gesture
    valid = positive.any(dim=1) & negative.any(dim=1)
    if not torch.any(valid):
        return embedding.sum() * 0.0
    hardest_positive = distance.masked_fill(~positive, float("-inf")).max(dim=1).values
    hardest_negative = distance.masked_fill(~negative, float("inf")).min(dim=1).values
    return F.relu(hardest_positive[valid] - hardest_negative[valid] + margin).mean()


def run_epoch(
    model, loader, criterion, device, variant, metric_weight,
    temperature, triplet_margin, optimizer=None,
):
    training = optimizer is not None
    model.train(training)
    total = total_ce = total_metric = 0.0
    correct = count = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for xb, db, user_y, gesture_y in loader:
            xb, db = xb.to(device), db.to(device)
            user_y, gesture_y = user_y.to(device), gesture_y.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            embedding, logits = model(xb, db)
            ce = criterion(logits, user_y)
            if variant == "hard-triplet":
                metric = same_gesture_batch_hard_triplet(
                    embedding, user_y, gesture_y, triplet_margin
                )
            else:
                metric = supcon.gesture_conditioned_supcon_loss(
                    embedding, user_y, gesture_y, temperature
                )
            loss = ce + metric_weight * metric
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            n = len(user_y)
            total += float(loss.item()) * n
            total_ce += float(ce.item()) * n
            total_metric += float(metric.item()) * n
            correct += int((logits.argmax(1) == user_y).sum().item())
            count += n
    n = max(count, 1)
    return total / n, total_ce / n, total_metric / n, correct / n


def main():
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("hard-triplet", "multiscale-supcon"), required=True)
    parser.add_argument(
        "--data",
        default=str(project_dir / "dataset" / "dataset_1955_recent8_updated_20260905.npz"),
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--metric-weight", type=float, default=0.20)
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--triplet-margin", type=float, default=0.20)
    parser.add_argument("--enroll", type=int, default=3)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = str(
            project_dir / "output" / args.variant.replace("-", "_")
        )

    base.seed_everything(args.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    _, X_seq, duration, meta, T, D = load_dataset(args.data)
    train_idx, val_idx = base.chronological_user_split(meta, TRAIN_USERS)
    gestures = sorted(np.unique(meta["gesture"]).tolist())
    gesture_to_label = {name: i for i, name in enumerate(gestures)}
    user_to_label = {name: i for i, name in enumerate(TRAIN_USERS)}
    user_y = np.full(len(X_seq), -1, dtype=np.int64)
    for user, label in user_to_label.items():
        user_y[meta["performer"] == user] = label
    gesture_y = np.asarray(
        [gesture_to_label[x] for x in meta["gesture"]], dtype=np.int64
    )
    seq_mean, seq_std = fit_sequence_stats(X_seq[train_idx])
    dur_mean, dur_std = fit_duration_stats(duration[train_idx])
    X_norm = apply_sequence_stats(X_seq, seq_mean, seq_std)
    duration_norm = apply_duration_stats(duration, dur_mean, dur_std)
    train_loader = supcon.make_loader(
        X_norm[train_idx], duration_norm[train_idx], user_y[train_idx],
        gesture_y[train_idx], args.batch_size, shuffle=True,
    )
    val_loader = supcon.make_loader(
        X_norm[val_idx], duration_norm[val_idx], user_y[val_idx],
        gesture_y[val_idx], args.batch_size, shuffle=False,
    )

    if args.variant == "hard-triplet":
        model = Embedding1DCNN(D, len(TRAIN_USERS), args.embedding_dim)
        architecture = "basic-1dcnn"
    else:
        model = MultiScaleEmbedding1DCNN(D, len(TRAIN_USERS), args.embedding_dim)
        architecture = "multiscale-supcon"
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("=" * 84)
    print(f"Metric embedding variant: {args.variant}")
    print("=" * 84)
    print(f"device={device} N={len(X_seq)} T={T} D={D} parameters={parameter_count:,}")
    print(f"metric_weight={args.metric_weight} temperature={args.temperature} margin={args.triplet_margin}")
    print("date/session usage: split only; not model input, loss, or pair rule")

    best_val_acc = -1.0
    best_epoch = 0
    best_state = None
    stale = 0
    stopped_epoch = args.epochs
    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(
            model, train_loader, criterion, device, args.variant,
            args.metric_weight, args.temperature, args.triplet_margin, optimizer,
        )
        va = run_epoch(
            model, val_loader, criterion, device, args.variant,
            args.metric_weight, args.temperature, args.triplet_margin, optimizer=None,
        )
        print(
            f"Epoch {epoch:02d} | train total={tr[0]:.4f} CE={tr[1]:.4f} "
            f"metric={tr[2]:.4f} acc={tr[3]:.3f} | val total={va[0]:.4f} "
            f"CE={va[1]:.4f} metric={va[2]:.4f} acc={va[3]:.3f}"
        )
        if va[3] > best_val_acc:
            best_val_acc = va[3]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            stopped_epoch = epoch
            print(f"Early stopping at epoch {epoch}")
            break

    if best_state is None:
        raise RuntimeError("No checkpoint was created")
    model.load_state_dict(best_state)
    model.eval()
    known_idx = np.where(np.isin(meta["performer"], TRAIN_USERS))[0]
    known_embeddings = base.extract_embeddings(
        model, X_norm[known_idx], duration_norm[known_idx], device
    )
    cal_labels, cal_scores = base.build_known_validation_scores(
        known_embeddings, known_idx, meta, enrollment_per_gesture=args.enroll
    )
    threshold, val_eer, val_eer_metrics = base.find_eer_threshold(cal_labels, cal_scores)
    rows, global_metrics, test_eer, test_eer_threshold = base.evaluate_unseen_users(
        model, X_norm, duration_norm, meta, device, threshold, args.enroll
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_hash = file_sha256(args.data)
    model_path = output_dir / f"embedding_1dcnn_{args.variant.replace('-', '_')}.pt"
    torch.save({
        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "architecture": architecture,
        "variant": args.variant,
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
        "metric_weight": args.metric_weight,
        "temperature": args.temperature,
        "triplet_margin": args.triplet_margin,
        "date_usage": "split_only",
        "parameter_count": parameter_count,
        "best_epoch": best_epoch,
        "stopped_epoch": stopped_epoch,
        "best_val_id_accuracy": best_val_acc,
        "validation_eer": val_eer,
        "unseen_metrics": global_metrics,
        "unseen_test_eer_analysis": test_eer,
    }, model_path)
    with (output_dir / "unseen_user_results.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "summary.txt").open("w", encoding="utf-8") as f:
        f.write(f"Metric embedding variant={args.variant}\n")
        f.write(f"architecture={architecture}\n")
        f.write("date_usage=split_only\n")
        f.write(f"dataset={Path(args.data).resolve()}\n")
        f.write(f"dataset_sha256={dataset_hash}\n")
        f.write(f"parameter_count={parameter_count}\n")
        f.write(f"metric_weight={args.metric_weight}\n")
        f.write(f"temperature={args.temperature}\n")
        f.write(f"triplet_margin={args.triplet_margin}\n")
        f.write(f"seed={args.seed}\n")
        f.write(f"best_epoch={best_epoch}\nstopped_epoch={stopped_epoch}\n")
        f.write(f"best_val_id_accuracy={best_val_acc:.6f}\n")
        f.write(f"validation_threshold={threshold:.6f}\nvalidation_eer={val_eer:.6f}\n")
        f.write(f"validation_far={val_eer_metrics['far']:.6f}\n")
        f.write(f"validation_frr={val_eer_metrics['frr']:.6f}\n")
        f.write(f"unseen_accuracy={global_metrics['accuracy']:.6f}\n")
        f.write(f"unseen_balanced_accuracy={global_metrics['balanced_accuracy']:.6f}\n")
        f.write(f"unseen_far={global_metrics['far']:.6f}\n")
        f.write(f"unseen_frr={global_metrics['frr']:.6f}\n")
        f.write(f"unseen_test_eer_analysis={test_eer:.6f}\n")
        f.write(f"unseen_test_eer_threshold_analysis={test_eer_threshold:.6f}\n")

    print("=" * 84)
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
