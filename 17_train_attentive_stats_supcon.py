"""Train an attentive-statistics 1D-CNN with SupCon authentication loss.

Compared with 16_train_supcon_embedding.py, only temporal pooling changes:
adaptive mean pooling -> learned attentive mean + standard-deviation pooling.
Date/session metadata remains split-only information.
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
TRAIN_USERS = base.TRAIN_USERS
UNSEEN_USERS = base.UNSEEN_USERS


class AttentiveStatsEmbedding1DCNN(nn.Module):
    """Basic convolution stack with learned attentive mean/std pooling."""

    def __init__(self, input_dim, num_classes, embedding_dim=128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Conv1d(64, 96, kernel_size=3, padding=1),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.Conv1d(96, 128, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.30),
        )
        self.attention = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(64, 1, kernel_size=1),
        )
        self.embedding_head = nn.Linear(128 * 2 + 1, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x, duration):
        frame_features = self.features(x.transpose(1, 2))
        weights = torch.softmax(self.attention(frame_features), dim=2)
        mean = torch.sum(weights * frame_features, dim=2)
        centered = frame_features - mean.unsqueeze(2)
        variance = torch.sum(weights * centered.square(), dim=2)
        std = torch.sqrt(variance.clamp_min(1e-5))
        pooled = torch.cat([mean, std, duration.unsqueeze(1)], dim=1)
        embedding = F.normalize(self.embedding_head(pooled), p=2, dim=1)
        logits = self.classifier(embedding)
        return embedding, logits


def main():
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default=str(project_dir / "dataset" / "dataset_1955_recent8_updated_20260905.npz"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(project_dir / "output" / "attentive_stats_supcon_auth"),
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--contrastive-weight", type=float, default=0.20)
    parser.add_argument("--temperature", type=float, default=0.10)
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
    gestures = sorted(np.unique(meta["gesture"]).tolist())
    gesture_to_label = {gesture: i for i, gesture in enumerate(gestures)}
    user_y = np.full(len(X_seq), -1, dtype=np.int64)
    for user, label in user_to_label.items():
        user_y[meta["performer"] == user] = label
    gesture_y = np.asarray(
        [gesture_to_label[x] for x in meta["gesture"]], dtype=np.int64
    )

    train_loader = supcon.make_loader(
        X_norm[train_idx], duration_norm[train_idx], user_y[train_idx],
        gesture_y[train_idx], args.batch_size, shuffle=True,
    )
    val_loader = supcon.make_loader(
        X_norm[val_idx], duration_norm[val_idx], user_y[val_idx],
        gesture_y[val_idx], args.batch_size, shuffle=False,
    )

    model = AttentiveStatsEmbedding1DCNN(
        D, len(TRAIN_USERS), args.embedding_dim
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=1e-4
    )

    parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("=" * 82)
    print("Attentive-statistics SupCon 1D-CNN user embedding")
    print("=" * 82)
    print(f"device={device} N={len(X_seq)} T={T} D={D} parameters={parameter_count:,}")
    print(f"train={len(train_idx)} val={len(val_idx)}")
    print(
        f"loss=CE + {args.contrastive_weight}*SupCon, "
        f"temperature={args.temperature}"
    )
    print("date/session usage: split only; not model input, loss, or pair rule")

    best_val_acc = -1.0
    best_epoch = 0
    best_state = None
    stale = 0
    stopped_epoch = args.epochs
    for epoch in range(1, args.epochs + 1):
        train_values = supcon.run_epoch(
            model, train_loader, criterion, device,
            args.contrastive_weight, args.temperature, optimizer,
        )
        val_values = supcon.run_epoch(
            model, val_loader, criterion, device,
            args.contrastive_weight, args.temperature, optimizer=None,
        )
        train_loss, train_ce, train_sc, train_acc = train_values
        val_loss, val_ce, val_sc, val_acc = val_values
        print(
            f"Epoch {epoch:02d} | train total={train_loss:.4f} "
            f"CE={train_ce:.4f} SupCon={train_sc:.4f} acc={train_acc:.3f} | "
            f"val total={val_loss:.4f} CE={val_ce:.4f} "
            f"SupCon={val_sc:.4f} acc={val_acc:.3f}"
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
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
        raise RuntimeError("No attentive-statistics checkpoint was created")
    model.load_state_dict(best_state)
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
    model_path = output_dir / "embedding_1dcnn_attentive_stats_supcon.pt"
    torch.save(
        {
            "model_state_dict": {
                key: value.detach().cpu()
                for key, value in model.state_dict().items()
            },
            "architecture": "attentive-stats-supcon",
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
            "training_loss": "cross_entropy+gesture_conditioned_supcon",
            "contrastive_weight": args.contrastive_weight,
            "temperature": args.temperature,
            "pooling": "learned_attentive_mean_and_std",
            "date_usage": "split_only",
            "parameter_count": parameter_count,
            "best_epoch": best_epoch,
            "stopped_epoch": stopped_epoch,
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
        f.write("Attentive-statistics SupCon 1D-CNN authentication\n")
        f.write("architecture=attentive-stats-supcon\n")
        f.write("pooling=learned_attentive_mean_and_std\n")
        f.write("date_usage=split_only\n")
        f.write(f"dataset={Path(args.data).resolve()}\n")
        f.write(f"dataset_sha256={dataset_hash}\n")
        f.write(f"parameter_count={parameter_count}\n")
        f.write(f"contrastive_weight={args.contrastive_weight}\n")
        f.write(f"temperature={args.temperature}\n")
        f.write(f"embedding_dim={args.embedding_dim}\n")
        f.write(f"seed={args.seed}\n")
        f.write(f"best_epoch={best_epoch}\nstopped_epoch={stopped_epoch}\n")
        f.write(f"best_val_id_accuracy={best_val_acc:.6f}\n")
        f.write(f"validation_threshold={threshold:.6f}\n")
        f.write(f"validation_eer={val_eer:.6f}\n")
        f.write(f"validation_far={val_eer_metrics['far']:.6f}\n")
        f.write(f"validation_frr={val_eer_metrics['frr']:.6f}\n")
        f.write(f"unseen_accuracy={global_metrics['accuracy']:.6f}\n")
        f.write(f"unseen_balanced_accuracy={global_metrics['balanced_accuracy']:.6f}\n")
        f.write(f"unseen_far={global_metrics['far']:.6f}\n")
        f.write(f"unseen_frr={global_metrics['frr']:.6f}\n")
        f.write(f"unseen_test_eer_analysis={test_eer:.6f}\n")
        f.write(f"unseen_test_eer_threshold_analysis={test_eer_threshold:.6f}\n")

    print("=" * 82)
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
