"""Gesture-conditioned projection heads with SupCon user authentication.

The temporal CNN is shared, while G1~G5 use separate lightweight embedding
projections. Date/session metadata is split-only information.
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


class GestureConditionedEmbedding1DCNN(nn.Module):
    def __init__(self, input_dim, num_classes, num_gestures=5, embedding_dim=128):
        super().__init__()
        self.num_gestures = int(num_gestures)
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
            nn.AdaptiveAvgPool1d(1),
        )
        self.embedding_heads = nn.ModuleList([
            nn.Linear(128 + 1, embedding_dim)
            for _ in range(self.num_gestures)
        ])
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def pooled_features(self, x, duration):
        h = self.features(x.transpose(1, 2)).squeeze(-1)
        return torch.cat([h, duration.unsqueeze(1)], dim=1)

    def embeddings_for_all_gestures(self, x, duration):
        h = self.pooled_features(x, duration)
        embeddings = [
            F.normalize(head(h), p=2, dim=1)
            for head in self.embedding_heads
        ]
        return torch.stack(embeddings, dim=1)

    def forward(self, x, duration, gesture_labels):
        all_embeddings = self.embeddings_for_all_gestures(x, duration)
        batch_index = torch.arange(x.shape[0], device=x.device)
        embedding = all_embeddings[batch_index, gesture_labels]
        return embedding, self.classifier(embedding)


def run_epoch(
    model,
    loader,
    criterion,
    device,
    contrastive_weight,
    temperature,
    optimizer=None,
):
    training = optimizer is not None
    model.train(training)
    total_loss = total_ce = total_supcon = 0.0
    total_correct = total_n = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for xb, db, user_y, gesture_y in loader:
            xb = xb.to(device)
            db = db.to(device)
            user_y = user_y.to(device)
            gesture_y = gesture_y.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            embedding, logits = model(xb, db, gesture_y)
            ce_loss = criterion(logits, user_y)
            sc_loss = supcon.gesture_conditioned_supcon_loss(
                embedding, user_y, gesture_y, temperature
            )
            loss = ce_loss + contrastive_weight * sc_loss
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            n = len(user_y)
            total_loss += float(loss.item()) * n
            total_ce += float(ce_loss.item()) * n
            total_supcon += float(sc_loss.item()) * n
            total_correct += int((logits.argmax(dim=1) == user_y).sum().item())
            total_n += n
    n = max(total_n, 1)
    return total_loss / n, total_ce / n, total_supcon / n, total_correct / n


@torch.no_grad()
def extract_conditioned_embeddings(
    model, X, duration, gesture_labels, device, batch_size=128
):
    loader = supcon.make_loader(
        X,
        duration,
        np.zeros(len(X), dtype=np.int64),
        gesture_labels,
        batch_size,
        shuffle=False,
    )
    model.eval()
    result = []
    for xb, db, _, gb in loader:
        embedding, _ = model(xb.to(device), db.to(device), gb.to(device))
        result.append(embedding.cpu().numpy())
    return np.concatenate(result, axis=0)


@torch.no_grad()
def extract_all_gesture_embeddings(model, X, duration, device, batch_size=128):
    dummy = np.zeros(len(X), dtype=np.int64)
    loader = supcon.make_loader(
        X, duration, dummy, dummy, batch_size, shuffle=False
    )
    model.eval()
    result = []
    for xb, db, _, _ in loader:
        embedding = model.embeddings_for_all_gestures(
            xb.to(device), db.to(device)
        )
        result.append(embedding.cpu().numpy())
    return np.concatenate(result, axis=0)


def evaluate_unseen(
    embeddings,
    meta,
    threshold,
    enrollment_per_gesture,
    gesture_to_label,
):
    performer = meta["performer"]
    gesture = meta["gesture"]
    session = meta["session"]
    rows = []
    global_labels = []
    global_scores = []
    for target_user in UNSEEN_USERS:
        user_idx = np.where(performer == target_user)[0]
        enroll_session = sorted(np.unique(session[user_idx]).tolist())[0]
        print(f"\n[UNSEEN] {target_user} enrollment_session={enroll_session}")
        for gesture_name in sorted(gesture_to_label):
            enroll_candidates = np.where(
                (performer == target_user)
                & (gesture == gesture_name)
                & (session == enroll_session)
            )[0]
            if len(enroll_candidates) < enrollment_per_gesture:
                continue
            enroll_idx = enroll_candidates[:enrollment_per_gesture]
            genuine_idx = np.where(
                (performer == target_user)
                & (gesture == gesture_name)
                & (session != enroll_session)
            )[0]
            impostor_idx = np.where(
                np.isin(performer, UNSEEN_USERS)
                & (performer != target_user)
                & (gesture == gesture_name)
            )[0]
            template = embeddings[enroll_idx].mean(axis=0)
            genuine_scores = base.cosine_scores(embeddings[genuine_idx], template)
            impostor_scores = base.cosine_scores(embeddings[impostor_idx], template)
            labels = np.concatenate([
                np.ones(len(genuine_scores), dtype=np.int64),
                np.zeros(len(impostor_scores), dtype=np.int64),
            ])
            scores = np.concatenate([genuine_scores, impostor_scores])
            metrics = base.binary_metrics(labels, scores, threshold)
            local_threshold, local_eer, _ = base.find_eer_threshold(labels, scores)
            rows.append({
                "user": target_user,
                "gesture": gesture_name,
                "enrollment_session": enroll_session,
                "enrollment_samples": len(enroll_idx),
                "genuine_samples": len(genuine_idx),
                "impostor_samples": len(impostor_idx),
                "threshold": threshold,
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "far": metrics["far"],
                "frr": metrics["frr"],
                "test_eer_analysis": local_eer,
                "test_eer_threshold_analysis": local_threshold,
            })
            global_labels.extend(labels.tolist())
            global_scores.extend(scores.tolist())
            print(
                f"  {gesture_name}: Genuine={len(genuine_idx):3d} "
                f"Impostor={len(impostor_idx):3d} | "
                f"BalAcc={metrics['balanced_accuracy']:.3f} "
                f"FAR={metrics['far']:.3f} FRR={metrics['frr']:.3f}"
            )
    labels = np.asarray(global_labels, dtype=np.int64)
    scores = np.asarray(global_scores, dtype=np.float64)
    metrics = base.binary_metrics(labels, scores, threshold)
    test_threshold, test_eer, _ = base.find_eer_threshold(labels, scores)
    return rows, metrics, test_eer, test_threshold


def main():
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default=str(project_dir / "dataset" / "dataset_1955_recent8_updated_20260905.npz"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(project_dir / "output" / "gesture_conditioned_supcon_auth"),
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

    model = GestureConditionedEmbedding1DCNN(
        D, len(TRAIN_USERS), len(gestures), args.embedding_dim
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=1e-4
    )
    parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("=" * 84)
    print("Gesture-conditioned projection + SupCon 1D-CNN")
    print("=" * 84)
    print(f"device={device} N={len(X_seq)} T={T} D={D} parameters={parameter_count:,}")
    print(f"train={len(train_idx)} val={len(val_idx)} heads={len(gestures)}")
    print("date/session usage: split only; not model input, loss, or pair rule")

    best_val_acc = -1.0
    best_epoch = 0
    best_state = None
    stale = 0
    stopped_epoch = args.epochs
    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(
            model, train_loader, criterion, device,
            args.contrastive_weight, args.temperature, optimizer,
        )
        va = run_epoch(
            model, val_loader, criterion, device,
            args.contrastive_weight, args.temperature, optimizer=None,
        )
        print(
            f"Epoch {epoch:02d} | train total={tr[0]:.4f} CE={tr[1]:.4f} "
            f"SupCon={tr[2]:.4f} acc={tr[3]:.3f} | "
            f"val total={va[0]:.4f} CE={va[1]:.4f} "
            f"SupCon={va[2]:.4f} acc={va[3]:.3f}"
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
        raise RuntimeError("No gesture-conditioned checkpoint was created")
    model.load_state_dict(best_state)
    model.eval()

    true_embeddings = extract_conditioned_embeddings(
        model, X_norm, duration_norm, gesture_y, device
    )
    known_idx = np.where(np.isin(meta["performer"], TRAIN_USERS))[0]
    cal_labels, cal_scores = base.build_known_validation_scores(
        true_embeddings[known_idx], known_idx, meta,
        enrollment_per_gesture=args.enroll,
    )
    threshold, val_eer, val_eer_metrics = base.find_eer_threshold(
        cal_labels, cal_scores
    )
    rows, global_metrics, test_eer, test_eer_threshold = evaluate_unseen(
        true_embeddings, meta, threshold, args.enroll, gesture_to_label
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_hash = file_sha256(args.data)
    model_path = output_dir / "embedding_1dcnn_gesture_conditioned_supcon.pt"
    torch.save({
        "model_state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "architecture": "gesture-conditioned-supcon",
        "input_dim": D,
        "seq_len": T,
        "embedding_dim": args.embedding_dim,
        "num_classes": len(TRAIN_USERS),
        "num_gestures": len(gestures),
        "gesture_classes": gestures,
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
        "training_loss": "cross_entropy+gesture_conditioned_supcon",
        "contrastive_weight": args.contrastive_weight,
        "temperature": args.temperature,
        "projection": "one_head_per_gesture",
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
        f.write("Gesture-conditioned projection SupCon 1D-CNN authentication\n")
        f.write("architecture=gesture-conditioned-supcon\n")
        f.write("projection=one_head_per_gesture\n")
        f.write("date_usage=split_only\n")
        f.write(f"dataset={Path(args.data).resolve()}\n")
        f.write(f"dataset_sha256={dataset_hash}\n")
        f.write(f"parameter_count={parameter_count}\n")
        f.write(f"contrastive_weight={args.contrastive_weight}\n")
        f.write(f"temperature={args.temperature}\n")
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
