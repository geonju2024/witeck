"""Train a basic 1D-CNN with gesture-conditioned supervised contrastive loss.

Session/date metadata is used only for data splitting and enrollment/test
separation. It is not a model input and is not used to define training pairs.
"""

from __future__ import annotations

import argparse
import copy
import csv
import importlib
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from two_stage_common import (
    apply_duration_stats,
    apply_sequence_stats,
    file_sha256,
    fit_duration_stats,
    fit_sequence_stats,
    load_dataset,
)


base = importlib.import_module("08_train_embedding")
Embedding1DCNN = base.Embedding1DCNN
TRAIN_USERS = base.TRAIN_USERS
UNSEEN_USERS = base.UNSEEN_USERS


def build_model(
    input_dim,
    num_classes,
    embedding_dim,
    sequence_mean=None,
    sequence_std=None,
):
    """Construction hook used by dataset-specific model wrappers."""
    return Embedding1DCNN(input_dim, num_classes, embedding_dim)


def make_loader(X, duration, user_y, gesture_y, batch_size, shuffle):
    dataset = TensorDataset(
        torch.from_numpy(X).float(),
        torch.from_numpy(duration).float(),
        torch.from_numpy(user_y).long(),
        torch.from_numpy(gesture_y).long(),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )


class BalancedUserGestureBatchSampler:
    """Put at least two examples of every user-gesture group in each batch."""

    def __init__(self, user_y, gesture_y, samples_per_group=2, seed=42):
        self.groups = []
        for user in sorted(np.unique(user_y).tolist()):
            for gesture in sorted(np.unique(gesture_y).tolist()):
                indices = np.where((user_y == user) & (gesture_y == gesture))[0]
                if len(indices):
                    self.groups.append(indices)
        self.samples_per_group = int(samples_per_group)
        self.batch_size = len(self.groups) * self.samples_per_group
        self.num_batches = max(1, math.ceil(len(user_y) / self.batch_size))
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        for _ in range(self.num_batches):
            batch = []
            for indices in self.groups:
                replace = len(indices) < self.samples_per_group
                chosen = rng.choice(
                    indices, size=self.samples_per_group, replace=replace
                )
                batch.extend(chosen.tolist())
            rng.shuffle(batch)
            yield batch


def make_balanced_loader(
    X, duration, user_y, gesture_y, samples_per_group, seed
):
    dataset = TensorDataset(
        torch.from_numpy(X).float(),
        torch.from_numpy(duration).float(),
        torch.from_numpy(user_y).long(),
        torch.from_numpy(gesture_y).long(),
    )
    sampler = BalancedUserGestureBatchSampler(
        user_y,
        gesture_y,
        samples_per_group=samples_per_group,
        seed=seed,
    )
    return DataLoader(dataset, batch_sampler=sampler, num_workers=0)


def gesture_conditioned_supcon_loss(
    embedding,
    user_labels,
    gesture_labels,
    temperature=0.10,
):
    """Contrast identities only against samples of the same gesture.

    Positive: same user and same gesture.
    Negative: different user and same gesture.
    Anchors without a positive partner in the batch are ignored.
    """
    n = embedding.shape[0]
    if n < 2:
        return embedding.sum() * 0.0

    similarity = embedding @ embedding.T / temperature
    eye = torch.eye(n, dtype=torch.bool, device=embedding.device)
    same_user = user_labels[:, None].eq(user_labels[None, :])
    same_gesture = gesture_labels[:, None].eq(gesture_labels[None, :])
    eligible = same_gesture & ~eye
    positive = same_user & same_gesture & ~eye
    positive_count = positive.sum(dim=1)
    valid_anchor = positive_count > 0
    if not torch.any(valid_anchor):
        return embedding.sum() * 0.0

    masked_similarity = similarity.masked_fill(~eligible, float("-inf"))
    log_denominator = torch.logsumexp(masked_similarity, dim=1)
    positive_mean = (
        similarity.masked_fill(~positive, 0.0).sum(dim=1)
        / positive_count.clamp_min(1)
    )
    loss = log_denominator - positive_mean
    return loss[valid_anchor].mean()


def run_epoch(
    model,
    loader,
    ce_criterion,
    device,
    contrastive_weight,
    temperature,
    optimizer=None,
):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_ce = 0.0
    total_supcon = 0.0
    total_correct = 0
    total_n = 0
    context = torch.enable_grad() if training else torch.no_grad()

    with context:
        for xb, db, user_y, gesture_y in loader:
            xb = xb.to(device)
            db = db.to(device)
            user_y = user_y.to(device)
            gesture_y = gesture_y.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)

            embedding, logits = model(xb, db)
            ce_loss = ce_criterion(logits, user_y)
            supcon_loss = gesture_conditioned_supcon_loss(
                embedding,
                user_y,
                gesture_y,
                temperature,
            )
            loss = ce_loss + contrastive_weight * supcon_loss
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

            total_loss += float(loss.item()) * len(user_y)
            total_ce += float(ce_loss.item()) * len(user_y)
            total_supcon += float(supcon_loss.item()) * len(user_y)
            total_correct += int((logits.argmax(dim=1) == user_y).sum().item())
            total_n += len(user_y)

    denom = max(total_n, 1)
    return (
        total_loss / denom,
        total_ce / denom,
        total_supcon / denom,
        total_correct / denom,
    )


def main():
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default=str(project_dir / "dataset" / "dataset_1955_recent8_updated_20260905.npz"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(project_dir / "output" / "supcon_embedding_auth"),
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
    parser.add_argument("--balanced-user-gesture-batches", action="store_true")
    parser.add_argument("--samples-per-group", type=int, default=2)
    parser.add_argument(
        "--selection-metric",
        choices=("id-accuracy", "auth-eer"),
        default="id-accuracy",
    )
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
        [gesture_to_label[x] for x in meta["gesture"]],
        dtype=np.int64,
    )

    if args.balanced_user_gesture_batches:
        train_loader = make_balanced_loader(
            X_norm[train_idx],
            duration_norm[train_idx],
            user_y[train_idx],
            gesture_y[train_idx],
            args.samples_per_group,
            args.seed,
        )
    else:
        train_loader = make_loader(
            X_norm[train_idx],
            duration_norm[train_idx],
            user_y[train_idx],
            gesture_y[train_idx],
            args.batch_size,
            shuffle=True,
        )
    val_loader = make_loader(
        X_norm[val_idx],
        duration_norm[val_idx],
        user_y[val_idx],
        gesture_y[val_idx],
        args.batch_size,
        shuffle=False,
    )

    model = build_model(
        D,
        len(TRAIN_USERS),
        args.embedding_dim,
        sequence_mean=seq_mean,
        sequence_std=seq_std,
    ).to(device)
    ce_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )

    print("=" * 78)
    print("Gesture-conditioned SupCon basic 1D-CNN user embedding")
    print("=" * 78)
    print(f"device={device} N={len(X_seq)} T={T} D={D}")
    print(f"train={len(train_idx)} val={len(val_idx)}")
    print(
        "batch_sampling="
        + (
            f"balanced_user_gesture({args.samples_per_group}_per_group)"
            if args.balanced_user_gesture_batches
            else "random"
        )
    )
    print(
        f"loss=CE + {args.contrastive_weight}*SupCon, "
        f"temperature={args.temperature}"
    )
    print("date/session usage: split only; not model input, loss, or pair rule")

    known_idx = np.where(np.isin(meta["performer"], TRAIN_USERS))[0]
    best_selection_value = -float("inf")
    best_val_acc = -1.0
    best_val_auth_eer = float("nan")
    best_epoch = 0
    best_state = None
    stale = 0
    stopped_epoch = args.epochs
    for epoch in range(1, args.epochs + 1):
        train_loss, train_ce, train_supcon, train_acc = run_epoch(
            model,
            train_loader,
            ce_criterion,
            device,
            args.contrastive_weight,
            args.temperature,
            optimizer,
        )
        val_loss, val_ce, val_supcon, val_acc = run_epoch(
            model,
            val_loader,
            ce_criterion,
            device,
            args.contrastive_weight,
            args.temperature,
            optimizer=None,
        )
        val_auth_eer = float("nan")
        if args.selection_metric == "auth-eer":
            epoch_embeddings = base.extract_embeddings(
                model,
                X_norm[known_idx],
                duration_norm[known_idx],
                device,
            )
            epoch_labels, epoch_scores = base.build_known_validation_scores(
                epoch_embeddings,
                known_idx,
                meta,
                enrollment_per_gesture=args.enroll,
            )
            _, val_auth_eer, _ = base.find_eer_threshold(
                epoch_labels,
                epoch_scores,
            )
            selection_value = -val_auth_eer
        else:
            selection_value = val_acc
        print(
            f"Epoch {epoch:02d} | "
            f"train total={train_loss:.4f} CE={train_ce:.4f} "
            f"SupCon={train_supcon:.4f} acc={train_acc:.3f} | "
            f"val total={val_loss:.4f} CE={val_ce:.4f} "
            f"SupCon={val_supcon:.4f} acc={val_acc:.3f}"
            + (
                f" auth_EER={val_auth_eer:.4f}"
                if args.selection_metric == "auth-eer"
                else ""
            )
        )
        if selection_value > best_selection_value:
            best_selection_value = selection_value
            best_val_acc = val_acc
            best_val_auth_eer = val_auth_eer
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
        raise RuntimeError("No SupCon checkpoint was created")
    model.load_state_dict(best_state)
    model.eval()

    known_embeddings = base.extract_embeddings(
        model,
        X_norm[known_idx],
        duration_norm[known_idx],
        device,
    )
    cal_labels, cal_scores = base.build_known_validation_scores(
        known_embeddings,
        known_idx,
        meta,
        enrollment_per_gesture=args.enroll,
    )
    threshold, val_eer, val_eer_metrics = base.find_eer_threshold(
        cal_labels,
        cal_scores,
    )
    rows, global_metrics, test_eer, test_eer_threshold = base.evaluate_unseen_users(
        model,
        X_norm,
        duration_norm,
        meta,
        device,
        threshold,
        args.enroll,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_hash = file_sha256(args.data)
    model_path = output_dir / "embedding_1dcnn_supcon.pt"
    torch.save(
        {
            "model_state_dict": {
                key: value.detach().cpu()
                for key, value in model.state_dict().items()
            },
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
            "positive_rule": "same_user_and_same_gesture",
            "eligible_comparison_rule": "same_gesture",
            "date_usage": "split_only",
            "selection_metric": args.selection_metric,
            "best_val_auth_eer": best_val_auth_eer,
            "batch_sampling": (
                "balanced_user_gesture"
                if args.balanced_user_gesture_batches
                else "random"
            ),
            "samples_per_group": args.samples_per_group,
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
        f.write("Gesture-conditioned SupCon basic 1D-CNN authentication\n")
        f.write("date_usage=split_only\n")
        f.write("positive_rule=same_user_and_same_gesture\n")
        f.write("eligible_comparison_rule=same_gesture\n")
        f.write(
            "batch_sampling="
            + (
                "balanced_user_gesture"
                if args.balanced_user_gesture_batches
                else "random"
            )
            + "\n"
        )
        f.write(f"samples_per_group={args.samples_per_group}\n")
        f.write(f"selection_metric={args.selection_metric}\n")
        if np.isfinite(best_val_auth_eer):
            f.write(f"best_val_auth_eer={best_val_auth_eer:.6f}\n")
        f.write(f"dataset={Path(args.data).resolve()}\n")
        f.write(f"dataset_sha256={dataset_hash}\n")
        f.write(f"seed={args.seed}\n")
        f.write(f"contrastive_weight={args.contrastive_weight}\n")
        f.write(f"temperature={args.temperature}\n")
        f.write(f"embedding_dim={args.embedding_dim}\n")
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

    print("=" * 78)
    print(f"best val identity accuracy={best_val_acc:.4f} at epoch {best_epoch}")
    print(f"validation threshold={threshold:.6f} EER={val_eer:.4f}")
    print(
        f"unseen Acc={global_metrics['accuracy']:.4f} "
        f"BalAcc={global_metrics['balanced_accuracy']:.4f} "
        f"FAR={global_metrics['far']:.4f} "
        f"FRR={global_metrics['frr']:.4f} "
        f"test EER={test_eer:.4f}"
    )
    print(f"Saved: {model_path}")


if __name__ == "__main__":
    main()
