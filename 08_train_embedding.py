"""
08_train_embedding.py

First embedding-based unseen-user authentication experiment.

Goal
----
Train ONE shared 1D-CNN encoder using known users P01~P07,
then enroll completely unseen users P08~P10 WITHOUT retraining.

Training:
    sequence -> 1D-CNN -> 128-D embedding -> identity classifier

Enrollment:
    unseen user's first-session samples -> embeddings -> mean template

Verification:
    new sample -> embedding
    -> cosine similarity with claimed user's template
    -> accept / reject using threshold calibrated ONLY on known users

Important:
- P08, P09, P10 are NEVER used for encoder training.
- P08, P09, P10 are NEVER used for threshold calibration.
- Verification is gesture-specific.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from two_stage_common import (
    load_dataset,
    fit_duration_stats,
    apply_duration_stats,
    fit_sequence_stats,
    apply_sequence_stats,
)


SEED = 42

TRAIN_USERS = [
    "P01",
    "P02",
    "P03",
    "P04",
    "P05",
    "P06",
    "P07",
]

UNSEEN_USERS = [
    "P08",
    "P09",
    "P10",
]


def seed_everything(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Embedding1DCNN(nn.Module):
    """
    Same general CNN backbone as the existing Small1DCNN,
    but produces a normalized embedding.

    During training, a classification head predicts known-user ID.
    During verification, only the embedding is used.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        embedding_dim: int = 128,
    ):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.20),

            nn.Conv1d(64, 96, kernel_size=3, padding=1),
            nn.BatchNorm1d(96),
            nn.ReLU(),

            nn.Conv1d(
                96,
                128,
                kernel_size=3,
                padding=2,
                dilation=2,
            ),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.30),

            nn.AdaptiveAvgPool1d(1),
        )

        self.embedding_head = nn.Linear(128 + 1, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x, duration):
        h = self.features(x.transpose(1, 2)).squeeze(-1)

        h = torch.cat(
            [h, duration.unsqueeze(1)],
            dim=1,
        )

        embedding = self.embedding_head(h)
        embedding = F.normalize(
            embedding,
            p=2,
            dim=1,
        )

        logits = self.classifier(embedding)

        return embedding, logits


def chronological_user_split(meta, users):
    """
    Known-user training split.

    For each person:
      latest recording session -> validation
      all earlier sessions      -> training

    This avoids putting samples from exactly the same recording session
    into both train and validation.
    """

    performer = meta["performer"]
    session = meta["session"]

    train_idx = []
    val_idx = []

    for user in users:
        idx = np.where(performer == user)[0]

        sessions = sorted(np.unique(session[idx]))

        if len(sessions) < 2:
            raise RuntimeError(
                f"{user} has fewer than two sessions."
            )

        val_session = sessions[-1]

        user_val = idx[session[idx] == val_session]
        user_train = idx[session[idx] != val_session]

        train_idx.extend(user_train.tolist())
        val_idx.extend(user_val.tolist())

        print(
            f"{user}: "
            f"train={len(user_train):3d}, "
            f"val={len(user_val):3d}, "
            f"val_session={val_session}"
        )

    return (
        np.asarray(train_idx, dtype=np.int64),
        np.asarray(val_idx, dtype=np.int64),
    )


def make_loader(
    X,
    duration,
    y,
    batch_size,
    shuffle,
):
    dataset = TensorDataset(
        torch.from_numpy(X).float(),
        torch.from_numpy(duration).float(),
        torch.from_numpy(y).long(),
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )


def run_epoch(
    model,
    loader,
    criterion,
    device,
    optimizer=None,
):
    training = optimizer is not None

    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_n = 0

    context = torch.enable_grad() if training else torch.no_grad()

    with context:
        for X, duration, y in loader:
            X = X.to(device)
            duration = duration.to(device)
            y = y.to(device)

            if training:
                optimizer.zero_grad()

            _, logits = model(X, duration)

            loss = criterion(logits, y)

            if training:
                loss.backward()
                optimizer.step()

            pred = logits.argmax(dim=1)

            total_loss += loss.item() * len(y)
            total_correct += int((pred == y).sum().item())
            total_n += len(y)

    return (
        total_loss / max(total_n, 1),
        total_correct / max(total_n, 1),
    )


@torch.no_grad()
def extract_embeddings(
    model,
    X,
    duration,
    device,
    batch_size=128,
):
    model.eval()

    dummy_y = np.zeros(len(X), dtype=np.int64)

    loader = make_loader(
        X,
        duration,
        dummy_y,
        batch_size,
        shuffle=False,
    )

    result = []

    for xb, db, _ in loader:
        xb = xb.to(device)
        db = db.to(device)

        emb, _ = model(xb, db)
        result.append(emb.cpu().numpy())

    return np.concatenate(result, axis=0)


def cosine_scores(embeddings, template):
    template = template / (
        np.linalg.norm(template) + 1e-12
    )

    embeddings = embeddings / (
        np.linalg.norm(
            embeddings,
            axis=1,
            keepdims=True,
        )
        + 1e-12
    )

    return embeddings @ template


def binary_metrics(labels, scores, threshold):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)

    pred = (scores >= threshold).astype(np.int64)

    genuine = labels == 1
    impostor = labels == 0

    tp = int(np.sum((pred == 1) & genuine))
    fn = int(np.sum((pred == 0) & genuine))
    fp = int(np.sum((pred == 1) & impostor))
    tn = int(np.sum((pred == 0) & impostor))

    far = fp / max(int(impostor.sum()), 1)
    frr = fn / max(int(genuine.sum()), 1)

    accuracy = (
        (tp + tn) / max(len(labels), 1)
    )

    balanced_accuracy = (
        (1.0 - far) + (1.0 - frr)
    ) / 2.0

    return {
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "far": float(far),
        "frr": float(frr),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def find_eer_threshold(labels, scores):
    """
    Find threshold where FAR and FRR are closest.

    This is used ONLY with validation data
    from known training identities.
    """

    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)

    thresholds = np.unique(scores)

    best = None

    for threshold in thresholds:
        m = binary_metrics(
            labels,
            scores,
            threshold,
        )

        gap = abs(m["far"] - m["frr"])
        eer = (m["far"] + m["frr"]) / 2.0

        candidate = (
            gap,
            eer,
            float(threshold),
            m,
        )

        if best is None or candidate[:2] < best[:2]:
            best = candidate

    _, eer, threshold, metrics = best

    return threshold, eer, metrics


def build_known_validation_scores(
    embeddings,
    indices,
    meta,
    enrollment_per_gesture=3,
):
    """
    Threshold calibration using KNOWN users only,
    but with cross-session enrollment / verification.

    For each known person:
      second-latest session -> enrollment
      latest session        -> genuine verification

    Other known users performing the same gesture
    in their latest session are used as impostors.
    """

    performer = meta["performer"]
    gesture = meta["gesture"]
    session = meta["session"]

    labels = []
    scores = []

    index_to_pos = {
        int(idx): pos
        for pos, idx in enumerate(indices)
    }

    gestures = sorted(np.unique(gesture[indices]))

    for target_user in TRAIN_USERS:
        user_idx = indices[
            performer[indices] == target_user
        ]

        user_sessions = sorted(
            np.unique(session[user_idx])
        )

        if len(user_sessions) < 2:
            continue

        enroll_session = user_sessions[-2]
        test_session = user_sessions[-1]

        for g in gestures:
            enroll_candidates = indices[
                (performer[indices] == target_user)
                & (gesture[indices] == g)
                & (session[indices] == enroll_session)
            ]

            genuine = indices[
                (performer[indices] == target_user)
                & (gesture[indices] == g)
                & (session[indices] == test_session)
            ]

            if len(enroll_candidates) < enrollment_per_gesture:
                continue

            if len(genuine) == 0:
                continue

            enroll = enroll_candidates[
                :enrollment_per_gesture
            ]

            template_embeddings = np.stack(
                [
                    embeddings[index_to_pos[int(i)]]
                    for i in enroll
                ]
            )

            template = template_embeddings.mean(axis=0)

            genuine_embeddings = np.stack(
                [
                    embeddings[index_to_pos[int(i)]]
                    for i in genuine
                ]
            )

            genuine_scores = cosine_scores(
                genuine_embeddings,
                template,
            )

            labels.extend(
                [1] * len(genuine_scores)
            )
            scores.extend(
                genuine_scores.tolist()
            )

            impostor = []

            for other_user in TRAIN_USERS:
                if other_user == target_user:
                    continue

                other_idx = indices[
                    performer[indices] == other_user
                ]

                other_sessions = sorted(
                    np.unique(session[other_idx])
                )

                if len(other_sessions) < 1:
                    continue

                other_test_session = other_sessions[-1]

                imp = indices[
                    (performer[indices] == other_user)
                    & (gesture[indices] == g)
                    & (session[indices] == other_test_session)
                ]

                impostor.extend(
                    imp.tolist()
                )

            if len(impostor):
                impostor = np.asarray(
                    impostor,
                    dtype=np.int64,
                )

                impostor_embeddings = np.stack(
                    [
                        embeddings[index_to_pos[int(i)]]
                        for i in impostor
                    ]
                )

                imp_scores = cosine_scores(
                    impostor_embeddings,
                    template,
                )

                labels.extend(
                    [0] * len(imp_scores)
                )
                scores.extend(
                    imp_scores.tolist()
                )

    return (
        np.asarray(labels, dtype=np.int64),
        np.asarray(scores, dtype=np.float64),
    )


def evaluate_unseen_users(
    model,
    X_all,
    duration_all,
    meta,
    device,
    threshold,
    enrollment_per_gesture,
):
    """
    P08/P09/P10 are completely unseen during training.

    For each unseen target user:
      earliest session = enrollment candidate session
      enrollment_per_gesture samples are used for template creation

    All later sessions of the same person = genuine.
    Other unseen people performing the same gesture = impostors.
    """

    performer = meta["performer"]
    gesture = meta["gesture"]
    session = meta["session"]

    rows = []
    global_labels = []
    global_scores = []

    gestures = sorted(np.unique(gesture))

    for target_user in UNSEEN_USERS:
        user_idx = np.where(
            performer == target_user
        )[0]

        user_sessions = sorted(
            np.unique(session[user_idx])
        )

        enroll_session = user_sessions[0]

        print()
        print(
            f"[UNSEEN] {target_user} "
            f"enrollment_session={enroll_session}"
        )

        for g in gestures:
            enrollment_candidates = np.where(
                (performer == target_user)
                & (gesture == g)
                & (session == enroll_session)
            )[0]

            if len(enrollment_candidates) < enrollment_per_gesture:
                print(
                    f"  {g}: SKIP "
                    f"(only {len(enrollment_candidates)} "
                    f"enrollment samples)"
                )
                continue

            enroll_idx = enrollment_candidates[
                :enrollment_per_gesture
            ]

            genuine_idx = np.where(
                (performer == target_user)
                & (gesture == g)
                & (session != enroll_session)
            )[0]

            impostor_idx = np.where(
                np.isin(performer, UNSEEN_USERS)
                & (performer != target_user)
                & (gesture == g)
            )[0]

            enroll_emb = extract_embeddings(
                model,
                X_all[enroll_idx],
                duration_all[enroll_idx],
                device,
            )

            template = enroll_emb.mean(axis=0)

            genuine_emb = extract_embeddings(
                model,
                X_all[genuine_idx],
                duration_all[genuine_idx],
                device,
            )

            impostor_emb = extract_embeddings(
                model,
                X_all[impostor_idx],
                duration_all[impostor_idx],
                device,
            )

            genuine_scores = cosine_scores(
                genuine_emb,
                template,
            )

            impostor_scores = cosine_scores(
                impostor_emb,
                template,
            )

            labels = np.concatenate(
                [
                    np.ones(
                        len(genuine_scores),
                        dtype=np.int64,
                    ),
                    np.zeros(
                        len(impostor_scores),
                        dtype=np.int64,
                    ),
                ]
            )

            scores = np.concatenate(
                [
                    genuine_scores,
                    impostor_scores,
                ]
            )

            metrics = binary_metrics(
                labels,
                scores,
                threshold,
            )

            local_threshold, local_eer, _ = (
                find_eer_threshold(
                    labels,
                    scores,
                )
            )

            row = {
                "user": target_user,
                "gesture": g,
                "enrollment_session": enroll_session,
                "enrollment_samples": len(enroll_idx),
                "genuine_samples": len(genuine_idx),
                "impostor_samples": len(impostor_idx),
                "threshold": threshold,
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics[
                    "balanced_accuracy"
                ],
                "far": metrics["far"],
                "frr": metrics["frr"],
                "test_eer_analysis": local_eer,
                "test_eer_threshold_analysis": local_threshold,
            }

            rows.append(row)

            global_labels.extend(labels.tolist())
            global_scores.extend(scores.tolist())

            print(
                f"  {g}: "
                f"Genuine={len(genuine_idx):3d} "
                f"Impostor={len(impostor_idx):3d} | "
                f"BalAcc={metrics['balanced_accuracy']:.3f} "
                f"FAR={metrics['far']:.3f} "
                f"FRR={metrics['frr']:.3f}"
            )

    global_labels = np.asarray(
        global_labels,
        dtype=np.int64,
    )
    global_scores = np.asarray(
        global_scores,
        dtype=np.float64,
    )

    global_metrics = binary_metrics(
        global_labels,
        global_scores,
        threshold,
    )

    test_eer_threshold, test_eer, _ = (
        find_eer_threshold(
            global_labels,
            global_scores,
        )
    )

    return (
        rows,
        global_metrics,
        test_eer,
        test_eer_threshold,
    )


def main():
    parser = argparse.ArgumentParser()

    default_data = (
        Path(os.environ["WITECH_ROOT"])
        / "derived"
        / "dataset.npz"
    )

    parser.add_argument(
        "--data",
        type=str,
        default=str(default_data),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--enroll",
        type=int,
        default=3,
        help="Enrollment samples per gesture",
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=8,
    )

    args = parser.parse_args()

    seed_everything()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)
    print("Embedding-based unseen-user authentication")
    print("=" * 70)
    print("device       :", device)
    print("dataset      :", args.data)
    print("train users  :", TRAIN_USERS)
    print("unseen users :", UNSEEN_USERS)
    print("embedding dim:", args.embedding_dim)
    print("enroll / G   :", args.enroll)
    print()

    (
        _,
        X_seq,
        duration,
        meta,
        T,
        D,
    ) = load_dataset(args.data)

    print(
        f"Dataset: N={len(X_seq)}, "
        f"T={T}, D={D}"
    )

    train_idx, val_idx = (
        chronological_user_split(
            meta,
            TRAIN_USERS,
        )
    )

    print()
    print(
        f"Known-user split: "
        f"train={len(train_idx)}, "
        f"val={len(val_idx)}"
    )

    # ------------------------------------------------------------
    # Fit normalization using TRAIN only
    # ------------------------------------------------------------

    seq_mean, seq_std = fit_sequence_stats(
        X_seq[train_idx]
    )

    dur_mean, dur_std = fit_duration_stats(
        duration[train_idx]
    )

    X_norm = apply_sequence_stats(
        X_seq,
        seq_mean,
        seq_std,
    )

    duration_norm = apply_duration_stats(
        duration,
        dur_mean,
        dur_std,
    )

    # ------------------------------------------------------------
    # Identity labels for known users
    # ------------------------------------------------------------

    user_to_label = {
        user: i
        for i, user in enumerate(TRAIN_USERS)
    }

    y_all = np.full(
        len(X_seq),
        -1,
        dtype=np.int64,
    )

    for user, label in user_to_label.items():
        y_all[
            meta["performer"] == user
        ] = label

    train_loader = make_loader(
        X_norm[train_idx],
        duration_norm[train_idx],
        y_all[train_idx],
        args.batch_size,
        shuffle=True,
    )

    val_loader = make_loader(
        X_norm[val_idx],
        duration_norm[val_idx],
        y_all[val_idx],
        args.batch_size,
        shuffle=False,
    )

    model = Embedding1DCNN(
        input_dim=D,
        num_classes=len(TRAIN_USERS),
        embedding_dim=args.embedding_dim,
    ).to(device)

    n_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Trainable parameters: {n_params:,}"
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )

    best_val_acc = -1.0
    best_state = None
    patience_count = 0

    print()
    print("Training encoder...")

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        train_loss, train_acc = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
        )

        val_loss, val_acc = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            optimizer=None,
        )

        print(
            f"Epoch {epoch:02d} | "
            f"train loss={train_loss:.4f} "
            f"acc={train_acc:.3f} | "
            f"val loss={val_loss:.4f} "
            f"acc={val_acc:.3f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc

            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

            patience_count = 0
        else:
            patience_count += 1

        if patience_count >= args.patience:
            print(
                f"Early stopping at epoch {epoch}"
            )
            break

    if best_state is None:
        raise RuntimeError(
            "No model checkpoint was created."
        )

    model.load_state_dict(best_state)
    model.to(device)

    print()
    print(
        f"Best validation ID accuracy: "
        f"{best_val_acc:.3f}"
    )

    # ------------------------------------------------------------
    # Calibrate verification threshold using KNOWN USERS ONLY
    # ------------------------------------------------------------

    print()
    print(
        "Calibrating verification threshold "
        "using known-user validation data..."
    )

    known_idx = np.where(
    np.isin(
        meta["performer"],
        TRAIN_USERS,
    )
)[0]

    known_embeddings = extract_embeddings(
        model,
        X_norm[known_idx],
        duration_norm[known_idx],
        device,
    )

    cal_labels, cal_scores = (
        build_known_validation_scores(
            known_embeddings,
            known_idx,
            meta,
            enrollment_per_gesture=args.enroll,
        )
    )

    if len(cal_labels) == 0:
        raise RuntimeError(
            "Threshold calibration produced no scores."
        )

    threshold, val_eer, val_eer_metrics = (
        find_eer_threshold(
            cal_labels,
            cal_scores,
        )
    )

    print(
        f"Validation threshold : {threshold:.6f}"
    )
    print(
        f"Validation EER       : {val_eer:.3f}"
    )
    print(
        f"Validation FAR       : "
        f"{val_eer_metrics['far']:.3f}"
    )
    print(
        f"Validation FRR       : "
        f"{val_eer_metrics['frr']:.3f}"
    )

    # ------------------------------------------------------------
    # Evaluate completely unseen identities
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print("UNSEEN USER EVALUATION")
    print("=" * 70)

    (
        rows,
        global_metrics,
        test_eer,
        test_eer_threshold,
    ) = evaluate_unseen_users(
        model,
        X_norm,
        duration_norm,
        meta,
        device,
        threshold,
        args.enroll,
    )

    print()
    print("=" * 70)
    print("FINAL UNSEEN-USER RESULT")
    print("=" * 70)

    print(
        f"Fixed threshold     : {threshold:.6f}"
    )
    print(
        f"Accuracy            : "
        f"{global_metrics['accuracy']:.3f}"
    )
    print(
        f"Balanced Accuracy   : "
        f"{global_metrics['balanced_accuracy']:.3f}"
    )
    print(
        f"FAR                 : "
        f"{global_metrics['far']:.3f}"
    )
    print(
        f"FRR                 : "
        f"{global_metrics['frr']:.3f}"
    )

    # Test EER is reported only as analysis.
    # It must NOT replace the validation-derived deployment threshold.
    print(
        f"Test EER (analysis) : "
        f"{test_eer:.3f}"
    )
    print(
        f"Test EER threshold  : "
        f"{test_eer_threshold:.6f}"
    )

    # ------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------

    output_dir = (
        Path(os.environ["WITECH_ROOT"])
        / "derived"
        / "embedding_experiment"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        output_dir
        / "embedding_1dcnn.pt"
    )

    torch.save(
        {
            "model_state_dict": {
                k: v.detach().cpu()
                for k, v in model.state_dict().items()
            },
            "input_dim": D,
            "embedding_dim": args.embedding_dim,
            "train_users": TRAIN_USERS,
            "unseen_users": UNSEEN_USERS,
            "threshold": threshold,
            "sequence_mean": seq_mean,
            "sequence_std": seq_std,
            "duration_mean": dur_mean,
            "duration_std": dur_std,
            "best_val_id_accuracy": best_val_acc,
        },
        model_path,
    )

    csv_path = (
        output_dir
        / "unseen_user_results.csv"
    )

    if rows:
        with open(
            csv_path,
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=list(
                    rows[0].keys()
                ),
            )

            writer.writeheader()
            writer.writerows(rows)

    summary_path = (
        output_dir
        / "summary.txt"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "Embedding-based unseen-user authentication\n"
        )
        f.write(
            f"train_users={TRAIN_USERS}\n"
        )
        f.write(
            f"unseen_users={UNSEEN_USERS}\n"
        )
        f.write(
            f"enrollment_per_gesture={args.enroll}\n"
        )
        f.write(
            f"best_val_id_accuracy={best_val_acc:.6f}\n"
        )
        f.write(
            f"validation_threshold={threshold:.6f}\n"
        )
        f.write(
            f"validation_eer={val_eer:.6f}\n"
        )
        f.write(
            f"unseen_accuracy="
            f"{global_metrics['accuracy']:.6f}\n"
        )
        f.write(
            f"unseen_balanced_accuracy="
            f"{global_metrics['balanced_accuracy']:.6f}\n"
        )
        f.write(
            f"unseen_far="
            f"{global_metrics['far']:.6f}\n"
        )
        f.write(
            f"unseen_frr="
            f"{global_metrics['frr']:.6f}\n"
        )
        f.write(
            f"unseen_test_eer_analysis="
            f"{test_eer:.6f}\n"
        )

    print()
    print("Saved:")
    print(" model  :", model_path)
    print(" csv    :", csv_path)
    print(" summary:", summary_path)


if __name__ == "__main__":
    main()