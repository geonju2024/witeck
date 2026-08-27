"""
09_train_siamese_embedding.py

Siamese / contrastive embedding experiment for unseen-user authentication.

Goal
----
Train ONE shared 1D-CNN encoder using P01~P07.

Positive pair:
    same performer + same gesture

Negative pair:
    different performer + same gesture

P08/P09/P10 are completely excluded from encoder training.

After training:
    unseen user's enrollment samples
        -> embeddings
        -> mean template

    verification sample
        -> embedding
        -> cosine similarity with template
        -> accept / reject

Comparison baseline:
    08_train_embedding.py
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
from torch.utils.data import Dataset, DataLoader

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


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Siamese1DCNN(nn.Module):
    """
    1D-CNN encoder.

    Input:
        (batch, T, D)

    Output:
        normalized embedding
    """

    def __init__(
        self,
        input_dim,
        embedding_dim=128,
    ):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(
                input_dim,
                64,
                kernel_size=5,
                padding=2,
            ),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.20),

            nn.Conv1d(
                64,
                96,
                kernel_size=3,
                padding=1,
            ),
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

        self.embedding_head = nn.Linear(
            128 + 1,
            embedding_dim,
        )

    def forward(
        self,
        x,
        duration,
    ):
        h = self.features(
            x.transpose(1, 2)
        ).squeeze(-1)

        h = torch.cat(
            [
                h,
                duration.unsqueeze(1),
            ],
            dim=1,
        )

        z = self.embedding_head(h)

        z = F.normalize(
            z,
            p=2,
            dim=1,
        )

        return z


def chronological_split(
    meta,
    users,
):
    """
    For each known user:

        latest session -> validation
        all previous sessions -> training
    """

    performer = meta["performer"]
    session = meta["session"]

    train_idx = []
    val_idx = []

    for user in users:
        idx = np.where(
            performer == user
        )[0]

        sessions = sorted(
            np.unique(
                session[idx]
            )
        )

        if len(sessions) < 2:
            raise RuntimeError(
                f"{user}: fewer than 2 sessions"
            )

        val_session = sessions[-1]

        tr = idx[
            session[idx] != val_session
        ]

        va = idx[
            session[idx] == val_session
        ]

        train_idx.extend(
            tr.tolist()
        )

        val_idx.extend(
            va.tolist()
        )

        print(
            f"{user}: "
            f"train={len(tr):3d}, "
            f"val={len(va):3d}, "
            f"val_session={val_session}"
        )

    return (
        np.asarray(
            train_idx,
            dtype=np.int64,
        ),
        np.asarray(
            val_idx,
            dtype=np.int64,
        ),
    )


def build_pairs(
    indices,
    meta,
    num_pairs,
    rng,
):
    """
    Build balanced Siamese pairs.

    Positive:
        same performer + same gesture

    Negative:
        different performer + same gesture
    """

    performer = meta["performer"]
    gesture = meta["gesture"]

    by_user_gesture = {}

    for user in TRAIN_USERS:
        for g in sorted(
            np.unique(gesture[indices])
        ):
            idx = indices[
                (performer[indices] == user)
                & (gesture[indices] == g)
            ]

            if len(idx) > 0:
                by_user_gesture[
                    (user, g)
                ] = idx

    positive_keys = [
        key
        for key, idx
        in by_user_gesture.items()
        if len(idx) >= 2
    ]

    gestures = sorted(
        np.unique(
            gesture[indices]
        )
    )

    pair_a = []
    pair_b = []
    targets = []

    half = num_pairs // 2

    # Positive pairs
    for _ in range(half):
        user, g = (
            positive_keys[
                rng.integers(
                    len(positive_keys)
                )
            ]
        )

        candidates = (
            by_user_gesture[
                (user, g)
            ]
        )

        chosen = rng.choice(
            candidates,
            size=2,
            replace=False,
        )

        pair_a.append(
            int(chosen[0])
        )

        pair_b.append(
            int(chosen[1])
        )

        targets.append(1.0)

    # Negative pairs
    for _ in range(
        num_pairs - half
    ):
        g = gestures[
            rng.integers(
                len(gestures)
            )
        ]

        valid_users = [
            u
            for u in TRAIN_USERS
            if (u, g)
            in by_user_gesture
        ]

        if len(valid_users) < 2:
            continue

        u1, u2 = rng.choice(
            valid_users,
            size=2,
            replace=False,
        )

        i1 = rng.choice(
            by_user_gesture[
                (u1, g)
            ]
        )

        i2 = rng.choice(
            by_user_gesture[
                (u2, g)
            ]
        )

        pair_a.append(
            int(i1)
        )

        pair_b.append(
            int(i2)
        )

        targets.append(-1.0)

    return (
        np.asarray(
            pair_a,
            dtype=np.int64,
        ),
        np.asarray(
            pair_b,
            dtype=np.int64,
        ),
        np.asarray(
            targets,
            dtype=np.float32,
        ),
    )


class PairDataset(Dataset):

    def __init__(
        self,
        X,
        duration,
        pair_a,
        pair_b,
        targets,
    ):
        self.X = X
        self.duration = duration

        self.pair_a = pair_a
        self.pair_b = pair_b
        self.targets = targets

    def __len__(self):
        return len(
            self.targets
        )

    def __getitem__(
        self,
        idx,
    ):
        a = self.pair_a[idx]
        b = self.pair_b[idx]

        return (
            torch.from_numpy(
                self.X[a]
            ).float(),
            torch.tensor(
                self.duration[a],
                dtype=torch.float32,
            ),

            torch.from_numpy(
                self.X[b]
            ).float(),
            torch.tensor(
                self.duration[b],
                dtype=torch.float32,
            ),

            torch.tensor(
                self.targets[idx],
                dtype=torch.float32,
            ),
        )


def run_pair_epoch(
    model,
    loader,
    criterion,
    device,
    optimizer=None,
):
    training = (
        optimizer is not None
    )

    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_n = 0

    context = (
        torch.enable_grad()
        if training
        else torch.no_grad()
    )

    with context:
        for (
            xa,
            da,
            xb,
            db,
            target,
        ) in loader:

            xa = xa.to(device)
            da = da.to(device)

            xb = xb.to(device)
            db = db.to(device)

            target = target.to(
                device
            )

            if training:
                optimizer.zero_grad()

            za = model(
                xa,
                da,
            )

            zb = model(
                xb,
                db,
            )

            loss = criterion(
                za,
                zb,
                target,
            )

            if training:
                loss.backward()
                optimizer.step()

            total_loss += (
                loss.item()
                * len(target)
            )

            total_n += len(
                target
            )

    return (
        total_loss
        / max(
            total_n,
            1,
        )
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

    output = []

    for start in range(
        0,
        len(X),
        batch_size,
    ):
        end = min(
            start + batch_size,
            len(X),
        )

        xb = torch.from_numpy(
            X[start:end]
        ).float().to(device)

        db = torch.from_numpy(
            duration[start:end]
        ).float().to(device)

        z = model(
            xb,
            db,
        )

        output.append(
            z.cpu().numpy()
        )

    return np.concatenate(
        output,
        axis=0,
    )


def cosine_scores(
    embeddings,
    template,
):
    template = (
        template
        / (
            np.linalg.norm(
                template
            )
            + 1e-12
        )
    )

    embeddings = (
        embeddings
        / (
            np.linalg.norm(
                embeddings,
                axis=1,
                keepdims=True,
            )
            + 1e-12
        )
    )

    return embeddings @ template


def binary_metrics(
    labels,
    scores,
    threshold,
):
    labels = np.asarray(
        labels,
        dtype=np.int64,
    )

    scores = np.asarray(
        scores,
        dtype=np.float64,
    )

    pred = (
        scores >= threshold
    ).astype(
        np.int64
    )

    genuine = (
        labels == 1
    )

    impostor = (
        labels == 0
    )

    tp = int(
        np.sum(
            (pred == 1)
            & genuine
        )
    )

    fn = int(
        np.sum(
            (pred == 0)
            & genuine
        )
    )

    fp = int(
        np.sum(
            (pred == 1)
            & impostor
        )
    )

    tn = int(
        np.sum(
            (pred == 0)
            & impostor
        )
    )

    far = (
        fp
        / max(
            int(
                impostor.sum()
            ),
            1,
        )
    )

    frr = (
        fn
        / max(
            int(
                genuine.sum()
            ),
            1,
        )
    )

    acc = (
        (tp + tn)
        / max(
            len(labels),
            1,
        )
    )

    bal = (
        (1.0 - far)
        + (1.0 - frr)
    ) / 2.0

    return {
        "accuracy": float(acc),
        "balanced_accuracy": float(bal),
        "far": float(far),
        "frr": float(frr),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def find_eer_threshold(
    labels,
    scores,
):
    thresholds = np.unique(
        scores
    )

    best = None

    for threshold in thresholds:

        m = binary_metrics(
            labels,
            scores,
            threshold,
        )

        gap = abs(
            m["far"]
            - m["frr"]
        )

        eer = (
            m["far"]
            + m["frr"]
        ) / 2.0

        candidate = (
            gap,
            eer,
            float(threshold),
            m,
        )

        if (
            best is None
            or candidate[:2]
            < best[:2]
        ):
            best = candidate

    _, eer, threshold, metrics = best

    return (
        threshold,
        eer,
        metrics,
    )


def build_known_calibration_scores(
    embeddings,
    indices,
    meta,
    enrollment_per_gesture,
):
    """
    Cross-session calibration.

    second-latest session:
        enrollment

    latest session:
        genuine verification

    Same gesture only.
    """

    performer = meta[
        "performer"
    ]

    gesture = meta[
        "gesture"
    ]

    session = meta[
        "session"
    ]

    index_to_pos = {
        int(idx): pos
        for pos, idx
        in enumerate(indices)
    }

    labels = []
    scores = []

    gestures = sorted(
        np.unique(
            gesture[indices]
        )
    )

    for target_user in TRAIN_USERS:

        user_idx = indices[
            performer[indices]
            == target_user
        ]

        sessions = sorted(
            np.unique(
                session[user_idx]
            )
        )

        if len(sessions) < 2:
            continue

        enroll_session = (
            sessions[-2]
        )

        test_session = (
            sessions[-1]
        )

        for g in gestures:

            enroll_idx = indices[
                (
                    performer[indices]
                    == target_user
                )
                & (
                    gesture[indices]
                    == g
                )
                & (
                    session[indices]
                    == enroll_session
                )
            ]

            genuine_idx = indices[
                (
                    performer[indices]
                    == target_user
                )
                & (
                    gesture[indices]
                    == g
                )
                & (
                    session[indices]
                    == test_session
                )
            ]

            if (
                len(enroll_idx)
                < enrollment_per_gesture
            ):
                continue

            if len(
                genuine_idx
            ) == 0:
                continue

            enroll_idx = (
                enroll_idx[
                    :enrollment_per_gesture
                ]
            )

            template = np.stack(
                [
                    embeddings[
                        index_to_pos[
                            int(i)
                        ]
                    ]
                    for i
                    in enroll_idx
                ]
            ).mean(
                axis=0
            )

            genuine_emb = np.stack(
                [
                    embeddings[
                        index_to_pos[
                            int(i)
                        ]
                    ]
                    for i
                    in genuine_idx
                ]
            )

            genuine_scores = (
                cosine_scores(
                    genuine_emb,
                    template,
                )
            )

            labels.extend(
                [1]
                * len(
                    genuine_scores
                )
            )

            scores.extend(
                genuine_scores.tolist()
            )

            impostor_idx = []

            for other_user in TRAIN_USERS:

                if (
                    other_user
                    == target_user
                ):
                    continue

                other_idx = indices[
                    performer[indices]
                    == other_user
                ]

                other_sessions = sorted(
                    np.unique(
                        session[
                            other_idx
                        ]
                    )
                )

                if not other_sessions:
                    continue

                other_test = (
                    other_sessions[-1]
                )

                imp = indices[
                    (
                        performer[indices]
                        == other_user
                    )
                    & (
                        gesture[indices]
                        == g
                    )
                    & (
                        session[indices]
                        == other_test
                    )
                ]

                impostor_idx.extend(
                    imp.tolist()
                )

            if impostor_idx:

                impostor_idx = (
                    np.asarray(
                        impostor_idx,
                        dtype=np.int64,
                    )
                )

                impostor_emb = np.stack(
                    [
                        embeddings[
                            index_to_pos[
                                int(i)
                            ]
                        ]
                        for i
                        in impostor_idx
                    ]
                )

                imp_scores = cosine_scores(
                    impostor_emb,
                    template,
                )

                labels.extend(
                    [0]
                    * len(
                        imp_scores
                    )
                )

                scores.extend(
                    imp_scores.tolist()
                )

    return (
        np.asarray(
            labels,
            dtype=np.int64,
        ),
        np.asarray(
            scores,
            dtype=np.float64,
        ),
    )


def evaluate_unseen(
    model,
    X,
    duration,
    meta,
    device,
    threshold,
    enrollment_per_gesture,
):
    performer = meta[
        "performer"
    ]

    gesture = meta[
        "gesture"
    ]

    session = meta[
        "session"
    ]

    gestures = sorted(
        np.unique(
            gesture
        )
    )

    rows = []

    global_labels = []
    global_scores = []

    for target_user in UNSEEN_USERS:

        user_idx = np.where(
            performer
            == target_user
        )[0]

        sessions = sorted(
            np.unique(
                session[user_idx]
            )
        )

        enroll_session = (
            sessions[0]
        )

        print()
        print(
            f"[UNSEEN] "
            f"{target_user} "
            f"enrollment_session="
            f"{enroll_session}"
        )

        for g in gestures:

            enroll_idx = np.where(
                (
                    performer
                    == target_user
                )
                & (
                    gesture == g
                )
                & (
                    session
                    == enroll_session
                )
            )[0]

            if (
                len(enroll_idx)
                < enrollment_per_gesture
            ):
                print(
                    f"  {g}: SKIP"
                )
                continue

            enroll_idx = (
                enroll_idx[
                    :enrollment_per_gesture
                ]
            )

            genuine_idx = np.where(
                (
                    performer
                    == target_user
                )
                & (
                    gesture == g
                )
                & (
                    session
                    != enroll_session
                )
            )[0]

            impostor_idx = np.where(
                np.isin(
                    performer,
                    UNSEEN_USERS,
                )
                & (
                    performer
                    != target_user
                )
                & (
                    gesture == g
                )
            )[0]

            enroll_emb = (
                extract_embeddings(
                    model,
                    X[enroll_idx],
                    duration[
                        enroll_idx
                    ],
                    device,
                )
            )

            template = (
                enroll_emb.mean(
                    axis=0
                )
            )

            genuine_emb = (
                extract_embeddings(
                    model,
                    X[genuine_idx],
                    duration[
                        genuine_idx
                    ],
                    device,
                )
            )

            impostor_emb = (
                extract_embeddings(
                    model,
                    X[impostor_idx],
                    duration[
                        impostor_idx
                    ],
                    device,
                )
            )

            genuine_scores = (
                cosine_scores(
                    genuine_emb,
                    template,
                )
            )

            impostor_scores = (
                cosine_scores(
                    impostor_emb,
                    template,
                )
            )

            labels = np.concatenate(
                [
                    np.ones(
                        len(
                            genuine_scores
                        ),
                        dtype=np.int64,
                    ),
                    np.zeros(
                        len(
                            impostor_scores
                        ),
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

            rows.append(
                {
                    "user": target_user,
                    "gesture": g,
                    "enrollment_session": enroll_session,
                    "enrollment_samples": len(
                        enroll_idx
                    ),
                    "genuine_samples": len(
                        genuine_idx
                    ),
                    "impostor_samples": len(
                        impostor_idx
                    ),
                    "threshold": threshold,
                    "accuracy": metrics[
                        "accuracy"
                    ],
                    "balanced_accuracy": metrics[
                        "balanced_accuracy"
                    ],
                    "far": metrics[
                        "far"
                    ],
                    "frr": metrics[
                        "frr"
                    ],
                    "test_eer_analysis": local_eer,
                    "test_eer_threshold_analysis": local_threshold,
                }
            )

            global_labels.extend(
                labels.tolist()
            )

            global_scores.extend(
                scores.tolist()
            )

            print(
                f"  {g}: "
                f"Genuine={len(genuine_idx):3d} "
                f"Impostor={len(impostor_idx):3d} | "
                f"BalAcc="
                f"{metrics['balanced_accuracy']:.3f} "
                f"FAR="
                f"{metrics['far']:.3f} "
                f"FRR="
                f"{metrics['frr']:.3f}"
            )

    global_labels = np.asarray(
        global_labels,
        dtype=np.int64,
    )

    global_scores = np.asarray(
        global_scores,
        dtype=np.float64,
    )

    metrics = binary_metrics(
        global_labels,
        global_scores,
        threshold,
    )

    test_threshold, test_eer, _ = (
        find_eer_threshold(
            global_labels,
            global_scores,
        )
    )

    return (
        rows,
        metrics,
        test_eer,
        test_threshold,
    )


def main():

    parser = argparse.ArgumentParser()

    default_data = (
        Path(
            os.environ[
                "WITECH_ROOT"
            ]
        )
        / "derived"
        / "dataset.npz"
    )

    parser.add_argument(
        "--data",
        default=str(
            default_data
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=40,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
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
        "--pairs",
        type=int,
        default=6000,
    )

    parser.add_argument(
        "--margin",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--enroll",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=7,
    )

    args = parser.parse_args()

    seed_everything()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)
    print(
        "Siamese / Contrastive "
        "Unseen-user Authentication"
    )
    print("=" * 70)

    print(
        "device       :",
        device,
    )

    print(
        "dataset      :",
        args.data,
    )

    print(
        "train users  :",
        TRAIN_USERS,
    )

    print(
        "unseen users :",
        UNSEEN_USERS,
    )

    print(
        "embedding dim:",
        args.embedding_dim,
    )

    print(
        "pairs/epoch  :",
        args.pairs,
    )

    print(
        "margin       :",
        args.margin,
    )

    print(
        "enroll / G   :",
        args.enroll,
    )

    print()

    (
        _,
        X_seq,
        duration,
        meta,
        T,
        D,
    ) = load_dataset(
        args.data
    )

    print(
        f"Dataset: "
        f"N={len(X_seq)}, "
        f"T={T}, "
        f"D={D}"
    )

    train_idx, val_idx = (
        chronological_split(
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

    # ---------------------------------------------------------
    # Normalization fitted only using known training data
    # ---------------------------------------------------------

    seq_mean, seq_std = (
        fit_sequence_stats(
            X_seq[train_idx]
        )
    )

    dur_mean, dur_std = (
        fit_duration_stats(
            duration[train_idx]
        )
    )

    X_norm = apply_sequence_stats(
        X_seq,
        seq_mean,
        seq_std,
    )

    duration_norm = (
        apply_duration_stats(
            duration,
            dur_mean,
            dur_std,
        )
    )

    model = Siamese1DCNN(
        input_dim=D,
        embedding_dim=args.embedding_dim,
    ).to(
        device
    )

    n_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Trainable parameters: "
        f"{n_params:,}"
    )

    # CosineEmbeddingLoss:
    #
    # target = +1 -> make embeddings similar
    # target = -1 -> make embeddings dissimilar
    criterion = (
        nn.CosineEmbeddingLoss(
            margin=args.margin
        )
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )

    best_val_loss = float(
        "inf"
    )

    best_state = None
    patience_count = 0

    rng = np.random.default_rng(
        SEED
    )

    print()
    print(
        "Training Siamese encoder..."
    )

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        # New random training pairs every epoch
        (
            train_a,
            train_b,
            train_target,
        ) = build_pairs(
            train_idx,
            meta,
            args.pairs,
            rng,
        )

        (
            val_a,
            val_b,
            val_target,
        ) = build_pairs(
            val_idx,
            meta,
            min(
                2000,
                args.pairs // 2,
            ),
            rng,
        )

        train_dataset = PairDataset(
            X_norm,
            duration_norm,
            train_a,
            train_b,
            train_target,
        )

        val_dataset = PairDataset(
            X_norm,
            duration_norm,
            val_a,
            val_b,
            val_target,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        )

        train_loss = run_pair_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
        )

        val_loss = run_pair_epoch(
            model,
            val_loader,
            criterion,
            device,
            optimizer=None,
        )

        print(
            f"Epoch {epoch:02d} | "
            f"train loss="
            f"{train_loss:.4f} | "
            f"val loss="
            f"{val_loss:.4f}"
        )

        if (
            val_loss
            < best_val_loss
        ):
            best_val_loss = val_loss

            best_state = {
                k: v.detach()
                .cpu()
                .clone()
                for k, v
                in model.state_dict()
                .items()
            }

            patience_count = 0

        else:
            patience_count += 1

        if (
            patience_count
            >= args.patience
        ):
            print(
                f"Early stopping "
                f"at epoch {epoch}"
            )
            break

    if best_state is None:
        raise RuntimeError(
            "No best model state"
        )

    model.load_state_dict(
        best_state
    )

    model.to(
        device
    )

    print()

    print(
        f"Best validation pair loss: "
        f"{best_val_loss:.4f}"
    )

    # ---------------------------------------------------------
    # Threshold calibration using known users only
    # ---------------------------------------------------------

    print()

    print(
        "Calibrating threshold "
        "using known users..."
    )

    known_idx = np.where(
        np.isin(
            meta["performer"],
            TRAIN_USERS,
        )
    )[0]

    known_embeddings = (
        extract_embeddings(
            model,
            X_norm[
                known_idx
            ],
            duration_norm[
                known_idx
            ],
            device,
        )
    )

    cal_labels, cal_scores = (
        build_known_calibration_scores(
            known_embeddings,
            known_idx,
            meta,
            args.enroll,
        )
    )

    (
        threshold,
        val_eer,
        val_metrics,
    ) = find_eer_threshold(
        cal_labels,
        cal_scores,
    )

    print(
        f"Validation threshold : "
        f"{threshold:.6f}"
    )

    print(
        f"Validation EER       : "
        f"{val_eer:.3f}"
    )

    print(
        f"Validation FAR       : "
        f"{val_metrics['far']:.3f}"
    )

    print(
        f"Validation FRR       : "
        f"{val_metrics['frr']:.3f}"
    )

    # ---------------------------------------------------------
    # Unseen user evaluation
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "UNSEEN USER EVALUATION"
    )
    print("=" * 70)

    (
        rows,
        global_metrics,
        test_eer,
        test_eer_threshold,
    ) = evaluate_unseen(
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
    print(
        "FINAL SIAMESE "
        "UNSEEN-USER RESULT"
    )
    print("=" * 70)

    print(
        f"Fixed threshold     : "
        f"{threshold:.6f}"
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

    print(
        f"Test EER (analysis) : "
        f"{test_eer:.3f}"
    )

    print(
        f"Test EER threshold  : "
        f"{test_eer_threshold:.6f}"
    )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    output_dir = (
        Path(
            os.environ[
                "WITECH_ROOT"
            ]
        )
        / "derived"
        / "siamese_embedding_experiment"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        output_dir
        / "siamese_1dcnn.pt"
    )

    torch.save(
        {
            "model_state_dict":
                {
                    k: v.detach()
                    .cpu()
                    for k, v
                    in model.state_dict()
                    .items()
                },
            "input_dim": D,
            "embedding_dim":
                args.embedding_dim,
            "train_users":
                TRAIN_USERS,
            "unseen_users":
                UNSEEN_USERS,
            "threshold":
                threshold,
            "margin":
                args.margin,
            "sequence_mean":
                seq_mean,
            "sequence_std":
                seq_std,
            "duration_mean":
                dur_mean,
            "duration_std":
                dur_std,
            "best_val_pair_loss":
                best_val_loss,
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
            writer.writerows(
                rows
            )

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
            "Siamese embedding "
            "unseen-user authentication\n"
        )

        f.write(
            f"train_users="
            f"{TRAIN_USERS}\n"
        )

        f.write(
            f"unseen_users="
            f"{UNSEEN_USERS}\n"
        )

        f.write(
            f"margin="
            f"{args.margin}\n"
        )

        f.write(
            f"enrollment_per_gesture="
            f"{args.enroll}\n"
        )

        f.write(
            f"validation_threshold="
            f"{threshold:.6f}\n"
        )

        f.write(
            f"validation_eer="
            f"{val_eer:.6f}\n"
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
    print(
        " model  :",
        model_path,
    )
    print(
        " csv    :",
        csv_path,
    )
    print(
        " summary:",
        summary_path,
    )


if __name__ == "__main__":
    main()