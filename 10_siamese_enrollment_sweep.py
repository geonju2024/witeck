"""
10_siamese_enrollment_sweep.py

Evaluate how the number of enrollment samples affects
unseen-user authentication performance.

The Siamese encoder is trained only ONCE.

Known users:
    P01~P07

Unseen users:
    P08~P10

Enrollment sizes:
    1, 3, 5

Important:
For enrollment=5, multiple enrollment sessions are allowed because
one session may not contain five samples per gesture.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from two_stage_common import (
    load_dataset,
    fit_duration_stats,
    apply_duration_stats,
    fit_sequence_stats,
    apply_sequence_stats,
)

from importlib import import_module


# ----------------------------------------------------------------------
# Reuse the already-tested Siamese implementation from experiment 09
# ----------------------------------------------------------------------

siamese09 = import_module("09_train_siamese_embedding")

Siamese1DCNN = siamese09.Siamese1DCNN
PairDataset = siamese09.PairDataset
build_pairs = siamese09.build_pairs
run_pair_epoch = siamese09.run_pair_epoch
extract_embeddings = siamese09.extract_embeddings
cosine_scores = siamese09.cosine_scores
binary_metrics = siamese09.binary_metrics
find_eer_threshold = siamese09.find_eer_threshold
seed_everything = siamese09.seed_everything

TRAIN_USERS = siamese09.TRAIN_USERS
UNSEEN_USERS = siamese09.UNSEEN_USERS

SEED = 42


# ----------------------------------------------------------------------
# Split known users
# ----------------------------------------------------------------------

def build_known_split(meta):
    """
    For P01~P07:

    last 3 sessions are held out from encoder training.

    Earlier sessions:
        Siamese encoder training

    second-last + third-last sessions:
        enrollment candidates for threshold calibration

    last session:
        verification for threshold calibration
    """

    performer = meta["performer"]
    session = meta["session"]

    train_idx = []

    calibration_enroll = {}
    calibration_test = {}

    for user in TRAIN_USERS:

        user_idx = np.where(
            performer == user
        )[0]

        sessions = sorted(
            np.unique(session[user_idx])
        )

        if len(sessions) < 5:
            raise RuntimeError(
                f"{user}: not enough sessions "
                f"for enrollment sweep"
            )

        # Last 3 sessions before the final session:
        # enrollment candidates
        enroll_sessions = sessions[-4:-1]

        # Final session:
        # calibration verification
        test_session = sessions[-1]

        # Everything before those 4 held-out sessions:
        # encoder training
        train_sessions = sessions[:-4]

        tr = user_idx[
            np.isin(
                session[user_idx],
                train_sessions,
            )
        ]

        en = user_idx[
            np.isin(
                session[user_idx],
                enroll_sessions,
            )
        ]

        te = user_idx[
            session[user_idx] == test_session
        ]

        train_idx.extend(
            tr.tolist()
        )

        calibration_enroll[user] = en
        calibration_test[user] = te

        print(
            f"{user}: "
            f"train={len(tr):3d} | "
            f"enroll={len(en):3d} "
            f"{list(enroll_sessions)} | "
            f"test={len(te):3d} "
            f"{test_session}"
        )

    return (
        np.asarray(
            train_idx,
            dtype=np.int64,
        ),
        calibration_enroll,
        calibration_test,
    )


# ----------------------------------------------------------------------
# Template
# ----------------------------------------------------------------------

def make_template(
    embeddings,
    index_to_pos,
    candidates,
    enrollment_size,
):
    """
    Take the first N enrollment candidates and average their embeddings.
    """

    if len(candidates) < enrollment_size:
        return None

    selected = candidates[
        :enrollment_size
    ]

    z = np.stack(
        [
            embeddings[
                index_to_pos[int(i)]
            ]
            for i in selected
        ]
    )

    template = z.mean(
        axis=0
    )

    template = (
        template
        / (
            np.linalg.norm(template)
            + 1e-12
        )
    )

    return template


# ----------------------------------------------------------------------
# Known-user threshold calibration
# ----------------------------------------------------------------------

def calibrate_threshold(
    embeddings,
    indices,
    meta,
    calibration_enroll,
    calibration_test,
    enrollment_size,
):
    performer = meta["performer"]
    gesture = meta["gesture"]

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

        enroll_pool = (
            calibration_enroll[
                target_user
            ]
        )

        genuine_pool = (
            calibration_test[
                target_user
            ]
        )

        for g in gestures:

            enroll_candidates = (
                enroll_pool[
                    gesture[enroll_pool] == g
                ]
            )

            genuine_idx = (
                genuine_pool[
                    gesture[genuine_pool] == g
                ]
            )

            template = make_template(
                embeddings,
                index_to_pos,
                enroll_candidates,
                enrollment_size,
            )

            if template is None:
                continue

            if len(genuine_idx) == 0:
                continue

            genuine_emb = np.stack(
                [
                    embeddings[
                        index_to_pos[int(i)]
                    ]
                    for i in genuine_idx
                ]
            )

            genuine_scores = (
                cosine_scores(
                    genuine_emb,
                    template,
                )
            )

            labels.extend(
                [1] * len(genuine_scores)
            )

            scores.extend(
                genuine_scores.tolist()
            )

            # Same gesture, different known users
            impostor_idx = []

            for other_user in TRAIN_USERS:

                if other_user == target_user:
                    continue

                other_test = (
                    calibration_test[
                        other_user
                    ]
                )

                imp = other_test[
                    gesture[other_test] == g
                ]

                impostor_idx.extend(
                    imp.tolist()
                )

            if impostor_idx:

                impostor_idx = np.asarray(
                    impostor_idx,
                    dtype=np.int64,
                )

                impostor_emb = np.stack(
                    [
                        embeddings[
                            index_to_pos[int(i)]
                        ]
                        for i in impostor_idx
                    ]
                )

                impostor_scores = (
                    cosine_scores(
                        impostor_emb,
                        template,
                    )
                )

                labels.extend(
                    [0]
                    * len(impostor_scores)
                )

                scores.extend(
                    impostor_scores.tolist()
                )

    labels = np.asarray(
        labels,
        dtype=np.int64,
    )

    scores = np.asarray(
        scores,
        dtype=np.float64,
    )

    if len(labels) == 0:
        raise RuntimeError(
            f"No calibration samples "
            f"for enrollment={enrollment_size}"
        )

    threshold, eer, metrics = (
        find_eer_threshold(
            labels,
            scores,
        )
    )

    return (
        threshold,
        eer,
        metrics,
    )


# ----------------------------------------------------------------------
# Unseen-user evaluation
# ----------------------------------------------------------------------

def evaluate_unseen(
    embeddings,
    indices,
    meta,
    threshold,
    enrollment_size,
):
    """
    Fair enrollment-size comparison.

    A (user, gesture) pair is evaluated ONLY if that pair has
    at least 5 enrollment samples in the shared enrollment pool.

    Therefore enroll=1, 3, 5 all use the exact same
    user/gesture combinations and the same test data.
    """

    performer = meta["performer"]
    gesture = meta["gesture"]
    session = meta["session"]

    index_to_pos = {
        int(idx): pos
        for pos, idx in enumerate(indices)
    }

    gestures = sorted(
        np.unique(
            gesture[indices]
        )
    )

    all_labels = []
    all_scores = []

    for target_user in UNSEEN_USERS:

        user_idx = indices[
            performer[indices] == target_user
        ]

        sessions = sorted(
            np.unique(
                session[user_idx]
            )
        )

        if len(sessions) < 4:
            raise RuntimeError(
                f"{target_user}: not enough sessions"
            )

        # Same enrollment pool for 1 / 3 / 5
        enroll_sessions = sessions[:3]

        # Same test sessions for 1 / 3 / 5
        test_sessions = sessions[3:]

        enroll_pool = user_idx[
            np.isin(
                session[user_idx],
                enroll_sessions,
            )
        ]

        genuine_pool = user_idx[
            np.isin(
                session[user_idx],
                test_sessions,
            )
        ]

        print(
            f"\n[UNSEEN] {target_user} | "
            f"enroll_sessions={list(enroll_sessions)} | "
            f"test_sessions={list(test_sessions)}"
        )

        for g in gestures:

            enroll_candidates = enroll_pool[
                gesture[enroll_pool] == g
            ]

            # --------------------------------------------------
            # IMPORTANT:
            # Keep only pairs that could support enroll=5.
            # This makes 1 / 3 / 5 directly comparable.
            # --------------------------------------------------
            if len(enroll_candidates) < 5:
                print(
                    f"  {g}: EXCLUDED "
                    f"(only {len(enroll_candidates)} "
                    f"enrollment samples; "
                    f"needs >=5 for fair sweep)"
                )
                continue

            genuine_idx = genuine_pool[
                gesture[genuine_pool] == g
            ]

            if len(genuine_idx) == 0:
                print(
                    f"  {g}: EXCLUDED "
                    f"(no genuine test samples)"
                )
                continue

            template = make_template(
                embeddings,
                index_to_pos,
                enroll_candidates,
                enrollment_size,
            )

            if template is None:
                raise RuntimeError(
                    f"Unexpected template failure: "
                    f"{target_user} {g} "
                    f"enroll={enrollment_size}"
                )

            impostor_idx = indices[
                np.isin(
                    performer[indices],
                    UNSEEN_USERS,
                )
                & (
                    performer[indices]
                    != target_user
                )
                & (
                    gesture[indices] == g
                )
                & np.isin(
                    session[indices],
                    test_sessions,
                )
            ]

            genuine_emb = np.stack(
                [
                    embeddings[
                        index_to_pos[int(i)]
                    ]
                    for i in genuine_idx
                ]
            )

            impostor_emb = np.stack(
                [
                    embeddings[
                        index_to_pos[int(i)]
                    ]
                    for i in impostor_idx
                ]
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

            all_labels.extend(
                labels.tolist()
            )

            all_scores.extend(
                scores.tolist()
            )

    all_labels = np.asarray(
        all_labels,
        dtype=np.int64,
    )

    all_scores = np.asarray(
        all_scores,
        dtype=np.float64,
    )

    if len(all_labels) == 0:
        raise RuntimeError(
            "No unseen-user evaluation samples remained."
        )

    metrics = binary_metrics(
        all_labels,
        all_scores,
        threshold,
    )

    test_threshold, test_eer, _ = (
        find_eer_threshold(
            all_labels,
            all_scores,
        )
    )

    return (
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
        default=str(default_data),
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
        "--patience",
        type=int,
        default=7,
    )

    args = parser.parse_args()

    seed_everything(SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)
    print(
        "Siamese Enrollment-Size Sweep"
    )
    print("=" * 70)

    print("device       :", device)
    print("dataset      :", args.data)
    print("train users  :", TRAIN_USERS)
    print("unseen users :", UNSEEN_USERS)
    print("enroll sizes : [1, 3, 5]")
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

    (
        train_idx,
        calibration_enroll,
        calibration_test,
    ) = build_known_split(
        meta
    )

    print(
        f"\nEncoder training samples: "
        f"{len(train_idx)}"
    )

    # --------------------------------------------------------------
    # Normalization
    # --------------------------------------------------------------

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

    # --------------------------------------------------------------
    # ONE encoder training
    # --------------------------------------------------------------

    model = Siamese1DCNN(
        input_dim=D,
        embedding_dim=args.embedding_dim,
    ).to(device)

    criterion = nn.CosineEmbeddingLoss(
        margin=args.margin
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )

    rng = np.random.default_rng(
        SEED
    )

    best_loss = float("inf")
    best_state = None
    patience_count = 0

    print(
        "\nTraining Siamese encoder ONCE..."
    )

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        (
            pair_a,
            pair_b,
            targets,
        ) = build_pairs(
            train_idx,
            meta,
            args.pairs,
            rng,
        )

        dataset = PairDataset(
            X_norm,
            duration_norm,
            pair_a,
            pair_b,
            targets,
        )

        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
        )

        train_loss = run_pair_epoch(
            model,
            loader,
            criterion,
            device,
            optimizer,
        )

        print(
            f"Epoch {epoch:02d} | "
            f"train loss="
            f"{train_loss:.4f}"
        )

        if train_loss < best_loss:

            best_loss = train_loss

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

    model.load_state_dict(
        best_state
    )

    model.to(device)

    print(
        f"\nBest training pair loss: "
        f"{best_loss:.4f}"
    )

    # --------------------------------------------------------------
    # Extract every embedding ONCE
    # --------------------------------------------------------------

    all_idx = np.arange(
        len(X_norm),
        dtype=np.int64,
    )

    print(
        "\nExtracting embeddings..."
    )

    all_embeddings = (
        extract_embeddings(
            model,
            X_norm,
            duration_norm,
            device,
        )
    )

    # --------------------------------------------------------------
    # Enrollment sweep
    # --------------------------------------------------------------

    results = []

    for enrollment_size in [
        1,
        3,
        5,
    ]:

        print()
        print("=" * 70)
        print(
            f"ENROLLMENT SIZE = "
            f"{enrollment_size}"
        )
        print("=" * 70)

        (
            threshold,
            val_eer,
            val_metrics,
        ) = calibrate_threshold(
            all_embeddings,
            all_idx,
            meta,
            calibration_enroll,
            calibration_test,
            enrollment_size,
        )

        print(
            f"Calibration threshold : "
            f"{threshold:.6f}"
        )

        print(
            f"Calibration EER       : "
            f"{val_eer:.3f}"
        )

        (
            test_metrics,
            test_eer,
            test_threshold,
        ) = evaluate_unseen(
            all_embeddings,
            all_idx,
            meta,
            threshold,
            enrollment_size,
        )

        results.append(
            {
                "enroll":
                    enrollment_size,
                "accuracy":
                    test_metrics[
                        "accuracy"
                    ],
                "balanced_accuracy":
                    test_metrics[
                        "balanced_accuracy"
                    ],
                "far":
                    test_metrics[
                        "far"
                    ],
                "frr":
                    test_metrics[
                        "frr"
                    ],
                "eer":
                    test_eer,
                "threshold":
                    threshold,
            }
        )

    # --------------------------------------------------------------
    # Final table
    # --------------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "ENROLLMENT SIZE COMPARISON"
    )
    print("=" * 70)

    print(
        f"{'Enroll':>6} "
        f"{'Accuracy':>10} "
        f"{'BalAcc':>10} "
        f"{'FAR':>8} "
        f"{'FRR':>8} "
        f"{'EER':>8}"
    )

    print("-" * 56)

    for r in results:

        print(
            f"{r['enroll']:>6d} "
            f"{r['accuracy']:>10.3f} "
            f"{r['balanced_accuracy']:>10.3f} "
            f"{r['far']:>8.3f} "
            f"{r['frr']:>8.3f} "
            f"{r['eer']:>8.3f}"
        )

    # --------------------------------------------------------------
    # Save summary
    # --------------------------------------------------------------

    output_dir = (
        Path(
            os.environ[
                "WITECH_ROOT"
            ]
        )
        / "derived"
        / "siamese_enrollment_sweep"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = (
        output_dir
        / "summary.csv"
    )

    import csv

    with open(
        summary_path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                results[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            results
        )

    print()
    print(
        "Saved summary:",
        summary_path,
    )


if __name__ == "__main__":
    main()