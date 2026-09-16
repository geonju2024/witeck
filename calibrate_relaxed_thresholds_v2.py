"""Calibrate a relaxed Dual-Head operating point on KNOWN-USER validation only.

Relaxed policy:
- NEVER use P08-P10 or any final-test users.
- For each head, search thresholds on the known-user validation score distribution.
- Keep validation FAR <= target_far.
- Among feasible thresholds, choose the one with the LOWEST FRR.
- If several thresholds have the same FRR, choose the HIGHEST threshold
  (more conservative / lower FAR).

This creates a genuinely more permissive operating point intended for demo use.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compute_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)

    pred = scores >= threshold
    genuine = labels == 1
    impostor = labels == 0

    tp = int(np.sum(pred & genuine))
    fn = int(np.sum((~pred) & genuine))
    fp = int(np.sum(pred & impostor))
    tn = int(np.sum((~pred) & impostor))

    far = fp / max(int(impostor.sum()), 1)
    frr = fn / max(int(genuine.sum()), 1)

    return {
        "threshold": float(threshold),
        "far": float(far),
        "frr": float(frr),
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
    }


def choose_relaxed_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
    target_far: float,
) -> dict:
    scores = np.asarray(scores, dtype=np.float64)

    # Add a value below the minimum score so "accept almost all" is searchable.
    eps = 1e-7
    candidates = np.unique(
        np.concatenate([
            scores,
            np.array([float(np.min(scores)) - eps], dtype=np.float64),
        ])
    )

    feasible = []
    for threshold in candidates:
        m = compute_metrics(labels, scores, float(threshold))
        if m["far"] <= target_far + 1e-12:
            feasible.append(m)

    if not feasible:
        raise RuntimeError(
            f"No threshold satisfies validation FAR <= {target_far:.2%}"
        )

    # Primary objective: lowest FRR.
    # Tie-break 1: lower FAR.
    # Tie-break 2: higher threshold (more conservative).
    feasible.sort(
        key=lambda m: (
            m["frr"],
            m["far"],
            -m["threshold"],
        )
    )
    return feasible[0]


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--trainer",
        default="45_train_shared_dual_head.py",
        help="Path to the trainer used to create the checkpoint.",
    )
    ap.add_argument(
        "--checkpoint",
        default=r"output\shared_dual_head\seed_40\shared_dual_head.pt",
    )
    ap.add_argument(
        "--data",
        default=r"dataset\dataset_1955_recent8_updated_20260905_hand_only.npz",
    )
    ap.add_argument(
        "--target-far",
        type=float,
        default=0.05,
        help="Maximum validation FAR allowed for each head. Default=0.05",
    )
    ap.add_argument(
        "--output",
        default=r"output\shared_dual_head\seed_40\relaxed_thresholds.json",
    )
    args = ap.parse_args()

    trainer_path = Path(args.trainer).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    data_path = Path(args.data).resolve()

    trainer = load_module(
        trainer_path,
        "dual_head_trainer_for_relaxed_calibration",
    )

    ckpt = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    # Same dataset loader and validation protocol as the trainer.
    _, X_seq, duration, meta, T, D = trainer.legacy.load_dataset(str(data_path))

    if int(D) != int(ckpt["input_dim"]):
        raise ValueError(
            f"Dataset D={D} != checkpoint input_dim={ckpt['input_dim']}"
        )

    if int(T) != int(ckpt["seq_len"]):
        raise ValueError(
            f"Dataset T={T} != checkpoint seq_len={ckpt['seq_len']}"
        )

    seq_mean = np.asarray(
        ckpt["sequence_mean"],
        dtype=np.float32,
    )
    seq_std = np.asarray(
        ckpt["sequence_std"],
        dtype=np.float32,
    )

    duration_mean = float(ckpt["duration_mean"])
    duration_std = float(ckpt["duration_std"])

    X_norm = (
        (X_seq - seq_mean)
        / np.maximum(seq_std, 1e-8)
    ).astype(np.float32)

    duration_norm = (
        (duration - duration_mean)
        / max(duration_std, 1e-8)
    ).astype(np.float32)

    train_users = list(ckpt["train_users"])

    known_idx = np.where(
        np.isin(meta["performer"], train_users)
    )[0]

    model = trainer.SharedDualHead1DCNN(
        input_dim=int(ckpt["input_dim"]),
        num_user_classes=len(train_users),
        embedding_dim=int(ckpt["embedding_dim"]),
    )

    model.load_state_dict(
        ckpt["model_state_dict"],
        strict=True,
    )

    device = torch.device("cpu")
    model.to(device)
    model.eval()

    gesture_emb, user_emb = trainer.extract_dual_embeddings(
        model,
        X_norm[known_idx],
        duration_norm[known_idx],
        device,
    )

    enrollment = int(
        ckpt.get("enrollment_per_gesture", 3)
    )

    # Recreate EXACTLY the same validation-score construction
    # that the Dual-Head trainer uses.
    user_labels, user_scores = trainer.build_user_validation_scores(
        user_emb,
        known_idx,
        meta,
        train_users,
        enrollment_per_gesture=enrollment,
    )

    gesture_labels, gesture_scores = trainer.build_gesture_validation_scores(
        gesture_emb,
        known_idx,
        meta,
        train_users,
        enrollment_per_gesture=enrollment,
    )

    default_user = compute_metrics(
        user_labels,
        user_scores,
        float(ckpt["user_threshold"]),
    )

    default_gesture = compute_metrics(
        gesture_labels,
        gesture_scores,
        float(ckpt["gesture_threshold"]),
    )

    relaxed_user = choose_relaxed_threshold(
        user_labels,
        user_scores,
        args.target_far,
    )

    relaxed_gesture = choose_relaxed_threshold(
        gesture_labels,
        gesture_scores,
        args.target_far,
    )

    result = {
        "source": "known-user validation only",
        "final_test_users_used": False,
        "policy": (
            "For each head, keep validation FAR <= target_far and "
            "choose the threshold with the lowest validation FRR."
        ),
        "target_far_per_head": float(args.target_far),

        "default": {
            "gesture": default_gesture,
            "user": default_user,
        },

        "demo_relaxed": {
            "gesture_threshold": relaxed_gesture["threshold"],
            "user_threshold": relaxed_user["threshold"],

            "gesture_validation_far": relaxed_gesture["far"],
            "gesture_validation_frr": relaxed_gesture["frr"],

            "user_validation_far": relaxed_user["far"],
            "user_validation_frr": relaxed_user["frr"],
        },

        "notes": [
            "P08-P10 were not used.",
            "This operating point is optimized only on known-user validation.",
            "A lower threshold can reduce FRR but may increase FAR.",
            "Do not present these validation values as unseen-free-gesture performance.",
        ],
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=" * 76)
    print("RELAXED OPERATING POINT - KNOWN-USER VALIDATION ONLY")
    print("=" * 76)
    print(f"Validation FAR limit per head: {args.target_far:.2%}")
    print()

    print("[DEFAULT]")
    print(
        f"Gesture Tg={default_gesture['threshold']:.9f} "
        f"FAR={default_gesture['far']:.2%} "
        f"FRR={default_gesture['frr']:.2%}"
    )
    print(
        f"User    Tu={default_user['threshold']:.9f} "
        f"FAR={default_user['far']:.2%} "
        f"FRR={default_user['frr']:.2%}"
    )

    print()
    print("[DEMO RELAXED]")
    print(
        f"Gesture Tg={relaxed_gesture['threshold']:.9f} "
        f"FAR={relaxed_gesture['far']:.2%} "
        f"FRR={relaxed_gesture['frr']:.2%}"
    )
    print(
        f"User    Tu={relaxed_user['threshold']:.9f} "
        f"FAR={relaxed_user['far']:.2%} "
        f"FRR={relaxed_user['frr']:.2%}"
    )

    print()
    print(f"saved: {output_path}")
    print()
    print("P08-P10 / final-test data used: NO")
    print("Use this JSON with finalize_release_v1_1_1.py")


if __name__ == "__main__":
    main()
