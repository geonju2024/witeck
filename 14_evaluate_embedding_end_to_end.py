"""Evaluate the basic embedding pipeline end to end.

gesture embedding -> predicted G1~G5 -> predicted-gesture enrollment template
-> cosine similarity -> accept/reject

For a genuine trial, both the gesture route and authentication must be correct.
For an impostor trial, acceptance on any routed template is a false accept.
"""

from __future__ import annotations

import argparse
import csv
import importlib
from pathlib import Path

import numpy as np
import torch

from two_stage_common import (
    apply_duration_stats,
    apply_sequence_stats,
    file_sha256,
    load_dataset,
    torch_load_compat,
)


embedding_module = importlib.import_module("08_train_embedding")
Embedding1DCNN = embedding_module.Embedding1DCNN
TRAIN_USERS = embedding_module.TRAIN_USERS
UNSEEN_USERS = embedding_module.UNSEEN_USERS
binary_score_metrics = embedding_module.binary_metrics
cosine_scores = embedding_module.cosine_scores
extract_embeddings = embedding_module.extract_embeddings


def binary_decision_metrics(labels, pred):
    labels = np.asarray(labels, dtype=np.int64)
    pred = np.asarray(pred, dtype=np.int64)
    genuine = labels == 1
    impostor = labels == 0
    tp = int(np.sum((pred == 1) & genuine))
    fn = int(np.sum((pred == 0) & genuine))
    fp = int(np.sum((pred == 1) & impostor))
    tn = int(np.sum((pred == 0) & impostor))
    far = fp / max(int(impostor.sum()), 1)
    frr = fn / max(int(genuine.sum()), 1)
    return {
        "accuracy": float((tp + tn) / max(len(labels), 1)),
        "balanced_accuracy": float(((1.0 - far) + (1.0 - frr)) / 2.0),
        "far": float(far),
        "frr": float(frr),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


@torch.no_grad()
def predict_gestures(model, X, duration, device, batch_size=128):
    loader = embedding_module.make_loader(
        X,
        duration,
        np.zeros(len(X), dtype=np.int64),
        batch_size,
        shuffle=False,
    )
    pred = []
    model.eval()
    for xb, db, _ in loader:
        _, logits = model(xb.to(device), db.to(device))
        pred.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(pred)


def load_model(checkpoint_path, current_hash, device):
    ckpt = torch_load_compat(checkpoint_path, device)
    if ckpt.get("dataset_sha256") != current_hash:
        raise RuntimeError(
            f"dataset SHA mismatch: {checkpoint_path}\n"
            f"checkpoint={ckpt.get('dataset_sha256')}\ncurrent={current_hash}"
        )
    model = Embedding1DCNN(
        input_dim=int(ckpt["input_dim"]),
        num_classes=int(ckpt["num_classes"]),
        embedding_dim=int(ckpt["embedding_dim"]),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return ckpt, model


def normalize_for_checkpoint(X_seq, duration, ckpt):
    X = apply_sequence_stats(X_seq, ckpt["sequence_mean"], ckpt["sequence_std"])
    d = apply_duration_stats(duration, ckpt["duration_mean"], ckpt["duration_std"])
    return X, d


def mean_metrics(rows, prefix):
    names = ("accuracy", "balanced_accuracy", "far", "frr")
    return {
        name: float(np.mean([row[f"{prefix}_{name}"] for row in rows]))
        for name in names
    }


def main():
    project_dir = Path(__file__).resolve().parent
    base_dir = project_dir / "output" / "basic_embedding_pipeline"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default=str(project_dir / "dataset" / "dataset_1955_recent8_updated_20260905.npz"),
    )
    parser.add_argument(
        "--gesture-checkpoint",
        default=str(base_dir / "gesture" / "gesture_embedding_1dcnn.pt"),
    )
    parser.add_argument(
        "--auth-checkpoint",
        default=str(base_dir / "user_auth" / "embedding_1dcnn.pt"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(base_dir / "end_to_end"),
    )
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    _, X_seq, duration, meta, _, _ = load_dataset(args.data)
    dataset_hash = file_sha256(args.data)
    gesture_ckpt, gesture_model = load_model(args.gesture_checkpoint, dataset_hash, device)
    auth_ckpt, auth_model = load_model(args.auth_checkpoint, dataset_hash, device)
    threshold = float(auth_ckpt["threshold"])
    enroll_count = int(auth_ckpt.get("enrollment_per_gesture", 3))

    X_gesture, d_gesture = normalize_for_checkpoint(X_seq, duration, gesture_ckpt)
    X_auth, d_auth = normalize_for_checkpoint(X_seq, duration, auth_ckpt)
    gesture_pred_label = predict_gestures(
        gesture_model, X_gesture, d_gesture, device, args.batch_size
    )
    gesture_classes = [str(x) for x in gesture_ckpt["classes"]]
    gesture_pred = np.asarray([gesture_classes[int(i)] for i in gesture_pred_label])
    auth_embeddings = extract_embeddings(
        auth_model, X_auth, d_auth, device, args.batch_size
    )

    performer = meta["performer"]
    gesture = meta["gesture"]
    session = meta["session"]
    gestures = sorted(np.unique(gesture).tolist())

    templates = {}
    enrollment_indices = {}
    for target_user in UNSEEN_USERS:
        user_idx = np.where(performer == target_user)[0]
        enroll_session = sorted(np.unique(session[user_idx]).tolist())[0]
        for g in gestures:
            candidates = np.where(
                (performer == target_user)
                & (gesture == g)
                & (session == enroll_session)
            )[0]
            if len(candidates) < enroll_count:
                raise RuntimeError(
                    f"{target_user}/{g}: need {enroll_count} enrollment samples, found {len(candidates)}"
                )
            selected = candidates[:enroll_count]
            templates[(target_user, g)] = auth_embeddings[selected].mean(axis=0)
            enrollment_indices[(target_user, g)] = selected

    rows = []
    trials = []
    global_labels = []
    global_direct_scores = []
    global_e2e_pred = []
    global_route_correct = []

    for target_user in UNSEEN_USERS:
        user_sessions = sorted(
            np.unique(session[performer == target_user]).tolist()
        )
        enroll_session = user_sessions[0]
        for true_gesture in gestures:
            genuine_idx = np.where(
                (performer == target_user)
                & (gesture == true_gesture)
                & (session != enroll_session)
            )[0]
            impostor_idx = np.where(
                np.isin(performer, UNSEEN_USERS)
                & (performer != target_user)
                & (gesture == true_gesture)
            )[0]
            trial_idx = np.concatenate([genuine_idx, impostor_idx])
            labels = np.concatenate([
                np.ones(len(genuine_idx), dtype=np.int64),
                np.zeros(len(impostor_idx), dtype=np.int64),
            ])

            direct_scores = cosine_scores(
                auth_embeddings[trial_idx], templates[(target_user, true_gesture)]
            )
            direct_pred = (direct_scores >= threshold).astype(np.int64)

            routed_scores = np.asarray([
                float(cosine_scores(
                    auth_embeddings[idx:idx + 1],
                    templates[(target_user, str(gesture_pred[idx]))],
                )[0])
                for idx in trial_idx
            ])
            route_correct = (gesture_pred[trial_idx] == true_gesture)
            routed_accept = routed_scores >= threshold
            e2e_pred = (
                routed_accept
                & ((labels == 0) | route_correct)
            ).astype(np.int64)

            direct_metrics = binary_score_metrics(labels, direct_scores, threshold)
            e2e_metrics = binary_decision_metrics(labels, e2e_pred)
            row = {
                "claimed_user": target_user,
                "true_gesture": true_gesture,
                "genuine_samples": len(genuine_idx),
                "impostor_samples": len(impostor_idx),
                "route_accuracy": float(route_correct.mean()),
            }
            for name in ("accuracy", "balanced_accuracy", "far", "frr"):
                row[f"direct_{name}"] = direct_metrics[name]
                row[f"end_to_end_{name}"] = e2e_metrics[name]
            rows.append(row)

            for pos, idx in enumerate(trial_idx):
                trials.append({
                    "claimed_user": target_user,
                    "sample_index": int(idx),
                    "actual_user": str(performer[idx]),
                    "session": str(session[idx]),
                    "true_gesture": true_gesture,
                    "predicted_gesture": str(gesture_pred[idx]),
                    "genuine": int(labels[pos]),
                    "route_correct": int(route_correct[pos]),
                    "direct_score": float(direct_scores[pos]),
                    "direct_accept": int(direct_pred[pos]),
                    "routed_score": float(routed_scores[pos]),
                    "end_to_end_accept": int(e2e_pred[pos]),
                })

            global_labels.extend(labels.tolist())
            global_direct_scores.extend(direct_scores.tolist())
            global_e2e_pred.extend(e2e_pred.tolist())
            global_route_correct.extend(route_correct.tolist())

    global_labels = np.asarray(global_labels, dtype=np.int64)
    direct_global = binary_score_metrics(
        global_labels, np.asarray(global_direct_scores), threshold
    )
    e2e_global = binary_decision_metrics(
        global_labels, np.asarray(global_e2e_pred)
    )
    direct_macro = mean_metrics(rows, "direct")
    e2e_macro = mean_metrics(rows, "end_to_end")
    route_accuracy = float(np.mean(global_route_correct))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "per_user_gesture_metrics.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "trial_results.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(trials[0].keys()))
        writer.writeheader()
        writer.writerows(trials)

    with (output_dir / "summary.txt").open("w", encoding="utf-8") as f:
        f.write("Basic 1D-CNN embedding end-to-end evaluation\n")
        f.write(f"dataset={Path(args.data).resolve()}\n")
        f.write(f"dataset_sha256={dataset_hash}\n")
        f.write(f"threshold={threshold:.6f}\n")
        f.write(f"enrollment_per_gesture={enroll_count}\n")
        f.write(f"trials={len(global_labels)}\n")
        f.write(f"route_accuracy={route_accuracy:.6f}\n")
        for prefix, result in (
            ("direct_global", direct_global),
            ("end_to_end_global", e2e_global),
            ("direct_macro", direct_macro),
            ("end_to_end_macro", e2e_macro),
        ):
            for name in ("accuracy", "balanced_accuracy", "far", "frr"):
                f.write(f"{prefix}_{name}={result[name]:.6f}\n")

    print("=" * 76)
    print("Basic 1D-CNN embedding end-to-end result")
    print("=" * 76)
    print(f"trials={len(global_labels)} threshold={threshold:.6f} route_acc={route_accuracy:.4f}")
    print(
        "Direct auth | "
        f"Acc={direct_global['accuracy']:.4f} BalAcc={direct_global['balanced_accuracy']:.4f} "
        f"FAR={direct_global['far']:.4f} FRR={direct_global['frr']:.4f}"
    )
    print(
        "End-to-end | "
        f"Acc={e2e_global['accuracy']:.4f} BalAcc={e2e_global['balanced_accuracy']:.4f} "
        f"FAR={e2e_global['far']:.4f} FRR={e2e_global['frr']:.4f}"
    )
    print(f"Saved: {output_dir / 'summary.txt'}")


if __name__ == "__main__":
    main()
