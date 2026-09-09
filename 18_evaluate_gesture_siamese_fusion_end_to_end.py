"""End-to-end gesture routing plus leakage-safe multi-gesture authentication.

Each transaction contains one sample for every requested gesture. Embedding1DCNN
predicts each gesture, the prediction routes the sample to a Siamese enrollment
template, and the routed cosine scores are averaged before thresholding.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import importlib.util
import itertools
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve


HERE = Path(__file__).resolve().parent
embedding_module = importlib.import_module("08_train_embedding")


class BaselineSiamese1DCNN(nn.Module):
    """Checkpoint-compatible baseline encoder used by the best prior run."""

    def __init__(self, input_dim: int, embedding_dim: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.20),
            nn.Conv1d(64, 96, kernel_size=3, padding=1),
            nn.BatchNorm1d(96), nn.ReLU(),
            nn.Conv1d(96, 128, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.30),
            nn.AdaptiveAvgPool1d(1),
        )
        self.embedding_head = nn.Linear(128 + 1, embedding_dim)

    def forward(self, x, duration):
        h = self.features(x.transpose(1, 2)).squeeze(-1)
        h = torch.cat([h, duration.unsqueeze(1)], dim=1)
        return F.normalize(self.embedding_head(h), p=2, dim=1)


@torch.no_grad()
def predict_gestures(model, X, duration, device, batch_size=128):
    loader = embedding_module.make_loader(
        X, duration, np.zeros(len(X), dtype=np.int64), batch_size, shuffle=False
    )
    output = []
    model.eval()
    for xb, db, _ in loader:
        _, logits = model(xb.to(device), db.to(device))
        output.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(output)


def load_protocol():
    source = HERE / "09_train_siamese_embedding.py"
    spec = importlib.util.spec_from_file_location("witeck_siamese_protocol", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load protocol: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def metrics(labels: np.ndarray, scores: np.ndarray, threshold: float | None = None):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1.0 - tpr
    i = int(np.nanargmin(np.abs(fpr - fnr)))
    eer = float((fpr[i] + fnr[i]) / 2.0)
    if threshold is None:
        negatives = scores[labels == 0]
        positives = scores[labels == 1]
        max_negative = float(np.max(negatives))
        min_positive = float(np.min(positives))
        threshold = (
            (max_negative + min_positive) / 2.0
            if max_negative < min_positive
            else float(thresholds[i])
        )
    pred = scores >= threshold
    far = float(np.mean(pred[labels == 0]))
    frr = float(np.mean(~pred[labels == 1]))
    return {
        "auc": float(roc_auc_score(labels, scores)),
        "eer": eer,
        "eer_threshold": float(thresholds[i]),
        "threshold": float(threshold),
        "accuracy": float(np.mean(pred == labels)),
        "balanced_accuracy": float(1.0 - (far + frr) / 2.0),
        "far": far,
        "frr": frr,
        "transactions": int(len(labels)),
        "genuine_transactions": int(np.sum(labels == 1)),
        "impostor_transactions": int(np.sum(labels == 0)),
    }


def ordered(indices: np.ndarray, names: np.ndarray) -> np.ndarray:
    return np.asarray(sorted(indices.tolist(), key=lambda i: (str(names[i]), int(i))))


def session_indices(meta, user: str, gesture: str, session: str) -> np.ndarray:
    return np.where(
        (meta["performer"] == user)
        & (meta["gesture"] == gesture)
        & (meta["session"] == session)
    )[0]


def build_templates(embeddings, meta, users, enroll_count, mode):
    templates = {}
    enrollment_sessions = {}
    gestures = sorted(np.unique(meta["gesture"]).tolist())
    names = meta.get("name", np.arange(len(meta["performer"])).astype(str))
    for user in users:
        user_idx = np.where(meta["performer"] == user)[0]
        sessions = sorted(np.unique(meta["session"][user_idx]).tolist())
        if len(sessions) < 2:
            continue
        enroll_session = sessions[-2] if mode == "calibration" else sessions[0]
        user_templates = {}
        for gesture in gestures:
            idx = ordered(session_indices(meta, user, gesture, enroll_session), names)
            if len(idx) >= enroll_count:
                template = embeddings[idx[:enroll_count]].mean(axis=0)
                template /= max(float(np.linalg.norm(template)), 1e-12)
                user_templates[gesture] = template
        templates[user] = user_templates
        enrollment_sessions[user] = enroll_session
    return templates, enrollment_sessions


def fused_transactions(
    embeddings,
    meta,
    target_users,
    probe_users,
    templates,
    enrollment_sessions,
    gesture_combo,
    mode,
    predicted_gestures=None,
):
    names = meta.get("name", np.arange(len(meta["performer"])).astype(str))
    labels, scores = [], []
    correct_routes = 0
    total_routes = 0
    for target in target_users:
        if target not in templates or any(g not in templates[target] for g in gesture_combo):
            continue
        for probe in probe_users:
            probe_idx = np.where(meta["performer"] == probe)[0]
            sessions = sorted(np.unique(meta["session"][probe_idx]).tolist())
            if not sessions:
                continue
            if mode == "calibration":
                sessions = [sessions[-1]]
            elif probe == target:
                sessions = [s for s in sessions if s != enrollment_sessions[target]]
            for session in sessions:
                per_gesture = []
                for gesture in gesture_combo:
                    idx = ordered(session_indices(meta, probe, gesture, session), names)
                    per_gesture.append(idx)
                count = min((len(x) for x in per_gesture), default=0)
                for j in range(count):
                    gesture_scores = []
                    for gesture, idx in zip(gesture_combo, per_gesture):
                        sample_idx = int(idx[j])
                        routed_gesture = (
                            gesture
                            if predicted_gestures is None
                            else str(predicted_gestures[sample_idx])
                        )
                        gesture_scores.append(
                            float(np.dot(
                                embeddings[sample_idx], templates[target][routed_gesture]
                            ))
                        )
                        correct_routes += int(routed_gesture == gesture)
                        total_routes += 1
                    scores.append(float(np.mean(gesture_scores)))
                    labels.append(int(probe == target))
    route_accuracy = correct_routes / max(total_routes, 1)
    return (
        np.asarray(labels, dtype=np.int64),
        np.asarray(scores, dtype=np.float64),
        float(route_accuracy),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gesture-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--enroll-counts", default="3,5,8")
    parser.add_argument("--min-gestures", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    protocol = load_protocol()
    protocol.seed_everything()
    device = protocol.select_device()
    _, X_raw, duration, meta, _, D = protocol.load_dataset(args.data)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    gesture_checkpoint = torch.load(
        args.gesture_checkpoint, map_location="cpu", weights_only=False
    )
    gesture_model = embedding_module.Embedding1DCNN(
        input_dim=int(gesture_checkpoint["input_dim"]),
        num_classes=int(gesture_checkpoint["num_classes"]),
        embedding_dim=int(gesture_checkpoint["embedding_dim"]),
    ).to(device)
    gesture_model.load_state_dict(gesture_checkpoint["model_state_dict"])
    gesture_model.eval()
    X_gesture = protocol.apply_sequence_stats(
        X_raw, gesture_checkpoint["sequence_mean"], gesture_checkpoint["sequence_std"]
    )
    duration_gesture = protocol.apply_duration_stats(
        duration, gesture_checkpoint["duration_mean"], gesture_checkpoint["duration_std"]
    )
    predicted_ids = predict_gestures(
        gesture_model, X_gesture, duration_gesture, device, args.batch_size
    )
    gesture_classes = [str(x) for x in gesture_checkpoint["classes"]]
    predicted_gestures = np.asarray(
        [gesture_classes[int(i)] for i in predicted_ids], dtype=str
    )

    X = protocol.apply_sequence_stats(
        X_raw, checkpoint["sequence_mean"], checkpoint["sequence_std"]
    )
    X[:, :, -1] = X_raw[:, :, -1]
    duration = protocol.apply_duration_stats(
        duration, checkpoint["duration_mean"], checkpoint["duration_std"]
    )
    state_keys = checkpoint["model_state_dict"].keys()
    model_class = (
        BaselineSiamese1DCNN
        if any(key.startswith("features.") for key in state_keys)
        else protocol.Siamese1DCNN
    )
    model = model_class(
        input_dim=int(checkpoint.get("input_dim", D)),
        embedding_dim=int(checkpoint.get("embedding_dim", 128)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    embeddings = protocol.extract_embeddings(
        model, X, duration, device, batch_size=args.batch_size
    )

    known_users = [str(x) for x in checkpoint.get("train_users", protocol.TRAIN_USERS)]
    unseen_users = [str(x) for x in checkpoint.get("unseen_users", protocol.UNSEEN_USERS)]
    gestures = sorted(np.unique(meta["gesture"]).tolist())
    enroll_counts = [int(x) for x in args.enroll_counts.split(",") if x.strip()]

    rows = []
    for enroll_count in enroll_counts:
        cal_templates, cal_sessions = build_templates(
            embeddings, meta, known_users, enroll_count, "calibration"
        )
        test_templates, test_sessions = build_templates(
            embeddings, meta, unseen_users, enroll_count, "test"
        )
        for size in range(args.min_gestures, len(gestures) + 1):
            for combo in itertools.combinations(gestures, size):
                cal_y, cal_s, cal_route_accuracy = fused_transactions(
                    embeddings, meta, known_users, known_users,
                    cal_templates, cal_sessions, combo, "calibration",
                    predicted_gestures,
                )
                if len(np.unique(cal_y)) < 2:
                    continue
                calibration = metrics(cal_y, cal_s)
                calibration["route_accuracy"] = cal_route_accuracy
                test_y, test_s, test_route_accuracy = fused_transactions(
                    embeddings, meta, unseen_users, unseen_users,
                    test_templates, test_sessions, combo, "test",
                    predicted_gestures,
                )
                if len(np.unique(test_y)) < 2:
                    continue
                final = metrics(test_y, test_s, calibration["threshold"])
                final["route_accuracy"] = test_route_accuracy
                rows.append({
                    "enroll_count": enroll_count,
                    "gestures": "+".join(combo),
                    "gesture_count": size,
                    "calibration": calibration,
                    "final_test": final,
                })

    if not rows:
        raise RuntimeError("No valid fusion configuration was produced")
    selected = max(
        rows,
        key=lambda r: (
            r["calibration"]["balanced_accuracy"],
            r["calibration"]["auc"],
            r["gesture_count"],
        ),
    )
    ranked = sorted(
        rows,
        key=lambda r: (
            r["calibration"]["balanced_accuracy"],
            r["calibration"]["auc"],
        ),
        reverse=True,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "selection_rule": "maximum calibration balanced accuracy; final test never used",
        "known_users": known_users,
        "unseen_users": unseen_users,
        "selected": selected,
        "all_configurations": ranked,
    }
    (output_dir / "fusion_results.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    with (output_dir / "fusion_results.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "selected", "enroll_count", "gestures", "gesture_count",
            "calibration_auc", "calibration_eer", "threshold",
            "calibration_balanced_accuracy", "final_auc", "final_eer",
            "final_accuracy", "final_balanced_accuracy", "final_far", "final_frr",
            "final_route_accuracy", "final_transactions",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in ranked:
            writer.writerow({
                "selected": row is selected,
                "enroll_count": row["enroll_count"],
                "gestures": row["gestures"],
                "gesture_count": row["gesture_count"],
                "calibration_auc": row["calibration"]["auc"],
                "calibration_eer": row["calibration"]["eer"],
                "threshold": row["calibration"]["threshold"],
                "calibration_balanced_accuracy": row["calibration"]["balanced_accuracy"],
                "final_auc": row["final_test"]["auc"],
                "final_eer": row["final_test"]["eer"],
                "final_accuracy": row["final_test"]["accuracy"],
                "final_balanced_accuracy": row["final_test"]["balanced_accuracy"],
                "final_far": row["final_test"]["far"],
                "final_frr": row["final_test"]["frr"],
                "final_route_accuracy": row["final_test"]["route_accuracy"],
                "final_transactions": row["final_test"]["transactions"],
            })

    print(json.dumps({"selected": selected, "top_calibration": ranked[:10]}, indent=2))


if __name__ == "__main__":
    main()



