"""Generate deployment preprocessing, threshold and enrollment reports.

All threshold choices use known-user cross-session validation scores only.
Unseen-user scores are reported after selection and never used to choose a value.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
import torch

from two_stage_common import apply_duration_stats, apply_sequence_stats, load_dataset


root = Path(__file__).resolve().parent
release = root / "archive" / "releases" / "ai_release_full"
data_path = root / "dataset" / "dataset_1955_recent8_updated_20260905_hand_only.npz"
auth_path = release / "weights" / "auth_handonly_supcon_v1.pt"
gesture_path = release / "weights" / "gesture_handonly_1dcnn_v1.pt"

base = importlib.import_module("08_train_embedding")
tailored = importlib.import_module("41_train_hand_only_supcon")


def metric_dict(labels, scores, threshold):
    result = base.binary_metrics(labels, scores, threshold)
    return {
        key: float(result[key])
        for key in ("accuracy", "balanced_accuracy", "far", "frr")
    }


def threshold_for_far(labels, scores, maximum_far):
    scores = np.asarray(scores, dtype=np.float64)
    thresholds = np.concatenate(
        [np.unique(scores), [np.nextafter(float(scores.max()), float("inf"))]]
    )
    candidates = []
    for threshold in thresholds:
        metrics = metric_dict(labels, scores, float(threshold))
        if metrics["far"] <= maximum_far + 1e-12:
            candidates.append(
                (metrics["frr"], maximum_far - metrics["far"], float(threshold), metrics)
            )
    if not candidates:
        raise RuntimeError(f"No threshold satisfies FAR <= {maximum_far}")
    _, _, threshold, metrics = min(candidates, key=lambda item: item[:3])
    return threshold, metrics


def calibration_scores(embeddings, meta, enrollment_count):
    performer = meta["performer"]
    gesture = meta["gesture"]
    session = meta["session"]
    by_gesture = {name: {"labels": [], "scores": []} for name in sorted(np.unique(gesture))}

    for target in base.TRAIN_USERS:
        target_indices = np.where(performer == target)[0]
        sessions = sorted(np.unique(session[target_indices]))
        enroll_session, test_session = sessions[-2], sessions[-1]
        for name in by_gesture:
            enrollment = np.where(
                (performer == target) & (gesture == name) & (session == enroll_session)
            )[0]
            genuine = np.where(
                (performer == target) & (gesture == name) & (session == test_session)
            )[0]
            if len(enrollment) < enrollment_count or not len(genuine):
                continue
            template = embeddings[enrollment[:enrollment_count]].mean(axis=0)
            impostors = []
            for other in base.TRAIN_USERS:
                if other == target:
                    continue
                other_indices = np.where(performer == other)[0]
                other_latest = sorted(np.unique(session[other_indices]))[-1]
                impostors.extend(
                    np.where(
                        (performer == other)
                        & (gesture == name)
                        & (session == other_latest)
                    )[0].tolist()
                )
            genuine_scores = base.cosine_scores(embeddings[genuine], template)
            impostor_scores = base.cosine_scores(
                embeddings[np.asarray(impostors, dtype=np.int64)], template
            )
            by_gesture[name]["labels"].extend([1] * len(genuine_scores))
            by_gesture[name]["scores"].extend(genuine_scores.tolist())
            by_gesture[name]["labels"].extend([0] * len(impostor_scores))
            by_gesture[name]["scores"].extend(impostor_scores.tolist())

    for values in by_gesture.values():
        values["labels"] = np.asarray(values["labels"], dtype=np.int64)
        values["scores"] = np.asarray(values["scores"], dtype=np.float64)
    labels = np.concatenate([values["labels"] for values in by_gesture.values()])
    scores = np.concatenate([values["scores"] for values in by_gesture.values()])
    return labels, scores, by_gesture


def unseen_scores(embeddings, meta, enrollment_count):
    performer = meta["performer"]
    gesture = meta["gesture"]
    session = meta["session"]
    by_gesture = {name: {"labels": [], "scores": []} for name in sorted(np.unique(gesture))}
    for target in base.UNSEEN_USERS:
        target_indices = np.where(performer == target)[0]
        enroll_session = sorted(np.unique(session[target_indices]))[0]
        for name in by_gesture:
            enrollment = np.where(
                (performer == target) & (gesture == name) & (session == enroll_session)
            )[0]
            if len(enrollment) < enrollment_count:
                continue
            genuine = np.where(
                (performer == target) & (gesture == name) & (session != enroll_session)
            )[0]
            impostors = np.where(
                np.isin(performer, base.UNSEEN_USERS)
                & (performer != target)
                & (gesture == name)
            )[0]
            template = embeddings[enrollment[:enrollment_count]].mean(axis=0)
            genuine_scores = base.cosine_scores(embeddings[genuine], template)
            impostor_scores = base.cosine_scores(embeddings[impostors], template)
            by_gesture[name]["labels"].extend([1] * len(genuine_scores))
            by_gesture[name]["scores"].extend(genuine_scores.tolist())
            by_gesture[name]["labels"].extend([0] * len(impostor_scores))
            by_gesture[name]["scores"].extend(impostor_scores.tolist())
    for values in by_gesture.values():
        values["labels"] = np.asarray(values["labels"], dtype=np.int64)
        values["scores"] = np.asarray(values["scores"], dtype=np.float64)
    labels = np.concatenate([values["labels"] for values in by_gesture.values()])
    scores = np.concatenate([values["scores"] for values in by_gesture.values()])
    return labels, scores, by_gesture


def combine_scheme_metrics(by_gesture, thresholds):
    labels = []
    predictions = []
    for name, values in by_gesture.items():
        labels.extend(values["labels"].tolist())
        predictions.extend((values["scores"] >= thresholds[name]).astype(np.int64).tolist())
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    genuine = labels == 1
    impostor = labels == 0
    far = float(np.mean(predictions[impostor] == 1))
    frr = float(np.mean(predictions[genuine] == 0))
    accuracy = float(np.mean(predictions == labels))
    return {
        "accuracy": accuracy,
        "balanced_accuracy": ((1.0 - far) + (1.0 - frr)) / 2.0,
        "far": far,
        "frr": frr,
    }


def main():
    checkpoint = torch.load(auth_path, map_location="cpu", weights_only=False)
    model = tailored.HandOnlySupCon1DCNN(
        checkpoint["input_dim"], checkpoint["num_classes"], checkpoint["embedding_dim"]
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    _, sequence, duration, meta, _, _ = load_dataset(str(data_path))
    normalized = apply_sequence_stats(
        sequence, checkpoint["sequence_mean"], checkpoint["sequence_std"]
    )
    normalized_duration = apply_duration_stats(
        duration, checkpoint["duration_mean"], checkpoint["duration_std"]
    )
    embeddings = base.extract_embeddings(
        model, normalized, normalized_duration, torch.device("cpu"), batch_size=256
    )

    enrollment_results = []
    n3_calibration = None
    n3_unseen = None
    for count in (1, 3, 4):
        cal_labels, cal_scores, cal_by_gesture = calibration_scores(embeddings, meta, count)
        test_labels, test_scores, test_by_gesture = unseen_scores(embeddings, meta, count)
        eer_threshold, eer, cal_eer_metrics = base.find_eer_threshold(cal_labels, cal_scores)
        enrollment_results.append(
            {
                "enrollment_takes": count,
                "enrollment_session_policy": "one_session",
                "validation_threshold_eer": float(eer_threshold),
                "validation_eer": float(eer),
                "validation_metrics": metric_dict(cal_labels, cal_scores, eer_threshold),
                "unseen_metrics": metric_dict(test_labels, test_scores, eer_threshold),
            }
        )
        if count == 3:
            n3_calibration = (cal_labels, cal_scores, cal_by_gesture)
            n3_unseen = (test_labels, test_scores, test_by_gesture)

    cal_labels, cal_scores, cal_by_gesture = n3_calibration
    test_labels, test_scores, test_by_gesture = n3_unseen
    eer_threshold, eer, _ = base.find_eer_threshold(cal_labels, cal_scores)
    far5_threshold, far5_val = threshold_for_far(cal_labels, cal_scores, 0.05)
    far1_threshold, far1_val = threshold_for_far(cal_labels, cal_scores, 0.01)
    operating_points = {
        "eer": {
            "threshold": float(eer_threshold),
            "validation": metric_dict(cal_labels, cal_scores, eer_threshold),
            "unseen_analysis": metric_dict(test_labels, test_scores, eer_threshold),
        },
        "far_5_percent": {
            "threshold": float(far5_threshold),
            "validation": far5_val,
            "unseen_analysis": metric_dict(test_labels, test_scores, far5_threshold),
        },
        "far_1_percent": {
            "threshold": float(far1_threshold),
            "validation": far1_val,
            "unseen_analysis": metric_dict(test_labels, test_scores, far1_threshold),
        },
    }

    gesture_thresholds = {}
    gesture_validation = {}
    for name, values in cal_by_gesture.items():
        threshold, gesture_eer, _ = base.find_eer_threshold(values["labels"], values["scores"])
        gesture_thresholds[name] = float(threshold)
        gesture_validation[name] = {
            "threshold": float(threshold),
            "eer": float(gesture_eer),
            **metric_dict(values["labels"], values["scores"], threshold),
        }
    scheme_comparison = {
        "global_eer": {
            "threshold": float(eer_threshold),
            "validation": metric_dict(cal_labels, cal_scores, eer_threshold),
            "unseen_analysis": metric_dict(test_labels, test_scores, eer_threshold),
        },
        "gesture_specific_eer": {
            "thresholds": gesture_thresholds,
            "validation_by_gesture": gesture_validation,
            "validation_combined": combine_scheme_metrics(cal_by_gesture, gesture_thresholds),
            "unseen_analysis_combined": combine_scheme_metrics(test_by_gesture, gesture_thresholds),
        },
    }

    threshold_payload = {
        "model_version": "handonly-supcon-v1.0.0",
        "scheme": "global",
        "selected_operating_point": "far_1_percent",
        "enrollment_takes_per_gesture": 3,
        "enrollment_gestures_per_user": 1,
        "enrollment_total_takes_per_user": 3,
        "enrollment_policy": "one_selected_personal_gesture",
        "selection_data": "known-user cross-session validation only",
        "operating_points": operating_points,
        "scheme_comparison": scheme_comparison,
        "enrollment_sweep": enrollment_results,
    }
    (release / "thresholds.json").write_text(
        json.dumps(threshold_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    gesture_checkpoint = torch.load(gesture_path, map_location="cpu", weights_only=False)
    preprocess = {
        "model_version": "handonly-supcon-v1.0.0",
        "gesture_model_version": "handonly-gesture-1dcnn-v1.0.0",
        "input": {
            "raw": "MediaPipe Hand 21 xyz points + tMs + camera width/height",
            "sequence_length": 32,
            "feature_dim": 127,
            "feature_layout": "hand_xyz_63+hand_velocity_63+valid_mask_1",
            "velocity": "normalized_coordinate_units_per_second",
            "right_hand_only": True,
            "minimum_input_frames": 8,
            "minimum_valid_frames": 8,
            "minimum_duration_ms": 750,
        },
        "authentication": {
            "sequence_mean": np.asarray(checkpoint["sequence_mean"]).reshape(-1).tolist(),
            "sequence_std": np.asarray(checkpoint["sequence_std"]).reshape(-1).tolist(),
            "duration_mean": float(checkpoint["duration_mean"]),
            "duration_std": float(checkpoint["duration_std"]),
            "embedding_dim": int(checkpoint["embedding_dim"]),
            "l2_normalized": True,
        },
        "gesture": {
            "sequence_mean": np.asarray(gesture_checkpoint["sequence_mean"]).reshape(-1).tolist(),
            "sequence_std": np.asarray(gesture_checkpoint["sequence_std"]).reshape(-1).tolist(),
            "duration_mean": float(gesture_checkpoint["duration_mean"]),
            "duration_std": float(gesture_checkpoint["duration_std"]),
            "classes": [str(value) for value in gesture_checkpoint["classes"]],
        },
    }
    (release / "preprocess.json").write_text(
        json.dumps(preprocess, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# Hand-only deployment calibration report",
        "",
        "Thresholds were selected on known-user cross-session validation only.",
        "Unseen-user results are post-selection analysis.",
        "",
        "## Global operating points (3 enrollment takes per gesture)",
        "",
        "| Point | Threshold | Validation FAR | Validation FRR | Unseen FAR | Unseen FRR | Unseen accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, key in (("EER", "eer"), ("FAR≤5%", "far_5_percent"), ("FAR≤1%", "far_1_percent")):
        item = operating_points[key]
        lines.append(
            f"| {label} | {item['threshold']:.6f} | {item['validation']['far']:.2%} | "
            f"{item['validation']['frr']:.2%} | {item['unseen_analysis']['far']:.2%} | "
            f"{item['unseen_analysis']['frr']:.2%} | {item['unseen_analysis']['accuracy']:.2%} |"
        )
    lines.extend(
        [
            "",
            "## Enrollment sweep (global validation-EER threshold)",
            "",
            "| Takes/gesture | Validation EER | Unseen accuracy | Unseen FAR | Unseen FRR |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for item in enrollment_results:
        unseen = item["unseen_metrics"]
        lines.append(
            f"| {item['enrollment_takes']} | {item['validation_eer']:.2%} | "
            f"{unseen['accuracy']:.2%} | {unseen['far']:.2%} | {unseen['frr']:.2%} |"
        )
    lines.extend(
        [
            "",
            "## Threshold scheme comparison (3 takes)",
            "",
            "| Scheme | Validation accuracy | Validation FAR | Validation FRR | Unseen accuracy | Unseen FAR | Unseen FRR |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for label, key in (("Global", "global_eer"), ("Gesture-specific", "gesture_specific_eer")):
        item = scheme_comparison[key]
        validation = item.get("validation", item.get("validation_combined"))
        unseen = item.get("unseen_analysis", item.get("unseen_analysis_combined"))
        lines.append(
            f"| {label} | {validation['accuracy']:.2%} | {validation['far']:.2%} | "
            f"{validation['frr']:.2%} | {unseen['accuracy']:.2%} | {unseen['far']:.2%} | "
            f"{unseen['frr']:.2%} |"
        )
    (release / "calibration_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))


if __name__ == "__main__":
    main()
