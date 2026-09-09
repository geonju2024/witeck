"""Validation-only score fusion for the two complementary SupCon models.

Fusion weights and thresholds are selected with known-user calibration trials.
P08~P10 trials are used once for final reporting and never for selection.
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


base = importlib.import_module("08_train_embedding")
evaluator = importlib.import_module("14_evaluate_embedding_end_to_end")
conditioned = importlib.import_module("18_train_gesture_conditioned_supcon")
TRAIN_USERS = base.TRAIN_USERS


def metrics(labels, pred):
    labels = np.asarray(labels, dtype=np.int64)
    pred = np.asarray(pred, dtype=np.int64)
    pos = labels == 1
    neg = labels == 0
    tp = int(np.sum((pred == 1) & pos))
    tn = int(np.sum((pred == 0) & neg))
    fp = int(np.sum((pred == 1) & neg))
    fn = int(np.sum((pred == 0) & pos))
    far = fp / max(int(neg.sum()), 1)
    frr = fn / max(int(pos.sum()), 1)
    return {
        "accuracy": (tp + tn) / max(len(labels), 1),
        "balanced_accuracy": ((1.0 - far) + (1.0 - frr)) / 2.0,
        "far": far,
        "frr": frr,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def candidate_thresholds(scores):
    values = np.unique(np.asarray(scores, dtype=np.float64))
    if len(values) == 1:
        return values
    mids = (values[:-1] + values[1:]) / 2.0
    return np.concatenate([[values[0] - 1e-8], mids, [values[-1] + 1e-8]])


def select_fusion(labels, score_a, score_b, objective="accuracy"):
    best = None
    for weight_a in np.linspace(0.0, 1.0, 21):
        fused = weight_a * score_a + (1.0 - weight_a) * score_b
        for threshold in candidate_thresholds(fused):
            result = metrics(labels, fused >= threshold)
            secondary = (
                result["balanced_accuracy"]
                if objective == "accuracy"
                else result["accuracy"]
            )
            key = (
                result[objective],
                secondary,
                -abs(float(weight_a) - 0.5),
            )
            if best is None or key > best[0]:
                best = (key, float(weight_a), float(threshold), result)
    _, weight_a, threshold, result = best
    return weight_a, threshold, result


def load_and_embed(checkpoint_path, X_seq, duration, dataset_hash, device):
    ckpt, model = evaluator.load_model(checkpoint_path, dataset_hash, device)
    X = apply_sequence_stats(X_seq, ckpt["sequence_mean"], ckpt["sequence_std"])
    d = apply_duration_stats(duration, ckpt["duration_mean"], ckpt["duration_std"])
    architecture = ckpt.get("architecture", "basic-1dcnn")
    if architecture == "gesture-conditioned-supcon":
        by_gesture = conditioned.extract_all_gesture_embeddings(model, X, d, device)
        names = [str(x) for x in ckpt["gesture_classes"]]
        mapping = {name: i for i, name in enumerate(names)}
        return ckpt, by_gesture, mapping
    embeddings = base.extract_embeddings(model, X, d, device)
    return ckpt, embeddings, None


def embedding_for(embeddings, mapping, indices, gesture_name):
    if mapping is None:
        return embeddings[indices]
    return embeddings[indices, mapping[gesture_name]]


def calibration_trials(
    embeddings_a, mapping_a, embeddings_b, mapping_b, meta, enroll_count
):
    performer, gesture, session = (
        meta["performer"], meta["gesture"], meta["session"]
    )
    labels, scores_a, scores_b, gestures = [], [], [], []
    for target in TRAIN_USERS:
        target_idx = np.where(performer == target)[0]
        sessions = sorted(np.unique(session[target_idx]).tolist())
        enroll_session, test_session = sessions[-2], sessions[-1]
        for gesture_name in sorted(np.unique(gesture).tolist()):
            enroll = np.where(
                (performer == target) & (gesture == gesture_name)
                & (session == enroll_session)
            )[0][:enroll_count]
            genuine = np.where(
                (performer == target) & (gesture == gesture_name)
                & (session == test_session)
            )[0]
            if len(enroll) < enroll_count or len(genuine) == 0:
                continue
            impostor = []
            for other in TRAIN_USERS:
                if other == target:
                    continue
                other_idx = np.where(performer == other)[0]
                other_test = sorted(np.unique(session[other_idx]).tolist())[-1]
                impostor.extend(np.where(
                    (performer == other) & (gesture == gesture_name)
                    & (session == other_test)
                )[0].tolist())
            trial_idx = np.concatenate([genuine, np.asarray(impostor, dtype=np.int64)])
            local_labels = np.concatenate([
                np.ones(len(genuine), dtype=np.int64),
                np.zeros(len(impostor), dtype=np.int64),
            ])
            template_a = embedding_for(
                embeddings_a, mapping_a, enroll, gesture_name
            ).mean(axis=0)
            template_b = embedding_for(
                embeddings_b, mapping_b, enroll, gesture_name
            ).mean(axis=0)
            score_a = base.cosine_scores(
                embedding_for(embeddings_a, mapping_a, trial_idx, gesture_name),
                template_a,
            )
            score_b = base.cosine_scores(
                embedding_for(embeddings_b, mapping_b, trial_idx, gesture_name),
                template_b,
            )
            labels.extend(local_labels.tolist())
            scores_a.extend(score_a.tolist())
            scores_b.extend(score_b.tolist())
            gestures.extend([gesture_name] * len(trial_idx))
    return (
        np.asarray(labels, dtype=np.int64),
        np.asarray(scores_a, dtype=np.float64),
        np.asarray(scores_b, dtype=np.float64),
        np.asarray(gestures, dtype=str),
    )


def read_trials(path):
    with Path(path).open(encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def aligned_test_trials(path_a, path_b):
    rows_a = read_trials(path_a)
    rows_b = read_trials(path_b)
    key = lambda row: (row["claimed_user"], row["sample_index"], row["true_gesture"])
    lookup_b = {key(row): row for row in rows_b}
    records = []
    for row_a in rows_a:
        row_b = lookup_b.get(key(row_a))
        if row_b is None:
            raise RuntimeError(f"Missing aligned fusion trial: {key(row_a)}")
        for field in ("genuine", "route_correct", "predicted_gesture"):
            if row_a[field] != row_b[field]:
                raise RuntimeError(f"Trial mismatch {key(row_a)}: {field}")
        records.append({
            "claimed_user": row_a["claimed_user"],
            "sample_index": int(row_a["sample_index"]),
            "true_gesture": row_a["true_gesture"],
            "predicted_gesture": row_a["predicted_gesture"],
            "label": int(row_a["genuine"]),
            "route_correct": int(row_a["route_correct"]),
            "direct_a": float(row_a["direct_score"]),
            "direct_b": float(row_b["direct_score"]),
            "routed_a": float(row_a["routed_score"]),
            "routed_b": float(row_b["routed_score"]),
        })
    if len(records) != len(rows_b):
        raise RuntimeError("Fusion trial files have different lengths")
    return records


def evaluate_policy(records, config, per_gesture=False):
    labels, direct_pred, e2e_pred = [], [], []
    output = []
    for row in records:
        gesture_name = row["true_gesture"]
        weight, threshold = config[gesture_name] if per_gesture else config["global"]
        direct_score = weight * row["direct_a"] + (1.0 - weight) * row["direct_b"]
        routed_score = weight * row["routed_a"] + (1.0 - weight) * row["routed_b"]
        direct_accept = int(direct_score >= threshold)
        routed_accept = int(routed_score >= threshold)
        final_accept = int(
            routed_accept
            and (row["label"] == 0 or row["route_correct"] == 1)
        )
        labels.append(row["label"])
        direct_pred.append(direct_accept)
        e2e_pred.append(final_accept)
        output.append({
            **row,
            "fusion_weight_supcon": weight,
            "fusion_threshold": threshold,
            "fused_direct_score": direct_score,
            "fused_direct_accept": direct_accept,
            "fused_routed_score": routed_score,
            "fused_end_to_end_accept": final_accept,
        })
    return metrics(labels, direct_pred), metrics(labels, e2e_pred), output


def main():
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default=str(project_dir / "dataset" / "dataset_1955_recent8_updated_20260905.npz"),
    )
    parser.add_argument(
        "--model-a",
        default=str(project_dir / "output" / "supcon_embedding_auth" / "embedding_1dcnn_supcon.pt"),
    )
    parser.add_argument(
        "--model-b",
        default=str(project_dir / "output" / "gesture_conditioned_supcon_auth" / "embedding_1dcnn_gesture_conditioned_supcon.pt"),
    )
    parser.add_argument(
        "--trials-a",
        default=str(project_dir / "output" / "supcon_embedding_auth" / "end_to_end" / "trial_results.csv"),
    )
    parser.add_argument(
        "--trials-b",
        default=str(project_dir / "output" / "gesture_conditioned_supcon_auth" / "end_to_end" / "trial_results.csv"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(project_dir / "output" / "supcon_score_fusion"),
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, X_seq, duration, meta, _, _ = load_dataset(args.data)
    dataset_hash = file_sha256(args.data)
    ckpt_a, emb_a, map_a = load_and_embed(
        args.model_a, X_seq, duration, dataset_hash, device
    )
    ckpt_b, emb_b, map_b = load_and_embed(
        args.model_b, X_seq, duration, dataset_hash, device
    )
    enroll_count = int(ckpt_a.get("enrollment_per_gesture", 3))
    if int(ckpt_b.get("enrollment_per_gesture", 3)) != enroll_count:
        raise RuntimeError("Enrollment counts differ between fusion models")
    cal_y, cal_a, cal_b, cal_g = calibration_trials(
        emb_a, map_a, emb_b, map_b, meta, enroll_count
    )

    global_weight, global_threshold, global_val = select_fusion(
        cal_y, cal_a, cal_b, objective="accuracy"
    )
    global_config = {"global": (global_weight, global_threshold)}
    gesture_config = {}
    gesture_val_rows = []
    for gesture_name in sorted(np.unique(cal_g).tolist()):
        mask = cal_g == gesture_name
        weight, threshold, result = select_fusion(
            cal_y[mask], cal_a[mask], cal_b[mask], objective="accuracy"
        )
        gesture_config[gesture_name] = (weight, threshold)
        gesture_val_rows.append({
            "gesture": gesture_name,
            "weight_supcon": weight,
            "weight_gesture_head": 1.0 - weight,
            "threshold": threshold,
            **result,
        })

    records = aligned_test_trials(args.trials_a, args.trials_b)
    global_direct, global_e2e, global_rows = evaluate_policy(
        records, global_config, per_gesture=False
    )
    gesture_direct, gesture_e2e, gesture_rows = evaluate_policy(
        records, gesture_config, per_gesture=True
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "validation_configuration.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        fields = list(gesture_val_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(gesture_val_rows)
    for name, rows in (("global", global_rows), ("per_gesture", gesture_rows)):
        with (output_dir / f"{name}_trial_results.csv").open(
            "w", newline="", encoding="utf-8-sig"
        ) as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    with (output_dir / "summary.txt").open("w", encoding="utf-8") as f:
        f.write("Validation-only SupCon score fusion\n")
        f.write(f"dataset_sha256={dataset_hash}\n")
        f.write("selection_objective=validation_accuracy\n")
        f.write("test_used_for_selection=false\n")
        f.write(f"global_weight_supcon={global_weight:.6f}\n")
        f.write(f"global_weight_gesture_head={1.0-global_weight:.6f}\n")
        f.write(f"global_threshold={global_threshold:.6f}\n")
        f.write(f"global_validation_accuracy={global_val['accuracy']:.6f}\n")
        for prefix, result in (
            ("global_direct", global_direct),
            ("global_end_to_end", global_e2e),
            ("per_gesture_direct", gesture_direct),
            ("per_gesture_end_to_end", gesture_e2e),
        ):
            for key in ("accuracy", "balanced_accuracy", "far", "frr"):
                f.write(f"{prefix}_{key}={result[key]:.6f}\n")

    print("=" * 80)
    print("Validation-only score fusion")
    print("=" * 80)
    print(
        f"Global validation choice: SupCon weight={global_weight:.2f}, "
        f"GestureHead weight={1-global_weight:.2f}, threshold={global_threshold:.6f}, "
        f"val Acc={global_val['accuracy']:.4f}"
    )
    print(
        f"Global fusion test: direct Acc={global_direct['accuracy']:.4f} "
        f"E2E Acc={global_e2e['accuracy']:.4f} "
        f"E2E BalAcc={global_e2e['balanced_accuracy']:.4f}"
    )
    print(
        f"Per-gesture fusion test: direct Acc={gesture_direct['accuracy']:.4f} "
        f"E2E Acc={gesture_e2e['accuracy']:.4f} "
        f"E2E BalAcc={gesture_e2e['balanced_accuracy']:.4f}"
    )
    print(f"Saved: {output_dir / 'summary.txt'}")


if __name__ == "__main__":
    main()
