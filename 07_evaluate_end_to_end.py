"""
07_evaluate_end_to_end.py

Evaluate ALL seven same-family two-stage systems:

Gesture classifier(model X)
    -> predicted gesture
    -> Authentication model(model X) for that gesture
    -> own / impostor

Compared against:
Direct Authentication(model X)
    -> true gesture's authentication model
    -> own / impostor

No training occurs in this file.

Important:
- Genuine success in two-stage mode requires BOTH:
      predicted gesture == true gesture
      AND authentication accepted
- An impostor accepted by any routed authentication model is False Accept.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import joblib
import numpy as np
import torch

import paths
from split_protocol import OWNER_MAP, build_splits
from two_stage_common import (
    MODEL_LABEL,
    MODEL_ORDER,
    RNNClassifier,
    Small1DCNN,
    SmallTransformer,
    apply_duration_stats,
    apply_sequence_stats,
    build_gesture_indices,
    file_sha256,
    load_dataset,
    model_dir,
    predict_torch_proba,
    torch_load_compat,
)


def binary_metrics(y_true, pred):
    y_true = np.asarray(y_true, dtype=np.int64)
    pred = np.asarray(pred, dtype=np.int64)

    pos = y_true == 1
    neg = y_true == 0

    tp = int(np.sum((pred == 1) & pos))
    tn = int(np.sum((pred == 0) & neg))
    fp = int(np.sum((pred == 1) & neg))
    fn = int(np.sum((pred == 0) & pos))

    far = fp / max(int(np.sum(neg)), 1)
    frr = fn / max(int(np.sum(pos)), 1)
    acc = (tp + tn) / max(len(y_true), 1)
    bal = ((1.0 - far) + (1.0 - frr)) / 2.0

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


def expected_path(model_key, task, gesture=None):
    d = model_dir(paths, model_key)

    if model_key in ("logreg", "svm", "rf"):
        if task == "gesture":
            return d / "gesture.joblib"
        return d / f"auth_{gesture}.joblib"

    if task == "gesture":
        return d / "gesture.pt"
    return d / f"auth_{gesture}.pt"


def validate_hash(saved_hash, current_hash, path):
    if saved_hash != current_hash:
        raise RuntimeError(
            f"dataset SHA-256 mismatch:\n"
            f"  checkpoint: {path}\n"
            f"  saved  : {saved_hash}\n"
            f"  current: {current_hash}\n"
            "같은 dataset.npz로 모델들을 다시 학습하세요."
        )


def load_gesture_model(model_key, current_hash, device):
    path = expected_path(model_key, "gesture")

    if not path.exists():
        raise FileNotFoundError(
            f"Gesture checkpoint not found: {path}\n"
            "먼저 python 06_train_gesture_models.py 를 실행하세요."
        )

    if model_key in ("logreg", "svm", "rf"):
        ckpt = joblib.load(path)
    else:
        ckpt = torch_load_compat(path, device)

    validate_hash(ckpt["dataset_sha256"], current_hash, path)
    return ckpt, path


def load_auth_model(model_key, gesture, current_hash, device):
    path = expected_path(model_key, "auth", gesture)

    if not path.exists():
        raise FileNotFoundError(
            f"Authentication checkpoint not found: {path}\n"
            "03 / 04 / 05 학습 스크립트를 먼저 실행하세요."
        )

    if model_key in ("logreg", "svm", "rf"):
        ckpt = joblib.load(path)
    else:
        ckpt = torch_load_compat(path, device)

    validate_hash(ckpt["dataset_sha256"], current_hash, path)
    return ckpt, path


def instantiate_torch_from_checkpoint(ckpt, device):
    key = ckpt["model_key"]

    if key in ("gru", "lstm"):
        model = RNNClassifier(
            input_dim=int(ckpt["input_dim"]),
            num_classes=int(ckpt["num_classes"]),
            cell=ckpt.get("cell", key),
            hidden_size=int(ckpt.get("hidden_size", 64)),
            dropout=float(ckpt.get("dropout", 0.2)),
        )

    elif key == "cnn":
        model = Small1DCNN(
            input_dim=int(ckpt["input_dim"]),
            num_classes=int(ckpt["num_classes"]),
        )

    elif key == "transformer":
        model = SmallTransformer(
            input_dim=int(ckpt["input_dim"]),
            seq_len=int(ckpt["seq_len"]),
            num_classes=int(ckpt["num_classes"]),
            d_model=int(ckpt.get("d_model", 128)),
            nhead=int(ckpt.get("nhead", 4)),
            num_layers=int(ckpt.get("num_layers", 2)),
            ff_dim=int(ckpt.get("ff_dim", 256)),
            dropout=float(ckpt.get("dropout", 0.2)),
        )

    else:
        raise ValueError(key)

    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model


def predict_gesture(
    model_key,
    ckpt,
    X_flat,
    X_seq,
    duration,
    indices,
    device,
    batch_size,
):
    if model_key in ("logreg", "svm", "rf"):
        dm = float(ckpt["duration_mean"])
        ds = float(ckpt["duration_std"])
        d = apply_duration_stats(duration[indices], dm, ds)
        X = np.concatenate([X_flat[indices], d[:, None]], axis=1)
        return ckpt["model"].predict(X).astype(np.int64)

    seq_mean = ckpt["seq_mean"].cpu().numpy()
    seq_std = ckpt["seq_std"].cpu().numpy()
    dm = float(ckpt["duration_mean"])
    ds = float(ckpt["duration_std"])

    X = apply_sequence_stats(X_seq[indices], seq_mean, seq_std)
    d = apply_duration_stats(duration[indices], dm, ds)

    model = instantiate_torch_from_checkpoint(ckpt, device)
    prob = predict_torch_proba(model, X, d, device, batch_size)
    return prob.argmax(axis=1).astype(np.int64)


def auth_score_one(
    model_key,
    ckpt,
    sample_idx,
    X_flat,
    X_seq,
    duration,
    device,
):
    if model_key in ("logreg", "svm", "rf"):
        dm = float(ckpt["duration_mean"])
        ds = float(ckpt["duration_std"])

        d = apply_duration_stats(
            duration[[sample_idx]],
            dm,
            ds,
        )

        X = np.concatenate(
            [X_flat[[sample_idx]], d[:, None]],
            axis=1,
        )

        score = float(
            ckpt["model"].predict_proba(X)[0, 1]
        )

    else:
        seq_mean = ckpt["seq_mean"].cpu().numpy()
        seq_std = ckpt["seq_std"].cpu().numpy()

        dm = float(ckpt["duration_mean"])
        ds = float(ckpt["duration_std"])

        X = apply_sequence_stats(
            X_seq[[sample_idx]],
            seq_mean,
            seq_std,
        )

        d = apply_duration_stats(
            duration[[sample_idx]],
            dm,
            ds,
        )

        model = instantiate_torch_from_checkpoint(
            ckpt,
            device,
        )

        score = float(
            predict_torch_proba(
                model,
                X,
                d,
                device,
                batch_size=1,
            )[0, 1]
        )

    accepted = int(
        score >= float(ckpt["threshold"])
    )

    return score, accepted


def macro_metrics_by_true_gesture(y_true, pred, true_gestures):
    rows = []

    for g in sorted(set(true_gestures)):
        m = true_gestures == g
        metrics = binary_metrics(y_true[m], pred[m])
        rows.append(metrics)

    keys = ["accuracy", "balanced_accuracy", "far", "frr"]
    return {
        key: float(np.mean([r[key] for r in rows]))
        for key in keys
    }


def evaluate_model_family(
    model_key,
    X_flat,
    X_seq,
    duration,
    meta,
    final_idx,
    classes,
    current_hash,
    device,
    batch_size,
    out_rows,
):
    gesture_ckpt, _ = load_gesture_model(
        model_key,
        current_hash,
        device,
    )

    auth_ckpts = {}
    for g in classes:
        auth_ckpts[g], _ = load_auth_model(
            model_key,
            g,
            current_hash,
            device,
        )

    gesture_pred_id = predict_gesture(
        model_key,
        gesture_ckpt,
        X_flat,
        X_seq,
        duration,
        final_idx,
        device,
        batch_size,
    )

    pred_gestures = np.array([
        classes[int(i)]
        for i in gesture_pred_id
    ])

    true_gestures = meta["gesture"][final_idx]
    performers = meta["performer"][final_idx]
    roles = meta["role"][final_idx]

    y_true = np.array([
        1
        if (
            roles[i] == "own"
            and performers[i] == OWNER_MAP[true_gestures[i]]
        )
        else 0
        for i in range(len(final_idx))
    ], dtype=np.int64)

    direct_pred = np.zeros(len(final_idx), dtype=np.int64)
    routed_pred = np.zeros(len(final_idx), dtype=np.int64)

    wrong_route = pred_gestures != true_gestures
    wrong_route_accepted = np.zeros(len(final_idx), dtype=bool)

    for i, sample_idx in enumerate(final_idx):
        true_g = true_gestures[i]
        pred_g = pred_gestures[i]

        direct_score, direct_accept = auth_score_one(
            model_key,
            auth_ckpts[true_g],
            int(sample_idx),
            X_flat,
            X_seq,
            duration,
            device,
        )
        direct_pred[i] = direct_accept

        routed_score, routed_accept = auth_score_one(
            model_key,
            auth_ckpts[pred_g],
            int(sample_idx),
            X_flat,
            X_seq,
            duration,
            device,
        )

        # Final decision:
        # genuine: correct gesture route AND routed auth accepted
        # impostor: any routed auth acceptance counts as false accept
        if y_true[i] == 1:
            routed_pred[i] = int(
                pred_g == true_g and routed_accept == 1
            )
        else:
            routed_pred[i] = int(
                routed_accept == 1
            )

        wrong_route_accepted[i] = bool(
            wrong_route[i]
            and routed_accept == 1
        )

        out_rows.append({
            "model": MODEL_LABEL[model_key],
            "sample_index": int(sample_idx),
            "true_gesture": true_g,
            "pred_gesture": pred_g,
            "performer": performers[i],
            "role": roles[i],
            "true_auth": int(y_true[i]),
            "direct_accept": int(direct_accept),
            "routed_accept_raw": int(routed_accept),
            "two_stage_final_accept": int(routed_pred[i]),
            "wrong_route": int(wrong_route[i]),
            "wrong_route_accepted": int(wrong_route_accepted[i]),
        })

    direct_global = binary_metrics(y_true, direct_pred)
    routed_global = binary_metrics(y_true, routed_pred)

    direct_macro = macro_metrics_by_true_gesture(
        y_true,
        direct_pred,
        true_gestures,
    )
    routed_macro = macro_metrics_by_true_gesture(
        y_true,
        routed_pred,
        true_gestures,
    )

    route_acc = float(
        np.mean(pred_gestures == true_gestures)
    )

    genuine_mask = y_true == 1
    impostor_mask = y_true == 0

    genuine_route_acc = float(
        np.mean(
            pred_gestures[genuine_mask]
            == true_gestures[genuine_mask]
        )
    )

    impostor_route_acc = float(
        np.mean(
            pred_gestures[impostor_mask]
            == true_gestures[impostor_mask]
        )
    )

    wrong_route_rate = float(
        np.mean(wrong_route)
    )

    wrong_route_accept_rate = (
        float(np.mean(wrong_route_accepted[wrong_route]))
        if np.any(wrong_route)
        else 0.0
    )

    return {
        "model_key": model_key,
        "direct_global": direct_global,
        "two_stage_global": routed_global,
        "direct_macro": direct_macro,
        "two_stage_macro": routed_macro,
        "gesture_route_accuracy": route_acc,
        "genuine_route_accuracy": genuine_route_acc,
        "impostor_route_accuracy": impostor_route_acc,
        "wrong_route_rate": wrong_route_rate,
        "wrong_route_accept_rate": wrong_route_accept_rate,
    }


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--data", default=str(paths.DATASET_NPZ))
    ap.add_argument("--models", default="all")
    ap.add_argument("--batch-size", type=int, default=64)

    args = ap.parse_args()

    if args.models.strip().lower() == "all":
        selected = MODEL_ORDER.copy()
    else:
        selected = [
            x.strip()
            for x in args.models.split(",")
            if x.strip()
        ]

    unknown = [
        x for x in selected
        if x not in MODEL_ORDER
    ]
    if unknown:
        raise ValueError(
            f"Unknown models: {unknown}"
        )

    data_path = str(
        paths.assert_external(
            args.data,
            "dataset.npz",
        )
    )

    current_hash = file_sha256(
        data_path
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    (
        X_flat,
        X_seq,
        duration,
        meta,
        T,
        D,
    ) = load_dataset(
        data_path
    )

    auth_splits = build_splits(
        meta
    )

    gesture_idx = build_gesture_indices(
        auth_splits
    )

    final_idx = gesture_idx[
        "final"
    ]

    classes = sorted(
        set(meta["gesture"])
    )

    all_sample_rows = []
    summaries = []

    for model_key in selected:
        print(
            "\n"
            + "#" * 100
        )
        print(
            f"{MODEL_LABEL[model_key]}"
        )
        print(
            "#" * 100
        )

        result = evaluate_model_family(
            model_key,
            X_flat,
            X_seq,
            duration,
            meta,
            final_idx,
            classes,
            current_hash,
            device,
            args.batch_size,
            all_sample_rows,
        )

        summaries.append(
            result
        )

        d = result[
            "direct_macro"
        ]
        t = result[
            "two_stage_macro"
        ]

        print(
            "\nMacro mean by true gesture"
        )
        print(
            f"Direct    : "
            f"Acc={d['accuracy']:.3f} "
            f"BalAcc={d['balanced_accuracy']:.3f} "
            f"FAR={d['far']:.3f} "
            f"FRR={d['frr']:.3f}"
        )
        print(
            f"Two-stage : "
            f"Acc={t['accuracy']:.3f} "
            f"BalAcc={t['balanced_accuracy']:.3f} "
            f"FAR={t['far']:.3f} "
            f"FRR={t['frr']:.3f}"
        )
        print(
            f"Gesture route accuracy={result['gesture_route_accuracy']:.3f}, "
            f"wrong-route rate={result['wrong_route_rate']:.3f}"
        )

    print(
        "\n"
        + "=" * 124
    )
    print(
        "ALL MODELS - END-TO-END FINAL TEST "
        "(macro mean over G1~G5)"
    )
    print(
        "=" * 124
    )
    print(
        f"{'Model':<22}"
        f"{'Direct Acc':>12}"
        f"{'Direct Bal':>12}"
        f"{'2Stage Acc':>12}"
        f"{'2Stage Bal':>12}"
        f"{'2Stage FAR':>12}"
        f"{'2Stage FRR':>12}"
        f"{'Route Acc':>11}"
    )
    print(
        "-" * 124
    )

    for r in summaries:
        d = r["direct_macro"]
        t = r["two_stage_macro"]

        print(
            f"{MODEL_LABEL[r['model_key']]:<22}"
            f"{d['accuracy']:>12.3f}"
            f"{d['balanced_accuracy']:>12.3f}"
            f"{t['accuracy']:>12.3f}"
            f"{t['balanced_accuracy']:>12.3f}"
            f"{t['far']:>12.3f}"
            f"{t['frr']:>12.3f}"
            f"{r['gesture_route_accuracy']:>11.3f}"
        )

    out_path = (
        Path(paths.DERIVED_DIR)
        / "runs"
        / "end_to_end_all_models_predictions.csv"
    )

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fields = [
        "model",
        "sample_index",
        "true_gesture",
        "pred_gesture",
        "performer",
        "role",
        "true_auth",
        "direct_accept",
        "routed_accept_raw",
        "two_stage_final_accept",
        "wrong_route",
        "wrong_route_accepted",
    ]

    with out_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )
        writer.writeheader()
        writer.writerows(
            all_sample_rows
        )

    print(
        f"\nSample-level CSV -> {out_path}"
    )


if __name__ == "__main__":
    main()
