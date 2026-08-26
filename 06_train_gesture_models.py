"""
06_train_gesture_models.py

Train G1~G5 gesture classifiers for ALL seven model families:

- Logistic Regression
- SVM-RBF
- Random Forest
- GRU
- LSTM
- 1D CNN
- Transformer

The gesture split is constructed by combining the exact Train / Validation /
Final indices already defined by split_protocol.py for each gesture.

Outputs:
- Accuracy
- Balanced Accuracy
- Macro Precision
- Macro Recall
- Macro F1
- Confusion Matrix
- one gesture-classifier checkpoint per model family
"""

from __future__ import annotations

import argparse
import csv

import joblib
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import paths
from runlog import add_log_arguments, start_run_log
from split_protocol import build_splits
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
    fit_duration_stats,
    fit_sequence_stats,
    load_dataset,
    model_dir,
    multiclass_metrics,
    predict_torch_proba,
    seed_everything,
    train_torch_model,
)


def parse_models(text):
    text = text.strip().lower()
    if text == "all":
        return MODEL_ORDER.copy()

    items = [x.strip() for x in text.split(",") if x.strip()]
    unknown = [x for x in items if x not in MODEL_ORDER]
    if unknown:
        raise ValueError(f"Unknown models: {unknown}. Choose from {MODEL_ORDER}")
    return items


def print_metrics(model_key, metrics, classes):
    print("\n" + "=" * 88)
    print(f"{MODEL_LABEL[model_key]} - GESTURE CLASSIFICATION FINAL TEST")
    print("=" * 88)
    print(f"Accuracy          : {metrics['accuracy']:.3f}")
    print(f"Balanced Accuracy : {metrics['balanced_accuracy']:.3f}")
    print(f"Macro Precision   : {metrics['macro_precision']:.3f}")
    print(f"Macro Recall      : {metrics['macro_recall']:.3f}")
    print(f"Macro F1          : {metrics['macro_f1']:.3f}")
    print("\nConfusion Matrix (row=true, col=pred)")
    print("classes:", classes)
    print(metrics["confusion_matrix"])
    print("\nGesture | Precision | Recall | F1")
    print("-" * 39)
    for i, gesture in enumerate(classes):
        print(
            f"{gesture:<8}| "
            f"{metrics['per_class_precision'][i]:>9.3f} | "
            f"{metrics['per_class_recall'][i]:>6.3f} | "
            f"{metrics['per_class_f1'][i]:>5.3f}"
        )


def save_metric_summaries(summary, per_class_rows):
    out_dir = paths.DERIVED_DIR / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / "gesture_models_summary.csv"
    summary_fields = [
        "model", "accuracy", "balanced_accuracy",
        "macro_precision", "macro_recall", "macro_f1",
    ]
    with summary_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary)

    per_class_path = out_dir / "gesture_models_per_class_metrics.csv"
    per_class_fields = ["model", "gesture", "precision", "recall", "f1"]
    with per_class_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=per_class_fields)
        writer.writeheader()
        writer.writerows(per_class_rows)

    return summary_path, per_class_path


def save_predictions(path, sample_idx, y_true, pred, classes, meta):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sample_index",
            "true_gesture",
            "pred_gesture",
            "performer",
            "session",
            "role",
        ])

        for idx, yt, yp in zip(sample_idx, y_true, pred):
            writer.writerow([
                int(idx),
                classes[int(yt)],
                classes[int(yp)],
                meta["performer"][idx],
                meta["session"][idx],
                meta["role"][idx],
            ])


def run_classical(
    model_key,
    X_flat,
    duration,
    y_all,
    split_idx,
    classes,
    args,
    dataset_hash,
):
    tr = split_idx["train"]
    va = split_idx["val"]
    te = split_idx["final"]

    d_mean, d_std = fit_duration_stats(duration[tr])

    dtr = apply_duration_stats(duration[tr], d_mean, d_std)
    dva = apply_duration_stats(duration[va], d_mean, d_std)
    dte = apply_duration_stats(duration[te], d_mean, d_std)

    Xtr = np.concatenate([X_flat[tr], dtr[:, None]], axis=1)
    Xva = np.concatenate([X_flat[va], dva[:, None]], axis=1)
    Xte = np.concatenate([X_flat[te], dte[:, None]], axis=1)

    ytr, yva, yte = y_all[tr], y_all[va], y_all[te]

    if model_key == "logreg":
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0,
                max_iter=3000,
                class_weight="balanced",
                random_state=args.seed,
            ),
        )
    elif model_key == "svm":
        model = make_pipeline(
            StandardScaler(),
            SVC(
                kernel="rbf",
                C=10.0,
                gamma="scale",
                class_weight="balanced",
                probability=True,
                random_state=args.seed,
            ),
        )
    elif model_key == "rf":
        model = RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=args.seed,
            n_jobs=-1,
        )
    else:
        raise ValueError(model_key)

    model.fit(Xtr, ytr)

    val_pred = model.predict(Xva)
    final_pred = model.predict(Xte)

    val_metrics = multiclass_metrics(yva, val_pred, classes)
    final_metrics = multiclass_metrics(yte, final_pred, classes)

    checkpoint = {
        "format_version": 1,
        "task": "gesture_classification",
        "model_key": model_key,
        "classes": classes,
        "model": model,
        "duration_mean": d_mean,
        "duration_std": d_std,
        "dataset_sha256": dataset_hash,
        "val_macro_f1": val_metrics["macro_f1"],
    }

    ckpt_path = model_dir(paths, model_key) / "gesture.joblib"
    joblib.dump(checkpoint, ckpt_path)

    print(
        f"[{MODEL_LABEL[model_key]}] "
        f"Val Macro-F1={val_metrics['macro_f1']:.3f} "
        f"-> {ckpt_path}"
    )

    return final_pred, final_metrics


def run_deep(
    model_key,
    X_seq,
    duration,
    y_all,
    split_idx,
    classes,
    T,
    D,
    args,
    device,
    dataset_hash,
):
    tr = split_idx["train"]
    va = split_idx["val"]
    te = split_idx["final"]

    seq_mean, seq_std = fit_sequence_stats(X_seq[tr])
    d_mean, d_std = fit_duration_stats(duration[tr])

    Xtr = apply_sequence_stats(X_seq[tr], seq_mean, seq_std)
    Xva = apply_sequence_stats(X_seq[va], seq_mean, seq_std)
    Xte = apply_sequence_stats(X_seq[te], seq_mean, seq_std)

    dtr = apply_duration_stats(duration[tr], d_mean, d_std)
    dva = apply_duration_stats(duration[va], d_mean, d_std)
    dte = apply_duration_stats(duration[te], d_mean, d_std)

    ytr, yva, yte = y_all[tr], y_all[va], y_all[te]

    num_classes = len(classes)

    if model_key in ("gru", "lstm"):
        model = RNNClassifier(
            input_dim=D,
            num_classes=num_classes,
            cell=model_key,
            hidden_size=64,
            dropout=0.2,
        ).to(device)
        lr = 1e-3
        arch = {
            "cell": model_key,
            "hidden_size": 64,
            "dropout": 0.2,
        }

    elif model_key == "cnn":
        model = Small1DCNN(
            input_dim=D,
            num_classes=num_classes,
        ).to(device)
        lr = 1e-3
        arch = {}

    elif model_key == "transformer":
        model = SmallTransformer(
            input_dim=D,
            seq_len=T,
            num_classes=num_classes,
            d_model=args.d_model,
            nhead=args.heads,
            num_layers=args.layers,
            ff_dim=args.ff_dim,
            dropout=args.dropout,
        ).to(device)
        lr = 5e-4
        arch = {
            "d_model": args.d_model,
            "nhead": args.heads,
            "num_layers": args.layers,
            "ff_dim": args.ff_dim,
            "dropout": args.dropout,
        }

    else:
        raise ValueError(model_key)

    model, best_epoch, best_loss = train_torch_model(
        model,
        Xtr, dtr, ytr,
        Xva, dva, yva,
        device,
        num_classes=num_classes,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=lr,
        patience=args.patience,
    )

    val_prob = predict_torch_proba(
        model, Xva, dva, device, args.batch_size
    )
    final_prob = predict_torch_proba(
        model, Xte, dte, device, args.batch_size
    )

    val_pred = val_prob.argmax(axis=1)
    final_pred = final_prob.argmax(axis=1)

    val_metrics = multiclass_metrics(yva, val_pred, classes)
    final_metrics = multiclass_metrics(yte, final_pred, classes)

    checkpoint = {
        "format_version": 1,
        "task": "gesture_classification",
        "model_key": model_key,
        "classes": classes,
        "input_dim": int(D),
        "seq_len": int(T),
        "num_classes": int(num_classes),
        "state_dict": model.state_dict(),
        "seq_mean": torch.from_numpy(seq_mean),
        "seq_std": torch.from_numpy(seq_std),
        "duration_mean": float(d_mean),
        "duration_std": float(d_std),
        "dataset_sha256": dataset_hash,
        "val_macro_f1": float(val_metrics["macro_f1"]),
        **arch,
    }

    ckpt_path = model_dir(paths, model_key) / "gesture.pt"
    torch.save(checkpoint, ckpt_path)

    print(
        f"[{MODEL_LABEL[model_key]}] "
        f"best_epoch={best_epoch}, val_loss={best_loss:.4f}, "
        f"Val Macro-F1={val_metrics['macro_f1']:.3f} "
        f"-> {ckpt_path}"
    )

    return final_pred, final_metrics


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--data", default=str(paths.DATASET_NPZ))
    ap.add_argument("--models", default="all")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)

    # Transformer settings
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--ff-dim", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.2)

    add_log_arguments(ap, paths.RUNS_DIR)
    args = ap.parse_args()

    selected = parse_models(args.models)
    data_path = str(paths.assert_external(args.data, "dataset.npz"))
    dataset_hash = file_sha256(data_path)

    start_run_log(
        "06_train_gesture_models",
        out_dir=args.log_dir,
        data_files=[data_path],
        extra={
            "models": selected,
            "task": "gesture_classification",
            "seed": args.seed,
        },
        enabled=not args.no_log,
    )

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_flat, X_seq, duration, meta, T, D = load_dataset(data_path)
    auth_splits = build_splits(meta)
    split_idx = build_gesture_indices(auth_splits)

    classes = sorted(set(meta["gesture"]))
    if classes != ["G1", "G2", "G3", "G4", "G5"]:
        raise ValueError(f"Expected G1~G5, found: {classes}")

    class_to_id = {g: i for i, g in enumerate(classes)}
    y_all = np.array(
        [class_to_id[g] for g in meta["gesture"]],
        dtype=np.int64,
    )

    print(f"dataset = {data_path}")
    print(f"device  = {device}")
    print(f"classes = {classes}")
    print(
        "gesture split sizes: "
        f"train={len(split_idx['train'])}, "
        f"val={len(split_idx['val'])}, "
        f"final={len(split_idx['final'])}"
    )

    summary = []
    per_class_rows = []

    for model_key in selected:
        seed_everything(args.seed)

        if model_key in ("logreg", "svm", "rf"):
            pred, metrics = run_classical(
                model_key,
                X_flat,
                duration,
                y_all,
                split_idx,
                classes,
                args,
                dataset_hash,
            )
        else:
            pred, metrics = run_deep(
                model_key,
                X_seq,
                duration,
                y_all,
                split_idx,
                classes,
                T,
                D,
                args,
                device,
                dataset_hash,
            )

        print_metrics(model_key, metrics, classes)

        pred_path = (
            model_dir(paths, model_key)
            / "gesture_final_predictions.csv"
        )
        save_predictions(
            pred_path,
            split_idx["final"],
            y_all[split_idx["final"]],
            pred,
            classes,
            meta,
        )

        summary.append({
            "model": model_key,
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"],
            "macro_f1": metrics["macro_f1"],
        })
        for i, gesture in enumerate(classes):
            per_class_rows.append({
                "model": model_key,
                "gesture": gesture,
                "precision": metrics["per_class_precision"][i],
                "recall": metrics["per_class_recall"][i],
                "f1": metrics["per_class_f1"][i],
            })

    print("\n" + "=" * 80)
    print("ALL MODELS - GESTURE CLASSIFICATION FINAL TEST")
    print("=" * 80)
    print(
        f"{'Model':<22}"
        f"{'Accuracy':>12}"
        f"{'BalAcc':>12}"
        f"{'Macro-F1':>12}"
    )
    print("-" * 80)

    for row in sorted(
        summary,
        key=lambda x: x["macro_f1"],
        reverse=True,
    ):
        print(
            f"{MODEL_LABEL[row['model']]:<22}"
            f"{row['accuracy']:>12.3f}"
            f"{row['balanced_accuracy']:>12.3f}"
            f"{row['macro_f1']:>12.3f}"
        )

    summary_path, per_class_path = save_metric_summaries(
        summary, per_class_rows
    )
    print(f"\nSummary CSV  -> {summary_path}")
    print(f"Per-class CSV -> {per_class_path}")


if __name__ == "__main__":
    main()
