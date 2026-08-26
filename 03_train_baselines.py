"""
03_train_baselines.py

Authentication models:
- Logistic Regression
- SVM-RBF
- Random Forest
- GRU
- LSTM

This version keeps the existing authentication task and additionally saves
the five trained authentication model families for the two-stage experiment.
"""

from __future__ import annotations

import argparse
from pathlib import Path

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
from split_protocol import (
    build_splits,
    evaluate_final,
    print_split_summary,
    validation_threshold,
)
from two_stage_common import (
    MODEL_LABEL,
    RNNClassifier,
    apply_duration_stats,
    apply_sequence_stats,
    file_sha256,
    fit_duration_stats,
    fit_sequence_stats,
    load_dataset,
    model_dir,
    predict_torch_proba,
    seed_everything,
    train_torch_model,
)


MODEL_ORDER = ["logreg", "svm", "rf", "gru", "lstm"]


def print_result(model_key, rows):
    print("\n" + "=" * 112)
    print(f"{MODEL_LABEL[model_key]} - AUTHENTICATION FINAL TEST")
    print("=" * 112)
    print(
        f"{'Gesture':<9}{'Owner':<8}"
        f"{'AUC':>9}{'EER':>9}{'1-EER':>10}"
        f"{'FAR':>9}{'FRR':>9}"
        f"{'Acc':>9}{'BalAcc':>9}"
    )
    print("-" * 112)

    for r in rows:
        print(
            f"{r['gesture']:<9}{r['owner']:<8}"
            f"{r['auc']:>9.3f}{r['eer']:>9.3f}{1-r['eer']:>10.3f}"
            f"{r['far']:>9.3f}{r['frr']:>9.3f}"
            f"{r['accuracy']:>9.3f}{r['balanced_accuracy']:>9.3f}"
        )

    keys = ["auc", "eer", "far", "frr", "accuracy", "balanced_accuracy"]
    mean = {k: float(np.mean([r[k] for r in rows])) for k in keys}

    print("-" * 112)
    print(
        f"{'MEAN':<17}"
        f"{mean['auc']:>9.3f}{mean['eer']:>9.3f}{1-mean['eer']:>10.3f}"
        f"{mean['far']:>9.3f}{mean['frr']:>9.3f}"
        f"{mean['accuracy']:>9.3f}{mean['balanced_accuracy']:>9.3f}"
    )


def run_classical(model_key, X_flat, duration, splits, seed, dataset_hash):
    rows = []
    save_dir = model_dir(paths, model_key)

    for gi, g in enumerate(sorted(splits)):
        s = splits[g]
        tr, va, te = s.train_idx, s.val_idx, s.final_idx

        d_mean, d_std = fit_duration_stats(duration[tr])

        dtr = apply_duration_stats(duration[tr], d_mean, d_std)
        dva = apply_duration_stats(duration[va], d_mean, d_std)
        dte = apply_duration_stats(duration[te], d_mean, d_std)

        Xtr = np.concatenate([X_flat[tr], dtr[:, None]], axis=1)
        Xva = np.concatenate([X_flat[va], dva[:, None]], axis=1)
        Xte = np.concatenate([X_flat[te], dte[:, None]], axis=1)

        if model_key == "logreg":
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=1.0,
                    max_iter=3000,
                    class_weight="balanced",
                    random_state=seed + gi,
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
                    random_state=seed + gi,
                ),
            )
        elif model_key == "rf":
            model = RandomForestClassifier(
                n_estimators=500,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=seed + gi,
                n_jobs=-1,
            )
        else:
            raise ValueError(model_key)

        model.fit(Xtr, s.train_y)

        val_score = model.predict_proba(Xva)[:, 1]
        final_score = model.predict_proba(Xte)[:, 1]

        val_eer, threshold = validation_threshold(s.val_y, val_score)
        metrics = evaluate_final(s.final_y, final_score, threshold)

        checkpoint = {
            "format_version": 1,
            "task": "authentication",
            "model_key": model_key,
            "gesture": g,
            "owner": s.owner,
            "model": model,
            "duration_mean": d_mean,
            "duration_std": d_std,
            "threshold": float(threshold),
            "val_eer": float(val_eer),
            "dataset_sha256": dataset_hash,
        }

        ckpt_path = save_dir / f"auth_{g}.joblib"
        joblib.dump(checkpoint, ckpt_path)

        print(
            f"[{MODEL_LABEL[model_key]} {g}] "
            f"Val EER={val_eer:.3f}, threshold={threshold:.4f} "
            f"-> {ckpt_path}"
        )

        rows.append({
            "gesture": g,
            "owner": s.owner,
            **metrics,
        })

    print_result(model_key, rows)


def run_rnn(model_key, X_seq, duration, splits, args, device, dataset_hash):
    rows = []
    save_dir = model_dir(paths, model_key)

    for gi, g in enumerate(sorted(splits)):
        seed_everything(args.seed + gi * 100)

        s = splits[g]
        tr, va, te = s.train_idx, s.val_idx, s.final_idx

        seq_mean, seq_std = fit_sequence_stats(X_seq[tr])
        d_mean, d_std = fit_duration_stats(duration[tr])

        Xtr = apply_sequence_stats(X_seq[tr], seq_mean, seq_std)
        Xva = apply_sequence_stats(X_seq[va], seq_mean, seq_std)
        Xte = apply_sequence_stats(X_seq[te], seq_mean, seq_std)

        dtr = apply_duration_stats(duration[tr], d_mean, d_std)
        dva = apply_duration_stats(duration[va], d_mean, d_std)
        dte = apply_duration_stats(duration[te], d_mean, d_std)

        model = RNNClassifier(
            input_dim=Xtr.shape[2],
            num_classes=2,
            cell=model_key,
            hidden_size=64,
            dropout=0.2,
        ).to(device)

        model, best_epoch, best_loss = train_torch_model(
            model,
            Xtr, dtr, s.train_y,
            Xva, dva, s.val_y,
            device,
            num_classes=2,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=1e-3,
            patience=args.patience,
        )

        val_score = predict_torch_proba(
            model, Xva, dva, device, args.batch_size
        )[:, 1]
        final_score = predict_torch_proba(
            model, Xte, dte, device, args.batch_size
        )[:, 1]

        val_eer, threshold = validation_threshold(s.val_y, val_score)
        metrics = evaluate_final(s.final_y, final_score, threshold)

        checkpoint = {
            "format_version": 1,
            "task": "authentication",
            "model_key": model_key,
            "gesture": g,
            "owner": s.owner,
            "input_dim": int(Xtr.shape[2]),
            "num_classes": 2,
            "cell": model_key,
            "hidden_size": 64,
            "dropout": 0.2,
            "state_dict": model.state_dict(),
            "seq_mean": torch.from_numpy(seq_mean),
            "seq_std": torch.from_numpy(seq_std),
            "duration_mean": float(d_mean),
            "duration_std": float(d_std),
            "threshold": float(threshold),
            "val_eer": float(val_eer),
            "dataset_sha256": dataset_hash,
        }

        ckpt_path = save_dir / f"auth_{g}.pt"
        torch.save(checkpoint, ckpt_path)

        print(
            f"[{MODEL_LABEL[model_key]} {g}] "
            f"best_epoch={best_epoch}, val_loss={best_loss:.4f}, "
            f"Val EER={val_eer:.3f}, threshold={threshold:.4f} "
            f"-> {ckpt_path}"
        )

        rows.append({
            "gesture": g,
            "owner": s.owner,
            **metrics,
        })

    print_result(model_key, rows)


def parse_models(text):
    text = text.strip().lower()
    if text == "all":
        return MODEL_ORDER.copy()

    models = [x.strip() for x in text.split(",") if x.strip()]
    unknown = [x for x in models if x not in MODEL_ORDER]
    if unknown:
        raise ValueError(f"Unknown models: {unknown}. Choose from {MODEL_ORDER}")
    return models


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--data", default=str(paths.DATASET_NPZ))
    ap.add_argument(
        "--models",
        default="all",
        help="all or comma-separated: logreg,svm,rf,gru,lstm",
    )
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)

    add_log_arguments(ap, paths.RUNS_DIR)
    args = ap.parse_args()

    data_path = str(paths.assert_external(args.data, "dataset.npz"))
    selected = parse_models(args.models)
    dataset_hash = file_sha256(data_path)

    start_run_log(
        "03_train_baselines",
        out_dir=args.log_dir,
        data_files=[data_path],
        extra={
            "models": selected,
            "protocol": "single_final_test",
            "save_auth_checkpoints": True,
            "seed": args.seed,
        },
        enabled=not args.no_log,
    )

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_flat, X_seq, duration, meta, T, D = load_dataset(data_path)
    splits = build_splits(meta)

    print(f"dataset = {data_path}")
    print(f"device  = {device}")
    print(f"models  = {selected}\n")
    print_split_summary(splits)

    for model_key in selected:
        if model_key in ("logreg", "svm", "rf"):
            run_classical(
                model_key, X_flat, duration, splits, args.seed, dataset_hash
            )
        else:
            run_rnn(
                model_key, X_seq, duration, splits, args, device, dataset_hash
            )


if __name__ == "__main__":
    main()
