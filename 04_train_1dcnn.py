"""
04_train_1dcnn.py

1D CNN authentication model.

Same authentication architecture as before.
This version additionally saves G1~G5 authentication checkpoints so that
07_evaluate_end_to_end.py can run the real two-stage pipeline.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

import paths
from runlog import add_log_arguments, start_run_log
from split_protocol import (
    build_splits,
    evaluate_final,
    print_split_summary,
    validation_threshold,
)
from two_stage_common import (
    Small1DCNN,
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


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--data", default=str(paths.DATASET_NPZ))
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)

    add_log_arguments(ap, paths.RUNS_DIR)
    args = ap.parse_args()

    data_path = str(paths.assert_external(args.data, "dataset.npz"))
    dataset_hash = file_sha256(data_path)

    start_run_log(
        "04_train_1dcnn",
        out_dir=args.log_dir,
        data_files=[data_path],
        extra={
            "protocol": "single_final_test",
            "save_auth_checkpoints": True,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "patience": args.patience,
            "seed": args.seed,
        },
        enabled=not args.no_log,
    )

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, X, duration, meta, T, D = load_dataset(data_path)
    splits = build_splits(meta)
    save_dir = model_dir(paths, "cnn")

    print(f"dataset = {data_path}")
    print(f"device  = {device}")
    print(f"X       = {X.shape} [N,T,D]\n")
    print_split_summary(splits)

    rows = []

    for gi, g in enumerate(sorted(splits)):
        seed_everything(args.seed + gi * 100)

        s = splits[g]
        tr, va, te = s.train_idx, s.val_idx, s.final_idx

        seq_mean, seq_std = fit_sequence_stats(X[tr])
        d_mean, d_std = fit_duration_stats(duration[tr])

        Xtr = apply_sequence_stats(X[tr], seq_mean, seq_std)
        Xva = apply_sequence_stats(X[va], seq_mean, seq_std)
        Xte = apply_sequence_stats(X[te], seq_mean, seq_std)

        dtr = apply_duration_stats(duration[tr], d_mean, d_std)
        dva = apply_duration_stats(duration[va], d_mean, d_std)
        dte = apply_duration_stats(duration[te], d_mean, d_std)

        model = Small1DCNN(
            input_dim=D,
            num_classes=2,
        ).to(device)

        model, best_epoch, best_loss = train_torch_model(
            model,
            Xtr, dtr, s.train_y,
            Xva, dva, s.val_y,
            device,
            num_classes=2,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
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
            "model_key": "cnn",
            "gesture": g,
            "owner": s.owner,
            "input_dim": int(D),
            "num_classes": 2,
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
            f"\n[{g}] owner={s.owner} "
            f"best_epoch={best_epoch} "
            f"val_loss={best_loss:.4f} "
            f"Val EER={val_eer:.3f} "
            f"threshold={threshold:.4f} "
            f"-> {ckpt_path}"
        )

        rows.append({
            "gesture": g,
            "owner": s.owner,
            **metrics,
        })

    print("\n" + "=" * 112)
    print("1D CNN - AUTHENTICATION FINAL TEST")
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


if __name__ == "__main__":
    main()
