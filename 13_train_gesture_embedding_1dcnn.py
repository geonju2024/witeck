"""Train the basic 1D-CNN as a gesture embedding model.

The encoder maps [B, 32, 169] sequences to an L2-normalized embedding.
A small classification head is used only to learn/predict G1~G5.
"""

from __future__ import annotations

import argparse
import copy
import csv
import importlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from split_protocol import build_splits
from two_stage_common import (
    apply_duration_stats,
    apply_sequence_stats,
    build_gesture_indices,
    class_weights,
    file_sha256,
    fit_duration_stats,
    fit_sequence_stats,
    load_dataset,
    multiclass_metrics,
    seed_everything,
)


embedding_module = importlib.import_module("08_train_embedding")
Embedding1DCNN = embedding_module.Embedding1DCNN
make_loader = embedding_module.make_loader
run_epoch = embedding_module.run_epoch


@torch.no_grad()
def predict(model, X, duration, device, batch_size):
    loader = make_loader(
        X,
        duration,
        np.zeros(len(X), dtype=np.int64),
        batch_size,
        shuffle=False,
    )
    embeddings = []
    predictions = []
    model.eval()
    for xb, db, _ in loader:
        emb, logits = model(xb.to(device), db.to(device))
        embeddings.append(emb.cpu().numpy())
        predictions.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(embeddings), np.concatenate(predictions)


def main():
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default=str(project_dir / "dataset" / "dataset_1955_recent8_updated_20260905.npz"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(project_dir / "output" / "basic_embedding_pipeline" / "gesture"),
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    _, X_seq, duration, meta, T, D = load_dataset(args.data)
    classes = sorted(np.unique(meta["gesture"]).tolist())
    class_to_label = {name: i for i, name in enumerate(classes)}
    y_all = np.asarray([class_to_label[x] for x in meta["gesture"]], dtype=np.int64)

    auth_splits = build_splits(meta)
    indices = build_gesture_indices(auth_splits)
    train_idx = indices["train"]
    val_idx = indices["val"]
    final_idx = indices["final"]

    seq_mean, seq_std = fit_sequence_stats(X_seq[train_idx])
    dur_mean, dur_std = fit_duration_stats(duration[train_idx])
    X_norm = apply_sequence_stats(X_seq, seq_mean, seq_std)
    duration_norm = apply_duration_stats(duration, dur_mean, dur_std)

    train_loader = make_loader(
        X_norm[train_idx], duration_norm[train_idx], y_all[train_idx],
        args.batch_size, shuffle=True,
    )
    val_loader = make_loader(
        X_norm[val_idx], duration_norm[val_idx], y_all[val_idx],
        args.batch_size, shuffle=False,
    )

    model = Embedding1DCNN(D, len(classes), args.embedding_dim).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights(y_all[train_idx], len(classes), device)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val_acc = -1.0
    best_epoch = 0
    best_state = None
    stale = 0

    print("=" * 72)
    print("Basic 1D-CNN gesture embedding training")
    print("=" * 72)
    print(f"device={device} dataset={args.data}")
    print(f"N={len(X_seq)} T={T} D={D} embedding={args.embedding_dim}")
    print(f"split train={len(train_idx)} val={len(val_idx)} final={len(final_idx)}")
    print(f"parameters={sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, device, optimizer=None
        )
        print(
            f"Epoch {epoch:02d} | train loss={train_loss:.4f} acc={train_acc:.3f} | "
            f"val loss={val_loss:.4f} acc={val_acc:.3f}"
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break

    if best_state is None:
        raise RuntimeError("No gesture checkpoint was created")
    model.load_state_dict(best_state)
    _, final_pred = predict(
        model, X_norm[final_idx], duration_norm[final_idx], device, args.batch_size
    )
    metrics = multiclass_metrics(y_all[final_idx], final_pred, classes)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "gesture_embedding_1dcnn.pt"
    dataset_hash = file_sha256(args.data)
    torch.save(
        {
            "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "input_dim": D,
            "seq_len": T,
            "embedding_dim": args.embedding_dim,
            "num_classes": len(classes),
            "classes": classes,
            "sequence_mean": seq_mean,
            "sequence_std": seq_std,
            "duration_mean": dur_mean,
            "duration_std": dur_std,
            "dataset_path": str(Path(args.data).resolve()),
            "dataset_sha256": dataset_hash,
            "best_epoch": best_epoch,
            "best_val_accuracy": best_val_acc,
            "final_metrics": metrics,
        },
        checkpoint_path,
    )

    with (output_dir / "final_predictions.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["sample_index", "performer", "session", "true_gesture", "pred_gesture", "correct"])
        for idx, pred in zip(final_idx, final_pred):
            true_name = str(meta["gesture"][idx])
            pred_name = classes[int(pred)]
            writer.writerow([
                int(idx), str(meta["performer"][idx]), str(meta["session"][idx]),
                true_name, pred_name, int(true_name == pred_name),
            ])

    with (output_dir / "summary.txt").open("w", encoding="utf-8") as f:
        f.write("Basic 1D-CNN gesture embedding classification\n")
        f.write(f"dataset={Path(args.data).resolve()}\n")
        f.write(f"dataset_sha256={dataset_hash}\n")
        f.write(f"embedding_dim={args.embedding_dim}\n")
        f.write(f"train_samples={len(train_idx)}\nval_samples={len(val_idx)}\nfinal_samples={len(final_idx)}\n")
        f.write(f"best_epoch={best_epoch}\nbest_val_accuracy={best_val_acc:.6f}\n")
        f.write(f"final_accuracy={metrics['accuracy']:.6f}\n")
        f.write(f"final_balanced_accuracy={metrics['balanced_accuracy']:.6f}\n")
        f.write(f"final_macro_f1={metrics['macro_f1']:.6f}\n")
        f.write("confusion_matrix=\n")
        f.write(np.array2string(metrics["confusion_matrix"]) + "\n")

    print("=" * 72)
    print(f"Final accuracy={metrics['accuracy']:.4f} macro-F1={metrics['macro_f1']:.4f}")
    print("confusion matrix:")
    print(metrics["confusion_matrix"])
    print(f"Saved: {checkpoint_path}")


if __name__ == "__main__":
    main()
