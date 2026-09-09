"""Train a gesture-oriented SupCon 1D-CNN embedding model.

Positive pairs share the same gesture regardless of performer. The final split
and normalization protocol are identical to 13_train_gesture_embedding_1dcnn.py.
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
import torch.nn.functional as F

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


def gesture_supcon_loss(embedding, labels, temperature=0.10):
    """Pull together all samples of the same gesture across performers."""
    count = embedding.shape[0]
    if count < 2:
        return embedding.sum() * 0.0
    similarity = embedding @ embedding.T / temperature
    eye = torch.eye(count, dtype=torch.bool, device=embedding.device)
    positive = labels[:, None].eq(labels[None, :]) & ~eye
    eligible = ~eye
    positive_count = positive.sum(dim=1)
    valid = positive_count > 0
    if not torch.any(valid):
        return embedding.sum() * 0.0
    log_denominator = torch.logsumexp(
        similarity.masked_fill(~eligible, float("-inf")), dim=1
    )
    positive_mean = (
        similarity.masked_fill(~positive, 0.0).sum(dim=1)
        / positive_count.clamp_min(1)
    )
    return (log_denominator - positive_mean)[valid].mean()


def run_epoch(
    model,
    loader,
    criterion,
    device,
    contrastive_weight,
    temperature,
    optimizer=None,
):
    training = optimizer is not None
    model.train(training)
    total_loss = total_ce = total_supcon = 0.0
    total_correct = total_count = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for xb, db, labels in loader:
            xb = xb.to(device)
            db = db.to(device)
            labels = labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            embedding, logits = model(xb, db)
            ce_loss = criterion(logits, labels)
            supcon_loss = gesture_supcon_loss(embedding, labels, temperature)
            loss = ce_loss + contrastive_weight * supcon_loss
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            n = len(labels)
            total_loss += float(loss.item()) * n
            total_ce += float(ce_loss.item()) * n
            total_supcon += float(supcon_loss.item()) * n
            total_correct += int((logits.argmax(dim=1) == labels).sum().item())
            total_count += n
    n = max(total_count, 1)
    return total_loss / n, total_ce / n, total_supcon / n, total_correct / n


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
    head_predictions = []
    model.eval()
    for xb, db, _ in loader:
        embedding, logits = model(xb.to(device), db.to(device))
        embeddings.append(embedding.cpu().numpy())
        head_predictions.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(embeddings), np.concatenate(head_predictions)


def main():
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default=str(project_dir / "dataset" / "dataset_1955_recent8_updated_20260905.npz"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(project_dir / "output" / "gesture_supcon_embedding"),
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--contrastive-weight", type=float, default=0.20)
    parser.add_argument("--temperature", type=float, default=0.10)
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

    indices = build_gesture_indices(build_splits(meta))
    train_idx, val_idx, final_idx = (
        indices["train"], indices["val"], indices["final"]
    )
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

    best_val_accuracy = -1.0
    best_epoch = 0
    best_state = None
    stale = 0
    stopped_epoch = args.epochs
    print("=" * 76)
    print("Gesture SupCon 1D-CNN embedding training")
    print("=" * 76)
    print(f"device={device} N={len(X_seq)} T={T} D={D}")
    print(f"split train={len(train_idx)} val={len(val_idx)} final={len(final_idx)}")
    print(
        f"loss=CE + {args.contrastive_weight}*gesture-SupCon, "
        f"temperature={args.temperature}"
    )

    for epoch in range(1, args.epochs + 1):
        train_values = run_epoch(
            model, train_loader, criterion, device,
            args.contrastive_weight, args.temperature, optimizer,
        )
        val_values = run_epoch(
            model, val_loader, criterion, device,
            args.contrastive_weight, args.temperature, optimizer=None,
        )
        print(
            f"Epoch {epoch:02d} | train loss={train_values[0]:.4f} "
            f"CE={train_values[1]:.4f} SupCon={train_values[2]:.4f} "
            f"acc={train_values[3]:.3f} | val loss={val_values[0]:.4f} "
            f"acc={val_values[3]:.3f}"
        )
        if val_values[3] > best_val_accuracy:
            best_val_accuracy = val_values[3]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            stopped_epoch = epoch
            print(f"Early stopping at epoch {epoch}")
            break

    if best_state is None:
        raise RuntimeError("No gesture SupCon checkpoint was created")
    model.load_state_dict(best_state)
    train_embeddings, _ = predict(
        model, X_norm[train_idx], duration_norm[train_idx], device, args.batch_size
    )
    final_embeddings, final_head_pred = predict(
        model, X_norm[final_idx], duration_norm[final_idx], device, args.batch_size
    )
    prototypes = np.stack(
        [train_embeddings[y_all[train_idx] == label].mean(axis=0)
         for label in range(len(classes))]
    )
    prototypes /= np.linalg.norm(prototypes, axis=1, keepdims=True).clip(min=1e-8)
    prototype_pred = (final_embeddings @ prototypes.T).argmax(axis=1)
    head_metrics = multiclass_metrics(y_all[final_idx], final_head_pred, classes)
    prototype_metrics = multiclass_metrics(y_all[final_idx], prototype_pred, classes)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_hash = file_sha256(args.data)
    checkpoint_path = output_dir / "gesture_embedding_supcon_1dcnn.pt"
    torch.save(
        {
            "model_state_dict": {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            },
            "architecture": "basic-1dcnn-gesture-supcon",
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
            "stopped_epoch": stopped_epoch,
            "best_val_accuracy": best_val_accuracy,
            "contrastive_weight": args.contrastive_weight,
            "temperature": args.temperature,
            "positive_rule": "same_gesture_regardless_of_user",
            "class_prototypes": prototypes,
            "final_head_metrics": head_metrics,
            "final_prototype_metrics": prototype_metrics,
        },
        checkpoint_path,
    )

    with (output_dir / "final_predictions.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "sample_index", "performer", "session", "true_gesture",
            "head_prediction", "prototype_prediction",
            "head_correct", "prototype_correct",
        ])
        for idx, head, prototype in zip(final_idx, final_head_pred, prototype_pred):
            truth = str(meta["gesture"][idx])
            writer.writerow([
                int(idx), str(meta["performer"][idx]), str(meta["session"][idx]), truth,
                classes[int(head)], classes[int(prototype)],
                int(truth == classes[int(head)]), int(truth == classes[int(prototype)]),
            ])

    summary_path = output_dir / "summary.txt"
    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write("Gesture SupCon 1D-CNN embedding classification\n")
        handle.write(f"dataset={Path(args.data).resolve()}\n")
        handle.write(f"dataset_sha256={dataset_hash}\n")
        handle.write("positive_rule=same_gesture_regardless_of_user\n")
        handle.write(f"contrastive_weight={args.contrastive_weight}\n")
        handle.write(f"temperature={args.temperature}\n")
        handle.write(f"embedding_dim={args.embedding_dim}\n")
        handle.write(
            f"train_samples={len(train_idx)}\nval_samples={len(val_idx)}\n"
            f"final_samples={len(final_idx)}\n"
        )
        handle.write(f"best_epoch={best_epoch}\nstopped_epoch={stopped_epoch}\n")
        handle.write(f"best_val_accuracy={best_val_accuracy:.6f}\n")
        handle.write(f"final_head_accuracy={head_metrics['accuracy']:.6f}\n")
        handle.write(
            f"final_head_balanced_accuracy={head_metrics['balanced_accuracy']:.6f}\n"
        )
        handle.write(f"final_head_macro_f1={head_metrics['macro_f1']:.6f}\n")
        handle.write(
            f"final_prototype_accuracy={prototype_metrics['accuracy']:.6f}\n"
        )
        handle.write(
            "final_prototype_balanced_accuracy="
            f"{prototype_metrics['balanced_accuracy']:.6f}\n"
        )
        handle.write(
            f"final_prototype_macro_f1={prototype_metrics['macro_f1']:.6f}\n"
        )
        handle.write("head_confusion_matrix=\n")
        handle.write(np.array2string(head_metrics["confusion_matrix"]) + "\n")
        handle.write("prototype_confusion_matrix=\n")
        handle.write(np.array2string(prototype_metrics["confusion_matrix"]) + "\n")

    print("=" * 76)
    print(
        f"Final head Acc={head_metrics['accuracy']:.4f} "
        f"macro-F1={head_metrics['macro_f1']:.4f}"
    )
    print(
        f"Final prototype Acc={prototype_metrics['accuracy']:.4f} "
        f"macro-F1={prototype_metrics['macro_f1']:.4f}"
    )
    print("Head confusion matrix:")
    print(head_metrics["confusion_matrix"])
    print(f"Saved: {checkpoint_path}")


if __name__ == "__main__":
    main()
