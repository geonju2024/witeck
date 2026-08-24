"""
05_train_transformer.py

Small Transformer authentication model - single Final Test version.

Architecture:
[N,32,169]
-> Linear(169, 128)
-> learnable positional embedding
-> TransformerEncoder x 2
   heads=4, FF=256, GELU, pre-norm
-> mean pooling over time
-> concat normalized duration_sec
-> binary classifier

Threshold is selected on Validation only.
"""

from __future__ import annotations

import argparse
import copy
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import paths
from runlog import add_log_arguments, start_run_log
from split_protocol import (
    build_splits,
    evaluate_final,
    print_split_summary,
    validation_threshold,
)


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class GestureTransformer(nn.Module):
    def __init__(
        self,
        input_dim,
        seq_len,
        d_model=128,
        nhead=4,
        num_layers=2,
        ff_dim=256,
        dropout=0.2,
    ):
        super().__init__()

        if d_model % nhead != 0:
            raise ValueError(
                "d_model must be divisible by nhead"
            )

        self.input_proj = nn.Linear(
            input_dim,
            d_model,
        )

        self.pos_embed = nn.Parameter(
            torch.zeros(
                1,
                seq_len,
                d_model,
            )
        )
        nn.init.normal_(
            self.pos_embed,
            mean=0.0,
            std=0.02,
        )

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
        )

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # pooled d_model + duration_sec
        self.head = nn.Linear(
            d_model + 1,
            2,
        )

    def forward(self, x, duration):
        h = self.input_proj(x)

        h = h + self.pos_embed[:, : h.shape[1]]

        h = self.encoder(h)

        h = h.mean(dim=1)
        h = self.norm(h)
        h = self.dropout(h)

        h = torch.cat(
            [h, duration.unsqueeze(1)],
            dim=1,
        )

        return self.head(h)


def load_dataset(path):
    z = np.load(
        path,
        allow_pickle=True,
    )

    X = z["X"].astype(np.float32)
    T = int(z["T"])
    D = int(z["D"])

    if X.shape[1] != T * D:
        raise ValueError(
            f"X.shape[1]={X.shape[1]} != T*D={T*D}"
        )

    if "duration_sec" not in z.files:
        raise ValueError(
            "dataset.npz에 duration_sec가 없습니다. "
            "최신 01/02 전처리 결과를 사용하세요."
        )

    X = X.reshape(
        len(X),
        T,
        D,
    )

    duration = z[
        "duration_sec"
    ].astype(
        np.float32
    ).reshape(-1)

    meta = {
        "gesture": z["gesture"].astype(str),
        "performer": z["performer"].astype(str),
        "session": z["session"].astype(str),
        "role": z["role"].astype(str),
    }

    return X, duration, meta, T, D


def normalize_sequence(train, val, final):
    mean = train.mean(
        axis=(0, 1),
        keepdims=True,
    )

    std = train.std(
        axis=(0, 1),
        keepdims=True,
    )

    std = np.where(
        std < 1e-6,
        1.0,
        std,
    )

    return (
        ((train - mean) / std).astype(np.float32),
        ((val - mean) / std).astype(np.float32),
        ((final - mean) / std).astype(np.float32),
    )


def normalize_duration(train, val, final):
    mean = float(
        np.mean(train)
    )

    std = float(
        np.std(train)
    )

    if std < 1e-6:
        std = 1.0

    return (
        ((train - mean) / std).astype(np.float32),
        ((val - mean) / std).astype(np.float32),
        ((final - mean) / std).astype(np.float32),
    )


def make_loader(
    X,
    duration,
    y,
    batch_size,
    shuffle,
):
    ds = TensorDataset(
        torch.from_numpy(X).float(),
        torch.from_numpy(duration).float(),
        torch.from_numpy(y).long(),
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
    )


def class_weights(y, device):
    counts = np.bincount(
        y,
        minlength=2,
    ).astype(np.float64)

    if np.any(counts == 0):
        raise ValueError(
            f"Train split missing a class: "
            f"{counts.tolist()}"
        )

    weights = len(y) / (
        2.0 * counts
    )

    return torch.tensor(
        weights,
        dtype=torch.float32,
        device=device,
    )


def validation_loss(
    model,
    loader,
    criterion,
    device,
):
    model.eval()

    total = 0.0
    n = 0

    with torch.no_grad():
        for xb, db, yb in loader:
            xb = xb.to(device)
            db = db.to(device)
            yb = yb.to(device)

            loss = criterion(
                model(xb, db),
                yb,
            )

            total += (
                float(loss.item())
                * len(yb)
            )
            n += len(yb)

    return total / max(n, 1)


def train_model(
    model,
    X_train,
    d_train,
    y_train,
    X_val,
    d_val,
    y_val,
    device,
    epochs,
    batch_size,
    lr,
    patience,
):
    train_loader = make_loader(
        X_train,
        d_train,
        y_train,
        batch_size,
        True,
    )

    val_loader = make_loader(
        X_val,
        d_val,
        y_val,
        batch_size,
        False,
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights(
            y_train,
            device,
        )
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-3,
    )

    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    stale = 0

    for epoch in range(
        1,
        epochs + 1,
    ):
        model.train()

        for xb, db, yb in train_loader:
            xb = xb.to(device)
            db = db.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(
                set_to_none=True
            )

            loss = criterion(
                model(xb, db),
                yb,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
            )

            optimizer.step()

        val_loss = validation_loss(
            model,
            val_loader,
            criterion,
            device,
        )

        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(
                model.state_dict()
            )
            stale = 0
        else:
            stale += 1

        if stale >= patience:
            break

    if best_state is not None:
        model.load_state_dict(
            best_state
        )

    return (
        model,
        best_epoch,
        best_loss,
    )


def predict_scores(
    model,
    X,
    duration,
    device,
    batch_size,
):
    dummy_y = np.zeros(
        len(X),
        dtype=np.int64,
    )

    loader = make_loader(
        X,
        duration,
        dummy_y,
        batch_size,
        False,
    )

    scores = []

    model.eval()

    with torch.no_grad():
        for xb, db, _ in loader:
            logits = model(
                xb.to(device),
                db.to(device),
            )

            prob = torch.softmax(
                logits,
                dim=1,
            )[:, 1]

            scores.append(
                prob.cpu().numpy()
            )

    return np.concatenate(
        scores
    ).astype(np.float64)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--data",
        default=str(
            paths.DATASET_NPZ
        ),
    )

    ap.add_argument(
        "--epochs",
        type=int,
        default=100,
    )

    ap.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    ap.add_argument(
        "--lr",
        type=float,
        default=5e-4,
    )

    ap.add_argument(
        "--patience",
        type=int,
        default=10,
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    ap.add_argument(
        "--d-model",
        type=int,
        default=128,
    )

    ap.add_argument(
        "--heads",
        type=int,
        default=4,
    )

    ap.add_argument(
        "--layers",
        type=int,
        default=2,
    )

    ap.add_argument(
        "--ff-dim",
        type=int,
        default=256,
    )

    ap.add_argument(
        "--dropout",
        type=float,
        default=0.2,
    )

    add_log_arguments(
        ap,
        paths.RUNS_DIR,
    )

    args = ap.parse_args()

    data_path = str(
        paths.assert_external(
            args.data,
            "dataset.npz",
        )
    )

    start_run_log(
        "05_train_transformer",
        out_dir=args.log_dir,
        data_files=[data_path],
        extra={
            "protocol": "single_final_test",
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "patience": args.patience,
            "d_model": args.d_model,
            "heads": args.heads,
            "layers": args.layers,
            "ff_dim": args.ff_dim,
            "dropout": args.dropout,
            "seed": args.seed,
        },
        enabled=not args.no_log,
    )

    seed_everything(
        args.seed
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    X, duration, meta, T, D = load_dataset(
        data_path
    )

    splits = build_splits(
        meta
    )

    print(
        f"dataset = {data_path}"
    )

    print(
        f"device  = {device}"
    )

    print(
        f"X       = {X.shape} "
        f"[N,T,D]\n"
    )

    print_split_summary(
        splits
    )

    rows = []

    for gi, g in enumerate(
        sorted(splits)
    ):
        seed_everything(
            args.seed
            + gi * 100
        )

        s = splits[g]

        tr = s.train_idx
        va = s.val_idx
        te = s.final_idx

        Xtr, Xva, Xte = (
            normalize_sequence(
                X[tr],
                X[va],
                X[te],
            )
        )

        dtr, dva, dte = (
            normalize_duration(
                duration[tr],
                duration[va],
                duration[te],
            )
        )

        model = GestureTransformer(
            input_dim=D,
            seq_len=T,
            d_model=args.d_model,
            nhead=args.heads,
            num_layers=args.layers,
            ff_dim=args.ff_dim,
            dropout=args.dropout,
        ).to(device)

        (
            model,
            best_epoch,
            best_loss,
        ) = train_model(
            model,
            Xtr,
            dtr,
            s.train_y,
            Xva,
            dva,
            s.val_y,
            device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            patience=args.patience,
        )

        val_score = predict_scores(
            model,
            Xva,
            dva,
            device,
            args.batch_size,
        )

        final_score = predict_scores(
            model,
            Xte,
            dte,
            device,
            args.batch_size,
        )

        (
            val_eer,
            threshold,
        ) = validation_threshold(
            s.val_y,
            val_score,
        )

        metrics = evaluate_final(
            s.final_y,
            final_score,
            threshold,
        )

        print(
            f"\n[{g}] "
            f"owner={s.owner} "
            f"best_epoch={best_epoch} "
            f"val_loss={best_loss:.4f} "
            f"Val EER={val_eer:.3f} "
            f"threshold={threshold:.4f}"
        )

        rows.append({
            "gesture": g,
            "owner": s.owner,
            **metrics,
        })

    print(
        "\n"
        + "=" * 112
    )

    print(
        "Transformer - FINAL TEST"
    )

    print(
        "=" * 112
    )

    print(
        f"{'Gesture':<9}"
        f"{'Owner':<8}"
        f"{'AUC':>9}"
        f"{'EER':>9}"
        f"{'1-EER':>10}"
        f"{'FAR':>9}"
        f"{'FRR':>9}"
        f"{'Acc':>9}"
        f"{'BalAcc':>9}"
    )

    print(
        "-" * 112
    )

    for r in rows:
        print(
            f"{r['gesture']:<9}"
            f"{r['owner']:<8}"
            f"{r['auc']:>9.3f}"
            f"{r['eer']:>9.3f}"
            f"{1-r['eer']:>10.3f}"
            f"{r['far']:>9.3f}"
            f"{r['frr']:>9.3f}"
            f"{r['accuracy']:>9.3f}"
            f"{r['balanced_accuracy']:>9.3f}"
        )

    keys = [
        "auc",
        "eer",
        "far",
        "frr",
        "accuracy",
        "balanced_accuracy",
    ]

    mean = {
        k: float(
            np.mean(
                [r[k] for r in rows]
            )
        )
        for k in keys
    }

    print(
        "-" * 112
    )

    print(
        f"{'MEAN':<17}"
        f"{mean['auc']:>9.3f}"
        f"{mean['eer']:>9.3f}"
        f"{1-mean['eer']:>10.3f}"
        f"{mean['far']:>9.3f}"
        f"{mean['frr']:>9.3f}"
        f"{mean['accuracy']:>9.3f}"
        f"{mean['balanced_accuracy']:>9.3f}"
    )

    print(
        "※ 1-EER은 예전 정확도 환산값과의 비교용이며 "
        "실제 Accuracy가 아닙니다."
    )


if __name__ == "__main__":
    main()