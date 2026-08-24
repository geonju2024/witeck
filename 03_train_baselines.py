"""
03_train_baselines.py

Final-test version.

Models:
- Logistic Regression
- SVM-RBF
- Random Forest
- GRU
- LSTM

All five models use the same Train / Validation / Final Test split
defined in split_protocol.py.

Threshold is selected on Validation only.
Final Test is evaluated once with that fixed threshold.
"""

from __future__ import annotations

import argparse
import copy
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader, TensorDataset

import paths
from runlog import add_log_arguments, start_run_log
from split_protocol import (
    build_splits,
    evaluate_final,
    print_split_summary,
    validation_threshold,
)


MODEL_ORDER = ["logreg", "svm", "rf", "gru", "lstm"]
MODEL_NAME = {
    "logreg": "Logistic Regression",
    "svm": "SVM-RBF",
    "rf": "Random Forest",
    "gru": "GRU",
    "lstm": "LSTM",
}


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset(path):
    z = np.load(path, allow_pickle=True)

    X_flat = z["X"].astype(np.float32)
    T = int(z["T"])
    D = int(z["D"])

    if X_flat.shape[1] != T * D:
        raise ValueError(f"X.shape[1]={X_flat.shape[1]} != T*D={T*D}")

    if "duration_sec" not in z.files:
        raise ValueError(
            "dataset.npz에 duration_sec가 없습니다. "
            "수정된 01_extract_landmarks.py와 02_build_features.py로 만든 "
            "최신 dataset.npz를 사용하세요."
        )

    X_seq = X_flat.reshape(len(X_flat), T, D)
    duration = z["duration_sec"].astype(np.float32).reshape(-1)

    meta = {
        "gesture": z["gesture"].astype(str),
        "performer": z["performer"].astype(str),
        "session": z["session"].astype(str),
        "role": z["role"].astype(str),
    }

    return X_flat, X_seq, duration, meta, T, D


def duration_norm(train, val, final):
    mean = float(np.mean(train))
    std = float(np.std(train))
    if std < 1e-6:
        std = 1.0

    return (
        ((train - mean) / std).astype(np.float32),
        ((val - mean) / std).astype(np.float32),
        ((final - mean) / std).astype(np.float32),
    )


def sequence_norm(train, val, final):
    mean = train.mean(axis=(0, 1), keepdims=True)
    std = train.std(axis=(0, 1), keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)

    return (
        ((train - mean) / std).astype(np.float32),
        ((val - mean) / std).astype(np.float32),
        ((final - mean) / std).astype(np.float32),
    )


def class_weights(y, device):
    counts = np.bincount(y.astype(np.int64), minlength=2).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError(f"Train split missing a class: {counts.tolist()}")

    w = len(y) / (2.0 * counts)
    return torch.tensor(w, dtype=torch.float32, device=device)


class RNNClassifier(nn.Module):
    def __init__(self, input_dim, cell="gru", hidden_size=64, dropout=0.2):
        super().__init__()

        rnn_cls = nn.GRU if cell == "gru" else nn.LSTM

        self.rnn = rnn_cls(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

        # duration_sec 1개를 global tempo feature로 추가
        self.head = nn.Linear(hidden_size + 1, 2)

    def forward(self, x, duration):
        _, hidden = self.rnn(x)

        if isinstance(hidden, tuple):  # LSTM
            hidden = hidden[0]

        h = hidden[-1]
        h = self.dropout(self.norm(h))
        h = torch.cat([h, duration.unsqueeze(1)], dim=1)

        return self.head(h)


def make_loader(X, duration, y, batch_size, shuffle):
    ds = TensorDataset(
        torch.from_numpy(X).float(),
        torch.from_numpy(duration).float(),
        torch.from_numpy(y).long(),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def eval_loss(model, loader, criterion, device):
    model.eval()
    total = 0.0
    n = 0

    with torch.no_grad():
        for xb, db, yb in loader:
            xb = xb.to(device)
            db = db.to(device)
            yb = yb.to(device)

            loss = criterion(model(xb, db), yb)
            total += float(loss.item()) * len(yb)
            n += len(yb)

    return total / max(n, 1)


def train_rnn(
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
    train_loader = make_loader(X_train, d_train, y_train, batch_size, True)
    val_loader = make_loader(X_val, d_val, y_val, batch_size, False)

    criterion = nn.CrossEntropyLoss(weight=class_weights(y_train, device))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-3,
    )

    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    stale = 0

    for epoch in range(1, epochs + 1):
        model.train()

        for xb, db, yb in train_loader:
            xb = xb.to(device)
            db = db.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb, db), yb)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        val_loss = eval_loss(model, val_loader, criterion, device)

        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1

        if stale >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_epoch, best_loss


def rnn_scores(model, X, duration, device, batch_size):
    dummy_y = np.zeros(len(X), dtype=np.int64)
    loader = make_loader(X, duration, dummy_y, batch_size, False)

    scores = []

    model.eval()
    with torch.no_grad():
        for xb, db, _ in loader:
            logits = model(xb.to(device), db.to(device))
            prob = torch.softmax(logits, dim=1)[:, 1]
            scores.append(prob.cpu().numpy())

    return np.concatenate(scores).astype(np.float64)


def print_result(model_name, rows):
    print("\n" + "=" * 112)
    print(f"{model_name} - FINAL TEST")
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
    print("※ 1-EER은 예전 '정확도 환산'과 비교하기 위한 참고값이며 실제 Accuracy가 아닙니다.")


def run_classical(model_key, X_flat, duration, splits, seed):
    rows = []

    for gi, g in enumerate(sorted(splits)):
        s = splits[g]

        tr, va, te = s.train_idx, s.val_idx, s.final_idx
        dtr, dva, dte = duration_norm(
            duration[tr],
            duration[va],
            duration[te],
        )

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

        print(
            f"[{MODEL_NAME[model_key]} {g}] "
            f"Val EER={val_eer:.3f}, threshold={threshold:.4f}"
        )

        rows.append({
            "gesture": g,
            "owner": s.owner,
            **metrics,
        })

    print_result(MODEL_NAME[model_key], rows)


def run_rnn(model_key, X_seq, duration, splits, args, device):
    rows = []

    for gi, g in enumerate(sorted(splits)):
        seed_everything(args.seed + gi * 100)

        s = splits[g]
        tr, va, te = s.train_idx, s.val_idx, s.final_idx

        Xtr, Xva, Xte = sequence_norm(
            X_seq[tr],
            X_seq[va],
            X_seq[te],
        )

        dtr, dva, dte = duration_norm(
            duration[tr],
            duration[va],
            duration[te],
        )

        model = RNNClassifier(
            input_dim=Xtr.shape[2],
            cell=model_key,
            hidden_size=64,
            dropout=0.2,
        ).to(device)

        model, best_epoch, best_loss = train_rnn(
            model,
            Xtr, dtr, s.train_y,
            Xva, dva, s.val_y,
            device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=1e-3,
            patience=args.patience,
        )

        val_score = rnn_scores(
            model, Xva, dva, device, args.batch_size
        )
        final_score = rnn_scores(
            model, Xte, dte, device, args.batch_size
        )

        val_eer, threshold = validation_threshold(s.val_y, val_score)
        metrics = evaluate_final(s.final_y, final_score, threshold)

        print(
            f"[{MODEL_NAME[model_key]} {g}] "
            f"best_epoch={best_epoch}, "
            f"val_loss={best_loss:.4f}, "
            f"Val EER={val_eer:.3f}, threshold={threshold:.4f}"
        )

        rows.append({
            "gesture": g,
            "owner": s.owner,
            **metrics,
        })

    print_result(MODEL_NAME[model_key], rows)


def parse_models(text):
    text = text.lower().strip()

    if text == "all":
        return MODEL_ORDER

    models = [x.strip() for x in text.split(",") if x.strip()]
    unknown = [x for x in models if x not in MODEL_ORDER]

    if unknown:
        raise ValueError(
            f"Unknown models: {unknown}. "
            f"Choose from {MODEL_ORDER}"
        )

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

    start_run_log(
        "03_train_baselines",
        out_dir=args.log_dir,
        data_files=[data_path],
        extra={
            "models": selected,
            "protocol": "single_final_test",
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "patience": args.patience,
            "seed": args.seed,
        },
        enabled=not args.no_log,
    )

    seed_everything(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    X_flat, X_seq, duration, meta, T, D = load_dataset(data_path)
    splits = build_splits(meta)

    print(f"dataset = {data_path}")
    print(f"device  = {device}")
    print(f"X       = {X_seq.shape} [N,T,D]")
    print(f"models  = {selected}\n")

    print_split_summary(splits)

    for model_key in selected:
        if model_key in ("logreg", "svm", "rf"):
            run_classical(
                model_key,
                X_flat,
                duration,
                splits,
                args.seed,
            )
        else:
            run_rnn(
                model_key,
                X_seq,
                duration,
                splits,
                args,
                device,
            )


if __name__ == "__main__":
    main()