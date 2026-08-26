"""
two_stage_common.py

Shared definitions for the two-stage experiment.

Stage 1: gesture classification
    G1 / G2 / G3 / G4 / G5

Stage 2: user authentication
    own / impostor

The same model families are used in both stages:
    Logistic Regression, SVM-RBF, Random Forest, GRU, LSTM, 1D CNN, Transformer
"""

from __future__ import annotations

import copy
import hashlib
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, TensorDataset


MODEL_ORDER = ["logreg", "svm", "rf", "gru", "lstm", "cnn", "transformer"]
MODEL_LABEL = {
    "logreg": "Logistic Regression",
    "svm": "SVM-RBF",
    "rf": "Random Forest",
    "gru": "GRU",
    "lstm": "LSTM",
    "cnn": "1D CNN",
    "transformer": "Transformer",
}


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def model_root(paths_module) -> Path:
    root = Path(paths_module.DERIVED_DIR) / "models" / "two_stage_all"
    root.mkdir(parents=True, exist_ok=True)
    return root


def model_dir(paths_module, model_key: str) -> Path:
    p = model_root(paths_module) / model_key
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_dataset(path: str):
    z = np.load(path, allow_pickle=True)

    X_flat = z["X"].astype(np.float32)
    T = int(z["T"])
    D = int(z["D"])

    if X_flat.shape[1] != T * D:
        raise ValueError(f"X.shape[1]={X_flat.shape[1]} != T*D={T*D}")

    if "duration_sec" not in z.files:
        raise ValueError(
            "dataset.npz에 duration_sec가 없습니다. "
            "최신 01_extract_landmarks.py / 02_build_features.py 결과를 사용하세요."
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


def fit_duration_stats(train: np.ndarray):
    mean = float(np.mean(train))
    std = float(np.std(train))
    if std < 1e-6:
        std = 1.0
    return mean, std


def apply_duration_stats(x: np.ndarray, mean: float, std: float):
    return ((x - mean) / std).astype(np.float32)


def fit_sequence_stats(train: np.ndarray):
    mean = train.mean(axis=(0, 1), keepdims=True).astype(np.float32)
    std = train.std(axis=(0, 1), keepdims=True).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean, std


def apply_sequence_stats(x: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return ((x - mean) / std).astype(np.float32)


def build_gesture_indices(auth_splits):
    """
    Authentication split의 G1~G5 index를 합쳐서 gesture classification split을 만든다.

    동일한 원본 sample이 두 split에 들어가는지 검사한다.
    """
    result = {}

    for name, attr in (
        ("train", "train_idx"),
        ("val", "val_idx"),
        ("final", "final_idx"),
    ):
        arr = np.concatenate([
            getattr(auth_splits[g], attr)
            for g in sorted(auth_splits)
        ]).astype(np.int64)

        if len(np.unique(arr)) != len(arr):
            raise AssertionError(f"gesture {name} split contains duplicate sample indices")

        result[name] = np.sort(arr)

    if np.intersect1d(result["train"], result["val"]).size:
        raise AssertionError("gesture Train/Validation overlap")
    if np.intersect1d(result["train"], result["final"]).size:
        raise AssertionError("gesture Train/Final overlap")
    if np.intersect1d(result["val"], result["final"]).size:
        raise AssertionError("gesture Validation/Final overlap")

    return result


class RNNClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        cell: str = "gru",
        hidden_size: int = 64,
        dropout: float = 0.2,
    ):
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
        self.head = nn.Linear(hidden_size + 1, num_classes)

    def forward(self, x, duration):
        _, hidden = self.rnn(x)

        if isinstance(hidden, tuple):
            hidden = hidden[0]

        h = hidden[-1]
        h = self.dropout(self.norm(h))
        h = torch.cat([h, duration.unsqueeze(1)], dim=1)
        return self.head(h)


class Small1DCNN(nn.Module):
    """
    기존 authentication 1D CNN backbone 유지.
    """
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.20),

            nn.Conv1d(64, 96, kernel_size=3, padding=1),
            nn.BatchNorm1d(96),
            nn.ReLU(),

            nn.Conv1d(
                96,
                128,
                kernel_size=3,
                padding=2,
                dilation=2,
            ),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.30),

            nn.AdaptiveAvgPool1d(1),
        )

        self.head = nn.Linear(128 + 1, num_classes)

    def forward(self, x, duration):
        h = self.features(x.transpose(1, 2)).squeeze(-1)
        h = torch.cat([h, duration.unsqueeze(1)], dim=1)
        return self.head(h)


class SmallTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        num_classes: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        ff_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()

        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d_model + 1, num_classes)

    def forward(self, x, duration):
        h = self.input_proj(x) + self.pos_embed[:, : x.shape[1]]
        h = self.encoder(h)
        h = self.dropout(self.norm(h.mean(dim=1)))
        h = torch.cat([h, duration.unsqueeze(1)], dim=1)
        return self.head(h)


def class_weights(y: np.ndarray, num_classes: int, device):
    counts = np.bincount(
        y.astype(np.int64),
        minlength=num_classes,
    ).astype(np.float64)

    if np.any(counts == 0):
        raise ValueError(f"Train split missing a class: counts={counts.tolist()}")

    weights = len(y) / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def make_loader(X, duration, y, batch_size, shuffle):
    ds = TensorDataset(
        torch.from_numpy(X).float(),
        torch.from_numpy(duration).float(),
        torch.from_numpy(y).long(),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def _eval_loss(model, loader, criterion, device):
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


def train_torch_model(
    model,
    X_train,
    d_train,
    y_train,
    X_val,
    d_val,
    y_val,
    device,
    *,
    num_classes: int,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    weight_decay: float = 1e-3,
):
    train_loader = make_loader(
        X_train, d_train, y_train, batch_size, True
    )
    val_loader = make_loader(
        X_val, d_val, y_val, batch_size, False
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights(y_train, num_classes, device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
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

        val_loss = _eval_loss(model, val_loader, criterion, device)

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


def predict_torch_proba(model, X, duration, device, batch_size):
    dummy_y = np.zeros(len(X), dtype=np.int64)
    loader = make_loader(X, duration, dummy_y, batch_size, False)

    out = []

    model.eval()
    with torch.no_grad():
        for xb, db, _ in loader:
            logits = model(xb.to(device), db.to(device))
            out.append(torch.softmax(logits, dim=1).cpu().numpy())

    return np.concatenate(out).astype(np.float64)


def multiclass_metrics(y_true, pred, classes):
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "macro_precision": float(
            precision_score(y_true, pred, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true, pred, average="macro", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(y_true, pred, average="macro", zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(
            y_true,
            pred,
            labels=np.arange(len(classes)),
        ),
    }


def torch_load_compat(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)
