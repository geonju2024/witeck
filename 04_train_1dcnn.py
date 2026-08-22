"""
04_train_1dcnn.py

dataset.npz의 시계열 특징을 1D CNN으로 학습/평가한다.

입력
----
02_build_features.py가 저장한 dataset.npz

    X             : [N, T*D]
    T             : 현재 32
    D             : 현재 169
    duration_sec  : 영상 전체 수행시간

1D CNN 입력
-----------
    [N, T, D]
    ->
    [N, D, T]

Conv1d는 시간축 T를 따라 이동한다.

Absolute tempo
--------------
CNN 시계열 특징에는 실제 시간 기준 velocity가 이미 포함되어 있고,
추가로 전체 수행시간 duration_sec를 global feature로 사용한다.

    CNN pooled feature
    +
    normalized duration_sec
    ->
    classifier

Split
-----
split_protocol.py의 동일한 고정 split을 사용한다.

    Train
    Validation
    Internal Test
    External Impostor Test

Validation 역할
---------------
    1) Early stopping
    2) 인증 threshold 결정

Test에서는 threshold를 다시 찾지 않는다.
Validation threshold를 그대로 적용하여 FAR / FRR을 계산한다.

사용법
------
    python 04_train_1dcnn.py
"""

from __future__ import annotations

import argparse
import random
import sys
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader, TensorDataset

import paths
from runlog import add_log_arguments, start_run_log
from split_protocol import (
    build_verification_split,
    describe_verification_split,
)


# =========================================================
# Reproducibility
# =========================================================
def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =========================================================
# Dataset
# =========================================================
def load_dataset(path: str):
    z = np.load(path, allow_pickle=True)

    X = z["X"].astype(np.float32)

    if "T" not in z.files or "D" not in z.files:
        raise ValueError(
            "dataset.npz에 T 또는 D가 없습니다. "
            "수정된 02_build_features.py로 다시 생성하세요."
        )

    if "duration_sec" not in z.files:
        raise ValueError(
            "dataset.npz에 duration_sec가 없습니다. "
            "수정된 01_extract_landmarks.py와 "
            "02_build_features.py로 다시 생성하세요."
        )

    T = int(z["T"])
    D = int(z["D"])

    if X.shape[1] != T * D:
        raise ValueError(
            f"X.shape[1]={X.shape[1]} != T*D={T*D}"
        )

    X = X.reshape(
        len(X),
        T,
        D,
    )

    duration = z["duration_sec"].astype(
        np.float32
    ).reshape(-1)

    if len(duration) != len(X):
        raise ValueError(
            "duration_sec 길이가 X와 다릅니다."
        )

    meta = {
        "gesture": z["gesture"].astype(str),
        "performer": z["performer"].astype(str),
        "session": z["session"].astype(str),
        "role": z["role"].astype(str),
    }

    canonical_hand = (
        bool(z["canonical_hand"])
        if "canonical_hand" in z.files
        else None
    )

    return (
        X,
        duration,
        meta,
        T,
        D,
        canonical_hand,
    )


# =========================================================
# Normalization
# =========================================================
def normalize_sequence_from_train(
    X_train: np.ndarray,
    *others,
):
    """
    train 데이터 통계만 사용하여 feature별 표준화.
    """
    mean = X_train.mean(
        axis=(0, 1),
        keepdims=True,
    )

    std = X_train.std(
        axis=(0, 1),
        keepdims=True,
    )

    std = np.where(
        std < 1e-6,
        1.0,
        std,
    )

    def norm(X):
        return (
            (X - mean) / std
        ).astype(np.float32)

    return (
        norm(X_train),
        *[
            norm(X)
            for X in others
        ],
    )


def normalize_duration_from_train(
    d_train: np.ndarray,
    *others,
):
    """
    duration도 train 통계만 사용해 z-score 표준화.
    """
    mean = float(
        d_train.mean()
    )

    std = float(
        d_train.std()
    )

    if std < 1e-6:
        std = 1.0

    def norm(d):
        return (
            (d - mean) / std
        ).astype(np.float32)

    return (
        norm(d_train),
        *[
            norm(d)
            for d in others
        ],
    )


# =========================================================
# Model
# =========================================================
class Small1DCNN(nn.Module):
    """
    입력
        x        : [B, T, D]
        duration : [B]

    내부적으로
        [B,T,D] -> [B,D,T]

    구조
        Conv1d(D -> 64, k=5)
        BatchNorm
        ReLU
        Dropout

        Conv1d(64 -> 96, k=3)
        BatchNorm
        ReLU

        Conv1d(96 -> 128, k=3, dilation=2)
        BatchNorm
        ReLU
        Dropout

        AdaptiveAvgPool1d(1)

        pooled 128
        + duration 1
        = 129

        Linear(129 -> n_classes)
    """

    def __init__(
        self,
        in_channels: int,
        n_classes: int,
    ) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(
                in_channels,
                64,
                kernel_size=5,
                padding=2,
            ),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.20),

            nn.Conv1d(
                64,
                96,
                kernel_size=3,
                padding=1,
            ),
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

        self.head = nn.Linear(
            128 + 1,
            n_classes,
        )

    def forward(
        self,
        x: torch.Tensor,
        duration: torch.Tensor,
    ) -> torch.Tensor:
        # [B,T,D] -> [B,D,T]
        x = x.permute(
            0,
            2,
            1,
        ).contiguous()

        x = self.features(
            x
        ).squeeze(-1)

        duration = duration.reshape(
            -1,
            1,
        )

        x = torch.cat(
            [
                x,
                duration,
            ],
            dim=1,
        )

        return self.head(
            x
        )


def count_parameters(
    model: nn.Module,
) -> int:
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


# =========================================================
# DataLoader
# =========================================================
def make_loader(
    X: np.ndarray,
    duration: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
):
    xt = torch.from_numpy(
        X.astype(np.float32)
    )

    dt = torch.from_numpy(
        duration.astype(np.float32)
    )

    yt = torch.from_numpy(
        y.astype(np.int64)
    )

    return DataLoader(
        TensorDataset(
            xt,
            dt,
            yt,
        ),
        batch_size=batch_size,
        shuffle=shuffle,
    )


# =========================================================
# Training
# =========================================================
def train_model(
    X_train,
    d_train,
    y_train,
    X_val,
    d_val,
    y_val,
    args,
    device,
    seed,
):
    seed_everything(
        seed
    )

    model = Small1DCNN(
        in_channels=X_train.shape[2],
        n_classes=2,
    ).to(
        device
    )

    train_loader = make_loader(
        X_train,
        d_train,
        y_train,
        args.batch_size,
        shuffle=True,
    )

    val_loader = make_loader(
        X_val,
        d_val,
        y_val,
        args.batch_size,
        shuffle=False,
    )

    # class imbalance 보정
    counts = np.bincount(
        y_train,
        minlength=2,
    ).astype(
        np.float32
    )

    weights = (
        counts.sum()
        / np.maximum(
            counts,
            1.0,
        )
    )

    weights = (
        weights
        / weights.mean()
    )

    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(
            weights,
            dtype=torch.float32,
            device=device,
        )
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_state = None
    best_val_loss = float(
        "inf"
    )

    bad_epochs = 0

    for epoch in range(
        args.epochs
    ):
        # -------------------------
        # Train
        # -------------------------
        model.train()

        for (
            xb,
            db,
            yb,
        ) in train_loader:
            xb = xb.to(
                device
            )

            db = db.to(
                device
            )

            yb = yb.to(
                device
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(
                xb,
                db,
            )

            loss = criterion(
                logits,
                yb,
            )

            loss.backward()

            optimizer.step()

        # -------------------------
        # Validation
        # -------------------------
        model.eval()

        total = 0.0
        n = 0

        with torch.no_grad():
            for (
                xb,
                db,
                yb,
            ) in val_loader:
                xb = xb.to(
                    device
                )

                db = db.to(
                    device
                )

                yb = yb.to(
                    device
                )

                logits = model(
                    xb,
                    db,
                )

                loss = criterion(
                    logits,
                    yb,
                )

                total += (
                    float(
                        loss.item()
                    )
                    * len(
                        yb
                    )
                )

                n += len(
                    yb
                )

        val_loss = (
            total
            / max(
                n,
                1,
            )
        )

        if val_loss < (
            best_val_loss
            - 1e-4
        ):
            best_val_loss = val_loss

            best_state = deepcopy(
                model.state_dict()
            )

            bad_epochs = 0

        else:
            bad_epochs += 1

            if (
                bad_epochs
                >= args.patience
            ):
                break

    if best_state is not None:
        model.load_state_dict(
            best_state
        )

    return model


@torch.no_grad()
def predict_score(
    model,
    X,
    duration,
    batch_size,
    device,
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
        shuffle=False,
    )

    model.eval()

    scores = []

    for (
        xb,
        db,
        _,
    ) in loader:
        xb = xb.to(
            device
        )

        db = db.to(
            device
        )

        p = torch.softmax(
            model(
                xb,
                db,
            ),
            dim=1,
        )[:, 1]

        scores.append(
            p.cpu().numpy()
        )

    return np.concatenate(
        scores
    )


# =========================================================
# Metrics / Threshold
# =========================================================
def eer_threshold(
    y_true: np.ndarray,
    score: np.ndarray,
):
    """
    Validation에서 threshold 결정.
    """
    fpr, tpr, thresholds = roc_curve(
        y_true,
        score,
    )

    fnr = 1.0 - tpr

    i = int(
        np.nanargmin(
            np.abs(
                fpr - fnr
            )
        )
    )

    eer = float(
        (
            fpr[i]
            + fnr[i]
        )
        / 2.0
    )

    threshold = float(
        thresholds[i]
    )

    auc = float(
        roc_auc_score(
            y_true,
            score,
        )
    )

    return (
        threshold,
        eer,
        auc,
    )


def fixed_threshold_metrics(
    y_true: np.ndarray,
    score: np.ndarray,
    threshold: float,
):
    """
    Validation threshold를 고정해서
    Test FAR/FRR을 계산한다.

    EER은 Test separability 참고용으로만 계산한다.
    """
    fpr, tpr, _ = roc_curve(
        y_true,
        score,
    )

    fnr = 1.0 - tpr

    i = int(
        np.nanargmin(
            np.abs(
                fpr - fnr
            )
        )
    )

    diagnostic_eer = float(
        (
            fpr[i]
            + fnr[i]
        )
        / 2.0
    )

    auc = float(
        roc_auc_score(
            y_true,
            score,
        )
    )

    pred = (
        score >= threshold
    ).astype(
        np.int64
    )

    n_pos = max(
        int(
            (y_true == 1).sum()
        ),
        1,
    )

    n_neg = max(
        int(
            (y_true == 0).sum()
        ),
        1,
    )

    tp = int(
        (
            (pred == 1)
            & (y_true == 1)
        ).sum()
    )

    tn = int(
        (
            (pred == 0)
            & (y_true == 0)
        ).sum()
    )

    far = float(
        (
            (pred == 1)
            & (y_true == 0)
        ).sum()
        / n_neg
    )

    frr = float(
        (
            (pred == 0)
            & (y_true == 1)
        ).sum()
        / n_pos
    )

    tpr_fixed = (
        tp / n_pos
    )

    tnr_fixed = (
        tn / n_neg
    )

    balanced_accuracy = float(
        (
            tpr_fixed
            + tnr_fixed
        )
        / 2.0
    )

    return {
        "auc": auc,
        "eer": diagnostic_eer,
        "far": far,
        "frr": frr,
        "balanced_accuracy": balanced_accuracy,
    }


# =========================================================
# One gesture experiment
# =========================================================
def run_gesture(
    gesture_name,
    X,
    duration,
    meta,
    args,
    device,
):
    split = build_verification_split(
        meta,
        gesture_name,
    )

    print(
        "\n"
        + describe_verification_split(
            meta,
            split,
        )
    )

    owner = split[
        "owner"
    ]

    performer = meta[
        "performer"
    ]

    y_all = (
        performer == owner
    ).astype(
        np.int64
    )

    train_idx = split[
        "train_idx"
    ]

    val_idx = split[
        "val_idx"
    ]

    test_idx = split[
        "test_idx"
    ]

    external_idx = split[
        "external_test_idx"
    ]

    # -----------------------------------------------------
    # train 통계만 사용해 normalization
    # -----------------------------------------------------
    (
        Xtr,
        Xva,
        Xte,
        Xex,
    ) = normalize_sequence_from_train(
        X[
            train_idx
        ],
        X[
            val_idx
        ],
        X[
            test_idx
        ],
        X[
            external_idx
        ],
    )

    (
        dtr,
        dva,
        dte,
        dex,
    ) = normalize_duration_from_train(
        duration[
            train_idx
        ],
        duration[
            val_idx
        ],
        duration[
            test_idx
        ],
        duration[
            external_idx
        ],
    )

    y_train = y_all[
        train_idx
    ]

    y_val = y_all[
        val_idx
    ]

    y_test = y_all[
        test_idx
    ]

    y_external = y_all[
        external_idx
    ]

    model = train_model(
        X_train=Xtr,
        d_train=dtr,
        y_train=y_train,

        X_val=Xva,
        d_val=dva,
        y_val=y_val,

        args=args,
        device=device,
        seed=args.seed,
    )

    # -----------------------------------------------------
    # Validation threshold
    # -----------------------------------------------------
    val_score = predict_score(
        model,
        Xva,
        dva,
        args.batch_size,
        device,
    )

    (
        threshold,
        val_eer,
        val_auc,
    ) = eer_threshold(
        y_val,
        val_score,
    )

    # -----------------------------------------------------
    # Internal Test
    # -----------------------------------------------------
    test_score = predict_score(
        model,
        Xte,
        dte,
        args.batch_size,
        device,
    )

    test_metrics = fixed_threshold_metrics(
        y_test,
        test_score,
        threshold,
    )

    # -----------------------------------------------------
    # External Impostor Test
    # -----------------------------------------------------
    external_score = predict_score(
        model,
        Xex,
        dex,
        args.batch_size,
        device,
    )

    external_metrics = fixed_threshold_metrics(
        y_external,
        external_score,
        threshold,
    )

    print(
        "\nValidation"
    )

    print(
        f"  threshold = {threshold:.4f}"
    )

    print(
        f"  AUC       = {val_auc:.3f}"
    )

    print(
        f"  EER       = {val_eer:.3f}"
    )

    print(
        "\nInternal Test "
        "(validation threshold 고정)"
    )

    print(
        f"  AUC       = {test_metrics['auc']:.3f}"
    )

    print(
        f"  EER       = {test_metrics['eer']:.3f}"
    )

    print(
        f"  FAR       = {test_metrics['far']:.3f}"
    )

    print(
        f"  FRR       = {test_metrics['frr']:.3f}"
    )

    print(
        f"  BalAcc    = {test_metrics['balanced_accuracy']:.3f}"
    )

    print(
        "\nExternal Impostor Test "
        "(validation threshold 고정)"
    )

    print(
        f"  AUC       = {external_metrics['auc']:.3f}"
    )

    print(
        f"  EER       = {external_metrics['eer']:.3f}"
    )

    print(
        f"  FAR       = {external_metrics['far']:.3f}"
    )

    print(
        f"  FRR       = {external_metrics['frr']:.3f}"
    )

    print(
        f"  BalAcc    = {external_metrics['balanced_accuracy']:.3f}"
    )

    return {
        "gesture": gesture_name,
        "owner": owner,
        "threshold": threshold,
        "val_auc": val_auc,
        "val_eer": val_eer,
        "test": test_metrics,
        "external": external_metrics,
        "params": count_parameters(
            model
        ),
    }


# =========================================================
# Final summary
# =========================================================
def print_summary(
    results,
    key,
    title,
):
    print(
        "\n"
        + "=" * 76
    )

    print(
        title
    )

    print(
        "=" * 76
    )

    print(
        f"{'Gesture':<10}"
        f"{'AUC':>9}"
        f"{'EER':>9}"
        f"{'FAR':>9}"
        f"{'FRR':>9}"
        f"{'BalAcc':>10}"
    )

    print(
        "-" * 56
    )

    for r in results:
        m = r[
            key
        ]

        print(
            f"{r['gesture']:<10}"
            f"{m['auc']:>9.3f}"
            f"{m['eer']:>9.3f}"
            f"{m['far']:>9.3f}"
            f"{m['frr']:>9.3f}"
            f"{m['balanced_accuracy']:>10.3f}"
        )

    vals = [
        r[
            key
        ]
        for r in results
    ]

    print(
        "-" * 56
    )

    print(
        f"{'MEAN':<10}"
        f"{np.mean([x['auc'] for x in vals]):>9.3f}"
        f"{np.mean([x['eer'] for x in vals]):>9.3f}"
        f"{np.mean([x['far'] for x in vals]):>9.3f}"
        f"{np.mean([x['frr'] for x in vals]):>9.3f}"
        f"{np.mean([x['balanced_accuracy'] for x in vals]):>10.3f}"
    )


# =========================================================
# Main
# =========================================================
def main():
    try:
        sys.stdout.reconfigure(
            encoding="utf-8"
        )
    except Exception:
        pass

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
        default=60,
    )

    ap.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    ap.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )

    ap.add_argument(
        "--weight-decay",
        type=float,
        default=1e-3,
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
        "04_train_1dcnn",
        out_dir=args.log_dir,
        data_files=[
            data_path
        ],
        extra={
            "protocol": (
                "fixed train/val/internal/external"
            ),
            "tempo": (
                "real-time velocity + duration"
            ),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "patience": args.patience,
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

    (
        X,
        duration,
        meta,
        T,
        D,
        canonical_hand,
    ) = load_dataset(
        data_path
    )

    print(
        f"데이터 = {data_path}"
    )

    print(
        f"device = {device}"
    )

    print(
        f"X = {X.shape} "
        f"[N={len(X)}, T={T}, D={D}]"
    )

    print(
        f"duration = "
        f"min {duration.min():.3f}s / "
        f"median {np.median(duration):.3f}s / "
        f"max {duration.max():.3f}s"
    )

    print(
        f"gestures = "
        f"{sorted(set(meta['gesture']))}"
    )

    if canonical_hand is not None:
        print(
            "좌우 정규화 = "
            + (
                "켬"
                if canonical_hand
                else "끔"
            )
        )

    probe = Small1DCNN(
        in_channels=D,
        n_classes=2,
    )

    print(
        f"1D CNN trainable params = "
        f"{count_parameters(probe):,}"
    )

    results = []

    for gesture_name in sorted(
        set(
            meta[
                "gesture"
            ]
        )
    ):
        print(
            "\n\n"
            + "#" * 76
        )

        print(
            f"# {gesture_name}"
        )

        print(
            "#" * 76
        )

        result = run_gesture(
            gesture_name,
            X,
            duration,
            meta,
            args,
            device,
        )

        results.append(
            result
        )

    print_summary(
        results,
        "test",
        "1D CNN - Internal Test Summary",
    )

    print_summary(
        results,
        "external",
        "1D CNN - External Impostor Test Summary",
    )


if __name__ == "__main__":
    main()