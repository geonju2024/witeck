"""
05_train_transformer.py

dataset.npz의 시계열 특징을 Transformer Encoder로 학습/평가한다.

입력
----
02_build_features.py가 저장한 dataset.npz

    X: [N, T*D]
    T: 시간 길이 (현재 32)
    D: 프레임당 feature 수 (현재 169)

Transformer에서는
    [N, T*D] -> [N, T, D]
로 복원한 뒤 사용한다.

기본 과제: verification
------------------------
제스처별 등록자 본인(own=1) vs 타인 모방(0)

평가 시
    1) 학습에서 보지 않은 등록자의 촬영 session
    2) 학습에서 보지 않은 impostor performer
를 동시에 hold-out한다.

즉
    다른 날의 본인
    +
    unseen impostor
를 평가한다.

선택 과제: gesture
-----------------
G1~G5 제스처 분류.
performer 단위 hold-out.

주요 인증 지표
--------------
AUC / EER / FAR / FRR

사용법
------
    python 05_train_transformer.py
    python 05_train_transformer.py --task both

Transformer 설정 예시
---------------------
    python 05_train_transformer.py \
        --d-model 128 \
        --nhead 4 \
        --num-layers 2 \
        --ff-dim 256 \
        --epochs 60
"""

from __future__ import annotations

import argparse
import random
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    StratifiedGroupKFold,
    train_test_split,
)
from torch.utils.data import DataLoader, TensorDataset

import paths
from runlog import add_log_arguments, start_run_log


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
            "02_build_features.py로 다시 생성하세요."
        )

    T = int(z["T"])
    D = int(z["D"])

    if X.ndim != 2:
        raise ValueError(
            f"X는 [N, T*D] 형태여야 합니다. 현재 shape={X.shape}"
        )

    if X.shape[1] != T * D:
        raise ValueError(
            f"X.shape[1]={X.shape[1]} != T*D={T*D}"
        )

    # [N, T*D] -> [N, T, D]
    X = X.reshape(len(X), T, D)

    required = [
        "gesture",
        "performer",
        "session",
        "role",
    ]

    missing = [
        key
        for key in required
        if key not in z.files
    ]

    if missing:
        raise ValueError(
            f"dataset.npz에 필요한 metadata가 없습니다: {missing}"
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

    return X, meta, T, D, canonical_hand


def normalize_from_train(
    X_train: np.ndarray,
    X_other: np.ndarray,
):
    """
    train fold에서만 feature별 mean/std를 계산한다.
    test 통계는 사용하지 않는다.
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

    return (
        (X_train - mean) / std,
        (X_other - mean) / std,
    )


def make_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
):
    xt = torch.from_numpy(
        X.astype(np.float32)
    )

    yt = torch.from_numpy(
        y.astype(np.int64)
    )

    return DataLoader(
        TensorDataset(xt, yt),
        batch_size=batch_size,
        shuffle=shuffle,
    )


def split_train_val_indices(
    y: np.ndarray,
    seed: int,
):
    """
    외부 test fold는 그대로 두고
    train fold 내부에서 early stopping용 validation을 만든다.
    """
    idx = np.arange(len(y))

    values, counts = np.unique(
        y,
        return_counts=True,
    )

    if (
        len(values) < 2
        or counts.min() < 2
        or len(y) < 12
    ):
        return idx, idx

    tr, va = train_test_split(
        idx,
        test_size=0.20,
        random_state=seed,
        stratify=y,
    )

    return tr, va


# =========================================================
# Transformer
# =========================================================
class GestureTransformer(nn.Module):
    """
    입력
        [B, T, D]

    처리
        D -> d_model projection
        + learnable positional embedding
        -> Transformer Encoder
        -> 시간축 mean pooling
        -> Linear classifier
    """

    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        ff_dim: int,
        dropout: float,
        n_classes: int,
    ) -> None:
        super().__init__()

        if d_model % nhead != 0:
            raise ValueError(
                f"d_model({d_model})은 nhead({nhead})로 나누어져야 합니다."
            )

        self.input_proj = nn.Linear(
            input_dim,
            d_model,
        )

        self.pos_embedding = nn.Parameter(
            torch.zeros(
                1,
                seq_len,
                d_model,
            )
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.norm = nn.LayerNorm(
            d_model
        )

        self.dropout = nn.Dropout(
            dropout
        )

        self.head = nn.Linear(
            d_model,
            n_classes,
        )

        nn.init.normal_(
            self.pos_embedding,
            mean=0.0,
            std=0.02,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        # [B,T,D] -> [B,T,d_model]
        x = self.input_proj(x)

        if x.shape[1] > self.pos_embedding.shape[1]:
            raise ValueError(
                "입력 sequence 길이가 positional embedding보다 깁니다."
            )

        x = (
            x
            + self.pos_embedding[
                :,
                :x.shape[1],
                :
            ]
        )

        x = self.encoder(x)

        # 전체 시간축 정보를 평균 pooling
        x = x.mean(dim=1)

        x = self.norm(x)
        x = self.dropout(x)

        return self.head(x)


def count_parameters(
    model: nn.Module,
) -> int:
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


# =========================================================
# Training
# =========================================================
def train_one_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_classes: int,
    args,
    seed: int,
    device: torch.device,
):
    seed_everything(seed)

    tr_idx, va_idx = split_train_val_indices(
        y_train,
        seed,
    )

    Xtr = X_train[tr_idx]
    ytr = y_train[tr_idx]

    Xva = X_train[va_idx]
    yva = y_train[va_idx]

    train_loader = make_loader(
        Xtr,
        ytr,
        args.batch_size,
        shuffle=True,
    )

    val_loader = make_loader(
        Xva,
        yva,
        args.batch_size,
        shuffle=False,
    )

    model = GestureTransformer(
        input_dim=X_train.shape[2],
        seq_len=X_train.shape[1],
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
        n_classes=n_classes,
    ).to(device)

    # class imbalance 보정
    counts = np.bincount(
        ytr,
        minlength=n_classes,
    ).astype(np.float32)

    weights = (
        counts.sum()
        / np.maximum(counts, 1.0)
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
    best_val_loss = float("inf")
    bad_epochs = 0

    for _ in range(args.epochs):

        # -------------------------
        # train
        # -------------------------
        model.train()

        for xb, yb in train_loader:

            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(xb)

            loss = criterion(
                logits,
                yb,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
            )

            optimizer.step()

        # -------------------------
        # validation
        # -------------------------
        model.eval()

        val_loss = 0.0
        n_val = 0

        with torch.no_grad():

            for xb, yb in val_loader:

                xb = xb.to(device)
                yb = yb.to(device)

                logits = model(xb)

                loss = criterion(
                    logits,
                    yb,
                )

                val_loss += (
                    float(loss.item())
                    * len(yb)
                )

                n_val += len(yb)

        val_loss /= max(
            n_val,
            1,
        )

        if val_loss < best_val_loss - 1e-4:

            best_val_loss = val_loss

            best_state = deepcopy(
                model.state_dict()
            )

            bad_epochs = 0

        else:

            bad_epochs += 1

            if bad_epochs >= args.patience:
                break

    if best_state is not None:
        model.load_state_dict(
            best_state
        )

    return model


@torch.no_grad()
def predict_proba(
    model: nn.Module,
    X: np.ndarray,
    batch_size: int,
    device: torch.device,
):
    model.eval()

    dummy_y = np.zeros(
        len(X),
        dtype=np.int64,
    )

    loader = make_loader(
        X,
        dummy_y,
        batch_size,
        shuffle=False,
    )

    probs = []

    for xb, _ in loader:

        xb = xb.to(device)

        p = torch.softmax(
            model(xb),
            dim=1,
        )

        probs.append(
            p.cpu().numpy()
        )

    return np.concatenate(
        probs,
        axis=0,
    )


# =========================================================
# Metrics
# =========================================================
def eer_metrics(
    y_true: np.ndarray,
    score: np.ndarray,
):
    fpr, tpr, thresholds = roc_curve(
        y_true,
        score,
    )

    fnr = 1.0 - tpr

    i = int(
        np.nanargmin(
            np.abs(fpr - fnr)
        )
    )

    eer = float(
        (fpr[i] + fnr[i])
        / 2.0
    )

    threshold = float(
        thresholds[i]
    )

    pred = (
        score >= threshold
    ).astype(int)

    n_pos = max(
        int((y_true == 1).sum()),
        1,
    )

    n_neg = max(
        int((y_true == 0).sum()),
        1,
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

    auc = float(
        roc_auc_score(
            y_true,
            score,
        )
    )

    return (
        eer,
        threshold,
        auc,
        far,
        frr,
    )


# =========================================================
# Verification folds
# =========================================================
def _split_list(items, k):
    n = len(items)

    return [
        items[
            i * n // k:
            (i + 1) * n // k
        ]
        for i in range(k)
    ]


def session_impostor_folds(
    y,
    performer,
    session,
):
    """
    이중분리:
        등록자의 session과 impostor performer를 동시에 hold-out.

    test:
        다른 날의 본인
        +
        학습에 없던 공격자
    """
    y = np.asarray(y)
    performer = np.asarray(performer)
    session = np.asarray(session)

    pos = np.where(
        y == 1
    )[0]

    sessions = sorted(
        set(
            session[pos]
        )
    )

    impostors = sorted(
        set(
            performer[
                y == 0
            ]
        )
    )

    K = min(
        len(sessions),
        len(impostors),
    )

    if K < 2:
        return None

    sess_groups = _split_list(
        sessions,
        K,
    )

    imp_groups = _split_list(
        impostors,
        K,
    )

    all_idx = np.arange(
        len(y)
    )

    folds = []

    for k in range(K):

        test_idx = np.concatenate(
            [
                pos[
                    np.isin(
                        session[pos],
                        sess_groups[k],
                    )
                ],

                np.where(
                    (y == 0)
                    & np.isin(
                        performer,
                        imp_groups[k],
                    )
                )[0],
            ]
        )

        train_idx = np.setdiff1d(
            all_idx,
            test_idx,
        )

        folds.append(
            (
                train_idx,
                test_idx,
            )
        )

    return folds


# =========================================================
# Task B: Verification
# =========================================================
def task_verification(
    X,
    meta,
    args,
    device,
):
    gesture = meta["gesture"]
    performer = meta["performer"]
    session = meta["session"]
    role = meta["role"]

    # role=own으로 제스처별 등록자를 자동 판정
    owners = {}

    for g in sorted(
        set(gesture)
    ):

        who = sorted(
            set(
                performer[
                    (gesture == g)
                    & (role == "own")
                ]
            )
        )

        if len(who) == 1:

            owners[g] = who[0]

        else:

            print(
                f"[{g}] 등록자 판정 불가: "
                f"{who} -> 생략"
            )

    print("\n" + "=" * 78)

    print(
        "Transformer - 본인 인증 "
        "(이중분리: 다른 날 본인 + unseen impostor)"
    )

    print("=" * 78)

    summary = []

    for gesture_no, g in enumerate(
        sorted(owners)
    ):

        owner = owners[g]

        mask = (
            gesture == g
        )

        Xg = X[mask]
        pg = performer[mask]
        sg = session[mask]
        rg = role[mask]

        y = (
            rg == "own"
        ).astype(
            np.int64
        )

        folds = session_impostor_folds(
            y,
            pg,
            sg,
        )

        if folds is None:

            print(
                f"[{g}] fold 생성 불가 -> 생략"
            )

            continue

        print(
            f"\n[{g}] "
            f"owner={owner} | "
            f"genuine={int(y.sum())}, "
            f"impostor={int((y == 0).sum())}, "
            f"folds={len(folds)}"
        )

        oof_score = np.full(
            len(y),
            np.nan,
            dtype=np.float32,
        )

        for fold_no, (
            tr,
            te,
        ) in enumerate(
            folds,
            1,
        ):

            Xtr, Xte = normalize_from_train(
                Xg[tr],
                Xg[te],
            )

            model = train_one_model(
                X_train=Xtr,
                y_train=y[tr],
                n_classes=2,
                args=args,
                seed=(
                    args.seed
                    + gesture_no * 100
                    + fold_no
                ),
                device=device,
            )

            prob = predict_proba(
                model,
                Xte,
                args.batch_size,
                device,
            )[:, 1]

            oof_score[te] = prob

            print(
                f"  fold {fold_no}: "
                f"train={len(tr):3d}, "
                f"test={len(te):3d}, "
                f"test genuine={int(y[te].sum()):2d}, "
                f"impostor={int((y[te] == 0).sum()):2d}"
            )

        valid = ~np.isnan(
            oof_score
        )

        (
            eer,
            threshold,
            auc,
            far,
            frr,
        ) = eer_metrics(
            y[valid],
            oof_score[valid],
        )

        print(
            f"  RESULT  "
            f"AUC={auc:.3f}  "
            f"EER={eer:.3f}  "
            f"FAR={far:.3f}  "
            f"FRR={frr:.3f}  "
            f"EER-threshold={threshold:.3f}"
        )

        summary.append(
            (
                g,
                owner,
                auc,
                eer,
                far,
                frr,
            )
        )

    if summary:

        print("\n" + "-" * 78)

        print("요약")

        print(
            f"{'gesture':<9}"
            f"{'owner':<8}"
            f"{'AUC':>9}"
            f"{'EER':>9}"
            f"{'FAR':>9}"
            f"{'FRR':>9}"
        )

        for (
            g,
            owner,
            auc,
            eer,
            far,
            frr,
        ) in summary:

            print(
                f"{g:<9}"
                f"{owner:<8}"
                f"{auc:>9.3f}"
                f"{eer:>9.3f}"
                f"{far:>9.3f}"
                f"{frr:>9.3f}"
            )

        print(
            f"{'MEAN':<17}"
            f"{np.mean([x[2] for x in summary]):>9.3f}"
            f"{np.mean([x[3] for x in summary]):>9.3f}"
            f"{np.mean([x[4] for x in summary]):>9.3f}"
            f"{np.mean([x[5] for x in summary]):>9.3f}"
        )


# =========================================================
# Task A: Gesture classification
# =========================================================
def task_gesture(
    X,
    meta,
    args,
    device,
):
    gesture = meta["gesture"]
    performer = meta["performer"]

    classes = sorted(
        set(gesture)
    )

    class_to_id = {
        c: i
        for i, c in enumerate(classes)
    }

    y = np.array(
        [
            class_to_id[g]
            for g in gesture
        ],
        dtype=np.int64,
    )

    cv = StratifiedGroupKFold(
        n_splits=min(
            5,
            len(set(performer)),
        ),
        shuffle=True,
        random_state=args.seed,
    )

    pred = np.full(
        len(y),
        -1,
        dtype=np.int64,
    )

    print("\n" + "=" * 78)

    print(
        f"Transformer - 제스처 분류 "
        f"({len(classes)}-class, performer hold-out)"
    )

    print("=" * 78)

    for fold_no, (
        tr,
        te,
    ) in enumerate(
        cv.split(
            X,
            y,
            groups=performer,
        ),
        1,
    ):

        Xtr, Xte = normalize_from_train(
            X[tr],
            X[te],
        )

        model = train_one_model(
            X_train=Xtr,
            y_train=y[tr],
            n_classes=len(classes),
            args=args,
            seed=(
                args.seed
                + fold_no
            ),
            device=device,
        )

        prob = predict_proba(
            model,
            Xte,
            args.batch_size,
            device,
        )

        pred[te] = prob.argmax(
            axis=1
        )

        print(
            f"fold {fold_no}: "
            f"train={len(tr)}, "
            f"test={len(te)}"
        )

    print(
        f"\nAccuracy = "
        f"{accuracy_score(y, pred):.3f}"
    )

    print(
        classification_report(
            y,
            pred,
            target_names=classes,
            digits=3,
            zero_division=0,
        )
    )

    print(
        "Confusion matrix "
        "(row=true, col=pred)"
    )

    print(
        confusion_matrix(
            y,
            pred,
        )
    )


# =========================================================
# Main
# =========================================================
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--data",
        default=str(
            paths.DATASET_NPZ
        ),
        help=(
            "dataset.npz 경로. "
            f"기본값: {paths.DATASET_NPZ}"
        ),
    )

    ap.add_argument(
        "--task",
        choices=[
            "verification",
            "gesture",
            "both",
        ],
        default="verification",
    )

    ap.add_argument(
        "--d-model",
        type=int,
        default=128,
    )

    ap.add_argument(
        "--nhead",
        type=int,
        default=4,
    )

    ap.add_argument(
        "--num-layers",
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
        default=0.20,
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
        default=5e-4,
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
        "05_train_transformer",
        out_dir=args.log_dir,
        data_files=[
            data_path
        ],
        extra={
            "task": args.task,
            "d_model": args.d_model,
            "nhead": args.nhead,
            "num_layers": args.num_layers,
            "ff_dim": args.ff_dim,
            "dropout": args.dropout,
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
        f"-> [N={len(X)}, T={T}, D={D}]"
    )

    print(
        f"gestures = "
        f"{sorted(set(meta['gesture']))}"
    )

    print(
        f"performers = "
        f"{sorted(set(meta['performer']))}"
    )

    if canonical_hand is not None:

        print(
            "좌우 정규화: "
            + (
                "켬"
                if canonical_hand
                else "끔"
            )
        )

    probe = GestureTransformer(
        input_dim=D,
        seq_len=T,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
        n_classes=2,
    )

    print(
        f"Transformer trainable params = "
        f"{count_parameters(probe):,}"
    )

    if args.task in (
        "gesture",
        "both",
    ):

        task_gesture(
            X,
            meta,
            args,
            device,
        )

    if args.task in (
        "verification",
        "both",
    ):

        task_verification(
            X,
            meta,
            args,
            device,
        )


if __name__ == "__main__":
    main()