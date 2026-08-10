"""
04_train_1dcnn.py

현재 dataset.npz의 시계열 특징을 1D CNN으로 학습/평가한다.

현재 02_build_features.py가 저장하는 X는 [N, T*D] 형태이지만,
1D CNN에서는 다시 [N, D, T]로 복원해서 시간축(T)에 convolution을 적용한다.

기본 과제: verification
  제스처별로 '등록자 본인(1) vs 타인 모방(0)'을 분류한다.
  평가 시 등록자의 촬영 세션과 타인 수행자를 동시에 hold-out하여
  다른 날의 본인 + 학습에 없던 공격자를 테스트한다.

선택 과제: gesture
  G1~G5 제스처 분류. 수행자 단위로 hold-out한다.

경로는 paths.py 가 정한다(공유 드라이브 WITECH). 인자 없이 실행하면
WITECH/derived/dataset.npz 를 읽고 WITECH/derived/runs/ 에 로그를 남긴다.

사용 예시:
  python 04_train_1dcnn.py
  python 04_train_1dcnn.py --task both --epochs 50
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
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from torch.utils.data import DataLoader, TensorDataset

import paths
from runlog import add_log_arguments, start_run_log


# -------------------------
# reproducibility
# -------------------------
def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -------------------------
# model
# -------------------------
class Small1DCNN(nn.Module):
    """
    입력: [B, D, T]
      B = batch
      D = 프레임당 feature 수 (현재 dataset.npz는 169)
      T = 시간 길이 (현재 dataset.npz는 32)

    Conv1d는 T축을 따라 움직이면서 제스처의 시간적 변화 패턴을 학습한다.
    """

    def __init__(self, in_channels: int, n_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.20),

            nn.Conv1d(64, 96, kernel_size=3, padding=1),
            nn.BatchNorm1d(96),
            nn.ReLU(),

            # dilation=2: 조금 더 넓은 시간 범위를 한 번에 본다.
            nn.Conv1d(96, 128, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.30),

            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(128, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x).squeeze(-1)
        return self.head(x)


# -------------------------
# data helpers
# -------------------------
def load_dataset(path: str):
    z = np.load(path, allow_pickle=True)
    X = z["X"].astype(np.float32)
    T = int(z["T"])
    D = int(z["D"])

    if X.shape[1] != T * D:
        raise ValueError(f"X.shape[1]={X.shape[1]} != T*D={T*D}")

    # 02_build_features.py에서 feat.reshape(-1)로 저장했으므로
    # [N, T*D] -> [N, T, D]
    X = X.reshape(len(X), T, D)

    meta = {
        "gesture": z["gesture"].astype(str),
        "performer": z["performer"].astype(str),
        "session": z["session"].astype(str),
        "role": z["role"].astype(str),
    }
    return X, meta, T, D


def normalize_from_train(X_train: np.ndarray, X_other: np.ndarray):
    """train 데이터에서만 mean/std를 구해 leakage를 막는다."""
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (X_train - mean) / std, (X_other - mean) / std


def to_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool):
    # Conv1d 입력은 [B, channels, time]
    xt = torch.from_numpy(X).permute(0, 2, 1).contiguous().float()
    yt = torch.from_numpy(y).long()
    ds = TensorDataset(xt, yt)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def split_train_val(X: np.ndarray, y: np.ndarray, seed: int):
    idx = np.arange(len(y))
    # 데이터가 너무 작거나 한 클래스가 1개뿐이면 validation 분리를 생략한다.
    vals, counts = np.unique(y, return_counts=True)
    if len(vals) < 2 or counts.min() < 2 or len(y) < 12:
        return X, y, X, y

    tr_idx, va_idx = train_test_split(
        idx,
        test_size=0.20,
        random_state=seed,
        stratify=y,
    )
    return X[tr_idx], y[tr_idx], X[va_idx], y[va_idx]


# -------------------------
# training
# -------------------------
def train_one_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_classes: int,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    device: torch.device,
):
    seed_everything(seed)

    Xtr, ytr, Xva, yva = split_train_val(X_train, y_train, seed)
    train_loader = to_loader(Xtr, ytr, batch_size, shuffle=True)
    val_loader = to_loader(Xva, yva, batch_size, shuffle=False)

    model = Small1DCNN(in_channels=X_train.shape[2], n_classes=n_classes).to(device)

    # class imbalance 보정
    counts = np.bincount(ytr, minlength=n_classes).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device)
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)

    best_state = None
    best_val = float("inf")
    bad_epochs = 0
    patience = 8

    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                loss = criterion(model(xb), yb)
                val_loss += float(loss.item()) * len(yb)
                n_val += len(yb)
        val_loss /= max(n_val, 1)

        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def predict_proba(model: nn.Module, X: np.ndarray, device: torch.device, batch_size: int):
    model.eval()
    dummy_y = np.zeros(len(X), dtype=np.int64)
    loader = to_loader(X, dummy_y, batch_size, shuffle=False)
    probs = []
    for xb, _ in loader:
        xb = xb.to(device)
        p = torch.softmax(model(xb), dim=1)
        probs.append(p.cpu().numpy())
    return np.concatenate(probs, axis=0)


# -------------------------
# verification folds
# -------------------------
def _split_list(items, k):
    n = len(items)
    return [items[i * n // k:(i + 1) * n // k] for i in range(k)]


def session_impostor_folds(y, performer, session):
    """
    test에는 동시에
      1) 학습에 없던 등록자 촬영일(session)
      2) 학습에 없던 impostor performer
    가 들어간다.
    """
    y = np.asarray(y)
    performer = np.asarray(performer)
    session = np.asarray(session)

    pos = np.where(y == 1)[0]
    sessions = sorted(set(session[pos]))
    impostors = sorted(set(performer[y == 0]))
    K = min(len(sessions), len(impostors))
    if K < 2:
        return None

    sess_groups = _split_list(sessions, K)
    imp_groups = _split_list(impostors, K)
    all_idx = np.arange(len(y))
    folds = []

    for k in range(K):
        test_idx = np.concatenate([
            pos[np.isin(session[pos], sess_groups[k])],
            np.where((y == 0) & np.isin(performer, imp_groups[k]))[0],
        ])
        train_idx = np.setdiff1d(all_idx, test_idx)
        folds.append((train_idx, test_idx))

    return folds


def eer_metrics(y_true: np.ndarray, score: np.ndarray):
    fpr, tpr, thresholds = roc_curve(y_true, score)
    fnr = 1.0 - tpr
    i = int(np.nanargmin(np.abs(fpr - fnr)))
    eer = float((fpr[i] + fnr[i]) / 2.0)
    threshold = float(thresholds[i])

    pred = (score >= threshold).astype(int)
    n_pos = max(int((y_true == 1).sum()), 1)
    n_neg = max(int((y_true == 0).sum()), 1)
    far = float(((pred == 1) & (y_true == 0)).sum() / n_neg)
    frr = float(((pred == 0) & (y_true == 1)).sum() / n_pos)
    auc = float(roc_auc_score(y_true, score))
    return eer, threshold, auc, far, frr


# -------------------------
# task B: authentication
# -------------------------
def task_verification(X, meta, args, device):
    gesture = meta["gesture"]
    performer = meta["performer"]
    session = meta["session"]
    role = meta["role"]

    # role=own에서 gesture별 등록자를 자동 추출
    owners = {}
    for g in sorted(set(gesture)):
        who = sorted(set(performer[(gesture == g) & (role == "own")]))
        if len(who) == 1:
            owners[g] = who[0]
        else:
            print(f"[{g}] 등록자 판정 불가: {who} -> 생략")

    print("\n" + "=" * 72)
    print("1D CNN - 본인 인증 (이중분리: 다른 날 본인 + unseen impostor)")
    print("=" * 72)

    summary = []

    for gi, g in enumerate(sorted(owners)):
        owner = owners[g]
        mask = gesture == g
        Xg = X[mask]
        pg = performer[mask]
        sg = session[mask]
        rg = role[mask]
        yg = (rg == "own").astype(np.int64)

        folds = session_impostor_folds(yg, pg, sg)
        if folds is None:
            print(f"[{g}] fold 생성 불가 -> 생략")
            continue

        print(
            f"\n[{g}] owner={owner} | genuine={int(yg.sum())}, "
            f"impostor={int((yg == 0).sum())}, folds={len(folds)}"
        )

        oof_score = np.full(len(yg), np.nan, dtype=np.float32)

        for fold_no, (tr, te) in enumerate(folds, 1):
            Xtr, Xte = normalize_from_train(Xg[tr], Xg[te])
            model = train_one_model(
                Xtr,
                yg[tr],
                n_classes=2,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                seed=args.seed + gi * 100 + fold_no,
                device=device,
            )
            prob = predict_proba(model, Xte, device, args.batch_size)[:, 1]
            oof_score[te] = prob
            print(
                f"  fold {fold_no}: train={len(tr):3d}, test={len(te):3d}, "
                f"test genuine={int(yg[te].sum()):2d}, impostor={int((yg[te] == 0).sum()):2d}"
            )

        valid = ~np.isnan(oof_score)
        eer, thr, auc, far, frr = eer_metrics(yg[valid], oof_score[valid])
        print(
            f"  RESULT  AUC={auc:.3f}  EER={eer:.3f}  "
            f"FAR={far:.3f}  FRR={frr:.3f}  EER-threshold={thr:.3f}"
        )
        summary.append((g, owner, auc, eer, far, frr))

    if summary:
        print("\n" + "-" * 72)
        print("요약")
        print(f"{'gesture':<9}{'owner':<8}{'AUC':>9}{'EER':>9}{'FAR':>9}{'FRR':>9}")
        for g, owner, auc, eer, far, frr in summary:
            print(f"{g:<9}{owner:<8}{auc:>9.3f}{eer:>9.3f}{far:>9.3f}{frr:>9.3f}")
        print(
            f"{'MEAN':<17}"
            f"{np.mean([x[2] for x in summary]):>9.3f}"
            f"{np.mean([x[3] for x in summary]):>9.3f}"
            f"{np.mean([x[4] for x in summary]):>9.3f}"
            f"{np.mean([x[5] for x in summary]):>9.3f}"
        )


# -------------------------
# task A: gesture classification
# -------------------------
def task_gesture(X, meta, args, device):
    gesture = meta["gesture"]
    performer = meta["performer"]
    classes = sorted(set(gesture))
    class_to_id = {c: i for i, c in enumerate(classes)}
    y = np.array([class_to_id[g] for g in gesture], dtype=np.int64)

    cv = StratifiedGroupKFold(
        n_splits=min(5, len(set(performer))),
        shuffle=True,
        random_state=args.seed,
    )

    pred = np.full(len(y), -1, dtype=np.int64)

    print("\n" + "=" * 72)
    print(f"1D CNN - 제스처 분류 ({len(classes)}-class, performer hold-out)")
    print("=" * 72)

    for fold_no, (tr, te) in enumerate(cv.split(X, y, groups=performer), 1):
        Xtr, Xte = normalize_from_train(X[tr], X[te])
        model = train_one_model(
            Xtr,
            y[tr],
            n_classes=len(classes),
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed + fold_no,
            device=device,
        )
        prob = predict_proba(model, Xte, device, args.batch_size)
        pred[te] = prob.argmax(axis=1)
        print(f"fold {fold_no}: train={len(tr)}, test={len(te)}")

    print(f"\nAccuracy = {accuracy_score(y, pred):.3f}")
    print(classification_report(y, pred, target_names=classes, digits=3, zero_division=0))
    print("Confusion matrix (row=true, col=pred)")
    print(confusion_matrix(y, pred))


# -------------------------
# main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(paths.DATASET_NPZ),
                    help=f"dataset.npz 경로. 기본값: {paths.DATASET_NPZ}")
    ap.add_argument(
        "--task",
        choices=["verification", "gesture", "both"],
        default="verification",
    )
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    add_log_arguments(ap, paths.RUNS_DIR)
    args = ap.parse_args()

    data_path = str(paths.assert_external(args.data, "dataset.npz"))

    # 이 시점부터의 출력은 derived/runs/ 아래 타임스탬프 로그 파일에도 함께 기록된다.
    start_run_log(
        "04_train_1dcnn",
        out_dir=args.log_dir,
        data_files=[data_path],
        extra={
            "task": args.task,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "seed": args.seed,
        },
        enabled=not args.no_log,
    )

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X, meta, T, D = load_dataset(data_path)
    print(f"데이터 = {data_path}")
    print(f"device = {device}")
    print(f"X = {X.shape} -> [N={len(X)}, T={T}, D={D}]")
    print(f"gestures = {sorted(set(meta['gesture']))}")
    print(f"performers = {sorted(set(meta['performer']))}")

    if args.task in ("gesture", "both"):
        task_gesture(X, meta, args, device)
    if args.task in ("verification", "both"):
        task_verification(X, meta, args, device)


if __name__ == "__main__":
    main()
