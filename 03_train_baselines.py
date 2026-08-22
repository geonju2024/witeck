"""
03_train_baselines.py

dataset.npz -> baseline 모델 5종을 같은 데이터와 같은 분할에서 학습/평가한다.

Baseline models
---------------
Classical ML
    1) Logistic Regression
    2) SVM-RBF
    3) Random Forest

Sequence DL
    4) GRU
    5) LSTM

입력
----
02_build_features.py가 저장한 dataset.npz

    X: [N, T*D]
    T: 시간 길이 (현재 32)
    D: 프레임당 feature 수 (현재 169)

Classical ML:
    [N, T*D] 그대로 사용

GRU / LSTM:
    [N, T*D] -> [N, T, D] 로 복원해서 사용

과제 A) Gesture classification
    "어떤 제스처인가?"
    performer 단위 hold-out

과제 B) Verification
    "등록자 본인(own)인가, 타인의 모방인가?"

Verification에서는 기존 코드의 네 평가 방식을 유지한다.
    혼합
    타인분리
    세션분리
    이중분리

특히 이중분리는
    학습에 없던 등록자의 다른 촬영일
    +
    학습에 없던 impostor performer
를 동시에 test에 넣는다.

주요 인증 지표
    AUC / EER / FAR / FRR

사용법
------
    python 03_train_baselines.py
    python 03_train_baselines.py --task verification
    python 03_train_baselines.py --task gesture
    python 03_train_baselines.py --task both
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import warnings
from copy import deepcopy

import numpy as np

warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    StratifiedGroupKFold,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import paths
from runlog import add_log_arguments, start_run_log


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset(path: str):
    z = np.load(path, allow_pickle=True)

    X_flat = z["X"].astype(np.float32)

    if "T" not in z.files or "D" not in z.files:
        raise ValueError(
            "dataset.npz에 T 또는 D가 없습니다. "
            "02_build_features.py로 다시 생성하세요."
        )

    T = int(z["T"])
    D = int(z["D"])

    if X_flat.ndim != 2:
        raise ValueError(f"X는 [N, T*D]여야 합니다. 현재 shape={X_flat.shape}")

    if X_flat.shape[1] != T * D:
        raise ValueError(f"X.shape[1]={X_flat.shape[1]} != T*D={T*D}")

    X_seq = X_flat.reshape(len(X_flat), T, D)

    required = ["gesture", "performer", "session", "role"]
    missing = [k for k in required if k not in z.files]
    if missing:
        raise ValueError(
            f"dataset.npz에 필요한 metadata가 없습니다: {missing}. "
            "02_build_features.py를 다시 실행하세요."
        )

    meta = {
        "gesture": z["gesture"].astype(str),
        "performer": z["performer"].astype(str),
        "session": z["session"].astype(str),
        "role": z["role"].astype(str),
    }

    canonical_hand = bool(z["canonical_hand"]) if "canonical_hand" in z.files else None

    return X_flat, X_seq, meta, T, D, canonical_hand


def classical_models():
    return {
        "LogReg": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, C=1.0),
        ),
        "SVM-RBF": make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", C=10, gamma="scale"),
        ),
        "RF": RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=0,
        ),
    }


def classical_score_method(name: str) -> str:
    return "predict_proba" if name == "RF" else "decision_function"


class RNNClassifier(nn.Module):
    def __init__(
        self,
        cell: str,
        input_dim: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        n_classes: int,
    ) -> None:
        super().__init__()

        cell = cell.lower()
        if cell not in {"gru", "lstm"}:
            raise ValueError(f"지원하지 않는 RNN cell: {cell}")

        rnn_dropout = dropout if num_layers > 1 else 0.0

        if cell == "gru":
            self.rnn = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=rnn_dropout,
            )
        else:
            self.rnn = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=rnn_dropout,
            )

        self.cell = cell
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.cell == "gru":
            _, h = self.rnn(x)
        else:
            _, (h, _) = self.rnn(x)

        x = h[-1]
        x = self.norm(x)
        x = self.dropout(x)
        return self.head(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def normalize_sequence_from_train(X_train: np.ndarray, X_other: np.ndarray):
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (X_train - mean) / std, (X_other - mean) / std


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool):
    xt = torch.from_numpy(X.astype(np.float32))
    yt = torch.from_numpy(y.astype(np.int64))

    return DataLoader(
        TensorDataset(xt, yt),
        batch_size=batch_size,
        shuffle=shuffle,
    )


def split_train_val_indices(y: np.ndarray, seed: int):
    idx = np.arange(len(y))
    values, counts = np.unique(y, return_counts=True)

    if len(values) < 2 or counts.min() < 2 or len(y) < 12:
        return idx, idx

    tr, va = train_test_split(
        idx,
        test_size=0.20,
        random_state=seed,
        stratify=y,
    )
    return tr, va


def train_rnn_model(
    cell: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_classes: int,
    args,
    seed: int,
    device: torch.device,
):
    seed_everything(seed)

    tr_idx, va_idx = split_train_val_indices(y_train, seed)

    Xtr = X_train[tr_idx]
    ytr = y_train[tr_idx]
    Xva = X_train[va_idx]
    yva = y_train[va_idx]

    train_loader = make_loader(Xtr, ytr, args.batch_size, shuffle=True)
    val_loader = make_loader(Xva, yva, args.batch_size, shuffle=False)

    model = RNNClassifier(
        cell=cell,
        input_dim=X_train.shape[2],
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.rnn_dropout,
        n_classes=n_classes,
    ).to(device)

    counts = np.bincount(ytr, minlength=n_classes).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()

    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device)
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
        model.train()

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)

            logits = model(xb)
            loss = criterion(logits, yb)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
            )

            optimizer.step()

        model.eval()

        val_loss = 0.0
        n_val = 0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                logits = model(xb)
                loss = criterion(logits, yb)

                val_loss += float(loss.item()) * len(yb)
                n_val += len(yb)

        val_loss /= max(n_val, 1)

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


@torch.no_grad()
def predict_rnn_proba(
    model: nn.Module,
    X: np.ndarray,
    batch_size: int,
    device: torch.device,
):
    model.eval()

    dummy_y = np.zeros(len(X), dtype=np.int64)
    loader = make_loader(X, dummy_y, batch_size, shuffle=False)

    probs = []

    for xb, _ in loader:
        xb = xb.to(device)

        p = torch.softmax(
            model(xb),
            dim=1,
        )

        probs.append(p.cpu().numpy())

    return np.concatenate(probs, axis=0)


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
    roc_auc = float(roc_auc_score(y_true, score))

    return eer, threshold, roc_auc, far, frr


def impostor_grouped_folds(y, performer, seed=0):
    y = np.asarray(y)
    performer = np.asarray(performer)

    pos = np.where(y == 1)[0]
    impostors = sorted(set(performer[y == 0]))

    K = len(impostors)

    if K < 2 or len(pos) < K:
        return None

    rng = np.random.RandomState(seed)

    pos_folds = np.array_split(
        pos[rng.permutation(len(pos))],
        K,
    )

    all_idx = np.arange(len(y))
    folds = []

    for k, who in enumerate(impostors):
        te = np.concatenate([
            pos_folds[k],
            np.where((y == 0) & (performer == who))[0],
        ])

        tr = np.setdiff1d(all_idx, te)
        folds.append((tr, te))

    return folds


def _split_list(items, k):
    n = len(items)
    return [
        items[i * n // k:(i + 1) * n // k]
        for i in range(k)
    ]


def session_folds(y, performer, session, seed=0):
    y = np.asarray(y)
    performer = np.asarray(performer)
    session = np.asarray(session)

    pos = np.where(y == 1)[0]
    sessions = sorted(set(session[pos]))

    K = len(sessions)

    if K < 2:
        return None

    neg = np.where(y == 0)[0]

    rng = np.random.RandomState(seed)

    neg_folds = np.array_split(
        neg[rng.permutation(len(neg))],
        K,
    )

    all_idx = np.arange(len(y))
    folds = []

    for k, sess in enumerate(sessions):
        te = np.concatenate([
            pos[session[pos] == sess],
            neg_folds[k],
        ])

        tr = np.setdiff1d(all_idx, te)
        folds.append((tr, te))

    return folds


def session_impostor_folds(y, performer, session):
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
        te = np.concatenate([
            pos[np.isin(session[pos], sess_groups[k])],
            np.where(
                (y == 0)
                & np.isin(performer, imp_groups[k])
            )[0],
        ])

        tr = np.setdiff1d(all_idx, te)
        folds.append((tr, te))

    return folds


SCHEME_ORDER = [
    "혼합",
    "타인분리",
    "세션분리",
    "이중분리",
]


def build_verification_schemes(y, performer, session):
    n_pos = int((y == 1).sum())

    schemes = [
        (
            "혼합",
            list(
                StratifiedKFold(
                    n_splits=min(5, n_pos),
                    shuffle=True,
                    random_state=0,
                ).split(
                    np.zeros(len(y)),
                    y,
                )
            ),
        )
    ]

    f = impostor_grouped_folds(
        y,
        performer,
    )
    if f is not None:
        schemes.append(("타인분리", f))

    if session is not None:
        f = session_folds(
            y,
            performer,
            session,
        )
        if f is not None:
            schemes.append(("세션분리", f))

        f = session_impostor_folds(
            y,
            performer,
            session,
        )
        if f is not None:
            schemes.append(("이중분리", f))

    return schemes


def evaluate_classical_verification(X_flat, y, folds):
    results = {}

    for name, model in classical_models().items():
        oof_score = np.full(
            len(y),
            np.nan,
            dtype=np.float64,
        )

        method = classical_score_method(name)

        for tr, te in folds:
            model.fit(
                X_flat[tr],
                y[tr],
            )

            score = getattr(model, method)(
                X_flat[te]
            )

            if np.ndim(score) == 2:
                score = score[:, 1]

            oof_score[te] = score

        valid = ~np.isnan(oof_score)

        results[name] = eer_metrics(
            y[valid],
            oof_score[valid],
        )

    return results


def evaluate_classical_gesture(X_flat, y, performer):
    n_groups = len(set(performer))

    cv = StratifiedGroupKFold(
        n_splits=min(5, n_groups),
        shuffle=True,
        random_state=0,
    )

    folds = list(
        cv.split(
            X_flat,
            y,
            groups=performer,
        )
    )

    out = {}

    for name, model in classical_models().items():
        pred = np.full(
            len(y),
            -1,
            dtype=np.int64,
        )

        for tr, te in folds:
            model.fit(
                X_flat[tr],
                y[tr],
            )

            pred[te] = model.predict(
                X_flat[te]
            )

        out[name] = pred

    return out, folds


def evaluate_rnn_verification(
    cell,
    X_seq,
    y,
    folds,
    args,
    device,
    base_seed,
):
    oof_score = np.full(
        len(y),
        np.nan,
        dtype=np.float32,
    )

    for fold_no, (tr, te) in enumerate(folds, 1):
        Xtr, Xte = normalize_sequence_from_train(
            X_seq[tr],
            X_seq[te],
        )

        model = train_rnn_model(
            cell=cell,
            X_train=Xtr,
            y_train=y[tr],
            n_classes=2,
            args=args,
            seed=base_seed + fold_no,
            device=device,
        )

        prob = predict_rnn_proba(
            model,
            Xte,
            args.batch_size,
            device,
        )[:, 1]

        oof_score[te] = prob

    valid = ~np.isnan(oof_score)

    return eer_metrics(
        y[valid],
        oof_score[valid],
    )


def evaluate_rnn_gesture(
    cell,
    X_seq,
    y,
    folds,
    args,
    device,
    base_seed,
    n_classes,
):
    pred = np.full(
        len(y),
        -1,
        dtype=np.int64,
    )

    for fold_no, (tr, te) in enumerate(folds, 1):
        Xtr, Xte = normalize_sequence_from_train(
            X_seq[tr],
            X_seq[te],
        )

        model = train_rnn_model(
            cell=cell,
            X_train=Xtr,
            y_train=y[tr],
            n_classes=n_classes,
            args=args,
            seed=base_seed + fold_no,
            device=device,
        )

        prob = predict_rnn_proba(
            model,
            Xte,
            args.batch_size,
            device,
        )

        pred[te] = prob.argmax(axis=1)

    return pred


def task_a_gesture(
    X_flat,
    X_seq,
    meta,
    args,
    device,
):
    gesture = meta["gesture"]
    performer = meta["performer"]

    classes = sorted(set(gesture))
    class_to_id = {
        c: i
        for i, c in enumerate(classes)
    }

    y = np.array(
        [class_to_id[g] for g in gesture],
        dtype=np.int64,
    )

    print("\n" + "=" * 78)
    print(
        f"과제 A) 제스처 분류 "
        f"({len(classes)}-class, performer hold-out)"
    )
    print("=" * 78)

    classical_pred, folds = evaluate_classical_gesture(
        X_flat,
        y,
        performer,
    )

    all_pred = dict(classical_pred)

    all_pred["GRU"] = evaluate_rnn_gesture(
        cell="gru",
        X_seq=X_seq,
        y=y,
        folds=folds,
        args=args,
        device=device,
        base_seed=args.seed + 1000,
        n_classes=len(classes),
    )

    all_pred["LSTM"] = evaluate_rnn_gesture(
        cell="lstm",
        X_seq=X_seq,
        y=y,
        folds=folds,
        args=args,
        device=device,
        base_seed=args.seed + 2000,
        n_classes=len(classes),
    )

    print(f"\n{'Model':<12}{'Accuracy':>12}")
    print("-" * 24)

    for name in [
        "LogReg",
        "SVM-RBF",
        "RF",
        "GRU",
        "LSTM",
    ]:
        acc = accuracy_score(
            y,
            all_pred[name],
        )

        print(
            f"{name:<12}"
            f"{acc:>12.3f}"
        )

    if args.gesture_detail:
        for name in [
            "LogReg",
            "SVM-RBF",
            "RF",
            "GRU",
            "LSTM",
        ]:
            print("\n" + "-" * 78)
            print(f"[{name}]")

            print(
                classification_report(
                    y,
                    all_pred[name],
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
                    all_pred[name],
                )
            )


def task_b_verification(
    X_flat,
    X_seq,
    meta,
    owners,
    args,
    device,
):
    gesture = meta["gesture"]
    performer = meta["performer"]
    session = meta["session"]

    print("\n" + "=" * 78)
    print(
        "과제 B) 본인 인증 "
        "(제스처별 own vs impostor)"
    )
    print("=" * 78)

    print(
        "모든 baseline 모델을 동일한 fold에서 비교합니다."
    )

    print(
        "[혼합]     같은 impostor가 train/test에 모두 등장"
    )
    print(
        "[타인분리] unseen impostor 평가"
    )
    print(
        "[세션분리] 다른 날의 본인 평가"
    )
    print(
        "[이중분리] 다른 날의 본인 + unseen impostor 동시 평가"
    )

    rows = []

    for gesture_no, g in enumerate(sorted(set(gesture))):
        owner = owners.get(g)

        if owner is None:
            print(f"\n[{g}] 등록자 정보 없음 -> 건너뜀")
            continue

        mask = gesture == g

        Xg_flat = X_flat[mask]
        Xg_seq = X_seq[mask]
        pg = performer[mask]
        sg = session[mask]

        y = (pg == owner).astype(np.int64)

        n_pos = int(y.sum())
        n_neg = int((y == 0).sum())
        n_sess = len(set(sg[y == 1]))
        n_impostors = len(set(pg[y == 0]))

        print(
            f"\n[{g}] "
            f"등록자={owner} | "
            f"본인={n_pos} | "
            f"타인={n_neg} | "
            f"타인 수행자={n_impostors}명 | "
            f"등록자 세션={n_sess}개"
        )

        if n_pos < 4 or n_neg < 4:
            print("샘플이 너무 적어 평가 생략")
            continue

        schemes = build_verification_schemes(
            y,
            pg,
            sg,
        )

        gesture_results = {}

        for scheme_no, (scheme_name, folds) in enumerate(schemes):
            print(
                f"\n  [{scheme_name}] "
                f"folds={len(folds)}"
            )

            result = {}

            classical = evaluate_classical_verification(
                Xg_flat,
                y,
                folds,
            )

            result.update(classical)

            result["GRU"] = evaluate_rnn_verification(
                cell="gru",
                X_seq=Xg_seq,
                y=y,
                folds=folds,
                args=args,
                device=device,
                base_seed=(
                    args.seed
                    + gesture_no * 10000
                    + scheme_no * 1000
                    + 100
                ),
            )

            result["LSTM"] = evaluate_rnn_verification(
                cell="lstm",
                X_seq=Xg_seq,
                y=y,
                folds=folds,
                args=args,
                device=device,
                base_seed=(
                    args.seed
                    + gesture_no * 10000
                    + scheme_no * 1000
                    + 500
                ),
            )

            gesture_results[scheme_name] = result

            print(
                f"  {'Model':<12}"
                f"{'AUC':>8}"
                f"{'EER':>8}"
                f"{'FAR':>8}"
                f"{'FRR':>8}"
            )

            for name in [
                "LogReg",
                "SVM-RBF",
                "RF",
                "GRU",
                "LSTM",
            ]:
                (
                    eer,
                    threshold,
                    roc_auc,
                    far,
                    frr,
                ) = result[name]

                print(
                    f"  {name:<12}"
                    f"{roc_auc:>8.3f}"
                    f"{eer:>8.3f}"
                    f"{far:>8.3f}"
                    f"{frr:>8.3f}"
                )

        rows.append(
            (
                g,
                owner,
                gesture_results,
            )
        )

    if not rows:
        return

    print("\n" + "=" * 78)
    print(
        "BASELINE SUMMARY "
        "(gesture별 metric의 단순 평균)"
    )
    print("=" * 78)

    used_schemes = [
        s
        for s in SCHEME_ORDER
        if any(
            s in result
            for _, _, result in rows
        )
    ]

    model_order = [
        "LogReg",
        "SVM-RBF",
        "RF",
        "GRU",
        "LSTM",
    ]

    for scheme in used_schemes:
        print(f"\n[{scheme}]")

        print(
            f"{'Model':<12}"
            f"{'AUC':>8}"
            f"{'EER':>8}"
            f"{'FAR':>8}"
            f"{'FRR':>8}"
        )

        for model_name in model_order:
            metrics = []

            for _, _, result in rows:
                if (
                    scheme in result
                    and model_name in result[scheme]
                ):
                    metrics.append(
                        result[scheme][model_name]
                    )

            if not metrics:
                continue

            mean_eer = float(
                np.mean([m[0] for m in metrics])
            )
            mean_auc = float(
                np.mean([m[2] for m in metrics])
            )
            mean_far = float(
                np.mean([m[3] for m in metrics])
            )
            mean_frr = float(
                np.mean([m[4] for m in metrics])
            )

            print(
                f"{model_name:<12}"
                f"{mean_auc:>8.3f}"
                f"{mean_eer:>8.3f}"
                f"{mean_far:>8.3f}"
                f"{mean_frr:>8.3f}"
            )

    print(
        "\n주 보고 지표는 [이중분리] 결과를 권장합니다."
    )


def extract_owners(meta, owners_path=None):
    gesture = meta["gesture"]
    performer = meta["performer"]
    role = meta["role"]

    owners = {}

    if role is not None:
        candidates = {}

        for g, p, r in zip(
            gesture,
            performer,
            role,
        ):
            if r == "own":
                candidates.setdefault(
                    g,
                    set(),
                ).add(p)

        for g, who in candidates.items():
            if len(who) == 1:
                owners[g] = next(iter(who))
            else:
                print(
                    f"경고: {g}의 own performer가 "
                    f"{sorted(who)}로 여러 명입니다."
                )

    if owners_path:
        with open(
            owners_path,
            encoding="utf-8-sig",
            newline="",
        ) as f:
            for row in csv.DictReader(f):
                owners[
                    row["gesture"].strip()
                ] = row[
                    "owner"
                ].strip()

    for g in sorted(set(gesture)):
        if g in owners:
            continue

        counts = {}

        for gg, pp in zip(
            gesture,
            performer,
        ):
            if gg == g:
                counts[pp] = (
                    counts.get(pp, 0)
                    + 1
                )

        if counts:
            owners[g] = max(
                counts,
                key=counts.get,
            )

            print(
                f"경고: {g} 등록자 정보 없음 -> "
                f"최다 수행자 {owners[g]}로 fallback"
            )

    return owners


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
        default=str(paths.DATASET_NPZ),
        help=(
            "dataset.npz 경로. "
            f"기본값: {paths.DATASET_NPZ}"
        ),
    )

    ap.add_argument(
        "--owners",
        help=(
            "선택: CSV 형식 gesture,owner. "
            "기본은 dataset.npz의 role=own으로 자동 판정."
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
        "--hidden-size",
        type=int,
        default=64,
    )

    ap.add_argument(
        "--num-layers",
        type=int,
        default=1,
    )

    ap.add_argument(
        "--rnn-dropout",
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

    ap.add_argument(
        "--gesture-detail",
        action="store_true",
        help=(
            "gesture task에서 모델별 classification report와 "
            "confusion matrix까지 출력"
        ),
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

    owners_path = (
        str(
            paths.assert_external(
                args.owners,
                "owners.csv",
            )
        )
        if args.owners
        else None
    )

    start_run_log(
        "03_train_baselines",
        out_dir=args.log_dir,
        data_files=[
            p
            for p in (
                data_path,
                owners_path,
            )
            if p
        ],
        extra={
            "task": args.task,
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "rnn_dropout": args.rnn_dropout,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "patience": args.patience,
            "seed": args.seed,
        },
        enabled=not args.no_log,
    )

    seed_everything(args.seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    (
        X_flat,
        X_seq,
        meta,
        T,
        D,
        canonical_hand,
    ) = load_dataset(data_path)

    print(f"데이터: {data_path}")
    print(f"device: {device}")
    print(f"X_flat = {X_flat.shape}")
    print(
        f"X_seq  = {X_seq.shape} "
        f"[N={len(X_seq)}, T={T}, D={D}]"
    )
    print(
        f"gestures   = "
        f"{sorted(set(meta['gesture']))}"
    )
    print(
        f"performers = "
        f"{sorted(set(meta['performer']))}"
    )
    print(
        f"sessions   = "
        f"{len(set(meta['session']))}개"
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

    gru_probe = RNNClassifier(
        cell="gru",
        input_dim=D,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.rnn_dropout,
        n_classes=2,
    )

    lstm_probe = RNNClassifier(
        cell="lstm",
        input_dim=D,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.rnn_dropout,
        n_classes=2,
    )

    print(
        f"GRU trainable params  = "
        f"{count_parameters(gru_probe):,}"
    )
    print(
        f"LSTM trainable params = "
        f"{count_parameters(lstm_probe):,}"
    )

    owners = extract_owners(
        meta,
        owners_path,
    )

    print(f"owners = {owners}")

    if args.task in (
        "gesture",
        "both",
    ):
        task_a_gesture(
            X_flat,
            X_seq,
            meta,
            args,
            device,
        )

    if args.task in (
        "verification",
        "both",
    ):
        task_b_verification(
            X_flat,
            X_seq,
            meta,
            owners,
            args,
            device,
        )


if __name__ == "__main__":
    main()