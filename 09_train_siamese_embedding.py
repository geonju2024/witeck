"""
09_train_siamese_embedding.py

Dilated Siamese 1D-CNN experiment for unseen-user authentication.

Goal
----
Train ONE shared 1D-CNN encoder using P01~P07.

Positive pair:
    same performer + same gesture

Negative pair:
    different performer + same gesture

P08/P09/P10 are completely excluded from encoder training.

After training:
    unseen user's enrollment samples
        -> embeddings
        -> mean template

    verification sample
        -> embedding
        -> cosine similarity with template
        -> accept / reject

Comparison baseline:
    08_train_embedding.py
"""

from __future__ import annotations

import argparse
import csv
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset, DataLoader

from two_stage_common import (
    load_dataset,
    fit_duration_stats,
    apply_duration_stats,
    fit_sequence_stats,
    apply_sequence_stats,
)
from lite_stats_siamese import LiteStatsSiamese1DCNN


SEED = 42

TRAIN_USERS = [
    "P01",
    "P02",
    "P03",
    "P04",
    "P05",
    "P06",
    "P07",
]

UNSEEN_USERS = [
    "P08",
    "P09",
    "P10",
]


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device():
    """Prefer CUDA, then Apple Metal (MPS), and finally CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")

    return torch.device("cpu")


class ResidualDilatedBlock(nn.Module):
    """시간 간격이 다른 패턴을 읽는 residual temporal block."""

    def __init__(
        self,
        channels,
        dilation,
        dropout=0.20,
    ):
        super().__init__()

        self.norm = nn.GroupNorm(8, channels)
        self.temporal = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
        )
        self.mix = nn.Conv1d(
            channels,
            channels,
            kernel_size=1,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = F.gelu(x)
        x = self.temporal(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.mix(x)
        return residual + x


class MaskedStatisticsPooling(nn.Module):
    """검출된 프레임만 사용해 mean/std/max를 계산한다."""

    def forward(self, x, valid_mask):
        # x: [B, C, T], valid_mask: [B, T]
        mask = valid_mask.to(dtype=x.dtype).unsqueeze(1)

        # 한 영상에서 손이 한 번도 검출되지 않았다면 NaN 대신 전체 프레임을 쓴다.
        empty = mask.sum(dim=2, keepdim=True) == 0
        if empty.any():
            mask = torch.where(empty, torch.ones_like(mask), mask)

        count = mask.sum(dim=2).clamp_min(1.0)
        mean = (x * mask).sum(dim=2) / count

        centered = x - mean.unsqueeze(2)
        variance = (centered.square() * mask).sum(dim=2) / count
        std = torch.sqrt(variance.clamp_min(1e-6))

        masked_x = x.masked_fill(mask == 0, torch.finfo(x.dtype).min)
        maximum = masked_x.max(dim=2).values

        return torch.cat([mean, std, maximum], dim=1)


class Siamese1DCNN(nn.Module):
    """
    사용자 인증용 dilated Siamese 1D-CNN encoder.

    입력은 기존 dataset.npz의 [B, 32, 169]를 그대로 사용한다.
    마지막 채널(valid mask)은 신원 특징으로 직접 학습하지 않고 pooling에서
    MediaPipe 미검출 프레임을 제외하는 용도로만 사용한다.

    dilation 1/2/4/8의 receptive field는 31프레임이므로 32프레임 동작의
    거의 전체 흐름을 한 번에 비교할 수 있다.
    """

    def __init__(
        self,
        input_dim,
        embedding_dim=128,
        channels=96,
        dropout=0.20,
    ):
        super().__init__()

        if input_dim < 2:
            raise ValueError("input_dim must include features and valid mask")
        if channels % 8 != 0:
            raise ValueError("channels must be divisible by 8 for GroupNorm")

        self.mask_index = input_dim - 1
        self.stem = nn.Conv1d(
            input_dim,
            channels,
            kernel_size=1,
        )
        self.blocks = nn.Sequential(
            *[
                ResidualDilatedBlock(
                    channels=channels,
                    dilation=dilation,
                    dropout=dropout,
                )
                for dilation in (1, 2, 4, 8)
            ]
        )
        self.final_norm = nn.GroupNorm(8, channels)
        self.pool = MaskedStatisticsPooling()
        self.embedding_head = nn.Sequential(
            nn.Linear(channels * 3, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, embedding_dim),
        )

    def forward(
        self,
        x,
        duration,
    ):
        del duration  # 기존 학습/평가 함수와 호출 형식을 호환하기 위해 유지한다.

        if x.ndim != 3:
            raise ValueError(f"Expected [B,T,D], got shape={tuple(x.shape)}")

        valid_mask = x[:, :, self.mask_index] > 0.5

        # 결측률 자체를 사용자 ID로 외우지 못하도록 mask 입력 채널은 0으로 만든다.
        features = x.clone()
        features[:, :, self.mask_index] = 0.0

        h = self.stem(features.transpose(1, 2))
        h = self.blocks(h)
        h = F.gelu(self.final_norm(h))
        h = self.pool(h, valid_mask)

        z = self.embedding_head(h)
        return F.normalize(z, p=2, dim=1)


class MultiKernelResidualBlock(nn.Module):
    """Local temporal patterns at three scales without global dilation."""

    def __init__(
        self,
        channels,
        kernels=(3, 5, 7),
        dropout=0.20,
    ):
        super().__init__()

        if not kernels or any(k <= 0 or k % 2 == 0 for k in kernels):
            raise ValueError("kernels must be positive odd integers")

        self.norm = nn.GroupNorm(8, channels)
        self.branches = nn.ModuleList(
            [
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size=kernel,
                    padding=kernel // 2,
                    groups=channels,
                    bias=False,
                )
                for kernel in kernels
            ]
        )
        self.mix = nn.Conv1d(
            channels * len(kernels),
            channels,
            kernel_size=1,
            bias=False,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = F.gelu(self.norm(x))
        x = torch.cat(
            [branch(x) for branch in self.branches],
            dim=1,
        )
        x = self.dropout(self.mix(x))
        return F.gelu(residual + x)


class TemporalPyramidStatisticsPooling(nn.Module):
    """Preserve coarse start/middle/end order with 1/2/4 temporal bins."""

    def __init__(self, levels=(1, 2, 4)):
        super().__init__()

        if not levels or any(level <= 0 for level in levels):
            raise ValueError("levels must contain positive integers")

        self.levels = tuple(levels)
        self.output_multiplier = 2 * sum(self.levels)

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError(f"Expected [B,C,T], got shape={tuple(x.shape)}")

        pooled = []

        for level in self.levels:
            if x.shape[-1] < level:
                raise ValueError(
                    f"Temporal length {x.shape[-1]} is smaller than level {level}"
                )

            for segment in torch.tensor_split(x, level, dim=-1):
                mean = segment.mean(dim=-1)
                variance = segment.var(
                    dim=-1,
                    correction=0,
                )
                std = torch.sqrt(variance.clamp_min(1e-6))
                pooled.extend([mean, std])

        return torch.cat(pooled, dim=1)


class PhasePyramidSiamese1DCNN(nn.Module):
    """Phase-preserving Siamese encoder for the 32-frame WITECK input."""

    def __init__(
        self,
        input_dim,
        embedding_dim=128,
        channels=96,
        kernels=(3, 5, 7),
        levels=(1, 2, 4),
        blocks=3,
        dropout=0.20,
    ):
        super().__init__()

        if input_dim < 2:
            raise ValueError("input_dim must include features and valid mask")
        if channels % 8 != 0:
            raise ValueError("channels must be divisible by 8 for GroupNorm")

        # The final hand-valid channel is quality metadata, not identity input.
        self.feature_dim = input_dim - 1
        self.stem = nn.Sequential(
            nn.Conv1d(
                self.feature_dim,
                channels,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(8, channels),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            *[
                MultiKernelResidualBlock(
                    channels=channels,
                    kernels=kernels,
                    dropout=dropout,
                )
                for _ in range(blocks)
            ]
        )
        self.pool = TemporalPyramidStatisticsPooling(levels=levels)
        pooled_dim = channels * self.pool.output_multiplier
        self.embedding_head = nn.Sequential(
            nn.Linear(pooled_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, embedding_dim),
        )

    def forward(self, x, duration):
        del duration

        if x.ndim != 3:
            raise ValueError(f"Expected [B,T,D], got shape={tuple(x.shape)}")
        if x.shape[-1] != self.feature_dim + 1:
            raise ValueError(
                f"Expected D={self.feature_dim + 1}, got D={x.shape[-1]}"
            )

        # Ignore the final valid-rate channel so capture quality cannot become
        # an identity shortcut. The repaired G1 samples do not need mask-aware
        # convolution.
        h = self.stem(x[:, :, : self.feature_dim].transpose(1, 2))
        h = self.blocks(h)
        h = self.pool(h)
        z = self.embedding_head(h)
        return F.normalize(z, p=2, dim=1)


def chronological_split(
    meta,
    users,
):
    """
    For each known user:

        latest 2 sessions -> validation/calibration
        all previous sessions -> encoder training

    The second-latest validation session is used for enrollment and the latest
    validation session is used for threshold-calibration probes. Neither session
    is visible while the encoder is trained.
    """

    performer = meta["performer"]
    session = meta["session"]

    train_idx = []
    val_idx = []

    for user in users:
        idx = np.where(
            performer == user
        )[0]

        sessions = sorted(
            np.unique(
                session[idx]
            )
        )

        if len(sessions) < 3:
            raise RuntimeError(
                f"{user}: fewer than 3 sessions"
            )

        val_sessions = sessions[-2:]

        tr = idx[
            ~np.isin(session[idx], val_sessions)
        ]

        va = idx[
            np.isin(session[idx], val_sessions)
        ]

        train_idx.extend(
            tr.tolist()
        )

        val_idx.extend(
            va.tolist()
        )

        print(
            f"{user}: "
            f"train={len(tr):3d}, "
            f"val={len(va):3d}, "
            f"val_sessions={list(val_sessions)}"
        )

    return (
        np.asarray(
            train_idx,
            dtype=np.int64,
        ),
        np.asarray(
            val_idx,
            dtype=np.int64,
        ),
    )


def build_pairs(
    indices,
    meta,
    num_pairs,
    rng,
):
    """
    Build balanced Siamese pairs.

    Positive:
        same performer + same gesture

    Negative:
        different performer + same gesture
    """

    performer = meta["performer"]
    gesture = meta["gesture"]
    session = meta["session"]

    by_user_gesture = {}

    for user in TRAIN_USERS:
        for g in sorted(
            np.unique(gesture[indices])
        ):
            idx = indices[
                (performer[indices] == user)
                & (gesture[indices] == g)
            ]

            if len(idx) > 0:
                by_user_gesture[
                    (user, g)
                ] = idx

    positive_keys = [
        key
        for key, idx
        in by_user_gesture.items()
        if len(idx) >= 2
    ]

    gestures = sorted(
        np.unique(
            gesture[indices]
        )
    )

    pair_a = []
    pair_b = []
    targets = []

    half = num_pairs // 2

    # Positive pairs
    for _ in range(half):
        user, g = (
            positive_keys[
                rng.integers(
                    len(positive_keys)
                )
            ]
        )

        candidates = (
            by_user_gesture[
                (user, g)
            ]
        )

        # 같은 촬영 환경을 외우지 않도록 가능한 경우 서로 다른 세션에서 뽑는다.
        candidate_sessions = np.unique(session[candidates])
        if len(candidate_sessions) >= 2:
            chosen_sessions = rng.choice(
                candidate_sessions,
                size=2,
                replace=False,
            )
            chosen = np.asarray(
                [
                    rng.choice(candidates[session[candidates] == s])
                    for s in chosen_sessions
                ],
                dtype=np.int64,
            )
        else:
            chosen = rng.choice(
                candidates,
                size=2,
                replace=False,
            )

        pair_a.append(
            int(chosen[0])
        )

        pair_b.append(
            int(chosen[1])
        )

        targets.append(1.0)

    # Negative pairs
    for _ in range(
        num_pairs - half
    ):
        g = gestures[
            rng.integers(
                len(gestures)
            )
        ]

        valid_users = [
            u
            for u in TRAIN_USERS
            if (u, g)
            in by_user_gesture
        ]

        if len(valid_users) < 2:
            continue

        u1, u2 = rng.choice(
            valid_users,
            size=2,
            replace=False,
        )

        i1 = rng.choice(
            by_user_gesture[
                (u1, g)
            ]
        )

        i2 = rng.choice(
            by_user_gesture[
                (u2, g)
            ]
        )

        pair_a.append(
            int(i1)
        )

        pair_b.append(
            int(i2)
        )

        targets.append(-1.0)

    return (
        np.asarray(
            pair_a,
            dtype=np.int64,
        ),
        np.asarray(
            pair_b,
            dtype=np.int64,
        ),
        np.asarray(
            targets,
            dtype=np.float32,
        ),
    )


class PairDataset(Dataset):

    def __init__(
        self,
        X,
        duration,
        pair_a,
        pair_b,
        targets,
    ):
        self.X = X
        self.duration = duration

        self.pair_a = pair_a
        self.pair_b = pair_b
        self.targets = targets

    def __len__(self):
        return len(
            self.targets
        )

    def __getitem__(
        self,
        idx,
    ):
        a = self.pair_a[idx]
        b = self.pair_b[idx]

        return (
            torch.from_numpy(
                self.X[a]
            ).float(),
            torch.tensor(
                self.duration[a],
                dtype=torch.float32,
            ),

            torch.from_numpy(
                self.X[b]
            ).float(),
            torch.tensor(
                self.duration[b],
                dtype=torch.float32,
            ),

            torch.tensor(
                self.targets[idx],
                dtype=torch.float32,
            ),
        )


def run_pair_epoch(
    model,
    loader,
    criterion,
    device,
    optimizer=None,
):
    training = (
        optimizer is not None
    )

    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_n = 0

    context = (
        torch.enable_grad()
        if training
        else torch.no_grad()
    )

    with context:
        for (
            xa,
            da,
            xb,
            db,
            target,
        ) in loader:

            xa = xa.to(device)
            da = da.to(device)

            xb = xb.to(device)
            db = db.to(device)

            target = target.to(
                device
            )

            if training:
                optimizer.zero_grad()

            # The two Siamese branches share exactly the same encoder.
            # Running them as one larger batch reduces Python/kernel-launch
            # overhead, which matters on CPU and Apple Silicon laptops.
            z_pair = model(
                torch.cat(
                    [xa, xb],
                    dim=0,
                ),
                torch.cat(
                    [da, db],
                    dim=0,
                ),
            )

            za, zb = z_pair.chunk(
                2,
                dim=0,
            )

            loss = criterion(
                za,
                zb,
                target,
            )

            if training:
                loss.backward()
                optimizer.step()

            total_loss += (
                loss.item()
                * len(target)
            )

            total_n += len(
                target
            )

    return (
        total_loss
        / max(
            total_n,
            1,
        )
    )


@torch.no_grad()
def extract_embeddings(
    model,
    X,
    duration,
    device,
    batch_size=128,
):
    model.eval()

    output = []

    for start in range(
        0,
        len(X),
        batch_size,
    ):
        end = min(
            start + batch_size,
            len(X),
        )

        xb = torch.from_numpy(
            X[start:end]
        ).float().to(device)

        db = torch.from_numpy(
            duration[start:end]
        ).float().to(device)

        z = model(
            xb,
            db,
        )

        output.append(
            z.cpu().numpy()
        )

    return np.concatenate(
        output,
        axis=0,
    )


def cosine_scores(
    embeddings,
    template,
):
    template = (
        template
        / (
            np.linalg.norm(
                template
            )
            + 1e-12
        )
    )

    embeddings = (
        embeddings
        / (
            np.linalg.norm(
                embeddings,
                axis=1,
                keepdims=True,
            )
            + 1e-12
        )
    )

    return embeddings @ template


def binary_metrics(
    labels,
    scores,
    threshold,
    compute_auc=True,
):
    labels = np.asarray(
        labels,
        dtype=np.int64,
    )

    scores = np.asarray(
        scores,
        dtype=np.float64,
    )

    pred = (
        scores >= threshold
    ).astype(
        np.int64
    )

    genuine = (
        labels == 1
    )

    impostor = (
        labels == 0
    )

    tp = int(
        np.sum(
            (pred == 1)
            & genuine
        )
    )

    fn = int(
        np.sum(
            (pred == 0)
            & genuine
        )
    )

    fp = int(
        np.sum(
            (pred == 1)
            & impostor
        )
    )

    tn = int(
        np.sum(
            (pred == 0)
            & impostor
        )
    )

    far = (
        fp
        / max(
            int(
                impostor.sum()
            ),
            1,
        )
    )

    frr = (
        fn
        / max(
            int(
                genuine.sum()
            ),
            1,
        )
    )

    acc = (
        (tp + tn)
        / max(
            len(labels),
            1,
        )
    )

    bal = (
        (1.0 - far)
        + (1.0 - frr)
    ) / 2.0

    auc = (
        float(roc_auc_score(labels, scores))
        if compute_auc and len(np.unique(labels)) == 2
        else float("nan")
    )

    return {
        "auc": auc,
        "accuracy": float(acc),
        "balanced_accuracy": float(bal),
        "far": float(far),
        "frr": float(frr),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def find_eer_threshold(
    labels,
    scores,
):
    thresholds = np.unique(
        scores
    )

    best = None

    for threshold in thresholds:

        m = binary_metrics(
            labels,
            scores,
            threshold,
            compute_auc=False,
        )

        gap = abs(
            m["far"]
            - m["frr"]
        )

        eer = (
            m["far"]
            + m["frr"]
        ) / 2.0

        candidate = (
            gap,
            eer,
            float(threshold),
            m,
        )

        if (
            best is None
            or candidate[:2]
            < best[:2]
        ):
            best = candidate

    _, eer, threshold, _ = best
    metrics = binary_metrics(
        labels,
        scores,
        threshold,
        compute_auc=True,
    )

    return (
        threshold,
        eer,
        metrics,
    )


def build_known_calibration_scores(
    embeddings,
    indices,
    meta,
    enrollment_per_gesture,
):
    """
    Cross-session calibration.

    second-latest session:
        enrollment

    latest session:
        genuine verification

    Same gesture only.
    """

    performer = meta[
        "performer"
    ]

    gesture = meta[
        "gesture"
    ]

    session = meta[
        "session"
    ]

    index_to_pos = {
        int(idx): pos
        for pos, idx
        in enumerate(indices)
    }

    labels = []
    scores = []

    gestures = sorted(
        np.unique(
            gesture[indices]
        )
    )

    for target_user in TRAIN_USERS:

        user_idx = indices[
            performer[indices]
            == target_user
        ]

        sessions = sorted(
            np.unique(
                session[user_idx]
            )
        )

        if len(sessions) < 2:
            continue

        enroll_session = (
            sessions[-2]
        )

        test_session = (
            sessions[-1]
        )

        for g in gestures:

            enroll_idx = indices[
                (
                    performer[indices]
                    == target_user
                )
                & (
                    gesture[indices]
                    == g
                )
                & (
                    session[indices]
                    == enroll_session
                )
            ]

            genuine_idx = indices[
                (
                    performer[indices]
                    == target_user
                )
                & (
                    gesture[indices]
                    == g
                )
                & (
                    session[indices]
                    == test_session
                )
            ]

            if (
                len(enroll_idx)
                < enrollment_per_gesture
            ):
                continue

            if len(
                genuine_idx
            ) == 0:
                continue

            enroll_idx = (
                enroll_idx[
                    :enrollment_per_gesture
                ]
            )

            template = np.stack(
                [
                    embeddings[
                        index_to_pos[
                            int(i)
                        ]
                    ]
                    for i
                    in enroll_idx
                ]
            ).mean(
                axis=0
            )

            genuine_emb = np.stack(
                [
                    embeddings[
                        index_to_pos[
                            int(i)
                        ]
                    ]
                    for i
                    in genuine_idx
                ]
            )

            genuine_scores = (
                cosine_scores(
                    genuine_emb,
                    template,
                )
            )

            labels.extend(
                [1]
                * len(
                    genuine_scores
                )
            )

            scores.extend(
                genuine_scores.tolist()
            )

            impostor_idx = []

            for other_user in TRAIN_USERS:

                if (
                    other_user
                    == target_user
                ):
                    continue

                other_idx = indices[
                    performer[indices]
                    == other_user
                ]

                other_sessions = sorted(
                    np.unique(
                        session[
                            other_idx
                        ]
                    )
                )

                if not other_sessions:
                    continue

                other_test = (
                    other_sessions[-1]
                )

                imp = indices[
                    (
                        performer[indices]
                        == other_user
                    )
                    & (
                        gesture[indices]
                        == g
                    )
                    & (
                        session[indices]
                        == other_test
                    )
                ]

                impostor_idx.extend(
                    imp.tolist()
                )

            if impostor_idx:

                impostor_idx = (
                    np.asarray(
                        impostor_idx,
                        dtype=np.int64,
                    )
                )

                impostor_emb = np.stack(
                    [
                        embeddings[
                            index_to_pos[
                                int(i)
                            ]
                        ]
                        for i
                        in impostor_idx
                    ]
                )

                imp_scores = cosine_scores(
                    impostor_emb,
                    template,
                )

                labels.extend(
                    [0]
                    * len(
                        imp_scores
                    )
                )

                scores.extend(
                    imp_scores.tolist()
                )

    return (
        np.asarray(
            labels,
            dtype=np.int64,
        ),
        np.asarray(
            scores,
            dtype=np.float64,
        ),
    )


def evaluate_unseen(
    model,
    X,
    duration,
    meta,
    device,
    threshold,
    enrollment_per_gesture,
):
    performer = meta[
        "performer"
    ]

    gesture = meta[
        "gesture"
    ]

    session = meta[
        "session"
    ]

    gestures = sorted(
        np.unique(
            gesture
        )
    )

    rows = []

    global_labels = []
    global_scores = []

    for target_user in UNSEEN_USERS:

        user_idx = np.where(
            performer
            == target_user
        )[0]

        sessions = sorted(
            np.unique(
                session[user_idx]
            )
        )

        enroll_session = (
            sessions[0]
        )

        print()
        print(
            f"[UNSEEN] "
            f"{target_user} "
            f"enrollment_session="
            f"{enroll_session}"
        )

        for g in gestures:

            enroll_idx = np.where(
                (
                    performer
                    == target_user
                )
                & (
                    gesture == g
                )
                & (
                    session
                    == enroll_session
                )
            )[0]

            if (
                len(enroll_idx)
                < enrollment_per_gesture
            ):
                print(
                    f"  {g}: SKIP"
                )
                continue

            enroll_idx = (
                enroll_idx[
                    :enrollment_per_gesture
                ]
            )

            genuine_idx = np.where(
                (
                    performer
                    == target_user
                )
                & (
                    gesture == g
                )
                & (
                    session
                    != enroll_session
                )
            )[0]

            impostor_idx = np.where(
                np.isin(
                    performer,
                    UNSEEN_USERS,
                )
                & (
                    performer
                    != target_user
                )
                & (
                    gesture == g
                )
            )[0]

            enroll_emb = (
                extract_embeddings(
                    model,
                    X[enroll_idx],
                    duration[
                        enroll_idx
                    ],
                    device,
                )
            )

            template = (
                enroll_emb.mean(
                    axis=0
                )
            )

            genuine_emb = (
                extract_embeddings(
                    model,
                    X[genuine_idx],
                    duration[
                        genuine_idx
                    ],
                    device,
                )
            )

            impostor_emb = (
                extract_embeddings(
                    model,
                    X[impostor_idx],
                    duration[
                        impostor_idx
                    ],
                    device,
                )
            )

            genuine_scores = (
                cosine_scores(
                    genuine_emb,
                    template,
                )
            )

            impostor_scores = (
                cosine_scores(
                    impostor_emb,
                    template,
                )
            )

            labels = np.concatenate(
                [
                    np.ones(
                        len(
                            genuine_scores
                        ),
                        dtype=np.int64,
                    ),
                    np.zeros(
                        len(
                            impostor_scores
                        ),
                        dtype=np.int64,
                    ),
                ]
            )

            scores = np.concatenate(
                [
                    genuine_scores,
                    impostor_scores,
                ]
            )

            metrics = binary_metrics(
                labels,
                scores,
                threshold,
            )

            local_threshold, local_eer, _ = (
                find_eer_threshold(
                    labels,
                    scores,
                )
            )

            rows.append(
                {
                    "user": target_user,
                    "gesture": g,
                    "enrollment_session": enroll_session,
                    "enrollment_samples": len(
                        enroll_idx
                    ),
                    "genuine_samples": len(
                        genuine_idx
                    ),
                    "impostor_samples": len(
                        impostor_idx
                    ),
                    "threshold": threshold,
                    "accuracy": metrics[
                        "accuracy"
                    ],
                    "balanced_accuracy": metrics[
                        "balanced_accuracy"
                    ],
                    "far": metrics[
                        "far"
                    ],
                    "frr": metrics[
                        "frr"
                    ],
                    "test_eer_analysis": local_eer,
                    "test_eer_threshold_analysis": local_threshold,
                }
            )

            global_labels.extend(
                labels.tolist()
            )

            global_scores.extend(
                scores.tolist()
            )

            print(
                f"  {g}: "
                f"Genuine={len(genuine_idx):3d} "
                f"Impostor={len(impostor_idx):3d} | "
                f"BalAcc="
                f"{metrics['balanced_accuracy']:.3f} "
                f"FAR="
                f"{metrics['far']:.3f} "
                f"FRR="
                f"{metrics['frr']:.3f}"
            )

    global_labels = np.asarray(
        global_labels,
        dtype=np.int64,
    )

    global_scores = np.asarray(
        global_scores,
        dtype=np.float64,
    )

    metrics = binary_metrics(
        global_labels,
        global_scores,
        threshold,
    )

    test_threshold, test_eer, _ = (
        find_eer_threshold(
            global_labels,
            global_scores,
        )
    )

    return (
        rows,
        metrics,
        test_eer,
        test_threshold,
    )


def main():

    parser = argparse.ArgumentParser()

    witech_root = os.environ.get("WITECH_ROOT")
    default_data = (
        Path(witech_root) / "derived" / "dataset.npz"
        if witech_root
        else Path("dataset.npz")
    )

    parser.add_argument(
        "--data",
        default=str(
            default_data
        ),
    )

    parser.add_argument(
        "--architecture",
        choices=(
            "dilated",
            "lite-stats",
            "phase-pyramid",
        ),
        default="dilated",
        help=(
            "Encoder architecture. 'dilated' preserves the previous model; "
            "'lite-stats' is the lightweight mean/std baseline; "
            "'phase-pyramid' keeps coarse temporal phase information."
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory for checkpoint, CSV, and summary.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=40,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--pairs",
        type=int,
        default=6000,
    )

    parser.add_argument(
        "--margin",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--enroll",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=7,
    )

    args = parser.parse_args()

    seed_everything()

    device = select_device()

    architecture_titles = {
        "dilated": "Dilated Siamese 1D-CNN",
        "lite-stats": "Lite-Stats Siamese 1D-CNN",
        "phase-pyramid": "Phase-Pyramid Siamese 1D-CNN",
    }
    architecture_title = architecture_titles[
        args.architecture
    ]

    print("=" * 70)
    print(
        f"{architecture_title} / Contrastive "
        "Unseen-user Authentication"
    )
    print("=" * 70)

    print(
        "device       :",
        device,
    )

    print(
        "dataset      :",
        args.data,
    )

    print(
        "architecture :",
        args.architecture,
    )

    print(
        "train users  :",
        TRAIN_USERS,
    )

    print(
        "unseen users :",
        UNSEEN_USERS,
    )

    print(
        "embedding dim:",
        args.embedding_dim,
    )

    print(
        "pairs/epoch  :",
        args.pairs,
    )

    print(
        "margin       :",
        args.margin,
    )

    print(
        "enroll / G   :",
        args.enroll,
    )

    print()

    (
        _,
        X_seq,
        duration,
        meta,
        T,
        D,
    ) = load_dataset(
        args.data
    )

    print(
        f"Dataset: "
        f"N={len(X_seq)}, "
        f"T={T}, "
        f"D={D}"
    )

    train_idx, val_idx = (
        chronological_split(
            meta,
            TRAIN_USERS,
        )
    )

    print()

    print(
        f"Known-user split: "
        f"train={len(train_idx)}, "
        f"val={len(val_idx)}"
    )

    # ---------------------------------------------------------
    # Normalization fitted only using known training data
    # ---------------------------------------------------------

    seq_mean, seq_std = (
        fit_sequence_stats(
            X_seq[train_idx]
        )
    )

    dur_mean, dur_std = (
        fit_duration_stats(
            duration[train_idx]
        )
    )

    X_norm = apply_sequence_stats(
        X_seq,
        seq_mean,
        seq_std,
    )

    # 마지막 채널은 hand-valid mask다. 모델이 masked pooling에 사용할 수 있도록
    # z-score 값이 아니라 원래의 0/1 값을 유지한다.
    X_norm[:, :, -1] = X_seq[:, :, -1]
    seq_mean[:, :, -1] = 0.0
    seq_std[:, :, -1] = 1.0

    duration_norm = (
        apply_duration_stats(
            duration,
            dur_mean,
            dur_std,
        )
    )

    model_classes = {
        "dilated": Siamese1DCNN,
        "lite-stats": LiteStatsSiamese1DCNN,
        "phase-pyramid": PhasePyramidSiamese1DCNN,
    }

    model = model_classes[args.architecture](
        input_dim=D,
        embedding_dim=args.embedding_dim,
    ).to(
        device
    )

    n_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Trainable parameters: "
        f"{n_params:,}"
    )

    # CosineEmbeddingLoss:
    #
    # target = +1 -> make embeddings similar
    # target = -1 -> make embeddings dissimilar
    criterion = (
        nn.CosineEmbeddingLoss(
            margin=args.margin
        )
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )

    best_val_loss = float(
        "inf"
    )

    best_state = None
    patience_count = 0

    rng = np.random.default_rng(
        SEED
    )

    print()
    print(
        f"Training {architecture_title} encoder..."
    )

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        # New random training pairs every epoch
        (
            train_a,
            train_b,
            train_target,
        ) = build_pairs(
            train_idx,
            meta,
            args.pairs,
            rng,
        )

        (
            val_a,
            val_b,
            val_target,
        ) = build_pairs(
            val_idx,
            meta,
            min(
                2000,
                args.pairs // 2,
            ),
            rng,
        )

        train_dataset = PairDataset(
            X_norm,
            duration_norm,
            train_a,
            train_b,
            train_target,
        )

        val_dataset = PairDataset(
            X_norm,
            duration_norm,
            val_a,
            val_b,
            val_target,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        )

        train_loss = run_pair_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
        )

        val_loss = run_pair_epoch(
            model,
            val_loader,
            criterion,
            device,
            optimizer=None,
        )

        print(
            f"Epoch {epoch:02d} | "
            f"train loss="
            f"{train_loss:.4f} | "
            f"val loss="
            f"{val_loss:.4f}"
        )

        if (
            val_loss
            < best_val_loss
        ):
            best_val_loss = val_loss

            best_state = {
                k: v.detach()
                .cpu()
                .clone()
                for k, v
                in model.state_dict()
                .items()
            }

            patience_count = 0

        else:
            patience_count += 1

        if (
            patience_count
            >= args.patience
        ):
            print(
                f"Early stopping "
                f"at epoch {epoch}"
            )
            break

    if best_state is None:
        raise RuntimeError(
            "No best model state"
        )

    model.load_state_dict(
        best_state
    )

    model.to(
        device
    )

    print()

    print(
        f"Best validation pair loss: "
        f"{best_val_loss:.4f}"
    )

    # ---------------------------------------------------------
    # Threshold calibration using known users only
    # ---------------------------------------------------------

    print()

    print(
        "Calibrating threshold "
        "using known users..."
    )

    known_idx = np.where(
        np.isin(
            meta["performer"],
            TRAIN_USERS,
        )
    )[0]

    known_embeddings = (
        extract_embeddings(
            model,
            X_norm[
                known_idx
            ],
            duration_norm[
                known_idx
            ],
            device,
        )
    )

    cal_labels, cal_scores = (
        build_known_calibration_scores(
            known_embeddings,
            known_idx,
            meta,
            args.enroll,
        )
    )

    (
        threshold,
        val_eer,
        val_metrics,
    ) = find_eer_threshold(
        cal_labels,
        cal_scores,
    )

    print(
        f"Validation threshold : "
        f"{threshold:.6f}"
    )

    print(
        f"Validation EER       : "
        f"{val_eer:.3f}"
    )

    print(
        f"Validation FAR       : "
        f"{val_metrics['far']:.3f}"
    )

    print(
        f"Validation FRR       : "
        f"{val_metrics['frr']:.3f}"
    )

    # ---------------------------------------------------------
    # Unseen user evaluation
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "UNSEEN USER EVALUATION"
    )
    print("=" * 70)

    (
        rows,
        global_metrics,
        test_eer,
        test_eer_threshold,
    ) = evaluate_unseen(
        model,
        X_norm,
        duration_norm,
        meta,
        device,
        threshold,
        args.enroll,
    )

    print()
    print("=" * 70)
    print(
        "FINAL SIAMESE "
        "UNSEEN-USER RESULT"
    )
    print("=" * 70)

    print(
        f"Fixed threshold     : "
        f"{threshold:.6f}"
    )

    print(
        f"AUC                 : "
        f"{global_metrics['auc']:.3f}"
    )

    print(
        f"Accuracy            : "
        f"{global_metrics['accuracy']:.3f}"
    )

    print(
        f"Balanced Accuracy   : "
        f"{global_metrics['balanced_accuracy']:.3f}"
    )

    print(
        f"FAR                 : "
        f"{global_metrics['far']:.3f}"
    )

    print(
        f"FRR                 : "
        f"{global_metrics['frr']:.3f}"
    )

    print(
        f"Test EER (analysis) : "
        f"{test_eer:.3f}"
    )

    print(
        f"Test EER threshold  : "
        f"{test_eer_threshold:.6f}"
    )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
    elif witech_root:
        output_dir = (
            Path(witech_root)
            / "derived"
            / f"siamese_{args.architecture.replace('-', '_')}_auth"
        )
    else:
        output_dir = (
            Path(args.data).expanduser().resolve().parent
            / f"siamese_{args.architecture.replace('-', '_')}_auth"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        output_dir
        / f"siamese_{args.architecture.replace('-', '_')}_1dcnn.pt"
    )

    torch.save(
        {
            "model_state_dict":
                {
                    k: v.detach()
                    .cpu()
                    for k, v
                    in model.state_dict()
                    .items()
                },
            "input_dim": D,
            "embedding_dim":
                args.embedding_dim,
            "architecture":
                args.architecture,
            "encoder":
                {
                    "dilated": "residual_dilated_1dcnn",
                    "lite-stats": "lite_statistics_1dcnn",
                    "phase-pyramid": "phase_pyramid_1dcnn",
                }[args.architecture],
            "pooling":
                {
                    "dilated": "masked_mean_std_max",
                    "lite-stats": "global_mean_std_plus_duration",
                    "phase-pyramid": "temporal_pyramid_mean_std_1_2_4",
                }[args.architecture],
            "dilations":
                (
                    (1, 2, 4, 8)
                    if args.architecture == "dilated"
                    else None
                ),
            "phase_levels":
                (
                    (1, 2, 4)
                    if args.architecture == "phase-pyramid"
                    else None
                ),
            "temporal_kernels":
                {
                    "dilated": (3,),
                    "lite-stats": (5, 3, 3),
                    "phase-pyramid": (3, 5, 7),
                }[args.architecture],
            "valid_mask_index":
                D - 1,
            "uses_valid_mask":
                args.architecture == "dilated",
            "uses_duration":
                args.architecture == "lite-stats",
            "trainable_parameters":
                n_params,
            "train_users":
                TRAIN_USERS,
            "unseen_users":
                UNSEEN_USERS,
            "threshold":
                threshold,
            "margin":
                args.margin,
            "sequence_mean":
                seq_mean,
            "sequence_std":
                seq_std,
            "duration_mean":
                dur_mean,
            "duration_std":
                dur_std,
            "best_val_pair_loss":
                best_val_loss,
        },
        model_path,
    )

    csv_path = (
        output_dir
        / "unseen_user_results.csv"
    )

    if rows:
        with open(
            csv_path,
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=list(
                    rows[0].keys()
                ),
            )

            writer.writeheader()
            writer.writerows(
                rows
            )

    summary_path = (
        output_dir
        / "summary.txt"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            f"{architecture_title} embedding "
            "unseen-user authentication\n"
        )

        f.write(
            f"architecture="
            f"{args.architecture}\n"
        )

        f.write(
            f"train_users="
            f"{TRAIN_USERS}\n"
        )

        f.write(
            f"unseen_users="
            f"{UNSEEN_USERS}\n"
        )

        f.write(
            f"margin="
            f"{args.margin}\n"
        )

        f.write(
            f"enrollment_per_gesture="
            f"{args.enroll}\n"
        )

        f.write(
            f"validation_threshold="
            f"{threshold:.6f}\n"
        )

        f.write(
            f"validation_eer="
            f"{val_eer:.6f}\n"
        )

        f.write(
            f"unseen_auc="
            f"{global_metrics['auc']:.6f}\n"
        )

        f.write(
            f"unseen_accuracy="
            f"{global_metrics['accuracy']:.6f}\n"
        )

        f.write(
            f"unseen_balanced_accuracy="
            f"{global_metrics['balanced_accuracy']:.6f}\n"
        )

        f.write(
            f"unseen_far="
            f"{global_metrics['far']:.6f}\n"
        )

        f.write(
            f"unseen_frr="
            f"{global_metrics['frr']:.6f}\n"
        )

        f.write(
            f"unseen_test_eer_analysis="
            f"{test_eer:.6f}\n"
        )

    print()
    print("Saved:")
    print(
        " model  :",
        model_path,
    )
    print(
        " csv    :",
        csv_path,
    )
    print(
        " summary:",
        summary_path,
    )


if __name__ == "__main__":
    main()
