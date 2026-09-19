"""Shared-backbone dual-head metric-learning trainer for WITECK hand-only D=127.

Purpose
-------
- Reuse the existing hand-only two-stream 1D-CNN feature extractor.
- Learn TWO normalized embeddings from one shared backbone:
    1) gesture_embedding: "is this the same registered gesture?"
    2) user_embedding:    "is this the same user?"
- Keep the existing closed-set user classifier only as an auxiliary training loss.
  It is NOT required at enrollment/inference time.

Important label rule for future personal-gesture data
-----------------------------------------------------
Shared gestures such as the legacy G1~G5 may keep global labels G1~G5.
A personal/free gesture must have a globally unique gesture id in metadata,
for example D01_PG1, D01_PG2, N01_PG1, etc. Do NOT label different
people's unrelated personal gestures all as just "PG1", because the gesture
head would incorrectly treat them as the same gesture.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


legacy = importlib.import_module("16_train_supcon_embedding")
base = importlib.import_module("08_train_embedding")

DEFAULT_TRAIN_USERS = list(legacy.TRAIN_USERS)
DEFAULT_UNSEEN_USERS = list(legacy.UNSEEN_USERS)


def parse_id_list(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class SharedDualHead1DCNN(nn.Module):
    """Hand-only shared two-stream backbone + gesture/user embedding heads."""

    def __init__(
        self,
        input_dim: int,
        num_user_classes: int,
        embedding_dim: int = 128,
    ):
        super().__init__()

        if int(input_dim) != 127:
            raise ValueError(
                f"SharedDualHead1DCNN expects hand-only D=127, got {input_dim}"
            )

        def branch(dropout: float) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv1d(63, 48, kernel_size=5, padding=2, bias=False),
                nn.BatchNorm1d(48),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv1d(
                    48,
                    64,
                    kernel_size=3,
                    padding=2,
                    dilation=2,
                    bias=False,
                ),
                nn.BatchNorm1d(64),
                nn.ReLU(),
            )

        self.position_branch = branch(0.15)
        self.velocity_branch = branch(0.20)

        # Existing model used:
        #   [position mean+std | velocity mean+std | duration]
        # = 64*4 + 1 = 257 dims.
        self.shared_projection = nn.Sequential(
            nn.Linear(64 * 4 + 1, 192),
            nn.LayerNorm(192),
            nn.ReLU(),
            nn.Dropout(0.20),
        )

        self.gesture_head = nn.Linear(192, embedding_dim)
        self.user_head = nn.Linear(192, embedding_dim)

        # Auxiliary CLOSED-SET classifier used only to stabilize training.
        # It is not used for new-user enrollment or final authentication.
        self.user_classifier = nn.Linear(embedding_dim, num_user_classes)

    @staticmethod
    def statistics_pool(features: torch.Tensor) -> torch.Tensor:
        mean = features.mean(dim=2)
        variance = features.var(dim=2, unbiased=False)
        std = torch.sqrt(variance.clamp_min(1e-6))
        return torch.cat([mean, std], dim=1)

    def forward(
        self,
        x: torch.Tensor,
        duration: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.ndim != 3 or x.shape[-1] != 127:
            raise ValueError(f"Expected [B,32,127], got {tuple(x.shape)}")

        position = x[:, :, :63].transpose(1, 2)
        velocity = x[:, :, 63:126].transpose(1, 2)

        position_stats = self.statistics_pool(
            self.position_branch(position)
        )
        velocity_stats = self.statistics_pool(
            self.velocity_branch(velocity)
        )

        fused = torch.cat(
            [
                position_stats,
                velocity_stats,
                duration.unsqueeze(1),
            ],
            dim=1,
        )

        shared = self.shared_projection(fused)

        gesture_embedding = F.normalize(
            self.gesture_head(shared),
            p=2,
            dim=1,
        )
        user_embedding = F.normalize(
            self.user_head(shared),
            p=2,
            dim=1,
        )
        user_logits = self.user_classifier(user_embedding)

        return gesture_embedding, user_embedding, user_logits


def supervised_contrastive_loss(
    embedding: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Standard supervised contrastive loss over all non-self batch pairs."""

    n = embedding.shape[0]
    if n < 2:
        return embedding.sum() * 0.0

    similarity = embedding @ embedding.T / temperature

    eye = torch.eye(
        n,
        dtype=torch.bool,
        device=embedding.device,
    )
    eligible = ~eye
    positive = labels[:, None].eq(labels[None, :]) & eligible

    positive_count = positive.sum(dim=1)
    valid_anchor = positive_count > 0

    if not bool(valid_anchor.any()):
        return embedding.sum() * 0.0

    masked_similarity = similarity.masked_fill(
        ~eligible,
        float("-inf"),
    )
    log_denominator = torch.logsumexp(
        masked_similarity,
        dim=1,
    )

    positive_sum = (
        similarity.masked_fill(~positive, 0.0).sum(dim=1)
    )
    positive_mean = positive_sum / positive_count.clamp_min(1)

    loss = log_denominator - positive_mean
    return loss[valid_anchor].mean()


def run_epoch(
    model,
    loader,
    ce_criterion,
    device,
    user_contrastive_weight,
    gesture_contrastive_weight,
    temperature,
    optimizer=None,
):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_ce = 0.0
    total_user_supcon = 0.0
    total_gesture_supcon = 0.0
    total_correct = 0
    total_n = 0

    context = torch.enable_grad() if training else torch.no_grad()

    with context:
        for xb, db, user_y, gesture_y in loader:
            xb = xb.to(device)
            db = db.to(device)
            user_y = user_y.to(device)
            gesture_y = gesture_y.to(device)

            if training:
                optimizer.zero_grad(set_to_none=True)

            gesture_emb, user_emb, user_logits = model(xb, db)

            ce_loss = ce_criterion(user_logits, user_y)

            user_supcon = supervised_contrastive_loss(
                user_emb,
                user_y,
                temperature,
            )
            gesture_supcon = supervised_contrastive_loss(
                gesture_emb,
                gesture_y,
                temperature,
            )

            loss = (
            ce_loss
            + user_contrastive_weight * user_supcon
            + gesture_contrastive_weight * gesture_supcon
            )

            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    5.0,
                )
                optimizer.step()

            n = len(user_y)
            total_loss += float(loss.item()) * n
            total_ce += float(ce_loss.item()) * n
            total_user_supcon += float(user_supcon.item()) * n
            total_gesture_supcon += float(gesture_supcon.item()) * n
            total_correct += int(
                (user_logits.argmax(dim=1) == user_y).sum().item()
            )
            total_n += n

    denom = max(total_n, 1)

    return (
        total_loss / denom,
        total_ce / denom,
        total_user_supcon / denom,
        total_gesture_supcon / denom,
        total_correct / denom,
    )


@torch.no_grad()
def extract_dual_embeddings(
    model,
    X,
    duration,
    device,
    batch_size=128,
):
    model.eval()

    dummy_user = np.zeros(len(X), dtype=np.int64)
    dummy_gesture = np.zeros(len(X), dtype=np.int64)

    loader = legacy.make_loader(
        X,
        duration,
        dummy_user,
        dummy_gesture,
        batch_size,
        shuffle=False,
    )

    gesture_result = []
    user_result = []

    for xb, db, _, _ in loader:
        xb = xb.to(device)
        db = db.to(device)

        gesture_emb, user_emb, _ = model(xb, db)

        gesture_result.append(
            gesture_emb.detach().cpu().numpy()
        )
        user_result.append(
            user_emb.detach().cpu().numpy()
        )

    return (
        np.concatenate(gesture_result, axis=0),
        np.concatenate(user_result, axis=0),
    )


def build_user_validation_scores(
    embeddings,
    indices,
    meta,
    train_users,
    enrollment_per_gesture=3,
):
    """Calibrate Tu.

    Positive:
      same user + same gesture across sessions

    Negative:
      different user + same gesture in latest available session

    This mirrors the old authentication calibration, but does not rely on
    hard-coded TRAIN_USERS inside 08_train_embedding.py.
    """

    performer = meta["performer"]
    gesture = meta["gesture"]
    session = meta["session"]

    labels = []
    scores = []

    index_to_pos = {
        int(idx): pos
        for pos, idx in enumerate(indices)
    }

    gestures = sorted(np.unique(gesture[indices]))

    for target_user in train_users:
        user_idx = indices[
            performer[indices] == target_user
        ]
        user_sessions = sorted(
            np.unique(session[user_idx])
        )

        if len(user_sessions) < 2:
            continue

        enroll_session = user_sessions[-2]
        test_session = user_sessions[-1]

        for g in gestures:
            enroll_candidates = indices[
                (performer[indices] == target_user)
                & (gesture[indices] == g)
                & (session[indices] == enroll_session)
            ]
            genuine = indices[
                (performer[indices] == target_user)
                & (gesture[indices] == g)
                & (session[indices] == test_session)
            ]

            if len(enroll_candidates) < enrollment_per_gesture:
                continue
            if len(genuine) == 0:
                continue

            enroll = enroll_candidates[:enrollment_per_gesture]

            template_embeddings = np.stack(
                [
                    embeddings[index_to_pos[int(i)]]
                    for i in enroll
                ]
            )
            template = template_embeddings.mean(axis=0)

            genuine_embeddings = np.stack(
                [
                    embeddings[index_to_pos[int(i)]]
                    for i in genuine
                ]
            )
            genuine_scores = base.cosine_scores(
                genuine_embeddings,
                template,
            )

            labels.extend([1] * len(genuine_scores))
            scores.extend(genuine_scores.tolist())

            impostor = []

            for other_user in train_users:
                if other_user == target_user:
                    continue

                other_idx = indices[
                    performer[indices] == other_user
                ]
                other_sessions = sorted(
                    np.unique(session[other_idx])
                )

                if not other_sessions:
                    continue

                other_test_session = other_sessions[-1]

                imp = indices[
                    (performer[indices] == other_user)
                    & (gesture[indices] == g)
                    & (session[indices] == other_test_session)
                ]
                impostor.extend(imp.tolist())

            if impostor:
                impostor_embeddings = np.stack(
                    [
                        embeddings[index_to_pos[int(i)]]
                        for i in impostor
                    ]
                )
                impostor_scores = base.cosine_scores(
                    impostor_embeddings,
                    template,
                )

                labels.extend(
                    [0] * len(impostor_scores)
                )
                scores.extend(
                    impostor_scores.tolist()
                )

    return (
        np.asarray(labels, dtype=np.int64),
        np.asarray(scores, dtype=np.float64),
    )


def build_gesture_validation_scores(
    embeddings,
    indices,
    meta,
    train_users,
    enrollment_per_gesture=3,
):
    """Calibrate Tg while holding identity fixed.

    Positive:
      same user + same gesture across sessions

    Negative:
      same user + DIFFERENT gesture in the latest session

    This directly teaches/calibrates the failure case found in the
    Direct-Template experiment: correct user performing the wrong gesture.
    """

    performer = meta["performer"]
    gesture = meta["gesture"]
    session = meta["session"]

    labels = []
    scores = []

    index_to_pos = {
        int(idx): pos
        for pos, idx in enumerate(indices)
    }

    for target_user in train_users:
        user_idx = indices[
            performer[indices] == target_user
        ]
        user_sessions = sorted(
            np.unique(session[user_idx])
        )

        if len(user_sessions) < 2:
            continue

        enroll_session = user_sessions[-2]
        test_session = user_sessions[-1]

        user_gestures = sorted(
            np.unique(gesture[user_idx])
        )

        for g in user_gestures:
            enroll_candidates = indices[
                (performer[indices] == target_user)
                & (gesture[indices] == g)
                & (session[indices] == enroll_session)
            ]
            genuine = indices[
                (performer[indices] == target_user)
                & (gesture[indices] == g)
                & (session[indices] == test_session)
            ]

            if len(enroll_candidates) < enrollment_per_gesture:
                continue
            if len(genuine) == 0:
                continue

            enroll = enroll_candidates[:enrollment_per_gesture]

            template_embeddings = np.stack(
                [
                    embeddings[index_to_pos[int(i)]]
                    for i in enroll
                ]
            )
            template = template_embeddings.mean(axis=0)

            genuine_embeddings = np.stack(
                [
                    embeddings[index_to_pos[int(i)]]
                    for i in genuine
                ]
            )
            genuine_scores = base.cosine_scores(
                genuine_embeddings,
                template,
            )

            labels.extend([1] * len(genuine_scores))
            scores.extend(genuine_scores.tolist())

            wrong_gesture = indices[
                (performer[indices] == target_user)
                & (gesture[indices] != g)
                & (session[indices] == test_session)
            ]

            if len(wrong_gesture):
                wrong_embeddings = np.stack(
                    [
                        embeddings[index_to_pos[int(i)]]
                        for i in wrong_gesture
                    ]
                )
                wrong_scores = base.cosine_scores(
                    wrong_embeddings,
                    template,
                )

                labels.extend([0] * len(wrong_scores))
                scores.extend(wrong_scores.tolist())

    return (
        np.asarray(labels, dtype=np.int64),
        np.asarray(scores, dtype=np.float64),
    )


def main() -> None:
    project_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        default=str(
            project_dir
            / "dataset"
            / "dataset_1955_recent8_updated_20260905_hand_only.npz"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(
            project_dir
            / "output"
            / "shared_dual_head"
        ),
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--user-contrastive-weight", type=float, default=0.20)
    parser.add_argument("--gesture-contrastive-weight", type=float, default=0.20)
    parser.add_argument("--enroll", type=int, default=3)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples-per-group", type=int, default=2)
    parser.add_argument(
        "--train-users",
        default=",".join(DEFAULT_TRAIN_USERS),
    )
    parser.add_argument(
        "--unseen-users",
        default=",".join(DEFAULT_UNSEEN_USERS),
    )

    args = parser.parse_args()

    train_users = parse_id_list(args.train_users)
    unseen_users = parse_id_list(args.unseen_users)

    base.seed_everything(args.seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    _, X_seq, duration, meta, T, D = legacy.load_dataset(args.data)

    train_idx, val_idx = base.chronological_user_split(
        meta,
        train_users,
    )

    seq_mean, seq_std = legacy.fit_sequence_stats(
        X_seq[train_idx]
    )
    dur_mean, dur_std = legacy.fit_duration_stats(
        duration[train_idx]
    )

    X_norm = legacy.apply_sequence_stats(
        X_seq,
        seq_mean,
        seq_std,
    )
    duration_norm = legacy.apply_duration_stats(
        duration,
        dur_mean,
        dur_std,
    )

    user_to_label = {
        user: i
        for i, user in enumerate(train_users)
    }

    # IMPORTANT:
    # For future FREE personal gestures, metadata gesture ids must be globally
    # unique, e.g. D01_PG1, D01_PG2, N01_PG1.
    gestures = sorted(
        np.unique(meta["gesture"]).tolist()
    )
    gesture_to_label = {
        gesture_id: i
        for i, gesture_id in enumerate(gestures)
    }

    user_y = np.full(
        len(X_seq),
        -1,
        dtype=np.int64,
    )
    for user, label in user_to_label.items():
        user_y[meta["performer"] == user] = label

    gesture_y = np.asarray(
        [
            gesture_to_label[x]
            for x in meta["gesture"]
        ],
        dtype=np.int64,
    )

    # Balanced user-gesture groups are especially useful for the two SupCon
    # losses because they make positive pairs much more likely to exist.
    train_loader = legacy.make_balanced_loader(
        X_norm[train_idx],
        duration_norm[train_idx],
        user_y[train_idx],
        gesture_y[train_idx],
        args.samples_per_group,
        args.seed,
    )

    val_loader = legacy.make_loader(
        X_norm[val_idx],
        duration_norm[val_idx],
        user_y[val_idx],
        gesture_y[val_idx],
        batch_size=128,
        shuffle=False,
    )

    model = SharedDualHead1DCNN(
        D,
        len(train_users),
        args.embedding_dim,
    ).to(device)

    ce_criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )

    print("=" * 78)
    print("WITECK Shared Backbone + Dual Embedding Head")
    print("=" * 78)
    print(f"device={device} N={len(X_seq)} T={T} D={D}")
    print(f"train={len(train_idx)} val={len(val_idx)}")
    print(f"train_users={train_users}")
    print(f"unseen_users={unseen_users}")
    print(
        "loss = CE(user) "
        f"+ {args.user_contrastive_weight}*UserSupCon "
        f"+ {args.gesture_contrastive_weight}*GestureSupCon"
    )
    print(
        "gesture labels are metadata gesture ids; "
        "personal/free gesture ids must be globally unique"
    )
    print(
        f"balanced user-gesture batches: "
        f"{args.samples_per_group} samples/group"
    )

    best_val_loss = float("inf")
    best_val_acc = -1.0
    best_epoch = 0
    best_state = None
    stale = 0
    stopped_epoch = args.epochs

    for epoch in range(1, args.epochs + 1):
        (
            train_loss,
            train_ce,
            train_user_supcon,
            train_gesture_supcon,
            train_acc,
        ) = run_epoch(
            model,
            train_loader,
            ce_criterion,
            device,
            args.user_contrastive_weight,
            args.gesture_contrastive_weight,
            args.temperature,
            optimizer,
        )

        (
            val_loss,
            val_ce,
            val_user_supcon,
            val_gesture_supcon,
            val_acc,
        ) = run_epoch(
            model,
            val_loader,
            ce_criterion,
            device,
            args.user_contrastive_weight,
            args.gesture_contrastive_weight,
            args.temperature,
            optimizer=None,
        )

        print(
            f"Epoch {epoch:02d} | "
            f"train total={train_loss:.4f} "
            f"CE={train_ce:.4f} "
            f"UserSupCon={train_user_supcon:.4f} "
            f"GestureSupCon={train_gesture_supcon:.4f} "
            f"acc={train_acc:.3f} | "
            f"val total={val_loss:.4f} "
            f"CE={val_ce:.4f} "
            f"UserSupCon={val_user_supcon:.4f} "
            f"GestureSupCon={val_gesture_supcon:.4f} "
            f"acc={val_acc:.3f}"
        )

        # Select by total validation loss so BOTH heads influence checkpointing.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
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
        raise RuntimeError("No dual-head checkpoint was created")

    model.load_state_dict(best_state)
    model.eval()

    known_idx = np.where(
        np.isin(meta["performer"], train_users)
    )[0]

    gesture_embeddings, user_embeddings = extract_dual_embeddings(
        model,
        X_norm[known_idx],
        duration_norm[known_idx],
        device,
    )

    user_cal_labels, user_cal_scores = build_user_validation_scores(
        user_embeddings,
        known_idx,
        meta,
        train_users,
        enrollment_per_gesture=args.enroll,
    )
    gesture_cal_labels, gesture_cal_scores = build_gesture_validation_scores(
        gesture_embeddings,
        known_idx,
        meta,
        train_users,
        enrollment_per_gesture=args.enroll,
    )

    if len(user_cal_labels) == 0:
        raise RuntimeError(
            "No user validation pairs were created. "
            "Check sessions/enrollment count."
        )
    if len(gesture_cal_labels) == 0:
        raise RuntimeError(
            "No gesture validation pairs were created. "
            "Check sessions/gesture diversity/enrollment count."
        )

    user_threshold, user_val_eer, user_val_metrics = (
        base.find_eer_threshold(
            user_cal_labels,
            user_cal_scores,
        )
    )
    gesture_threshold, gesture_val_eer, gesture_val_metrics = (
        base.find_eer_threshold(
            gesture_cal_labels,
            gesture_cal_scores,
        )
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = output_dir / "shared_dual_head.pt"

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    torch.save(
        {
            "model_state_dict": {
                key: value.detach().cpu()
                for key, value in model.state_dict().items()
            },
            "architecture": "hand-only-shared-two-stream-dual-head",
            "input_dim": D,
            "seq_len": T,
            "embedding_dim": args.embedding_dim,
            "feature_layout": "hand_xyz_63+hand_velocity_63+valid_mask_1",
            "streams": ["hand_position", "hand_velocity"],
            "pooling": "per_stream_mean+std",
            "valid_mask_usage": "excluded_from_identity_input",
            "parameter_count": parameter_count,
            "train_users": train_users,
            "unseen_users": unseen_users,
            "gesture_threshold": float(gesture_threshold),
            "user_threshold": float(user_threshold),
            "gesture_validation_eer": float(gesture_val_eer),
            "user_validation_eer": float(user_val_eer),
            "gesture_validation_metrics": gesture_val_metrics,
            "user_validation_metrics": user_val_metrics,
            "sequence_mean": seq_mean,
            "sequence_std": seq_std,
            "duration_mean": dur_mean,
            "duration_std": dur_std,
            "dataset_path": str(Path(args.data).resolve()),
            "dataset_sha256": file_sha256(args.data),
            "enrollment_per_gesture": args.enroll,
            "seed": args.seed,
            "temperature": args.temperature,
            "user_contrastive_weight": args.user_contrastive_weight,
            "gesture_contrastive_weight": args.gesture_contrastive_weight,
            "training_loss": (
                "user_cross_entropy"
                "+user_supcon"
                "+gesture_supcon"
            ),
            "gesture_positive_rule": "same_gesture_id",
            "gesture_negative_calibration_rule": (
                "same_user_and_different_gesture"
            ),
            "user_positive_rule": "same_user",
            "user_negative_calibration_rule": (
                "different_user_and_same_gesture_when_available"
            ),
            "best_epoch": best_epoch,
            "stopped_epoch": stopped_epoch,
            "best_val_loss": float(best_val_loss),
            "best_val_id_accuracy": float(best_val_acc),
        },
        model_path,
    )

    summary_path = output_dir / "summary.txt"
    summary_path.write_text(
        "\n".join(
            [
                "WITECK Shared Backbone + Dual Embedding Head",
                "architecture=hand-only-shared-two-stream-dual-head",
                f"device={device}",
                f"dataset={Path(args.data).resolve()}",
                f"train_users={train_users}",
                f"unseen_users={unseen_users}",
                f"parameter_count={parameter_count}",
                f"best_epoch={best_epoch}",
                f"stopped_epoch={stopped_epoch}",
                f"best_val_loss={best_val_loss:.6f}",
                f"best_val_id_accuracy={best_val_acc:.6f}",
                "",
                "[Gesture Head]",
                f"threshold={gesture_threshold:.9f}",
                f"validation_eer={gesture_val_eer:.6f}",
                f"validation_far={gesture_val_metrics['far']:.6f}",
                f"validation_frr={gesture_val_metrics['frr']:.6f}",
                "positive_rule=same_gesture_id",
                "negative_calibration=same_user_and_different_gesture",
                "",
                "[User Head]",
                f"threshold={user_threshold:.9f}",
                f"validation_eer={user_val_eer:.6f}",
                f"validation_far={user_val_metrics['far']:.6f}",
                f"validation_frr={user_val_metrics['frr']:.6f}",
                "positive_rule=same_user",
                "negative_calibration=different_user_and_same_gesture_when_available",
                "",
                "[Final Decision]",
                "accept = gesture_score >= gesture_threshold AND user_score >= user_threshold",
                "",
                "NOTE: final unseen-user/unseen-gesture evaluation belongs in",
                "46_evaluate_shared_dual_head.py and must not tune these thresholds.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 78)
    print("Calibration complete")
    print("=" * 78)
    print(
        f"Gesture: threshold={gesture_threshold:.6f} "
        f"EER={gesture_val_eer:.4f} "
        f"FAR={gesture_val_metrics['far']:.4f} "
        f"FRR={gesture_val_metrics['frr']:.4f}"
    )
    print(
        f"User:    threshold={user_threshold:.6f} "
        f"EER={user_val_eer:.4f} "
        f"FAR={user_val_metrics['far']:.4f} "
        f"FRR={user_val_metrics['frr']:.4f}"
    )
    print(f"Saved model: {model_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
