"""Train the WITECK dual-head encoder with service-aligned metric learning.

This experiment intentionally keeps the model architecture from
shared_dual_head_model.py unchanged.  It changes only the training problem:

* batches contain user/gesture positives and hard negatives on purpose;
* both heads use supervised contrastive and batch-hard triplet losses;
* closed-set user CE is optional and defaults to a small auxiliary weight;
* thresholds are calibrated on held-out sessions of training users only;
* P08-P10 remain untouched for the final evaluator.

The unchanged architecture means checkpoints produced here can be evaluated by
evaluate_baseline_model.py (or its unseen-G5 variant).
"""

from __future__ import annotations

import argparse
import copy
import importlib
import math
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


train_base = importlib.import_module("shared_dual_head_model")
legacy = importlib.import_module("train_supcon_embedding")
base = importlib.import_module("train_embedding_baseline")
common = importlib.import_module("model_utils")

SharedDualHead1DCNN = train_base.SharedDualHead1DCNN
DEFAULT_TRAIN_USERS = list(legacy.TRAIN_USERS)
DEFAULT_UNSEEN_USERS = list(legacy.UNSEEN_USERS)


def parse_id_list(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def choose_one(
    rng: np.random.Generator,
    candidates: np.ndarray,
) -> int | None:
    if len(candidates) == 0:
        return None
    return int(rng.choice(candidates))


class ServiceMetricBatchSampler:
    """Build batches that contain the comparisons used by the service.

    For each sampled anchor, the sampler tries to add:

    * user positive: same performer, preferably another gesture/session;
    * user hard negative: another performer doing the same gesture;
    * gesture positive: same gesture, preferably another session;
    * gesture hard negative: same performer doing another gesture.

    The same batch therefore supports both user and gesture metric losses.
    Session is used only to choose independent examples.  It is never passed to
    the model as an input feature.
    """

    ROLE_NAMES = (
        "user_positive",
        "user_hard_negative",
        "gesture_positive",
        "gesture_hard_negative",
    )

    def __init__(
        self,
        user_y: np.ndarray,
        gesture_y: np.ndarray,
        session: np.ndarray,
        batch_size: int = 48,
        seed: int = 42,
        batches_per_epoch: int | None = None,
        performer: np.ndarray | None = None,
        imitation_target: np.ndarray | None = None,
        use_imitation_negatives: bool = False,
    ) -> None:
        self.user_y = np.asarray(user_y)
        self.gesture_y = np.asarray(gesture_y)
        self.session = np.asarray(session).astype(str)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = 0
        self.use_imitation_negatives = bool(use_imitation_negatives)

        n = len(self.user_y)
        if not (n == len(self.gesture_y) == len(self.session)):
            raise ValueError("user/gesture/session arrays must have equal length")
        if n < 2:
            raise ValueError("metric sampler needs at least two samples")

        if self.use_imitation_negatives:
            if performer is None or imitation_target is None:
                raise ValueError(
                    "imitation negatives need performer and imitation_target"
                )
            self.performer = np.asarray(performer).astype(str)
            self.imitation_target = np.asarray(imitation_target).astype(str)
            if not (n == len(self.performer) == len(self.imitation_target)):
                raise ValueError(
                    "performer/imitation_target must match the other arrays"
                )
            self.role_names = (
                "user_positive",
                "imitation_negative",
                "user_hard_negative",
                "gesture_positive",
                "gesture_hard_negative",
            )
        else:
            self.performer = None
            self.imitation_target = None
            self.role_names = self.ROLE_NAMES

        minimum_batch = 1 + len(self.role_names)
        if self.batch_size < minimum_batch:
            raise ValueError(f"batch_size must be at least {minimum_batch}")

        self.indices = np.arange(n, dtype=np.int64)
        self.batches_per_epoch = (
            max(1, math.ceil(n / self.batch_size))
            if batches_per_epoch is None
            else int(batches_per_epoch)
        )
        self.last_epoch_stats: dict[str, int] = {}

    def __len__(self) -> int:
        return self.batches_per_epoch

    def _candidates(
        self,
        anchor: int,
        role: str,
    ) -> tuple[np.ndarray, bool]:
        u = self.user_y[anchor]
        g = self.gesture_y[anchor]
        s = self.session[anchor]
        not_self = self.indices != anchor

        if role == "user_positive":
            preferred = self.indices[
                not_self
                & (self.user_y == u)
                & (self.gesture_y != g)
                & (self.session != s)
            ]
            if len(preferred):
                return preferred, False
            fallback = self.indices[
                not_self & (self.user_y == u) & (self.session != s)
            ]
            if len(fallback):
                return fallback, True
            return self.indices[not_self & (self.user_y == u)], True

        if role == "imitation_negative":
            # Someone else deliberately copying this anchor's performer.  The
            # gesture is left free on purpose: the user head has to reject an
            # impersonator whichever gesture they attempt.  No fallback, so a
            # missing count means the dataset holds no such attack.
            target = self.performer[anchor]
            return (
                self.indices[
                    (self.performer != target)
                    & (self.imitation_target == target)
                ],
                False,
            )

        if role == "user_hard_negative":
            preferred = self.indices[
                (self.user_y != u) & (self.gesture_y == g)
            ]
            if len(preferred):
                return preferred, False
            return self.indices[self.user_y != u], True

        if role == "gesture_positive":
            preferred = self.indices[
                not_self & (self.gesture_y == g) & (self.session != s)
            ]
            if len(preferred):
                return preferred, False
            return self.indices[not_self & (self.gesture_y == g)], True

        if role == "gesture_hard_negative":
            preferred = self.indices[
                (self.user_y == u) & (self.gesture_y != g)
            ]
            if len(preferred):
                return preferred, False
            return self.indices[self.gesture_y != g], True

        raise ValueError(f"Unknown sampler role: {role}")

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        stats: Counter[str] = Counter()

        for _ in range(self.batches_per_epoch):
            batch: list[int] = []
            in_batch: set[int] = set()
            attempts = 0

            while len(batch) < self.batch_size and attempts < self.batch_size * 8:
                attempts += 1
                anchor = int(rng.choice(self.indices))

                if anchor not in in_batch:
                    batch.append(anchor)
                    in_batch.add(anchor)
                    stats["anchor"] += 1
                    if len(batch) >= self.batch_size:
                        break

                for role in self.role_names:
                    candidates, used_fallback = self._candidates(anchor, role)
                    candidates = np.asarray(
                        [x for x in candidates.tolist() if int(x) not in in_batch],
                        dtype=np.int64,
                    )
                    selected = choose_one(rng, candidates)
                    if selected is None:
                        stats[f"missing_{role}"] += 1
                        continue
                    batch.append(selected)
                    in_batch.add(selected)
                    stats[role] += 1
                    if used_fallback:
                        stats[f"fallback_{role}"] += 1
                    if len(batch) >= self.batch_size:
                        break

            if len(batch) < 2:
                raise RuntimeError("Could not construct a metric-learning batch")

            rng.shuffle(batch)
            yield batch

        self.last_epoch_stats = dict(stats)
        fields = " ".join(
            f"{name}={stats.get(name, 0)}"
            for name in ("anchor",) + self.role_names
        )
        fallbacks = sum(
            value for name, value in stats.items() if name.startswith("fallback_")
        )
        missing = sum(
            value for name, value in stats.items() if name.startswith("missing_")
        )
        print(
            f"[ServiceMetricSampler epoch={self.epoch + 1}] "
            f"{fields} fallback={fallbacks} missing={missing}"
        )
        self.epoch += 1


def make_metric_loader(
    X: np.ndarray,
    duration: np.ndarray,
    user_y: np.ndarray,
    gesture_y: np.ndarray,
    session: np.ndarray,
    batch_size: int,
    seed: int,
    performer: np.ndarray | None = None,
    imitation_target: np.ndarray | None = None,
    use_imitation_negatives: bool = False,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(X).float(),
        torch.from_numpy(duration).float(),
        torch.from_numpy(user_y).long(),
        torch.from_numpy(gesture_y).long(),
    )
    sampler = ServiceMetricBatchSampler(
        user_y=user_y,
        gesture_y=gesture_y,
        session=session,
        batch_size=batch_size,
        seed=seed,
        performer=performer,
        imitation_target=imitation_target,
        use_imitation_negatives=use_imitation_negatives,
    )
    return DataLoader(dataset, batch_sampler=sampler, num_workers=0)


def batch_hard_triplet_loss(
    embedding: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.20,
) -> torch.Tensor:
    """Batch-hard triplet loss using cosine distance on L2 embeddings."""

    n = embedding.shape[0]
    if n < 3:
        return embedding.sum() * 0.0

    distance = 1.0 - embedding @ embedding.T
    eye = torch.eye(n, dtype=torch.bool, device=embedding.device)
    positive = labels[:, None].eq(labels[None, :]) & ~eye
    negative = ~labels[:, None].eq(labels[None, :])
    valid = positive.any(dim=1) & negative.any(dim=1)

    if not bool(valid.any()):
        return embedding.sum() * 0.0

    hardest_positive = distance.masked_fill(~positive, float("-inf")).max(dim=1).values
    hardest_negative = distance.masked_fill(~negative, float("inf")).min(dim=1).values
    losses = torch.relu(hardest_positive - hardest_negative + margin)
    return losses[valid].mean()


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    weights: dict[str, float],
    temperature: float,
    triplet_margin: float,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: Counter[str] = Counter()
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
            ce = nn.functional.cross_entropy(user_logits, user_y)
            user_supcon = train_base.supervised_contrastive_loss(
                user_emb, user_y, temperature
            )
            gesture_supcon = train_base.supervised_contrastive_loss(
                gesture_emb, gesture_y, temperature
            )
            user_triplet = batch_hard_triplet_loss(
                user_emb, user_y, triplet_margin
            )
            gesture_triplet = batch_hard_triplet_loss(
                gesture_emb, gesture_y, triplet_margin
            )

            loss = (
                weights["ce"] * ce
                + weights["user_supcon"] * user_supcon
                + weights["gesture_supcon"] * gesture_supcon
                + weights["user_triplet"] * user_triplet
                + weights["gesture_triplet"] * gesture_triplet
            )

            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

            n = len(user_y)
            values = {
                "loss": loss,
                "ce": ce,
                "user_supcon": user_supcon,
                "gesture_supcon": gesture_supcon,
                "user_triplet": user_triplet,
                "gesture_triplet": gesture_triplet,
            }
            for name, value in values.items():
                totals[name] += float(value.item()) * n
            totals["correct"] += int((user_logits.argmax(dim=1) == user_y).sum())
            total_n += n

    denom = max(total_n, 1)
    result = {name: totals[name] / denom for name in totals}
    result["accuracy"] = totals["correct"] / denom
    return result


def calibration_scores_from_split(
    gesture_embeddings: np.ndarray,
    user_embeddings: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    meta: dict[str, np.ndarray],
    train_users: list[str],
    enroll: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build threshold pairs without using final unseen users.

    Templates come only from train_idx.  Queries come only from val_idx.
    """

    performer = meta["performer"]
    gesture = meta["gesture"]
    session = meta["session"]
    user_labels: list[int] = []
    user_scores: list[float] = []
    gesture_labels: list[int] = []
    gesture_scores: list[float] = []

    for target_user in train_users:
        gestures = sorted(np.unique(gesture[train_idx][performer[train_idx] == target_user]))
        for target_gesture in gestures:
            enrollment = train_idx[
                (performer[train_idx] == target_user)
                & (gesture[train_idx] == target_gesture)
            ]
            if len(enrollment) < enroll:
                continue

            enrollment_sessions = sorted(np.unique(session[enrollment]))
            chosen_session = enrollment_sessions[-1]
            enrollment = enrollment[session[enrollment] == chosen_session][:enroll]
            if len(enrollment) < enroll:
                continue

            gesture_template = gesture_embeddings[enrollment].mean(axis=0)
            user_template = user_embeddings[enrollment].mean(axis=0)

            genuine = val_idx[
                (performer[val_idx] == target_user)
                & (gesture[val_idx] == target_gesture)
            ]
            if len(genuine) == 0:
                continue

            gesture_labels.extend([1] * len(genuine))
            gesture_scores.extend(
                base.cosine_scores(gesture_embeddings[genuine], gesture_template).tolist()
            )
            user_labels.extend([1] * len(genuine))
            user_scores.extend(
                base.cosine_scores(user_embeddings[genuine], user_template).tolist()
            )

            wrong_gesture = val_idx[
                (performer[val_idx] == target_user)
                & (gesture[val_idx] != target_gesture)
            ]
            if len(wrong_gesture):
                gesture_labels.extend([0] * len(wrong_gesture))
                gesture_scores.extend(
                    base.cosine_scores(
                        gesture_embeddings[wrong_gesture], gesture_template
                    ).tolist()
                )

            same_gesture_impostor = val_idx[
                (performer[val_idx] != target_user)
                & (gesture[val_idx] == target_gesture)
            ]
            if len(same_gesture_impostor):
                user_labels.extend([0] * len(same_gesture_impostor))
                user_scores.extend(
                    base.cosine_scores(
                        user_embeddings[same_gesture_impostor], user_template
                    ).tolist()
                )

    return (
        np.asarray(user_labels, dtype=np.int64),
        np.asarray(user_scores, dtype=np.float64),
        np.asarray(gesture_labels, dtype=np.int64),
        np.asarray(gesture_scores, dtype=np.float64),
    )


def format_epoch(prefix: str, values: dict[str, float]) -> str:
    keys = (
        "loss",
        "ce",
        "user_supcon",
        "gesture_supcon",
        "user_triplet",
        "gesture_triplet",
        "accuracy",
    )
    return prefix + " " + " ".join(f"{key}={values.get(key, 0.0):.4f}" for key in keys)


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default=str(
            project_dir / "dataset" / "dataset_1955_recent8_updated_20260905_hand_only.npz"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(project_dir / "output" / "shared_dual_head_metric"),
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--triplet-margin", type=float, default=0.20)
    parser.add_argument("--user-ce-weight", type=float, default=0.10)
    parser.add_argument("--user-contrastive-weight", type=float, default=1.0)
    parser.add_argument("--gesture-contrastive-weight", type=float, default=1.0)
    parser.add_argument("--user-triplet-weight", type=float, default=0.50)
    parser.add_argument("--gesture-triplet-weight", type=float, default=0.50)
    parser.add_argument("--enroll", type=int, default=3)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument(
        "--selection-metric",
        choices=("total_loss", "user_metric"),
        default="total_loss",
        help=(
            "Early-stopping checkpoint criterion. user_metric uses only "
            "user SupCon + user triplet validation losses."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-users", default=",".join(DEFAULT_TRAIN_USERS))
    parser.add_argument("--unseen-users", default=",".join(DEFAULT_UNSEEN_USERS))
    parser.add_argument(
        "--heldout-gesture",
        default=None,
        help=(
            "Optional gesture ID excluded from training/validation, e.g. G5. "
            "Use this only for an unseen-gesture proxy experiment."
        ),
    )
    parser.add_argument(
        "--imitation-negatives",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Put a real impersonator of the anchor's performer in the batch. "
            "Pass --no-imitation-negatives to reproduce the v1.0.0 sampler."
        ),
    )
    args = parser.parse_args()

    train_users = parse_id_list(args.train_users)
    unseen_users = parse_id_list(args.unseen_users)
    base.seed_everything(args.seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    _, X_seq, duration, meta, T, D = legacy.load_dataset(args.data)
    common.describe_metadata(meta)
    for warning in common.validate_metadata(meta):
        print(f"[metadata] {warning}")
    train_idx, val_idx = base.chronological_user_split(meta, train_users)
    if args.heldout_gesture:
        train_idx = train_idx[meta["gesture"][train_idx] != args.heldout_gesture]
        val_idx = val_idx[meta["gesture"][val_idx] != args.heldout_gesture]
        if len(train_idx) == 0 or len(val_idx) == 0:
            raise RuntimeError(
                f"No train/validation samples remain after holding out "
                f"gesture={args.heldout_gesture}"
            )

    seq_mean, seq_std = legacy.fit_sequence_stats(X_seq[train_idx])
    dur_mean, dur_std = legacy.fit_duration_stats(duration[train_idx])
    X_norm = legacy.apply_sequence_stats(X_seq, seq_mean, seq_std)
    duration_norm = legacy.apply_duration_stats(duration, dur_mean, dur_std)

    user_to_label = {user: i for i, user in enumerate(train_users)}
    gesture_ids = sorted(np.unique(meta["gesture"]).tolist())
    gesture_to_label = {gesture: i for i, gesture in enumerate(gesture_ids)}
    user_y = np.full(len(X_seq), -1, dtype=np.int64)
    for user, label in user_to_label.items():
        user_y[meta["performer"] == user] = label
    gesture_y = np.asarray(
        [gesture_to_label[value] for value in meta["gesture"]], dtype=np.int64
    )

    if np.any(user_y[train_idx] < 0) or np.any(user_y[val_idx] < 0):
        raise RuntimeError("Train/validation split contains an unmapped user")

    gesture_owners = meta["_schema"]["gesture_owners"]
    imitation_target = meta["imitation_target"]

    train_loader = make_metric_loader(
        X_norm[train_idx],
        duration_norm[train_idx],
        user_y[train_idx],
        gesture_y[train_idx],
        meta["session"][train_idx],
        args.batch_size,
        args.seed,
        performer=meta["performer"][train_idx],
        imitation_target=imitation_target[train_idx],
        use_imitation_negatives=args.imitation_negatives,
    )
    val_loader = legacy.make_loader(
        X_norm[val_idx],
        duration_norm[val_idx],
        user_y[val_idx],
        gesture_y[val_idx],
        batch_size=128,
        shuffle=False,
    )

    model = SharedDualHead1DCNN(D, len(train_users), args.embedding_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    weights = {
        "ce": args.user_ce_weight,
        "user_supcon": args.user_contrastive_weight,
        "gesture_supcon": args.gesture_contrastive_weight,
        "user_triplet": args.user_triplet_weight,
        "gesture_triplet": args.gesture_triplet_weight,
    }

    print("=" * 78)
    print("WITECK service-aligned dual-head metric training")
    print("=" * 78)
    print(f"device={device} N={len(X_seq)} T={T} D={D}")
    print(f"train={len(train_idx)} val={len(val_idx)} final_users={unseen_users}")
    print(f"heldout_gesture={args.heldout_gesture}")
    print(f"weights={weights} margin={args.triplet_margin}")
    print("date/session usage: sampler and split only; never a model feature")
    print(f"gesture_owners={gesture_owners}")
    if args.imitation_negatives:
        train_performer = meta["performer"][train_idx]
        attacked = np.isin(train_performer, imitation_target[train_idx])
        print(
            "imitation negatives: ON "
            f"({int(attacked.sum())}/{len(train_idx)} training samples "
            "belong to a performer somebody else imitates)"
        )
    else:
        print("imitation negatives: OFF (v1.0.0 sampler)")

    best_selection_value = float("inf")
    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    best_val_accuracy = 0.0
    stale = 0
    stopped_epoch = args.epochs

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            weights,
            args.temperature,
            args.triplet_margin,
            optimizer,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            device,
            weights,
            args.temperature,
            args.triplet_margin,
        )
        print(f"Epoch {epoch:02d} | {format_epoch('train', train_metrics)}")
        print(f"           | {format_epoch('val  ', val_metrics)}")

        if args.selection_metric == "user_metric":
            selection_value = (
                weights["user_supcon"] * val_metrics["user_supcon"]
                + weights["user_triplet"] * val_metrics["user_triplet"]
            )
        else:
            selection_value = val_metrics["loss"]

        if selection_value < best_selection_value:
            best_selection_value = selection_value
            best_val_loss = val_metrics["loss"]
            best_val_accuracy = val_metrics["accuracy"]
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
        raise RuntimeError("No checkpoint was created")
    model.load_state_dict(best_state)
    model.eval()

    gesture_embeddings, user_embeddings = train_base.extract_dual_embeddings(
        model, X_norm, duration_norm, device
    )
    (
        user_cal_labels,
        user_cal_scores,
        gesture_cal_labels,
        gesture_cal_scores,
    ) = calibration_scores_from_split(
        gesture_embeddings,
        user_embeddings,
        train_idx,
        val_idx,
        meta,
        train_users,
        args.enroll,
    )
    if len(np.unique(user_cal_labels)) < 2:
        raise RuntimeError("User calibration did not produce both classes")
    if len(np.unique(gesture_cal_labels)) < 2:
        raise RuntimeError("Gesture calibration did not produce both classes")

    user_threshold, user_eer, user_metrics = base.find_eer_threshold(
        user_cal_labels, user_cal_scores
    )
    gesture_threshold, gesture_eer, gesture_metrics = base.find_eer_threshold(
        gesture_cal_labels, gesture_cal_scores
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "shared_dual_head_metric.pt"
    parameter_count = sum(p.numel() for p in model.parameters())
    checkpoint = {
        "model_state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "architecture": getattr(
            model,
            "architecture_name",
            "hand-only-shared-two-stream-dual-head",
        ),
        "experiment": getattr(
            model,
            "experiment_name",
            "service-metric-hard-negative-v1",
        ),
        "input_dim": D,
        "seq_len": T,
        "embedding_dim": args.embedding_dim,
        "feature_layout": "hand_xyz_63+hand_velocity_63+valid_mask_1",
        "streams": ["hand_position", "hand_velocity"],
        "pooling": getattr(
            model,
            "pooling_name",
            "per_stream_mean+std",
        ),
        "valid_mask_usage": "excluded_from_identity_input",
        "parameter_count": parameter_count,
        "train_users": train_users,
        "unseen_users": unseen_users,
        "gesture_threshold": float(gesture_threshold),
        "user_threshold": float(user_threshold),
        "gesture_validation_eer": float(gesture_eer),
        "user_validation_eer": float(user_eer),
        "gesture_validation_metrics": gesture_metrics,
        "user_validation_metrics": user_metrics,
        "sequence_mean": seq_mean,
        "sequence_std": seq_std,
        "duration_mean": dur_mean,
        "duration_std": dur_std,
        "dataset_path": str(Path(args.data).resolve()),
        "dataset_sha256": train_base.file_sha256(args.data),
        "enrollment_per_gesture": args.enroll,
        "seed": args.seed,
        "temperature": args.temperature,
        "triplet_margin": args.triplet_margin,
        "loss_weights": weights,
        "training_loss": "user_ce+user_supcon+gesture_supcon+user_triplet+gesture_triplet",
        "sampler": "service_metric_anchor_four_roles",
        "user_positive_rule": "same_user_prefer_different_gesture_and_session",
        "user_hard_negative_rule": "different_user_prefer_same_gesture",
        "imitation_negatives": bool(args.imitation_negatives),
        "imitation_negative_rule": (
            "different_performer_whose_imitation_target_is_the_anchor_performer"
            if args.imitation_negatives
            else None
        ),
        "gesture_owners": gesture_owners,
        "gesture_positive_rule": "same_gesture_prefer_different_session",
        "gesture_hard_negative_rule": "same_user_different_gesture",
        "threshold_calibration": "train_templates_vs_heldout_sessions_of_train_users",
        "heldout_gesture": args.heldout_gesture,
        "best_epoch": best_epoch,
        "stopped_epoch": stopped_epoch,
        "selection_metric": args.selection_metric,
        "best_selection_value": float(best_selection_value),
        "best_val_loss": float(best_val_loss),
        "best_val_id_accuracy": float(best_val_accuracy),
    }
    torch.save(checkpoint, model_path)

    summary_path = output_dir / "summary.txt"
    summary_path.write_text(
        "\n".join(
            [
                "WITECK service-aligned dual-head metric training",
                f"checkpoint={model_path.resolve()}",
                f"dataset={Path(args.data).resolve()}",
                f"device={device}",
                f"train_users={train_users}",
                f"untouched_final_users={unseen_users}",
                f"heldout_gesture={args.heldout_gesture}",
                f"imitation_negatives={args.imitation_negatives}",
                f"loss_weights={weights}",
                f"triplet_margin={args.triplet_margin}",
                f"best_epoch={best_epoch}",
                f"stopped_epoch={stopped_epoch}",
                f"selection_metric={args.selection_metric}",
                f"best_selection_value={best_selection_value:.6f}",
                f"best_val_loss={best_val_loss:.6f}",
                f"best_val_id_accuracy={best_val_accuracy:.6f}",
                f"gesture_validation_eer={gesture_eer:.6f}",
                f"user_validation_eer={user_eer:.6f}",
                "",
                "[Gesture Head]",
                f"threshold={gesture_threshold:.9f}",
                f"validation_eer={gesture_eer:.6f}",
                f"validation_far={gesture_metrics['far']:.6f}",
                f"validation_frr={gesture_metrics['frr']:.6f}",
                "",
                "[User Head]",
                f"threshold={user_threshold:.9f}",
                f"validation_eer={user_eer:.6f}",
                f"validation_far={user_metrics['far']:.6f}",
                f"validation_frr={user_metrics['frr']:.6f}",
                "",
                "Final evaluation must use P08-P10 without threshold tuning.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Saved model: {model_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
