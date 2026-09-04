from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


FEATURE_KEYS = ("X", "features", "data", "sequences")
USER_KEYS = (
    "user_ids", "users", "subjects", "subject_ids", "y_user",
    "performer", "performers",
)
GESTURE_KEYS = ("gesture_ids", "gestures", "y_gesture", "gesture")


def _find_key(store, candidates: tuple[str, ...]) -> str:
    for key in candidates:
        if key in store:
            return key
    raise KeyError(f"none of {candidates} found; available keys: {list(store.keys())}")


@dataclass
class WiteckArrays:
    x: np.ndarray
    user_ids: np.ndarray
    gesture_ids: np.ndarray

    @classmethod
    def from_npz(cls, path: str | Path) -> "WiteckArrays":
        with np.load(path, allow_pickle=True) as store:
            x = np.asarray(store[_find_key(store, FEATURE_KEYS)], dtype=np.float32)
            users = np.asarray(store[_find_key(store, USER_KEYS)])
            gestures = np.asarray(store[_find_key(store, GESTURE_KEYS)])
            if x.ndim == 2:
                t = int(np.asarray(store["T"]).item()) if "T" in store else 32
                if x.shape[1] % t:
                    raise ValueError(
                        f"flattened feature width {x.shape[1]} is not divisible by T={t}"
                    )
                d = int(np.asarray(store["D"]).item()) if "D" in store else x.shape[1] // t
                if t * d != x.shape[1]:
                    raise ValueError(
                        f"flattened feature width {x.shape[1]} does not match T={t}, D={d}"
                    )
                x = x.reshape(len(x), t, d)
        if x.ndim != 3:
            raise ValueError(f"features must be [N,T,D], received {x.shape}")
        if not (len(x) == len(users) == len(gestures)):
            raise ValueError("features, user_ids and gesture_ids must have equal length")
        return cls(x=x, user_ids=users.astype(str), gesture_ids=gestures.astype(str))


class FeatureStandardizer:
    """Per-feature normalization fitted only on training sequences."""

    def fit(self, x: np.ndarray) -> "FeatureStandardizer":
        self.mean = x.mean(axis=(0, 1), keepdims=True).astype(np.float32)
        self.std = np.maximum(x.std(axis=(0, 1), keepdims=True), 1e-6).astype(np.float32)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if not hasattr(self, "mean"):
            raise RuntimeError("standardizer is not fitted")
        return ((x - self.mean) / self.std).astype(np.float32)


class PairDataset(Dataset):
    """Deterministic same-gesture pairs, balanced by pair index."""

    def __init__(
        self,
        x: np.ndarray,
        user_ids: np.ndarray,
        gesture_ids: np.ndarray,
        pairs_per_epoch: int | None = None,
        seed: int = 42,
    ) -> None:
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.users = np.asarray(user_ids).astype(str)
        self.gestures = np.asarray(gesture_ids).astype(str)
        self.seed = seed
        self.epoch = 0
        self.pairs_per_epoch = pairs_per_epoch or max(2048, len(x) * 4)
        self.by_gesture: dict[str, np.ndarray] = {
            g: np.flatnonzero(self.gestures == g) for g in np.unique(self.gestures)
        }
        self.valid_positive = [
            i for i in range(len(x))
            if np.sum((self.users == self.users[i]) & (self.gestures == self.gestures[i])) > 1
        ]
        self.valid_negative = [
            i for i in range(len(x))
            if np.any(self.users[self.by_gesture[self.gestures[i]]] != self.users[i])
        ]
        if not self.valid_positive or not self.valid_negative:
            raise ValueError("dataset needs repeated genuine samples and cross-user same-gesture samples")

    def __len__(self) -> int:
        return self.pairs_per_epoch

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        rng = np.random.default_rng(self.seed + index + self.epoch * self.pairs_per_epoch)
        positive = index % 2 == 0
        pool = self.valid_positive if positive else self.valid_negative
        anchor = int(pool[rng.integers(len(pool))])
        same_gesture = self.by_gesture[self.gestures[anchor]]
        if positive:
            choices = same_gesture[
                (self.users[same_gesture] == self.users[anchor]) & (same_gesture != anchor)
            ]
        else:
            choices = same_gesture[self.users[same_gesture] != self.users[anchor]]
        partner = int(choices[rng.integers(len(choices))])
        return self.x[anchor], self.x[partner], torch.tensor(float(positive))


class SequenceDataset(Dataset):
    def __init__(self, x: np.ndarray, labels: np.ndarray) -> None:
        self.x = torch.as_tensor(x, dtype=torch.float32)
        names, encoded = np.unique(np.asarray(labels).astype(str), return_inverse=True)
        self.labels = torch.as_tensor(encoded, dtype=torch.long)
        self.class_names = names

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return self.x[index], self.labels[index]
