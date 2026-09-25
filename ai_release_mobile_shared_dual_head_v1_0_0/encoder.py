"""Stable backend interface for WITECK mobile Shared Dual Head v1.0.0."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

try:
    from .features import build_hand_features
    from .model_defs import SharedDualHead1DCNN
except ImportError:
    from features import build_hand_features
    from model_defs import SharedDualHead1DCNN


MODEL_VERSION = "witeck-mobile-shared-dual-head-g1g24-v1.0.0"
EMBEDDING_DIM = 128
ENROLLMENT_TAKES = 3

_ROOT = Path(__file__).resolve().parent
_WEIGHT_PATH = _ROOT / "weights" / "shared_dual_head.pt"
_LOCK = threading.RLock()
_MODEL: SharedDualHead1DCNN | None = None
_CHECKPOINT: dict | None = None


def _unit(value: np.ndarray, axis: int = -1) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    return value / np.maximum(np.linalg.norm(value, axis=axis, keepdims=True), 1e-12)


def load_model(device: str = "cpu") -> dict:
    """Load weights once during FastAPI startup. Repeated calls are harmless."""

    global _MODEL, _CHECKPOINT
    with _LOCK:
        if _MODEL is None:
            target = torch.device(device)
            checkpoint = torch.load(_WEIGHT_PATH, map_location=target, weights_only=False)
            if checkpoint.get("architecture") != "hand-only-shared-two-stream-dual-head":
                raise RuntimeError("unexpected checkpoint architecture")
            model = SharedDualHead1DCNN(
                input_dim=int(checkpoint["input_dim"]),
                num_user_classes=len(checkpoint["train_users"]),
                embedding_dim=int(checkpoint["embedding_dim"]),
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            _MODEL = model.to(target).eval()
            _CHECKPOINT = checkpoint
    return release_metadata()


def _ensure_loaded() -> tuple[SharedDualHead1DCNN, dict]:
    if _MODEL is None:
        load_model("cpu")
    assert _MODEL is not None and _CHECKPOINT is not None
    return _MODEL, _CHECKPOINT


def _forward(features: np.ndarray, durations: np.ndarray):
    model, checkpoint = _ensure_loaded()
    mean = np.asarray(checkpoint["sequence_mean"], dtype=np.float32)
    std = np.asarray(checkpoint["sequence_std"], dtype=np.float32)
    duration_mean = float(checkpoint["duration_mean"])
    duration_std = float(checkpoint["duration_std"])
    x = ((features.astype(np.float32) - mean) / std).astype(np.float32)
    d = ((durations.astype(np.float32) - duration_mean) / duration_std).astype(np.float32)
    device = next(model.parameters()).device
    with _LOCK, torch.inference_mode():
        gesture, user, _ = model(
            torch.from_numpy(x).to(device),
            torch.from_numpy(d).to(device),
        )
    return (
        _unit(gesture.cpu().numpy(), axis=1),
        _unit(user.cpu().numpy(), axis=1),
    )


def embed_preprocessed(features: np.ndarray, duration_sec: float):
    """Embed an already-built ``[32,127]`` feature tensor."""

    features = np.asarray(features, dtype=np.float32)
    if features.shape != (32, 127):
        raise ValueError(f"expected [32,127], got {features.shape}")
    gesture, user = _forward(
        features[None, ...], np.asarray([duration_sec], dtype=np.float32)
    )
    return gesture[0], user[0]


def embed(frames: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(gesture_embedding[128], user_embedding[128])``."""

    features, duration = build_hand_features(frames)
    return embed_preprocessed(features, duration)


def embed_batch(list_of_frames: Sequence[Any]) -> tuple[np.ndarray, np.ndarray]:
    """Return two L2-normalized arrays, each with shape ``[N,128]``."""

    prepared = [build_hand_features(payload) for payload in list_of_frames]
    if not prepared:
        empty = np.empty((0, EMBEDDING_DIM), dtype=np.float32)
        return empty, empty.copy()
    features = np.stack([item[0] for item in prepared]).astype(np.float32)
    durations = np.asarray([item[1] for item in prepared], dtype=np.float32)
    return _forward(features, durations)


def enroll(takes: Sequence[Any]) -> dict[str, np.ndarray | str | int]:
    """Create both templates from at least three captures; no retraining occurs."""

    if len(takes) < ENROLLMENT_TAKES:
        raise ValueError(f"at least {ENROLLMENT_TAKES} enrollment takes are required")
    gesture, user = embed_batch(takes)
    return {
        "model_version": MODEL_VERSION,
        "takes": len(takes),
        "gesture_template": _unit(gesture.mean(axis=0)),
        "user_template": _unit(user.mean(axis=0)),
    }


def verify(
    frames: Any,
    gesture_template: np.ndarray,
    user_template: np.ndarray,
    *,
    gesture_threshold: float | None = None,
    user_threshold: float | None = None,
) -> dict[str, float | bool | str]:
    """Verify one capture. Both heads must pass."""

    _, checkpoint = _ensure_loaded()
    gesture_embedding, user_embedding = embed(frames)
    gesture_score = float(gesture_embedding @ _unit(gesture_template))
    user_score = float(user_embedding @ _unit(user_template))
    tg = float(
        checkpoint["gesture_threshold"]
        if gesture_threshold is None
        else gesture_threshold
    )
    tu = float(
        checkpoint["user_threshold"] if user_threshold is None else user_threshold
    )
    return {
        "model_version": MODEL_VERSION,
        "gesture_score": gesture_score,
        "user_score": user_score,
        "gesture_threshold": tg,
        "user_threshold": tu,
        "gesture_passed": gesture_score >= tg,
        "user_passed": user_score >= tu,
        "passed": gesture_score >= tg and user_score >= tu,
    }


def release_metadata() -> dict:
    with (_ROOT / "manifest.json").open(encoding="utf-8") as handle:
        result = json.load(handle)
    if _MODEL is not None:
        result["device"] = str(next(_MODEL.parameters()).device)
    return result
