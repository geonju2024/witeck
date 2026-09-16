"""Stable FastAPI integration surface for WITECK Shared Dual-Head."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

try:
    from .features import InvalidSequenceError, build_hand_features
    from .model_defs import SharedDualHead1DCNN
except ImportError:
    from features import InvalidSequenceError, build_hand_features
    from model_defs import SharedDualHead1DCNN


MODEL_VERSION = "shared-dual-head-v1.1.1"
EMBEDDING_DIM = 128

_ROOT = Path(__file__).resolve().parent
_MODEL_PATH = _ROOT / "weights" / "shared_dual_head_v1.pt"
_PREPROCESS_PATH = _ROOT / "preprocess.json"
_THRESHOLDS_PATH = _ROOT / "thresholds.json"

_STATE_LOCK = threading.RLock()
_MODEL = None
_CKPT = None
_DEVICE = None
_PREPROCESS = None
_THRESHOLDS = None


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _construct_model(device: torch.device):
    ckpt = torch.load(_MODEL_PATH, map_location=device, weights_only=False)
    if int(ckpt["input_dim"]) != 127:
        raise RuntimeError("release weight does not match the D=127 feature contract")

    train_users = ckpt.get("train_users")
    if train_users is not None:
        num_user_classes = len(train_users)
    else:
        # Fallback for checkpoints that explicitly saved num_user_classes.
        num_user_classes = int(ckpt["num_user_classes"])

    model = SharedDualHead1DCNN(
        input_dim=int(ckpt["input_dim"]),
        num_user_classes=num_user_classes,
        embedding_dim=int(ckpt["embedding_dim"]),
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    return model.to(device).eval(), ckpt


def load_model(device: str = "cpu"):
    """Load the Dual-Head model once. Repeated calls are harmless and thread-safe."""

    global _MODEL, _CKPT, _DEVICE, _PREPROCESS, _THRESHOLDS
    requested_device = torch.device(device)

    with _STATE_LOCK:
        if _MODEL is None:
            if not _MODEL_PATH.exists():
                raise FileNotFoundError(
                    f"Missing release weight: {_MODEL_PATH}. "
                    "Run finalize_release.py with the trained checkpoint first."
                )
            _MODEL, _CKPT = _construct_model(requested_device)
            _DEVICE = requested_device
            _PREPROCESS = _load_json(_PREPROCESS_PATH)
            _THRESHOLDS = _load_json(_THRESHOLDS_PATH)

    return {
        "model_version": MODEL_VERSION,
        "device": str(next(_MODEL.parameters()).device),
        "embedding_dim": EMBEDDING_DIM,
    }


def _ensure_loaded():
    if _MODEL is None:
        load_model("cpu")


def _normalization():
    if _PREPROCESS is None:
        _ensure_loaded()
    section = _PREPROCESS["dual_head"]
    mean = np.asarray(section["sequence_mean"], dtype=np.float32)
    std = np.asarray(section["sequence_std"], dtype=np.float32)
    duration_mean = float(section["duration_mean"])
    duration_std = float(section["duration_std"])
    return mean, std, duration_mean, duration_std


def _normalize(features: np.ndarray, durations: np.ndarray):
    mean, std, duration_mean, duration_std = _normalization()
    features = (features - mean) / np.maximum(std, 1e-8)
    durations = (durations - duration_mean) / max(duration_std, 1e-8)
    return features.astype(np.float32), durations.astype(np.float32)


def _prepare_one(payload: Any):
    features, duration = build_hand_features(payload)
    return features, np.float32(duration)


def _prepare_batch(payloads: Sequence[Any]):
    if not isinstance(payloads, Sequence) or isinstance(payloads, (str, bytes)):
        raise InvalidSequenceError("payloads must be a sequence")
    if len(payloads) == 0:
        raise InvalidSequenceError("payloads must not be empty")

    built = [_prepare_one(payload) for payload in payloads]
    features = np.stack([item[0] for item in built], axis=0)
    durations = np.asarray([item[1] for item in built], dtype=np.float32)
    return features, durations


@torch.inference_mode()
def _forward_batch(payloads: Sequence[Any]):
    _ensure_loaded()
    features, durations = _prepare_batch(payloads)
    features, durations = _normalize(features, durations)

    xb = torch.from_numpy(features).to(_DEVICE)
    db = torch.from_numpy(durations).to(_DEVICE)

    # Preserve the previous release's concurrency contract:
    # backend code does NOT need to add a separate model-forward lock.
    with _STATE_LOCK:
        gesture, user = _MODEL(xb, db)

    return (
        gesture.detach().cpu().numpy().astype(np.float32, copy=False),
        user.detach().cpu().numpy().astype(np.float32, copy=False),
    )


def embed_gesture(payload: Any) -> np.ndarray:
    """Return one L2-normalized 128-D gesture embedding from raw landmark payload."""
    gesture, _ = _forward_batch([payload])
    return gesture[0]


def embed_user(payload: Any) -> np.ndarray:
    """Return one L2-normalized 128-D user embedding from raw landmark payload."""
    _, user = _forward_batch([payload])
    return user[0]


def embed_gesture_batch(payloads: Sequence[Any]) -> np.ndarray:
    """Return [N,128] L2-normalized gesture embeddings."""
    gesture, _ = _forward_batch(payloads)
    return gesture


def embed_user_batch(payloads: Sequence[Any]) -> np.ndarray:
    """Return [N,128] L2-normalized user embeddings."""
    _, user = _forward_batch(payloads)
    return user


def embed_both(payload: Any):
    """Efficiently compute both embeddings with one model forward pass."""
    gesture, user = _forward_batch([payload])
    return {
        "gesture_embedding": gesture[0],
        "user_embedding": user[0],
    }


def embed_both_batch(payloads: Sequence[Any]):
    """Efficiently compute both embedding matrices with one model forward pass."""
    gesture, user = _forward_batch(payloads)
    return {
        "gesture_embeddings": gesture,
        "user_embeddings": user,
    }


def get_thresholds() -> dict:
    _ensure_loaded()
    return dict(_THRESHOLDS)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Optional helper. Backend may instead compute normalized-vector dot products."""
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    a = a / max(float(np.linalg.norm(a)), 1e-12)
    b = b / max(float(np.linalg.norm(b)), 1e-12)
    return float(np.dot(a, b))
