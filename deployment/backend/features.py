"""Runtime feature builder matching the final G1-G24 mobile preprocessing."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence

import numpy as np


T_OUT = 32
FEATURE_DIM = 127
MIN_VALID_FRAMES = 8
WRIST = 0
MID_MCP = 9
POSITION_CLIP = 10.0
VELOCITY_CLIP = 100.0


class InvalidSequenceError(ValueError):
    """The capture cannot be transformed into a reliable model input."""


def _points(value: Any) -> np.ndarray:
    if value is None:
        raise InvalidSequenceError("landmarks are missing")
    if isinstance(value, np.ndarray):
        points = value.astype(np.float32, copy=False)
    else:
        points = np.asarray(
            [
                [p["x"], p["y"], p.get("z", 0.0)]
                if isinstance(p, Mapping)
                else p
                for p in value
            ],
            dtype=np.float32,
        )
    if points.shape != (21, 3) or not np.all(np.isfinite(points)):
        raise InvalidSequenceError(f"landmarks must be finite [21,3], got {points.shape}")
    return points


def _resample(seq: np.ndarray, times: np.ndarray) -> np.ndarray:
    if len(seq) == 1:
        return np.repeat(seq, T_OUT, axis=0).astype(np.float32)
    destination = np.linspace(float(times[0]), float(times[-1]), T_OUT)
    flat = seq.reshape(len(seq), -1)
    output = np.stack(
        [np.interp(destination, times, flat[:, column]) for column in range(flat.shape[1])],
        axis=1,
    )
    return output.astype(np.float32).reshape(T_OUT, *seq.shape[1:])


def _velocity(seq: np.ndarray, times: np.ndarray) -> np.ndarray:
    if len(seq) <= 1:
        return np.zeros_like(seq, dtype=np.float32)
    flat = seq.reshape(len(seq), -1)
    output = np.empty_like(flat, dtype=np.float32)
    for column in range(flat.shape[1]):
        output[:, column] = np.gradient(
            flat[:, column], times, edge_order=1
        ).astype(np.float32)
    return output.reshape(seq.shape)


def _payload_parts(payload: Mapping[str, Any]):
    frames = payload.get("frames")
    if not isinstance(frames, Sequence) or isinstance(frames, (str, bytes)):
        raise InvalidSequenceError("payload.frames must be a sequence")

    video = payload.get("video") or {}
    width = video.get("width", payload.get("width", payload.get("imageWidth")))
    height = video.get("height", payload.get("height", payload.get("imageHeight")))
    fps = video.get("fps", payload.get("fps", 30.0))
    total_frames = video.get("total_frames", payload.get("totalFrames"))

    if width is None or height is None or float(width) <= 0 or float(height) <= 0:
        raise InvalidSequenceError("positive camera width and height are required")
    fps = float(fps)
    if not np.isfinite(fps) or fps <= 1e-6:
        fps = 30.0

    detected: list[tuple[int, float, np.ndarray, str | None]] = []
    max_index = -1
    for fallback_index, frame in enumerate(frames):
        if not isinstance(frame, Mapping):
            raise InvalidSequenceError("each frame must be an object")
        frame_index = int(frame.get("frame_index", frame.get("frameIndex", fallback_index)))
        max_index = max(max_index, frame_index)
        value = frame.get("landmarks", frame.get("points", frame.get("hand")))
        if value is None or frame.get("valid", True) is False:
            continue
        timestamp = frame.get("tMs")
        if timestamp is None:
            timestamp = frame_index * 1000.0 / fps
        side = frame.get("handedness", payload.get("majority_handedness"))
        detected.append((frame_index, float(timestamp), _points(value), side))

    if len(detected) < MIN_VALID_FRAMES:
        raise InvalidSequenceError(
            f"at least {MIN_VALID_FRAMES} valid hand frames are required; got {len(detected)}"
        )
    detected.sort(key=lambda item: item[1])
    times = np.asarray([item[1] for item in detected], dtype=np.float64) / 1000.0
    times -= times[0]
    if not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 1e-8):
        raise InvalidSequenceError("detected timestamps must be strictly increasing")

    if total_frames is None:
        total_frames = max_index + 1
    total_frames = int(total_frames)
    if total_frames <= 0:
        raise InvalidSequenceError("totalFrames must be positive")
    return detected, times, float(width), float(height), fps, total_frames


def build_hand_features(payload: Mapping[str, Any]) -> tuple[np.ndarray, float]:
    """Convert mobile landmark payload to ``features[32,127], duration_sec``.

    The payload may use the JSON schema produced by
    ``extract_mobile_landmarks.py`` or the equivalent app schema.
    Undetected frames may be omitted, but every detected frame must retain its
    original frame index and timestamp.
    """

    detected, times, width, height, fps, total_frames = _payload_parts(payload)
    hand = np.stack([item[2] for item in detected]).astype(np.float32)

    ratio = width / height
    hand[..., 0] *= ratio
    hand[..., 2] *= ratio
    hand -= hand[:, WRIST : WRIST + 1, :]
    span = np.linalg.norm(hand[:, MID_MCP, :2], axis=1)
    hand /= np.maximum(span, 1e-6)[:, None, None]

    sides = [str(item[3]).strip().lower() for item in detected if item[3] is not None]
    majority = str(payload.get("majority_handedness", "")).strip().lower()
    if not majority and sides:
        majority = Counter(sides).most_common(1)[0][0]
    if majority in {"left", "l"}:
        hand[..., 0] *= -1.0

    velocity = _velocity(hand, times)
    hand32 = _resample(hand, times).reshape(T_OUT, 63)
    velocity32 = _resample(velocity, times).reshape(T_OUT, 63)

    full_valid = np.zeros((total_frames, 1), dtype=np.float32)
    for frame_index, _, _, _ in detected:
        if 0 <= frame_index < total_frames:
            full_valid[frame_index, 0] = 1.0
    full_times = np.arange(total_frames, dtype=np.float64) / fps
    valid32 = _resample(full_valid, full_times).reshape(T_OUT, 1)

    features = np.concatenate([hand32, velocity32, valid32], axis=1).astype(np.float32)

    # Same collapse-aware stabilization used by the final merged NPZ.
    collapse = (
        np.any(np.abs(features[:, :63]) > POSITION_CLIP, axis=1)
        & np.any(np.abs(features[:, 63:126]) > VELOCITY_CLIP, axis=1)
    )
    features[collapse, :63] = np.clip(
        features[collapse, :63], -POSITION_CLIP, POSITION_CLIP
    )
    features[collapse, 63:126] = np.clip(
        features[collapse, 63:126], -VELOCITY_CLIP, VELOCITY_CLIP
    )

    if features.shape != (T_OUT, FEATURE_DIM) or not np.all(np.isfinite(features)):
        raise InvalidSequenceError("feature generation produced an invalid tensor")
    duration_sec = float(total_frames) / fps
    return features, duration_sec
