"""Runtime feature builder matching the hand-only training preprocessing.

Public input is MediaPipe Hand landmarks (21 xyz points) plus per-frame ``tMs``.
The output is always a float32 array with shape ``[32, 127]`` and duration.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


T_OUT = 32
FEATURE_DIM = 127
MIN_INPUT_FRAMES = 8
MIN_VALID_FRAMES = 8
MIN_DURATION_MS = 750.0
WRIST = 0
MID_MCP = 9


class InvalidSequenceError(ValueError):
    """Raised when a capture is too short or malformed for reliable inference."""


def _points_array(value: Any) -> np.ndarray:
    if value is None:
        raise InvalidSequenceError("landmarks are missing")
    if isinstance(value, np.ndarray):
        points = value.astype(np.float32, copy=False)
    else:
        points = np.asarray(
            [[p["x"], p["y"], p.get("z", 0.0)] if isinstance(p, Mapping) else p for p in value],
            dtype=np.float32,
        )
    if points.shape != (21, 3) or not np.all(np.isfinite(points)):
        raise InvalidSequenceError(f"landmarks must be finite [21,3], got {points.shape}")
    return points


def _unpack_input(payload: Any):
    if isinstance(payload, Mapping):
        frames = payload.get("frames")
        width = payload.get("width") or payload.get("imageWidth")
        height = payload.get("height") or payload.get("imageHeight")
        default_handedness = payload.get("handedness")
    else:
        frames = payload
        width = height = default_handedness = None
    if not isinstance(frames, Sequence) or isinstance(frames, (str, bytes)):
        raise InvalidSequenceError("frames must be a sequence")
    return frames, width, height, default_handedness


def _parse_frames(payload: Any, require_right_hand: bool):
    frames, width, height, default_handedness = _unpack_input(payload)
    if len(frames) < MIN_INPUT_FRAMES:
        raise InvalidSequenceError(
            f"at least {MIN_INPUT_FRAMES} captured frames are required; got {len(frames)}"
        )

    xyz = np.zeros((len(frames), 21, 3), dtype=np.float32)
    valid = np.zeros(len(frames), dtype=np.uint8)
    times_ms = np.empty(len(frames), dtype=np.float64)
    handedness_values: list[str] = []

    for index, frame in enumerate(frames):
        if not isinstance(frame, Mapping):
            raise InvalidSequenceError("each frame must be an object containing tMs and landmarks")
        if "tMs" not in frame:
            raise InvalidSequenceError(f"frame {index} has no tMs")
        times_ms[index] = float(frame["tMs"])
        frame_width = frame.get("width") or frame.get("imageWidth")
        frame_height = frame.get("height") or frame.get("imageHeight")
        width = width or frame_width
        height = height or frame_height
        side = frame.get("handedness", default_handedness)
        if side is not None:
            handedness_values.append(str(side).strip().lower())

        points = frame.get("landmarks", frame.get("points", frame.get("hand")))
        if points is None or frame.get("valid", True) is False:
            continue
        try:
            xyz[index] = _points_array(points)
            valid[index] = 1
        except InvalidSequenceError:
            if frame.get("valid") is True:
                raise

    if not np.all(np.isfinite(times_ms)) or np.any(np.diff(times_ms) <= 0):
        raise InvalidSequenceError("tMs must be finite and strictly increasing")
    if int(valid.sum()) < MIN_VALID_FRAMES:
        raise InvalidSequenceError(
            f"at least {MIN_VALID_FRAMES} valid hand frames are required; got {int(valid.sum())}"
        )
    duration_ms = float(times_ms[-1] - times_ms[0])
    if duration_ms < MIN_DURATION_MS:
        raise InvalidSequenceError(
            f"capture duration must be at least {MIN_DURATION_MS:.0f} ms; got {duration_ms:.1f} ms"
        )
    if width is None or height is None or float(width) <= 0 or float(height) <= 0:
        raise InvalidSequenceError("positive camera width and height are required for aspect correction")

    if require_right_hand:
        known = [x for x in handedness_values if x in {"left", "right", "l", "r"}]
        if any(x in {"left", "l"} for x in known):
            raise InvalidSequenceError("this model version accepts right-hand captures only")
        # If the app cannot produce handedness, the UI/API must already enforce right hand.

    return xyz, valid, times_ms, float(width), float(height)


def _interpolate_missing(seq: np.ndarray, valid: np.ndarray, times: np.ndarray) -> np.ndarray:
    indices = np.where(valid > 0)[0]
    flat = seq.reshape(len(seq), -1).copy()
    if len(indices) != len(seq):
        for column in range(flat.shape[1]):
            flat[:, column] = np.interp(times, times[indices], flat[indices, column])
    return flat.reshape(seq.shape).astype(np.float32)


def _resample(seq: np.ndarray, times: np.ndarray) -> np.ndarray:
    destination = np.linspace(float(times[0]), float(times[-1]), T_OUT)
    flat = seq.reshape(len(seq), -1)
    output = np.stack(
        [np.interp(destination, times, flat[:, column]) for column in range(flat.shape[1])],
        axis=1,
    )
    return output.astype(np.float32).reshape(T_OUT, *seq.shape[1:])


def _velocity(seq: np.ndarray, times_seconds: np.ndarray) -> np.ndarray:
    flat = seq.reshape(len(seq), -1)
    result = np.empty_like(flat, dtype=np.float32)
    for column in range(flat.shape[1]):
        result[:, column] = np.gradient(
            flat[:, column], times_seconds, edge_order=1
        ).astype(np.float32)
    return result.reshape(seq.shape)


def build_hand_features(payload: Any, *, require_right_hand: bool = True):
    """Convert raw frames to ``(features[32,127], duration_sec)``.

    Captures with 8--31 valid frames are linearly resampled to 32. A capture is
    rejected when it has fewer than eight valid frames, lasts under 750 ms, or
    omits the camera width/height required by the training-time aspect correction.
    """

    hand, valid, times_ms, width, height = _parse_frames(payload, require_right_hand)
    times_seconds = ((times_ms - times_ms[0]) / 1000.0).astype(np.float32)

    hand[..., 0] *= width / height
    hand[..., 2] *= width / height
    hand = _interpolate_missing(hand, valid, times_seconds)
    hand -= hand[:, WRIST : WRIST + 1, :]
    span = np.linalg.norm(hand[:, MID_MCP, :2], axis=1)
    hand /= np.maximum(span, 1e-6)[:, None, None]

    velocity = _velocity(hand, times_seconds)
    hand32 = _resample(hand, times_seconds).reshape(T_OUT, 63)
    velocity32 = _resample(velocity, times_seconds).reshape(T_OUT, 63)
    valid32 = _resample(valid.astype(np.float32)[:, None], times_seconds)
    features = np.concatenate([hand32, velocity32, valid32], axis=1).astype(np.float32)
    if features.shape != (T_OUT, FEATURE_DIM) or not np.all(np.isfinite(features)):
        raise InvalidSequenceError("feature generation produced an invalid tensor")

    median_step = float(np.median(np.diff(times_seconds)))
    duration_sec = float(times_seconds[-1] + median_step)
    return features, duration_sec
