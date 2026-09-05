from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SequenceAugmenter:
    """Lightweight train-only augmentation for standardized WITECK sequences."""

    noise_std: float = 0.02
    frame_mask_probability: float = 0.35
    max_mask_frames: int = 4
    temporal_crop_probability: float = 0.50
    min_crop_ratio: float = 0.85
    max_shift_frames: int = 2
    valid_mask_index: int | None = 168
    invalid_mask_value: float = 0.0

    def __post_init__(self) -> None:
        if self.noise_std < 0.0:
            raise ValueError("noise_std must be non-negative")
        for name, value in (
            ("frame_mask_probability", self.frame_mask_probability),
            ("temporal_crop_probability", self.temporal_crop_probability),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.max_mask_frames < 0 or self.max_shift_frames < 0:
            raise ValueError("mask and shift sizes must be non-negative")
        if not 0.0 < self.min_crop_ratio <= 1.0:
            raise ValueError("min_crop_ratio must be in (0, 1]")

    def __call__(
        self,
        sequence: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        out = np.asarray(sequence, dtype=np.float32).copy()
        if out.ndim != 2:
            raise ValueError(f"sequence must be [T,D], received {out.shape}")
        time_steps, feature_count = out.shape
        if time_steps < 2:
            return out

        mask_index = self.valid_mask_index
        if mask_index is not None and not 0 <= mask_index < feature_count:
            mask_index = None

        if (
            self.temporal_crop_probability > 0.0
            and rng.random() < self.temporal_crop_probability
        ):
            ratio = rng.uniform(self.min_crop_ratio, 1.0)
            crop_steps = max(2, int(round(time_steps * ratio)))
            crop_steps = min(crop_steps, time_steps)
            start = int(rng.integers(time_steps - crop_steps + 1))
            source = out[start : start + crop_steps]
            positions = np.linspace(
                0.0,
                crop_steps - 1,
                time_steps,
                dtype=np.float32,
            )
            lower = np.floor(positions).astype(np.int64)
            upper = np.minimum(lower + 1, crop_steps - 1)
            weight = (positions - lower).reshape(-1, 1)
            out = (
                source[lower] * (1.0 - weight)
                + source[upper] * weight
            ).astype(np.float32)

        if self.max_shift_frames > 0:
            shift = int(
                rng.integers(
                    -self.max_shift_frames,
                    self.max_shift_frames + 1,
                )
            )
            if shift:
                out = np.roll(out, shift, axis=0)
                exposed = slice(0, shift) if shift > 0 else slice(shift, None)
                out[exposed] = 0.0
                if mask_index is not None:
                    out[exposed, mask_index] = self.invalid_mask_value

        continuous = np.ones(feature_count, dtype=bool)
        if mask_index is not None:
            continuous[mask_index] = False
        if self.noise_std > 0.0:
            out[:, continuous] += rng.normal(
                0.0,
                self.noise_std,
                size=(time_steps, int(continuous.sum())),
            ).astype(np.float32)

        if (
            self.max_mask_frames > 0
            and rng.random() < self.frame_mask_probability
        ):
            length = int(rng.integers(1, min(self.max_mask_frames, time_steps) + 1))
            start = int(rng.integers(time_steps - length + 1))
            out[start : start + length, continuous] = 0.0
            if mask_index is not None:
                out[start : start + length, mask_index] = self.invalid_mask_value

        return out.astype(np.float32, copy=False)
