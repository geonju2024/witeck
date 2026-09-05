from __future__ import annotations

from typing import Any

import numpy as np


SOURCE_MISSING = 0
SOURCE_TRACKING = 1
SOURCE_STATIC_FULL = 2
SOURCE_STATIC_ENHANCED = 3
SOURCE_POSE_CROP = 4
SOURCE_POSE_CROP_ENHANCED = 5

SOURCE_LABELS = np.array(
    [
        "missing",
        "tracking",
        "static_full",
        "static_enhanced",
        "pose_crop",
        "pose_crop_enhanced",
    ]
)


def enhance_bgr(frame: np.ndarray) -> np.ndarray:
    """Increase local luminance contrast without changing image geometry."""
    import cv2

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    luminance, channel_a, channel_b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = cv2.merge((clahe.apply(luminance), channel_a, channel_b))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


def pose_hand_boxes(
    pose_landmarks: Any,
    width: int,
    height: int,
    crop_scale: float = 2.5,
    min_visibility: float = 0.3,
) -> list[tuple[int, int, int, int]]:
    """Build square hand ROIs from MediaPipe elbow/wrist pose landmarks."""
    if pose_landmarks is None or width <= 0 or height <= 0:
        return []

    points = pose_landmarks.landmark
    boxes = []
    # MediaPipe Pose: left elbow/wrist=13/15, right elbow/wrist=14/16.
    for elbow_index, wrist_index in ((13, 15), (14, 16)):
        elbow = points[elbow_index]
        wrist = points[wrist_index]
        if min(float(elbow.visibility), float(wrist.visibility)) < min_visibility:
            continue

        elbow_xy = np.array([elbow.x * width, elbow.y * height], dtype=np.float32)
        wrist_xy = np.array([wrist.x * width, wrist.y * height], dtype=np.float32)
        forearm = float(np.linalg.norm(wrist_xy - elbow_xy))
        side = max(0.28 * min(width, height), crop_scale * forearm)
        side = min(side, 0.70 * max(width, height))

        # Shift slightly beyond the wrist toward the hand.
        center = wrist_xy + 0.20 * (wrist_xy - elbow_xy)
        half = side / 2.0
        x0 = max(0, int(round(center[0] - half)))
        y0 = max(0, int(round(center[1] - half)))
        x1 = min(width, int(round(center[0] + half)))
        y1 = min(height, int(round(center[1] + half)))
        if x1 - x0 >= 32 and y1 - y0 >= 32:
            boxes.append((x0, y0, x1, y1))
    return boxes


def candidates_from_result(
    result: Any,
    source: int,
    frame_width: int,
    frame_height: int,
    box: tuple[int, int, int, int] | None = None,
) -> list[dict[str, Any]]:
    """Convert a MediaPipe Hands result to full-frame normalized candidates."""
    if result is None or not result.multi_hand_landmarks:
        return []

    candidates = []
    for index, hand_landmarks in enumerate(result.multi_hand_landmarks):
        points = np.asarray(
            [[point.x, point.y, point.z] for point in hand_landmarks.landmark],
            dtype=np.float32,
        )
        if box is not None:
            x0, y0, x1, y1 = box
            crop_width = x1 - x0
            crop_height = y1 - y0
            points[:, 0] = (x0 + points[:, 0] * crop_width) / frame_width
            points[:, 1] = (y0 + points[:, 1] * crop_height) / frame_height
            points[:, 2] *= crop_width / frame_width

        side = -1
        score = 0.0
        if result.multi_handedness and index < len(result.multi_handedness):
            classification = result.multi_handedness[index].classification[0]
            side = 1 if classification.label == "Right" else 0
            score = float(classification.score)

        candidates.append(
            {
                "landmarks": points,
                "handedness": side,
                "score": score,
                "source": int(source),
            }
        )
    return candidates


def deduplicate_candidates(
    candidates: list[dict[str, Any]],
    max_hands: int = 2,
    wrist_distance: float = 0.08,
) -> list[dict[str, Any]]:
    """Keep the highest-confidence spatially distinct hand detections."""
    selected: list[dict[str, Any]] = []
    for candidate in sorted(candidates, key=lambda item: item["score"], reverse=True):
        wrist = candidate["landmarks"][0, :2]
        duplicate = False
        for kept in selected:
            kept_wrist = kept["landmarks"][0, :2]
            sides_compatible = (
                candidate["handedness"] < 0
                or kept["handedness"] < 0
                or candidate["handedness"] == kept["handedness"]
            )
            if sides_compatible and np.linalg.norm(wrist - kept_wrist) < wrist_distance:
                duplicate = True
                break
        if not duplicate:
            selected.append(candidate)
        if len(selected) == max_hands:
            break
    return selected


def recover_missing_hands(
    frame: np.ndarray,
    rgb: np.ndarray,
    pose_landmarks: Any,
    detector: Any,
    frame_width: int,
    frame_height: int,
    pose_crop_scale: float = 2.5,
) -> list[dict[str, Any]]:
    """Retry hand detection from cheap/global to focused/pose-guided inputs."""
    import cv2

    candidates = candidates_from_result(
        detector.process(rgb),
        SOURCE_STATIC_FULL,
        frame_width,
        frame_height,
    )
    if candidates:
        return deduplicate_candidates(candidates)

    enhanced_rgb = cv2.cvtColor(enhance_bgr(frame), cv2.COLOR_BGR2RGB)
    enhanced_rgb.flags.writeable = False
    candidates = candidates_from_result(
        detector.process(enhanced_rgb),
        SOURCE_STATIC_ENHANCED,
        frame_width,
        frame_height,
    )
    if candidates:
        return deduplicate_candidates(candidates)

    crop_candidates = []
    for box in pose_hand_boxes(
        pose_landmarks,
        frame_width,
        frame_height,
        crop_scale=pose_crop_scale,
    ):
        x0, y0, x1, y1 = box
        crop = frame[y0:y1, x0:x1]
        if crop.size == 0:
            continue

        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop_rgb.flags.writeable = False
        local = candidates_from_result(
            detector.process(crop_rgb),
            SOURCE_POSE_CROP,
            frame_width,
            frame_height,
            box=box,
        )
        if not local:
            enhanced_crop_rgb = cv2.cvtColor(
                enhance_bgr(crop),
                cv2.COLOR_BGR2RGB,
            )
            enhanced_crop_rgb.flags.writeable = False
            local = candidates_from_result(
                detector.process(enhanced_crop_rgb),
                SOURCE_POSE_CROP_ENHANCED,
                frame_width,
                frame_height,
                box=box,
            )
        crop_candidates.extend(local)

    return deduplicate_candidates(crop_candidates)
