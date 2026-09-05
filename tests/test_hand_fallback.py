from types import SimpleNamespace

import numpy as np

from witeck_auth.hand_fallback import deduplicate_candidates, pose_hand_boxes


def _point(x, y, visibility=1.0):
    return SimpleNamespace(x=x, y=y, visibility=visibility)


def test_pose_hand_boxes_are_valid_and_clamped():
    points = [_point(0.5, 0.5) for _ in range(33)]
    points[13] = _point(0.20, 0.50)
    points[15] = _point(0.05, 0.50)
    points[14] = _point(0.80, 0.50)
    points[16] = _point(0.95, 0.50)
    boxes = pose_hand_boxes(SimpleNamespace(landmark=points), 1000, 600)
    assert len(boxes) == 2
    for x0, y0, x1, y1 in boxes:
        assert 0 <= x0 < x1 <= 1000
        assert 0 <= y0 < y1 <= 600


def test_low_visibility_pose_is_not_used_for_crop():
    points = [_point(0.5, 0.5) for _ in range(33)]
    points[13] = _point(0.4, 0.5, visibility=0.1)
    points[15] = _point(0.5, 0.5)
    points[14] = _point(0.6, 0.5, visibility=0.1)
    points[16] = _point(0.5, 0.5)
    assert pose_hand_boxes(SimpleNamespace(landmark=points), 640, 480) == []


def test_candidate_deduplication_keeps_best_score():
    def candidate(wrist_x, side, score):
        landmarks = np.zeros((21, 3), dtype=np.float32)
        landmarks[0, 0] = wrist_x
        return {
            "landmarks": landmarks,
            "handedness": side,
            "score": score,
            "source": 4,
        }

    selected = deduplicate_candidates(
        [candidate(0.10, 1, 0.6), candidate(0.12, 1, 0.9), candidate(0.80, 0, 0.7)]
    )
    assert len(selected) == 2
    assert [item["score"] for item in selected] == [0.9, 0.7]
