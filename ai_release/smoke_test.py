"""Offline smoke test for raw preprocessing, validation, model load and embeddings."""

from __future__ import annotations

import numpy as np

try:
    from .encoder import (
        embed_both,
        embed_both_batch,
        embed_gesture,
        embed_user,
        load_model,
    )
    from .features import InvalidSequenceError
except ImportError:
    from encoder import embed_both, embed_both_batch, embed_gesture, embed_user, load_model
    from features import InvalidSequenceError


def sample_payload():
    frames = []
    for frame_index in range(12):
        points = []
        for landmark_index in range(21):
            points.append(
                {
                    "x": 0.45 + 0.006 * landmark_index + 0.001 * frame_index,
                    "y": 0.55 - 0.004 * landmark_index,
                    "z": -0.002 * landmark_index,
                }
            )
        frames.append({"tMs": frame_index * 75.0, "landmarks": points})
    return {
        "width": 1920,
        "height": 1080,
        "handedness": "Right",
        "frames": frames,
    }


if __name__ == "__main__":
    payload = sample_payload()
    print(load_model())

    u = embed_user(payload)
    g = embed_gesture(payload)
    both = embed_both(payload)
    batch = embed_both_batch([payload, payload])

    assert u.shape == (128,)
    assert g.shape == (128,)
    assert batch["user_embeddings"].shape == (2, 128)
    assert batch["gesture_embeddings"].shape == (2, 128)
    assert np.isclose(np.linalg.norm(u), 1.0, atol=1e-5)
    assert np.isclose(np.linalg.norm(g), 1.0, atol=1e-5)
    assert np.allclose(u, both["user_embedding"])
    assert np.allclose(g, both["gesture_embedding"])

    # Input validation regression check.
    bad = dict(payload)
    bad["frames"] = payload["frames"][:4]
    try:
        embed_user(bad)
        raise AssertionError("InvalidSequenceError was not raised")
    except InvalidSequenceError:
        pass

    print("user_embedding_norm", float(np.linalg.norm(u)))
    print("gesture_embedding_norm", float(np.linalg.norm(g)))
    print("input_validation", "passed")
    print("smoke test passed")
