# WITECK Shared Dual-Head AI release v1.1.0

This package preserves the v1.0.0 backend contract where possible while replacing
the old classifier/authentication stack with the Shared Dual-Head encoder.

## Public API

```python
from ai_release import (
    MODEL_VERSION,
    InvalidSequenceError,
    load_model,
    embed_user,
    embed_gesture,
    embed_user_batch,
    embed_gesture_batch,
    embed_both,
    embed_both_batch,
)
```

### Input

Pass the same raw MediaPipe landmark payload used by v1.0.0.

`features.py` performs:
- raw landmark parsing
- minimum frame / valid-frame checks
- 750 ms minimum duration check
- strictly increasing timestamp validation
- camera width/height validation
- right-hand validation
- interpolation of missing detections
- aspect correction
- wrist-relative coordinates
- hand-size normalization
- velocity calculation
- resampling to `[32,127]`

Malformed captures raise `InvalidSequenceError`.

### Embeddings

- Gesture embedding: 128-D float32, L2-normalized
- User embedding: 128-D float32, L2-normalized

For best throughput, `embed_both()` / `embed_both_batch()` computes both heads in one
forward pass.

### Threading contract

The release contains an internal `threading.RLock` around model loading and forward
passes. The backend should not add another model-forward lock unless it has a separate
application-level reason to do so.

### Matching ownership

The AI module returns embeddings. The backend owns:
- template averaging/storage
- cosine similarity
- threshold selection
- final authentication decision

For the current 1-user / 1-personal-gesture policy, the app does not need to send a
`gestureId` during authentication. `userId` is sufficient for the backend to retrieve
that user's one gesture template and one user template.

### Default decision

- Tg = 0.902032
- Tu = 0.342350
- accept iff both thresholds pass

See `calibration_report.md` for P08-P10 results and limitations.

## Weight installation

The template ZIP intentionally does not contain the new model weight because it exists
only on the training PC. Run `finalize_release.py` next to this folder and point it at:

`output\shared_dual_head\seed_40\shared_dual_head.pt`

The finalizer copies the weight, extracts the exact normalization statistics from the
checkpoint, updates thresholds, writes SHA-256 manifest entries, runs static validation,
and creates the final backend ZIP.
