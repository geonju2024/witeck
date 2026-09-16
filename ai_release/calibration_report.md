# WITECK Shared Dual-Head deployment calibration report

## Threshold provenance

The selected default thresholds were calibrated on the known-user validation split only.

- Gesture threshold `Tg`: **0.902032**
- User threshold `Tu`: **0.342350**
- Final decision: `gesture_score >= Tg AND user_score >= Tu`

The validation Gesture EER of 0% must **not** be presented as unseen-free-gesture performance.
The current gesture set is still the existing G1-G5 set.

## P08-P10 post-selection evaluation

Target users: P08, P09, P10  
Enrollment: 3 takes per registered gesture  
Trials: 7,275

| Metric | Result |
|---|---:|
| Accuracy | 95.77% |
| Balanced Accuracy | 84.37% |
| Genuine FRR | 28.04% |
| Combined Attack FAR | 3.22% |
| Wrong-Gesture FAR | 1.52% |
| Same-Gesture Impostor FAR | 17.34% |
| Random Impostor FAR | 0.13% |

### Interpretation

- The Dual-Head structure substantially reduced wrong-gesture acceptance.
- The current main bottlenecks are genuine rejection and same-gesture impostors.
- P08-P10 are unseen users, but their gestures are still from G1-G5.
- Therefore these numbers are **Unseen User + Known Gesture** results, not
  **Unseen User + Unseen Free Gesture** results.

## Relaxed demo operating point

A relaxed demo operating point was calibrated using known-user validation data only.

- Gesture threshold Tg: 0.902032
- User threshold Tu: 0.293309

Validation results:

- Gesture FAR: 0.00%
- Gesture FRR: 0.00%
- User FAR: 4.76%
- User FRR: 2.86%

P08-P10 and other final-test users were not used to derive this operating point.

This operating point is intended as a more permissive demo setting to reduce genuine rejection.
It should not be interpreted as unseen-user or unseen-free-gesture performance.