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

Not included yet.

A second operating point must be selected from development/validation score distributions.
It must not be tuned using P08-P10 final-test results, because doing so would contaminate
the unseen-user evaluation.
