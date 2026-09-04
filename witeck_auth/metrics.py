from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score


def verification_metrics(labels, scores, threshold: float | None = None) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1.0 - tpr
    eer_index = int(np.nanargmin(np.abs(fpr - fnr)))
    eer_threshold = float(thresholds[eer_index])
    selected = eer_threshold if threshold is None else float(threshold)
    predictions = scores >= selected
    far = float(np.mean(predictions[labels == 0]))
    frr = float(np.mean(~predictions[labels == 1]))
    return {
        "auc": float(roc_auc_score(labels, scores)),
        "eer": float((fpr[eer_index] + fnr[eer_index]) / 2.0),
        "eer_threshold": eer_threshold,
        "threshold": selected,
        "far": far,
        "frr": frr,
        "balanced_accuracy": float(1.0 - (far + frr) / 2.0),
    }
