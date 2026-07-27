import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.models import load_model

from gesture_auth_config import (
    DEFAULT_AUTH_THRESHOLD,
    FEATURE_SIZE,
    METADATA_PATH,
    MODEL_PATH,
    SEQUENCE_LENGTH,
    THRESHOLD_PATH,
)


def load_dataset():
    df = pd.read_csv(METADATA_PATH)
    required = {"file_path", "user_id", "auth_label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"metadata.csv missing required columns: {sorted(missing)}")

    samples = []
    labels = []
    rows = []

    for _, row in df.iterrows():
        file_path = str(row["file_path"]).replace("\\", os.sep).replace("/", os.sep)
        if not os.path.exists(file_path):
            print(f"Skipping missing file: {file_path}")
            continue

        data = np.load(file_path)
        if data.shape != (SEQUENCE_LENGTH, FEATURE_SIZE):
            print(f"Skipping invalid shape: {file_path}, shape={data.shape}")
            continue

        auth_label = str(row["auth_label"]).strip().lower()
        if auth_label not in {"success", "fail"}:
            print(f"Skipping invalid auth_label: {auth_label}")
            continue

        samples.append(data)
        labels.append(1 if auth_label == "success" else 0)
        rows.append(row.to_dict())

    if not samples:
        raise ValueError("No valid samples found.")

    return np.asarray(samples, dtype=np.float32), np.asarray(labels, dtype=np.int32), pd.DataFrame(rows)


def find_best_threshold(y_true, probabilities):
    candidates = sorted(set(float(p) for p in probabilities))
    thresholds = [0.0, 1.0]
    thresholds.extend(candidates)
    thresholds.extend((a + b) / 2 for a, b in zip(candidates, candidates[1:]))

    best = None
    for threshold in thresholds:
        y_pred = (probabilities >= threshold).astype(np.int32)
        acc = accuracy_score(y_true, y_pred)
        key = (acc, threshold)
        if best is None or key > best[0]:
            best = (key, threshold, y_pred)

    return best[1], best[2], best[0][0]


def evaluate_at_threshold(y_true, probabilities, threshold):
    y_pred = (probabilities >= threshold).astype(np.int32)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "confusion_matrix_labels": ["fail", "success"],
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["fail", "success"],
            zero_division=0,
            output_dict=True,
        ),
    }


def main():
    x, y, df = load_dataset()
    model = load_model(MODEL_PATH)
    probabilities = model.predict(x, verbose=0).reshape(-1)

    best_threshold, _, best_accuracy = find_best_threshold(y, probabilities)
    default_report = evaluate_at_threshold(y, probabilities, DEFAULT_AUTH_THRESHOLD)
    best_report = evaluate_at_threshold(y, probabilities, best_threshold)

    user_count = int(df["user_id"].nunique())
    report = {
        "sample_count": int(len(y)),
        "user_count": user_count,
        "success_count": int(np.sum(y == 1)),
        "fail_count": int(np.sum(y == 0)),
        "default_threshold_report": default_report,
        "best_threshold_report": best_report,
        "person_recognition_note": (
            "Only one user_id is present, so this dataset can verify gesture authentication "
            "accuracy but cannot prove multi-person recognition accuracy."
        ),
    }

    with open(THRESHOLD_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "threshold": float(DEFAULT_AUTH_THRESHOLD),
                "best_threshold_on_current_dataset": float(best_threshold),
                "best_accuracy_on_current_dataset": float(best_accuracy),
                "sample_count": int(len(y)),
            },
            f,
            indent=2,
        )

    print(f"Samples: {len(y)}")
    print(f"Users: {user_count}")
    print(f"Default threshold: {DEFAULT_AUTH_THRESHOLD:.4f}")
    print(f"Default accuracy: {default_report['accuracy']:.4f}")
    print(f"Best threshold on current dataset: {best_threshold:.4f}")
    print(f"Best accuracy on current dataset: {best_accuracy:.4f}")
    print("Confusion matrix at default threshold [fail, success]:")
    print(np.asarray(default_report["confusion_matrix"]))
    if user_count < 2:
        print(report["person_recognition_note"])

    with open("gesture_auth_evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
