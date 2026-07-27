import json
import os

import numpy as np
import pandas as pd

from gesture_auth_config import (
    FEATURE_SIZE,
    METADATA_PATH,
    MIN_PERSON_SAMPLES_PER_USER,
    MIN_PERSON_USERS,
    PERSON_DATASET_AUDIT_PATH,
    RECOMMENDED_PERSON_SAMPLES_PER_USER,
    SEQUENCE_LENGTH,
)


def audit_dataset():
    if not os.path.exists(METADATA_PATH):
        return {
            "ready_for_person_recognition": False,
            "reason": f"Missing metadata file: {METADATA_PATH}",
        }

    df = pd.read_csv(METADATA_PATH)
    required = {"file_path", "user_id", "gesture_label", "sequence_length", "feature_size"}
    missing = sorted(required - set(df.columns))
    if missing:
        return {
            "ready_for_person_recognition": False,
            "reason": f"metadata.csv missing required columns: {missing}",
        }

    valid_rows = []
    invalid_rows = []
    for index, row in df.iterrows():
        file_path = str(row["file_path"]).replace("\\", os.sep).replace("/", os.sep)
        user_id = str(row["user_id"]).strip()
        issue = None

        if not user_id:
            issue = "empty user_id"
        elif not os.path.exists(file_path):
            issue = "missing file"
        else:
            try:
                data = np.load(file_path)
                if data.shape != (SEQUENCE_LENGTH, FEATURE_SIZE):
                    issue = f"invalid shape {data.shape}"
            except Exception as exc:
                issue = f"could not load npy: {exc}"

        record = {
            "row": int(index),
            "file_path": str(row["file_path"]),
            "user_id": user_id,
            "gesture_label": str(row["gesture_label"]),
        }
        if issue:
            record["issue"] = issue
            invalid_rows.append(record)
        else:
            valid_rows.append(record)

    valid_df = pd.DataFrame(valid_rows)
    if valid_df.empty:
        user_counts = {}
        gesture_counts = {}
    else:
        user_counts = valid_df["user_id"].value_counts().sort_index().to_dict()
        gesture_counts = (
            valid_df.groupby(["user_id", "gesture_label"]).size().sort_index().astype(int).to_dict()
        )
        gesture_counts = {f"{user}|{gesture}": count for (user, gesture), count in gesture_counts.items()}

    users_with_too_few_samples = {
        user: int(count)
        for user, count in user_counts.items()
        if count < MIN_PERSON_SAMPLES_PER_USER
    }
    users_below_recommended_samples = {
        user: int(count)
        for user, count in user_counts.items()
        if count < RECOMMENDED_PERSON_SAMPLES_PER_USER
    }

    ready = (
        len(user_counts) >= MIN_PERSON_USERS
        and not users_with_too_few_samples
        and len(valid_rows) > 0
    )

    if len(user_counts) < MIN_PERSON_USERS:
        reason = f"Need at least {MIN_PERSON_USERS} users; found {len(user_counts)}."
    elif users_with_too_few_samples:
        reason = (
            f"Each user needs at least {MIN_PERSON_SAMPLES_PER_USER} valid samples; "
            f"too few: {users_with_too_few_samples}."
        )
    else:
        reason = "Ready for person recognition training/evaluation."

    return {
        "ready_for_person_recognition": ready,
        "reason": reason,
        "valid_sample_count": len(valid_rows),
        "invalid_sample_count": len(invalid_rows),
        "user_count": len(user_counts),
        "samples_per_user": {user: int(count) for user, count in user_counts.items()},
        "samples_per_user_gesture": gesture_counts,
        "invalid_rows": invalid_rows,
        "minimum_users": MIN_PERSON_USERS,
        "minimum_samples_per_user": MIN_PERSON_SAMPLES_PER_USER,
        "recommended_samples_per_user": RECOMMENDED_PERSON_SAMPLES_PER_USER,
        "users_below_recommended_samples": users_below_recommended_samples,
    }


def main():
    report = audit_dataset()
    with open(PERSON_DATASET_AUDIT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
