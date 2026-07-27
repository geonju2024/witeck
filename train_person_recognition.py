import argparse
import json
import math
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from gesture_auth_config import (
    FEATURE_SIZE,
    METADATA_PATH,
    MIN_PERSON_SAMPLES_PER_USER,
    MIN_PERSON_USERS,
    PERSON_REPEATED_EVAL_SPLITS,
    PERSON_TARGET_ACCURACY,
    PERSON_TEST_SIZE,
    RECOMMENDED_PERSON_SAMPLES_PER_USER,
    SEQUENCE_LENGTH,
)
from person_model_registry import (
    DEFAULT_PERSON_MODEL_TYPE,
    PERSON_MODEL_TYPES,
    build_keras_model,
    build_random_forest,
    extract_random_forest_features,
    is_keras_model,
    person_labels_path,
    person_model_path,
    person_report_path,
    predict_probabilities,
    save_person_model,
    validate_model_type,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train one person-recognition model at a time."
    )
    parser.add_argument(
        "--model",
        choices=PERSON_MODEL_TYPES,
        default=DEFAULT_PERSON_MODEL_TYPE,
        help=f"Model architecture to train (default: {DEFAULT_PERSON_MODEL_TYPE}).",
    )
    parser.add_argument(
        "--save-below-target",
        action="store_true",
        help="Save the final model even when repeated accuracy is below the configured target.",
    )
    return parser.parse_args()


def load_person_dataset():
    df = pd.read_csv(METADATA_PATH)
    required = {"file_path", "user_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"metadata.csv missing required columns: {sorted(missing)}")

    samples = []
    users = []

    for _, row in df.iterrows():
        user_id = str(row["user_id"]).strip()
        file_path = str(row["file_path"]).replace("\\", os.sep).replace("/", os.sep)
        if not user_id:
            continue
        if not os.path.exists(file_path):
            print(f"Skipping missing file: {file_path}")
            continue

        data = np.load(file_path)
        if data.shape != (SEQUENCE_LENGTH, FEATURE_SIZE):
            print(f"Skipping invalid shape: {file_path}, shape={data.shape}")
            continue

        samples.append(data)
        users.append(user_id)

    if not samples:
        raise ValueError("No valid samples found.")

    counts = pd.Series(users).value_counts().sort_index()
    if len(counts) < MIN_PERSON_USERS:
        raise ValueError(
            f"Person recognition needs at least {MIN_PERSON_USERS} users, but metadata has "
            f"{len(counts)}: {counts.to_dict()}"
        )

    too_small = counts[counts < MIN_PERSON_SAMPLES_PER_USER]
    if not too_small.empty:
        raise ValueError(
            f"Each user needs at least {MIN_PERSON_SAMPLES_PER_USER} samples for a basic split. "
            f"Too few: {too_small.to_dict()}"
        )

    return np.asarray(samples, dtype=np.float32), np.asarray(users), counts


def write_report(model_type, report):
    path = person_report_path(model_type)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return path


def build_model(model_type, class_count, random_state):
    if is_keras_model(model_type):
        return build_keras_model(
            model_type=model_type,
            sequence_length=SEQUENCE_LENGTH,
            feature_size=FEATURE_SIZE,
            class_count=class_count,
        )
    return build_random_forest(random_state=random_state)


def fit_model(model_type, model, x_train, y_train, x_test=None, y_test=None):
    if not is_keras_model(model_type):
        model.fit(extract_random_forest_features(x_train), y_train)
        return None

    from tensorflow.keras.callbacks import EarlyStopping

    fit_kwargs = {}
    callbacks = []
    if x_test is not None and y_test is not None:
        fit_kwargs["validation_data"] = (x_test, y_test)
        callbacks.append(
            EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
        )

    history = model.fit(
        x_train,
        y_train,
        epochs=80,
        batch_size=8,
        callbacks=callbacks,
        verbose=0,
        **fit_kwargs,
    )
    return len(history.history.get("loss", []))


def train_and_evaluate_split(
    model_type, x, y, class_count, class_names, random_state
):
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=PERSON_TEST_SIZE,
        random_state=random_state,
        stratify=y,
    )

    model = build_model(model_type, class_count, random_state)
    epochs_trained = fit_model(model_type, model, x_train, y_train, x_test, y_test)
    probabilities = predict_probabilities(model_type, model, x_test)
    y_pred = np.argmax(probabilities, axis=1)
    accuracy = float(accuracy_score(y_test, y_pred))
    labels = list(range(class_count))

    result = {
        "accuracy": accuracy,
        "train_count": int(len(x_train)),
        "test_count": int(len(x_test)),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=labels).tolist(),
        "classification_report": classification_report(
            y_test,
            y_pred,
            labels=labels,
            target_names=class_names,
            zero_division=0,
            output_dict=True,
        ),
    }
    if epochs_trained is not None:
        result["epochs_trained"] = int(epochs_trained)

    if is_keras_model(model_type):
        from tensorflow.keras import backend

        backend.clear_session()
    return result


def train_final_model(model_type, x, y, class_count):
    model = build_model(model_type, class_count, random_state=42)
    fit_model(model_type, model, x, y)
    return model


def main(args):
    model_type = validate_model_type(args.model)
    x, users, counts = load_person_dataset()

    encoder = LabelEncoder()
    y = encoder.fit_transform(users)
    class_names = encoder.classes_.tolist()
    class_count = len(class_names)
    test_count = int(math.ceil(len(y) * PERSON_TEST_SIZE))
    if test_count < class_count:
        raise ValueError(
            f"test_size={PERSON_TEST_SIZE} gives only {test_count} test samples for "
            f"{class_count} users. Collect more samples or increase PERSON_TEST_SIZE."
        )

    split_results = []
    for offset in range(PERSON_REPEATED_EVAL_SPLITS):
        result = train_and_evaluate_split(
            model_type=model_type,
            x=x,
            y=y,
            class_count=class_count,
            class_names=class_names,
            random_state=42 + offset,
        )
        split_results.append(result)
        print(
            f"{model_type} split {offset + 1}/{PERSON_REPEATED_EVAL_SPLITS}: "
            f"accuracy={result['accuracy']:.4f}"
        )

    accuracies = [result["accuracy"] for result in split_results]
    mean_accuracy = float(np.mean(accuracies))
    min_accuracy = float(np.min(accuracies))
    best_index = int(np.argmax(accuracies))
    target_met = (
        mean_accuracy >= PERSON_TARGET_ACCURACY
        and min_accuracy >= PERSON_TARGET_ACCURACY
    )

    model_path = person_model_path(model_type)
    labels_path = person_labels_path(model_type)
    should_save = target_met or args.save_below_target
    report = {
        "model_type": model_type,
        "model_path": model_path,
        "labels_path": labels_path,
        "model_saved": should_save,
        "target_accuracy": PERSON_TARGET_ACCURACY,
        "target_met": target_met,
        "target_rule": (
            "mean_accuracy and min_accuracy across repeated stratified splits "
            "must both meet target_accuracy"
        ),
        "mean_accuracy": mean_accuracy,
        "min_accuracy": min_accuracy,
        "best_accuracy": float(np.max(accuracies)),
        "best_split_index": best_index,
        "sample_count": int(len(users)),
        "user_count": class_count,
        "samples_per_user": {user: int(count) for user, count in counts.to_dict().items()},
        "minimum_samples_per_user": MIN_PERSON_SAMPLES_PER_USER,
        "recommended_samples_per_user": RECOMMENDED_PERSON_SAMPLES_PER_USER,
        "test_size": PERSON_TEST_SIZE,
        "repeated_eval_splits": PERSON_REPEATED_EVAL_SPLITS,
        "classes": class_names,
        "splits": split_results,
    }
    report_path = write_report(model_type, report)

    print(f"Model type: {model_type}")
    print(f"Users: {counts.to_dict()}")
    print(f"Mean accuracy: {mean_accuracy:.4f}")
    print(f"Min accuracy: {min_accuracy:.4f}")
    print(f"Target accuracy: {PERSON_TARGET_ACCURACY:.2f}")
    print(f"Target met: {target_met}")
    print(f"Report: {report_path}")

    if should_save:
        final_model = train_final_model(model_type, x, y, class_count)
        save_person_model(model_type, final_model, model_path)
        with open(labels_path, "w", encoding="utf-8") as f:
            json.dump(
                {"model_type": model_type, "classes": class_names},
                f,
                indent=2,
            )
        print(f"Saved model: {model_path}")
        print(f"Saved labels: {labels_path}")
    else:
        print(
            f"Model not saved because repeated accuracy is below {PERSON_TARGET_ACCURACY:.2f}. "
            "Use --save-below-target to keep it for comparison."
        )


if __name__ == "__main__":
    parsed_args = parse_args()
    try:
        main(parsed_args)
    except ValueError as exc:
        model_type = validate_model_type(parsed_args.model)
        report = {
            "model_type": model_type,
            "target_accuracy": PERSON_TARGET_ACCURACY,
            "target_met": False,
            "ready_for_training": False,
            "reason": str(exc),
        }
        write_report(model_type, report)
        print(f"Person recognition training skipped: {exc}")
        sys.exit(1)
