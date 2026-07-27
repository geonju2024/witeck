import argparse
import json
import os
import sys

from gesture_auth_config import PERSON_DATASET_AUDIT_PATH, PERSON_TARGET_ACCURACY
from person_model_registry import (
    DEFAULT_PERSON_MODEL_TYPE,
    PERSON_MODEL_TYPES,
    person_labels_path,
    person_model_path,
    person_report_path,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check the completion gate for one selected person model."
    )
    parser.add_argument(
        "--model",
        choices=PERSON_MODEL_TYPES,
        default=DEFAULT_PERSON_MODEL_TYPE,
        help=f"Model to check (default: {DEFAULT_PERSON_MODEL_TYPE}).",
    )
    return parser.parse_args()


def read_json(path):
    if not os.path.exists(path):
        return None, f"Missing file: {path}"
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f), None
    except json.JSONDecodeError as exc:
        return None, f"Invalid JSON in {path}: {exc}"


def check_goal(model_type):
    failures = []
    model_path = person_model_path(model_type)
    labels_path = person_labels_path(model_type)
    report_path = person_report_path(model_type)

    audit, error = read_json(PERSON_DATASET_AUDIT_PATH)
    if error:
        failures.append(error)
        audit = {}

    report, error = read_json(report_path)
    if error:
        failures.append(error)
        report = {}

    labels, error = read_json(labels_path)
    if error:
        failures.append(error)
        labels = {}

    if not os.path.exists(model_path):
        failures.append(f"Missing model: {model_path}")

    if audit.get("ready_for_person_recognition") is not True:
        failures.append(
            "Dataset is not ready for person recognition: "
            f"{audit.get('reason', 'no reason recorded')}"
        )

    if report.get("model_type") not in (None, model_type):
        failures.append(
            f"Report belongs to {report.get('model_type')}, not {model_type}."
        )

    if labels.get("model_type") not in (None, model_type):
        failures.append(
            f"Labels belong to {labels.get('model_type')}, not {model_type}."
        )

    if report.get("target_met") is not True:
        failures.append(
            "Person-recognition target is not met: "
            f"{report.get('reason', 'target_met is not true')}"
        )

    target_accuracy = float(report.get("target_accuracy", PERSON_TARGET_ACCURACY))
    if target_accuracy < PERSON_TARGET_ACCURACY:
        failures.append(
            f"Report target_accuracy is {target_accuracy}, below required "
            f"{PERSON_TARGET_ACCURACY}."
        )

    for metric_name in ("mean_accuracy", "min_accuracy"):
        value = report.get(metric_name)
        if value is None:
            failures.append(f"Missing metric in report: {metric_name}")
        elif float(value) < PERSON_TARGET_ACCURACY:
            failures.append(
                f"{metric_name}={float(value):.4f}, below required "
                f"{PERSON_TARGET_ACCURACY:.2f}."
            )

    audit_user_count = audit.get("user_count")
    label_classes = labels.get("classes") if isinstance(labels, dict) else None
    if audit_user_count is not None and label_classes is not None:
        if int(audit_user_count) != len(label_classes):
            failures.append(
                f"Label count {len(label_classes)} does not match audited "
                f"user_count {audit_user_count}."
            )

    return failures, model_path, labels_path, report_path


def main():
    args = parse_args()
    failures, model_path, labels_path, report_path = check_goal(args.model)
    if failures:
        print(f"Person recognition goal ({args.model}): NOT MET")
        for failure in failures:
            print(f"- {failure}")
        sys.exit(1)

    print(f"Person recognition goal ({args.model}): MET")
    print(f"Required accuracy: {PERSON_TARGET_ACCURACY:.2f}")
    print(f"Model: {model_path}")
    print(f"Labels: {labels_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
