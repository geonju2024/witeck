"""Run CE-weight ablations for the unseen-G5 metric-learning experiment.

Each run trains on P01-P07 with G5 excluded, calibrates thresholds using only
held-out sessions of P01-P07/G1-G4, and evaluates P08-P10 on G5 without tuning
the thresholds on those final users.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


def run(command: list[str], cwd: Path) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def parse_summary(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def one_minus(value: str) -> str:
    if not value:
        return ""
    return f"{1.0 - float(value):.6f}"


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default=str(
            project_dir / "dataset" / "dataset_1955_recent8_updated_20260905_hand_only.npz"
        ),
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device-python", default=sys.executable)
    parser.add_argument(
        "--output-root",
        default=str(project_dir / "output" / "g5_metric_loss_ablation"),
    )
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []

    for ce_weight in (1.0, 0.1, 0.0):
        tag = f"ce_{ce_weight:.1f}"
        train_dir = output_root / tag / "train"
        eval_dir = output_root / tag / "eval_unseen_g5"
        checkpoint = train_dir / "shared_dual_head_metric.pt"

        run(
            [
                args.device_python,
                "train_hard_negative_model.py",
                "--data",
                args.data,
                "--output-dir",
                str(train_dir),
                "--epochs",
                str(args.epochs),
                "--seed",
                str(args.seed),
                "--user-ce-weight",
                str(ce_weight),
                "--heldout-gesture",
                "G5",
            ],
            project_dir,
        )
        run(
            [
                args.device_python,
                "evaluate_unseen_gesture.py",
                "--data",
                args.data,
                "--checkpoint",
                str(checkpoint),
                "--output-dir",
                str(eval_dir),
            ],
            project_dir,
        )

        train_summary = parse_summary(train_dir / "summary.txt")
        eval_summary = parse_summary(eval_dir / "summary.txt")
        rows.append(
            {
                "ce_weight": str(ce_weight),
                "seed": str(args.seed),
                "best_epoch": train_summary.get("best_epoch", ""),
                "user_validation_eer": train_summary.get(
                    "user_validation_eer", ""
                ),
                "gesture_validation_eer": train_summary.get(
                    "gesture_validation_eer", ""
                ),
                "genuine_accept_rate": one_minus(
                    eval_summary.get("genuine_frr", "")
                ),
                "wrong_gesture_far": eval_summary.get("wrong_gesture_far", ""),
                "same_gesture_impostor_far": eval_summary.get(
                    "same_gesture_impostor_far", ""
                ),
                "random_impostor_far": eval_summary.get(
                    "random_impostor_far", ""
                ),
            }
        )

    csv_path = output_root / "comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved comparison: {csv_path}")


if __name__ == "__main__":
    main()
