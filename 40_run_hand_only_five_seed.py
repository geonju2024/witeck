"""Five-seed benchmark for the hand-only gesture/SupCon/E2E pipeline."""

from __future__ import annotations

import csv
import os
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


SEEDS = (40, 41, 42, 43, 44)
METRICS = (
    "gesture_accuracy",
    "gesture_balanced_accuracy",
    "gesture_macro_f1",
    "auth_accuracy",
    "auth_balanced_accuracy",
    "auth_far",
    "auth_frr",
    "auth_eer_analysis",
    "e2e_accuracy",
    "e2e_balanced_accuracy",
    "e2e_far",
    "e2e_frr",
    "route_accuracy",
)


def parse_summary(path: Path):
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        try:
            values[key.strip()] = float(value.strip())
        except ValueError:
            values[key.strip()] = value.strip()
    return values


def run_logged(command, log_path, root, environment):
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(
            command,
            cwd=root,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )


def main() -> None:
    root = Path(__file__).resolve().parent
    data = root / "dataset" / "dataset_1955_recent8_updated_20260905_hand_only.npz"
    output_root = root / "output" / "hand_only_five_seed"
    output_root.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PYTHONPYCACHEPREFIX"] = "/tmp/witeck-hand-only-five-seed-pycache"
    rows = []

    for seed in SEEDS:
        seed_dir = output_root / f"seed_{seed}"
        gesture_dir = seed_dir / "gesture"
        auth_dir = seed_dir / "user_auth"
        e2e_dir = seed_dir / "end_to_end"
        for directory in (gesture_dir, auth_dir, e2e_dir):
            directory.mkdir(parents=True, exist_ok=True)

        jobs = []
        gesture_checkpoint = gesture_dir / "gesture_embedding_1dcnn.pt"
        if not gesture_checkpoint.exists():
            jobs.append(
                (
                    [
                        sys.executable,
                        str(root / "13_train_gesture_embedding_1dcnn.py"),
                        "--data", str(data),
                        "--output-dir", str(gesture_dir),
                        "--seed", str(seed),
                    ],
                    gesture_dir / "training.log",
                )
            )
        auth_checkpoint = auth_dir / "embedding_1dcnn_supcon.pt"
        if not auth_checkpoint.exists():
            jobs.append(
                (
                    [
                        sys.executable,
                        str(root / "16_train_supcon_embedding.py"),
                        "--data", str(data),
                        "--output-dir", str(auth_dir),
                        "--seed", str(seed),
                    ],
                    auth_dir / "training.log",
                )
            )
        if jobs:
            print(f"seed={seed}: training gesture and authentication", flush=True)
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [
                    executor.submit(run_logged, command, log, root, environment)
                    for command, log in jobs
                ]
                for future in futures:
                    future.result()

        e2e_summary_path = e2e_dir / "summary.txt"
        if not e2e_summary_path.exists():
            print(f"seed={seed}: strict end-to-end evaluation", flush=True)
            run_logged(
                [
                    sys.executable,
                    str(root / "14_evaluate_embedding_end_to_end.py"),
                    "--data", str(data),
                    "--gesture-checkpoint", str(gesture_checkpoint),
                    "--auth-checkpoint", str(auth_checkpoint),
                    "--output-dir", str(e2e_dir),
                ],
                e2e_dir / "evaluation.log",
                root,
                environment,
            )

        gesture = parse_summary(gesture_dir / "summary.txt")
        auth = parse_summary(auth_dir / "summary.txt")
        end_to_end = parse_summary(e2e_summary_path)
        row = {
            "seed": seed,
            "gesture_accuracy": gesture["final_accuracy"],
            "gesture_balanced_accuracy": gesture["final_balanced_accuracy"],
            "gesture_macro_f1": gesture["final_macro_f1"],
            "auth_accuracy": auth["unseen_accuracy"],
            "auth_balanced_accuracy": auth["unseen_balanced_accuracy"],
            "auth_far": auth["unseen_far"],
            "auth_frr": auth["unseen_frr"],
            "auth_eer_analysis": auth["unseen_test_eer_analysis"],
            "e2e_accuracy": end_to_end["end_to_end_global_accuracy"],
            "e2e_balanced_accuracy": end_to_end[
                "end_to_end_global_balanced_accuracy"
            ],
            "e2e_far": end_to_end["end_to_end_global_far"],
            "e2e_frr": end_to_end["end_to_end_global_frr"],
            "route_accuracy": end_to_end["route_accuracy"],
        }
        rows.append(row)
        print(
            f"seed={seed}: gesture={row['gesture_accuracy']:.4f} "
            f"auth={row['auth_accuracy']:.4f} e2e={row['e2e_accuracy']:.4f}",
            flush=True,
        )

    with (output_root / "all_runs.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with (output_root / "aggregate_summary.txt").open(
        "w", encoding="utf-8"
    ) as handle:
        handle.write("Hand-only D=127 five-seed pipeline benchmark\n")
        handle.write(f"dataset={data.resolve()}\n")
        handle.write(f"seeds={list(SEEDS)}\n")
        handle.write("gesture_model=basic_1dcnn_embedding\n")
        handle.write("auth_model=gesture_conditioned_supcon_basic_1dcnn\n")
        for metric in METRICS:
            values = [float(row[metric]) for row in rows]
            handle.write(f"{metric}_mean={statistics.mean(values):.6f}\n")
            handle.write(f"{metric}_std={statistics.stdev(values):.6f}\n")
            handle.write(f"{metric}_min={min(values):.6f}\n")
            handle.write(f"{metric}_max={max(values):.6f}\n")

    print("Hand-only five-seed benchmark complete", flush=True)


if __name__ == "__main__":
    main()
