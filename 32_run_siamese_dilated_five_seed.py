"""Run 09_train_siamese_embedding.py (dilated) over five seeds."""

from __future__ import annotations

import csv
import os
import statistics
import subprocess
import sys
from pathlib import Path


SEEDS = (40, 41, 42, 43, 44)
METRICS = (
    "unseen_auc",
    "unseen_accuracy",
    "unseen_balanced_accuracy",
    "unseen_far",
    "unseen_frr",
    "unseen_test_eer_analysis",
)


def parse_summary(path: Path):
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        try:
            result[key.strip()] = float(value.strip())
        except ValueError:
            result[key.strip()] = value.strip()
    return result


def main():
    project_dir = Path(__file__).resolve().parent
    data_path = project_dir / "dataset" / "dataset_1955_recent8_updated_20260905.npz"
    output_root = project_dir / "output" / "siamese_dilated_five_seed"
    output_root.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PYTHONPYCACHEPREFIX"] = "/tmp/witeck-siamese-pycache"
    rows = []

    for run_number, seed in enumerate(SEEDS, start=1):
        run_dir = output_root / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        summary_path = run_dir / "summary.txt"
        checkpoint_path = run_dir / "siamese_dilated_1dcnn.pt"
        if not (summary_path.exists() and checkpoint_path.exists()):
            print(f"[{run_number}/5] seed={seed} training", flush=True)
            command = [
                sys.executable,
                str(project_dir / "09_train_siamese_embedding.py"),
                "--data", str(data_path),
                "--architecture", "dilated",
                "--seed", str(seed),
                "--output-dir", str(run_dir),
            ]
            with (run_dir / "training.log").open("w", encoding="utf-8") as log:
                subprocess.run(
                    command,
                    cwd=project_dir,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
        else:
            print(f"[{run_number}/5] seed={seed} already complete", flush=True)

        values = parse_summary(summary_path)
        row = {"seed": seed, **{name: values[name] for name in METRICS}}
        rows.append(row)
        print(
            f"[{run_number}/5] seed={seed} accuracy={row['unseen_accuracy']:.4f} "
            f"eer={row['unseen_test_eer_analysis']:.4f}",
            flush=True,
        )

    with (output_root / "all_runs.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    aggregates = {}
    for metric in METRICS:
        values = [float(row[metric]) for row in rows]
        aggregates[metric] = {
            "mean": statistics.mean(values),
            "std": statistics.stdev(values),
            "min": min(values),
            "max": max(values),
        }

    with (output_root / "aggregate_summary.txt").open("w", encoding="utf-8") as handle:
        handle.write("model=Dilated Siamese 1D-CNN\n")
        handle.write(f"seeds={list(SEEDS)}\n")
        handle.write("source=09_train_siamese_embedding.py\n")
        handle.write("split=latest_two_known_sessions_for_validation\n")
        for metric, values in aggregates.items():
            for statistic, value in values.items():
                handle.write(f"{metric}_{statistic}={value:.6f}\n")

    print("\n=== Dilated Siamese 1D-CNN five-seed result ===", flush=True)
    for metric, values in aggregates.items():
        print(
            f"{metric}={values['mean']:.4f}±{values['std']:.4f} "
            f"[{values['min']:.4f}, {values['max']:.4f}]",
            flush=True,
        )


if __name__ == "__main__":
    main()
