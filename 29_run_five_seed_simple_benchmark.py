"""Run five simple authentication models over five fixed random seeds."""

from __future__ import annotations

import csv
import os
import statistics
import subprocess
import sys
from pathlib import Path


SEEDS = (40, 41, 42, 43, 44)
MODELS = (
    {
        "slug": "ce_only",
        "name": "CE-only 1D-CNN",
        "script": "08_train_embedding.py",
        "checkpoint": "embedding_1dcnn.pt",
        "extra": (),
    },
    {
        "slug": "arcface",
        "name": "ArcFace 1D-CNN",
        "script": "15_train_arcface_embedding.py",
        "checkpoint": "embedding_1dcnn_arcface.pt",
        "extra": (),
    },
    {
        "slug": "supcon",
        "name": "SupCon 1D-CNN",
        "script": "16_train_supcon_embedding.py",
        "checkpoint": "embedding_1dcnn_supcon.pt",
        "extra": (),
    },
    {
        "slug": "cosface",
        "name": "CosFace 1D-CNN",
        "script": "27_train_cosface_embedding.py",
        "checkpoint": "embedding_1dcnn_cosface.pt",
        "extra": (),
    },
    {
        "slug": "center_loss",
        "name": "Center-loss 1D-CNN",
        "script": "28_train_center_loss_embedding.py",
        "checkpoint": "embedding_1dcnn_center_loss.pt",
        "extra": ("--contrastive-weight", "0.05"),
    },
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


def run_logged(command, log_path, project_dir, environment):
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(
            command,
            cwd=project_dir,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )


def main():
    project_dir = Path(__file__).resolve().parent
    benchmark_dir = project_dir / "output" / "five_seed_simple_benchmark"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    gesture_checkpoint = (
        project_dir
        / "output"
        / "basic_embedding_pipeline"
        / "gesture"
        / "gesture_embedding_1dcnn.pt"
    )
    if not gesture_checkpoint.exists():
        raise FileNotFoundError(f"Fixed gesture checkpoint not found: {gesture_checkpoint}")

    environment = os.environ.copy()
    environment["PYTHONPYCACHEPREFIX"] = "/tmp/witeck-five-seed-pycache"
    all_rows = []

    for model in MODELS:
        print(f"\n=== {model['name']} ===", flush=True)
        for run_number, seed in enumerate(SEEDS, start=1):
            run_dir = benchmark_dir / model["slug"] / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            auth_summary = run_dir / "summary.txt"
            checkpoint = run_dir / model["checkpoint"]
            if not (auth_summary.exists() and checkpoint.exists()):
                print(f"[{run_number}/5] seed={seed} training", flush=True)
                command = [
                    sys.executable,
                    str(project_dir / model["script"]),
                    "--seed", str(seed),
                    "--output-dir", str(run_dir),
                    *model["extra"],
                ]
                run_logged(
                    command,
                    run_dir / "training.log",
                    project_dir,
                    environment,
                )
            else:
                print(f"[{run_number}/5] seed={seed} training already complete", flush=True)

            e2e_dir = run_dir / "end_to_end"
            e2e_dir.mkdir(parents=True, exist_ok=True)
            e2e_summary = e2e_dir / "summary.txt"
            if not e2e_summary.exists():
                print(f"[{run_number}/5] seed={seed} end-to-end", flush=True)
                command = [
                    sys.executable,
                    str(project_dir / "14_evaluate_embedding_end_to_end.py"),
                    "--gesture-checkpoint", str(gesture_checkpoint),
                    "--auth-checkpoint", str(checkpoint),
                    "--output-dir", str(e2e_dir),
                ]
                run_logged(
                    command,
                    run_dir / "end_to_end.log",
                    project_dir,
                    environment,
                )

            auth = parse_summary(auth_summary)
            e2e = parse_summary(e2e_summary)
            row = {
                "model": model["name"],
                "slug": model["slug"],
                "seed": seed,
                "auth_accuracy": auth["unseen_accuracy"],
                "auth_balanced_accuracy": auth["unseen_balanced_accuracy"],
                "auth_far": auth["unseen_far"],
                "auth_frr": auth["unseen_frr"],
                "auth_eer_analysis": auth["unseen_test_eer_analysis"],
                "e2e_accuracy": e2e["end_to_end_global_accuracy"],
                "e2e_balanced_accuracy": e2e["end_to_end_global_balanced_accuracy"],
                "e2e_far": e2e["end_to_end_global_far"],
                "e2e_frr": e2e["end_to_end_global_frr"],
                "route_accuracy": e2e["route_accuracy"],
            }
            all_rows.append(row)
            print(
                f"[{run_number}/5] seed={seed} "
                f"auth={row['auth_accuracy']:.4f} e2e={row['e2e_accuracy']:.4f}",
                flush=True,
            )

    fieldnames = list(all_rows[0])
    with (benchmark_dir / "all_runs.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    metric_names = fieldnames[3:]
    aggregate_rows = []
    for model in MODELS:
        rows = [row for row in all_rows if row["slug"] == model["slug"]]
        aggregate = {"model": model["name"], "slug": model["slug"], "runs": len(rows)}
        for metric in metric_names:
            values = [float(row[metric]) for row in rows]
            aggregate[f"{metric}_mean"] = statistics.mean(values)
            aggregate[f"{metric}_std"] = statistics.stdev(values)
            aggregate[f"{metric}_min"] = min(values)
            aggregate[f"{metric}_max"] = max(values)
        aggregate_rows.append(aggregate)

        model_dir = benchmark_dir / model["slug"]
        with (model_dir / "aggregate_summary.txt").open("w", encoding="utf-8") as handle:
            handle.write(f"model={model['name']}\n")
            handle.write(f"seeds={list(SEEDS)}\n")
            handle.write("gesture_model=fixed_basic_1dcnn\n")
            for metric in metric_names:
                handle.write(f"{metric}_mean={aggregate[f'{metric}_mean']:.6f}\n")
                handle.write(f"{metric}_std={aggregate[f'{metric}_std']:.6f}\n")
                handle.write(f"{metric}_min={aggregate[f'{metric}_min']:.6f}\n")
                handle.write(f"{metric}_max={aggregate[f'{metric}_max']:.6f}\n")

    aggregate_fields = list(aggregate_rows[0])
    with (benchmark_dir / "aggregate_comparison.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=aggregate_fields)
        writer.writeheader()
        writer.writerows(aggregate_rows)

    ranked = sorted(
        aggregate_rows,
        key=lambda row: row["auth_accuracy_mean"],
        reverse=True,
    )
    with (benchmark_dir / "summary.txt").open("w", encoding="utf-8") as handle:
        handle.write("Five-seed simple 1D-CNN authentication benchmark\n")
        handle.write(f"seeds={list(SEEDS)}\n")
        handle.write("gesture_model=fixed_basic_1dcnn\n")
        for rank, row in enumerate(ranked, start=1):
            handle.write(
                f"rank_{rank}={row['model']} | "
                f"auth_accuracy={row['auth_accuracy_mean']:.6f}+-"
                f"{row['auth_accuracy_std']:.6f} | "
                f"e2e_accuracy={row['e2e_accuracy_mean']:.6f}+-"
                f"{row['e2e_accuracy_std']:.6f}\n"
            )

    print("\n=== Final ranking by mean authentication accuracy ===", flush=True)
    for rank, row in enumerate(ranked, start=1):
        print(
            f"{rank}. {row['model']}: "
            f"auth={row['auth_accuracy_mean']:.4f}±{row['auth_accuracy_std']:.4f}, "
            f"e2e={row['e2e_accuracy_mean']:.4f}±{row['e2e_accuracy_std']:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
