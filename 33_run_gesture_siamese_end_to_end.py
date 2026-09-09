"""Run gesture embedding, Siamese authentication, and end-to-end evaluation."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(command, project_dir):
    print("\n$ " + " ".join(str(x) for x in command), flush=True)
    subprocess.run(command, cwd=project_dir, check=True)


def main():
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default=str(
            project_dir / "dataset" / "dataset_1955_recent8_updated_20260905.npz"
        ),
    )
    parser.add_argument(
        "--output-root",
        default=str(project_dir / "output" / "gesture_siamese_pipeline"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--siamese-architecture",
        choices=("dilated", "lite-stats", "phase-pyramid"),
        default="dilated",
    )
    parser.add_argument("--gesture-epochs", type=int, default=50)
    parser.add_argument("--siamese-epochs", type=int, default=40)
    args = parser.parse_args()

    data_path = Path(args.data).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    gesture_dir = output_root / "gesture"
    auth_dir = output_root / "user_auth"
    e2e_dir = output_root / "end_to_end"
    gesture_checkpoint = gesture_dir / "gesture_embedding_1dcnn.pt"
    auth_checkpoint = (
        auth_dir
        / f"siamese_{args.siamese_architecture.replace('-', '_')}_1dcnn.pt"
    )

    run(
        [
            sys.executable,
            str(project_dir / "13_train_gesture_embedding_1dcnn.py"),
            "--data", str(data_path),
            "--output-dir", str(gesture_dir),
            "--epochs", str(args.gesture_epochs),
            "--seed", str(args.seed),
        ],
        project_dir,
    )
    run(
        [
            sys.executable,
            str(project_dir / "09_train_siamese_embedding.py"),
            "--data", str(data_path),
            "--architecture", args.siamese_architecture,
            "--output-dir", str(auth_dir),
            "--epochs", str(args.siamese_epochs),
            "--seed", str(args.seed),
        ],
        project_dir,
    )
    run(
        [
            sys.executable,
            str(project_dir / "14_evaluate_embedding_end_to_end.py"),
            "--data", str(data_path),
            "--gesture-checkpoint", str(gesture_checkpoint),
            "--auth-checkpoint", str(auth_checkpoint),
            "--output-dir", str(e2e_dir),
        ],
        project_dir,
    )

    print("\nPipeline complete", flush=True)
    print(f"gesture summary : {gesture_dir / 'summary.txt'}", flush=True)
    print(f"auth summary    : {auth_dir / 'summary.txt'}", flush=True)
    print(f"end-to-end      : {e2e_dir / 'summary.txt'}", flush=True)


if __name__ == "__main__":
    main()
