"""Run the lightweight WITECK authentication experiment.

The training and unseen-user evaluation protocol is shared with
09_train_siamese_embedding.py. Laptop-friendly defaults are supplied here, so
the experiment can be started with:

    python 12_train_lite_stats_siamese.py
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


DEFAULTS = {
    "--architecture": "lite-stats",
    "--data": "dataset/dataset_1955_recent8_updated_20260905.npz",
    "--output-dir": "output/lite_stats_auth",
    "--embedding-dim": "64",
    "--epochs": "30",
    "--pairs": "2000",
    "--batch-size": "256",
    "--patience": "5",
}


def main() -> None:
    for option, value in reversed(tuple(DEFAULTS.items())):
        if option not in sys.argv:
            sys.argv[1:1] = [option, value]

    training_script = Path(__file__).with_name(
        "09_train_siamese_embedding.py"
    )
    runpy.run_path(str(training_script), run_name="__main__")


if __name__ == "__main__":
    main()

