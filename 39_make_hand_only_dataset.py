"""Derive a D=127 hand-only dataset from the existing D=169 feature NPZ.

The source feature layout is fixed as:
    hand xyz 0:63, hand velocity 63:126,
    pose xyz 126:147, pose velocity 147:168, valid mask 168.

No sample, label, split metadata, or duration value is changed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=str(root / "dataset" / "dataset_1955_recent8_updated_20260905.npz"),
    )
    parser.add_argument(
        "--output",
        default=str(
            root / "dataset" / "dataset_1955_recent8_updated_20260905_hand_only.npz"
        ),
    )
    args = parser.parse_args()

    source_path = Path(args.input)
    output_path = Path(args.output)
    with np.load(source_path, allow_pickle=True) as source:
        T = int(source["T"])
        D = int(source["D"])
        X = source["X"].astype(np.float32)
        if T != 32 or D != 169 or X.shape[1] != T * D:
            raise ValueError(
                f"Expected source [N,32*169], got X={X.shape}, T={T}, D={D}"
            )

        sequence = X.reshape(len(X), T, D)
        hand_only = np.concatenate(
            [sequence[:, :, :126], sequence[:, :, 168:169]], axis=2
        ).astype(np.float32)
        if hand_only.shape != (len(X), 32, 127):
            raise AssertionError(hand_only.shape)

        payload = {key: source[key] for key in source.files if key not in {"X", "D"}}
        payload.update(
            {
                "X": hand_only.reshape(len(X), -1),
                "D": np.int64(127),
                "feature_layout": np.array(
                    "hand_xyz_63+hand_velocity_63+valid_mask_1"
                ),
                "hand_only": np.bool_(True),
                "source_dataset": np.array(str(source_path.resolve())),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)
    print(f"source={source_path.resolve()}")
    print(f"output={output_path.resolve()}")
    print(f"X={payload['X'].shape} T={int(payload['T'])} D={int(payload['D'])}")
    print(f"feature_layout={payload['feature_layout']}")


if __name__ == "__main__":
    main()
