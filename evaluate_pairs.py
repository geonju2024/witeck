from __future__ import annotations

import argparse
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

from witeck_auth.data import PairDataset, WiteckArrays
from witeck_auth.metrics import verification_metrics
from witeck_auth.models import InceptionTimeSiamese, TCNSiamese


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--pairs", type=int, default=10000)
    parser.add_argument("--threshold", type=float)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_cls = InceptionTimeSiamese if checkpoint["model_name"] == "inception" else TCNSiamese
    model = model_cls(input_dim=int(checkpoint["input_dim"]))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    arrays = WiteckArrays.from_npz(args.data)
    x = (arrays.x - checkpoint["normalization_mean"]) / checkpoint["normalization_std"]
    dataset = PairDataset(
        x, arrays.user_ids, arrays.gesture_ids, pairs_per_epoch=args.pairs, seed=2026
    )
    loader = DataLoader(dataset, batch_size=args.batch_size)
    labels, scores = [], []
    with torch.no_grad():
        for left, right, target in loader:
            output = model(left, right)
            labels.extend(target.numpy())
            scores.extend(output["similarity"].numpy())
    threshold = args.threshold
    if threshold is None:
        threshold = checkpoint.get("threshold")
    if threshold is None:
        raise ValueError(
            "no locked threshold available; train with --validation-data or pass --threshold"
        )
    print(json.dumps(verification_metrics(labels, scores, threshold), indent=2))


if __name__ == "__main__":
    main()
