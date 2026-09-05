from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from witeck_auth.augmentation import SequenceAugmenter
from witeck_auth.data import FeatureStandardizer, WiteckArrays
from witeck_auth.losses import remap_episode_targets
from witeck_auth.models import Prototypical1DCNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--episodes-per-epoch", type=int, default=100)
    parser.add_argument("--support", type=int, default=3)
    parser.add_argument("--query", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-augmentation", action="store_true")
    parser.add_argument("--noise-std", type=float, default=0.02)
    parser.add_argument("--frame-mask-probability", type=float, default=0.35)
    parser.add_argument("--max-mask-frames", type=int, default=4)
    parser.add_argument("--temporal-crop-probability", type=float, default=0.50)
    parser.add_argument("--min-crop-ratio", type=float, default=0.85)
    parser.add_argument("--max-shift-frames", type=int, default=2)
    return parser.parse_args()


def sample_episode(
    x,
    user_ids,
    gesture_ids,
    support_count,
    query_count,
    rng,
    augmenter=None,
):
    # One gesture per episode prevents the encoder from solving identity through gesture content.
    gesture = rng.choice(np.unique(gesture_ids))
    gesture_index = np.flatnonzero(gesture_ids == gesture)
    classes = np.unique(user_ids[gesture_index])
    eligible = [
        c for c in classes
        if np.sum((user_ids == c) & (gesture_ids == gesture)) >= support_count + query_count
    ]
    if len(eligible) < 2:
        raise ValueError("prototypical training requires at least two eligible user classes")
    support_idx, query_idx = [], []
    for label in eligible:
        candidates = np.flatnonzero((user_ids == label) & (gesture_ids == gesture))
        chosen = rng.choice(candidates, support_count + query_count, replace=False)
        support_idx.extend(chosen[:support_count])
        query_idx.extend(chosen[support_count:])
    class_map = {name: i for i, name in enumerate(eligible)}
    sy = torch.tensor([class_map[user_ids[i]] for i in support_idx], dtype=torch.long)
    qy = torch.tensor([class_map[user_ids[i]] for i in query_idx], dtype=torch.long)
    support_x = x[support_idx]
    query_x = x[query_idx]
    if augmenter is not None:
        support_x = np.stack([augmenter(sequence, rng) for sequence in support_x])
        query_x = np.stack([augmenter(sequence, rng) for sequence in query_x])
    return torch.tensor(support_x), sy, torch.tensor(query_x), qy


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    arrays = WiteckArrays.from_npz(args.data)
    scaler = FeatureStandardizer().fit(arrays.x)
    x = scaler.transform(arrays.x)
    valid_mask_index = 168 if x.shape[-1] == 169 else None
    invalid_mask_value = 0.0
    if valid_mask_index is not None:
        invalid_mask_value = float(
            -scaler.mean[0, 0, valid_mask_index]
            / scaler.std[0, 0, valid_mask_index]
        )
    augmenter = None
    if not args.no_augmentation:
        augmenter = SequenceAugmenter(
            noise_std=args.noise_std,
            frame_mask_probability=args.frame_mask_probability,
            max_mask_frames=args.max_mask_frames,
            temporal_crop_probability=args.temporal_crop_probability,
            min_crop_ratio=args.min_crop_ratio,
            max_shift_frames=args.max_shift_frames,
            valid_mask_index=valid_mask_index,
            invalid_mask_value=invalid_mask_value,
        )
    model = Prototypical1DCNN(input_dim=x.shape[-1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for _ in range(args.episodes_per_epoch):
            sx, sy, qx, qy = sample_episode(
                x,
                arrays.user_ids,
                arrays.gesture_ids,
                args.support,
                args.query,
                rng,
                augmenter=augmenter,
            )
            sx, sy, qx, qy = sx.to(device), sy.to(device), qx.to(device), qy.to(device)
            output = model(sx, sy, qx)
            targets = remap_episode_targets(qy, output["classes"])
            loss = F.cross_entropy(output["logits"], targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item()
        print(json.dumps({"epoch": epoch + 1, "loss": total / args.episodes_per_epoch}))

    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": "prototypical",
            "model_state": model.state_dict(),
            "input_dim": x.shape[-1],
            "normalization_mean": scaler.mean,
            "normalization_std": scaler.std,
            "augmentation": None if augmenter is None else {
                "noise_std": augmenter.noise_std,
                "frame_mask_probability": augmenter.frame_mask_probability,
                "max_mask_frames": augmenter.max_mask_frames,
                "temporal_crop_probability": augmenter.temporal_crop_probability,
                "min_crop_ratio": augmenter.min_crop_ratio,
                "max_shift_frames": augmenter.max_shift_frames,
            },
        },
        destination,
    )


if __name__ == "__main__":
    main()
