from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from witeck_auth.data import FeatureStandardizer, PairDataset, WiteckArrays
from witeck_auth.losses import SiameseVerificationLoss
from witeck_auth.metrics import verification_metrics
from witeck_auth.models import InceptionTimeSiamese, TCNSiamese


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="NPZ with X/user_ids/gesture_ids")
    parser.add_argument("--validation-data", help="Optional disjoint validation NPZ")
    parser.add_argument("--model", choices=("inception", "tcn"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    labels, scores = [], []
    for left, right, target in loader:
        output = model(left.to(device), right.to(device))
        labels.extend(target.numpy())
        scores.extend(output["similarity"].cpu().numpy())
    return verification_metrics(labels, scores)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    arrays = WiteckArrays.from_npz(args.data)
    scaler = FeatureStandardizer().fit(arrays.x)
    x = scaler.transform(arrays.x)
    dataset = PairDataset(x, arrays.user_ids, arrays.gesture_ids, seed=args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    validation_loader = None
    if args.validation_data:
        validation = WiteckArrays.from_npz(args.validation_data)
        validation_x = scaler.transform(validation.x)
        validation_dataset = PairDataset(
            validation_x,
            validation.user_ids,
            validation.gesture_ids,
            pairs_per_epoch=max(4096, len(validation_x) * 8),
            seed=4242,
        )
        validation_loader = DataLoader(
            validation_dataset, batch_size=args.batch_size, shuffle=False
        )
    input_dim = x.shape[-1]
    model = InceptionTimeSiamese(input_dim) if args.model == "inception" else TCNSiamese(input_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = SiameseVerificationLoss()

    last_metrics = {}
    best_eer = float("inf")
    best_state = None
    selected_threshold = None
    for epoch in range(args.epochs):
        dataset.set_epoch(epoch)
        model.train()
        labels, scores = [], []
        running_loss = 0.0
        for left, right, target in loader:
            left, right, target = left.to(device), right.to(device), target.to(device)
            output = model(left, right)
            loss = criterion(output, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(target)
            labels.extend(target.detach().cpu().numpy())
            scores.extend(output["similarity"].detach().cpu().numpy())
        scheduler.step()
        last_metrics = verification_metrics(labels, scores)
        record = {"epoch": epoch + 1, "loss": running_loss / len(dataset), "train": last_metrics}
        if validation_loader is not None:
            validation_metrics = evaluate(model, validation_loader, device)
            record["validation"] = validation_metrics
            if validation_metrics["eer"] < best_eer:
                best_eer = validation_metrics["eer"]
                selected_threshold = validation_metrics["eer_threshold"]
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(json.dumps(record))

    if best_state is not None:
        model.load_state_dict(best_state)

    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": args.model,
            "model_state": model.state_dict(),
            "input_dim": input_dim,
            "threshold": selected_threshold,
            "normalization_mean": scaler.mean,
            "normalization_std": scaler.std,
        },
        destination,
    )


if __name__ == "__main__":
    main()
