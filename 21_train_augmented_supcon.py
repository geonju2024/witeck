"""Two-view augmented SupCon training for unseen-user authentication.

The backbone and 128-D embedding are identical to the basic 1D-CNN. During
training only, each sequence is transformed into two mild views. The validity
mask is never noised. Session/date metadata remains split-only information.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import torch


supcon = importlib.import_module("16_train_supcon_embedding")

NOISE_STD = 0.025
MAX_SHIFT = 2
DURATION_NOISE_STD = 0.015


def augment_sequence(x: torch.Tensor) -> torch.Tensor:
    """Apply small phase and coordinate perturbations in normalized space."""
    batch, frames, channels = x.shape
    shifts = torch.randint(
        -MAX_SHIFT,
        MAX_SHIFT + 1,
        (batch, 1),
        device=x.device,
    )
    source = torch.arange(frames, device=x.device).view(1, frames) - shifts
    source = source.clamp(0, frames - 1)
    shifted = x.gather(1, source.unsqueeze(2).expand(-1, -1, channels))

    result = shifted.clone()
    # Channels 0:168 are coordinates/velocities; channel 168 is valid mask.
    feature_channels = min(168, channels)
    result[:, :, :feature_channels] += (
        torch.randn_like(result[:, :, :feature_channels]) * NOISE_STD
    )
    return result


def augmented_run_epoch(
    model,
    loader,
    ce_criterion,
    device,
    contrastive_weight,
    temperature,
    optimizer=None,
):
    training = optimizer is not None
    model.train(training)
    total_loss = total_ce = total_supcon = 0.0
    total_correct = total_n = 0
    context = torch.enable_grad() if training else torch.no_grad()

    with context:
        for xb, db, user_y, gesture_y in loader:
            xb = xb.to(device)
            db = db.to(device)
            user_y = user_y.to(device)
            gesture_y = gesture_y.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
                view1 = augment_sequence(xb)
                view2 = augment_sequence(xb)
                duration1 = db + torch.randn_like(db) * DURATION_NOISE_STD
                duration2 = db + torch.randn_like(db) * DURATION_NOISE_STD
                both_x = torch.cat([view1, view2], dim=0)
                both_duration = torch.cat([duration1, duration2], dim=0)
                both_user = torch.cat([user_y, user_y], dim=0)
                both_gesture = torch.cat([gesture_y, gesture_y], dim=0)
                embedding, logits = model(both_x, both_duration)
                ce_loss = ce_criterion(logits, both_user)
                metric_loss = supcon.gesture_conditioned_supcon_loss(
                    embedding,
                    both_user,
                    both_gesture,
                    temperature,
                )
                prediction = (
                    logits[: len(user_y)] + logits[len(user_y) :]
                ).argmax(dim=1)
            else:
                embedding, logits = model(xb, db)
                ce_loss = ce_criterion(logits, user_y)
                metric_loss = supcon.gesture_conditioned_supcon_loss(
                    embedding,
                    user_y,
                    gesture_y,
                    temperature,
                )
                prediction = logits.argmax(dim=1)

            loss = ce_loss + contrastive_weight * metric_loss
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

            n = len(user_y)
            total_loss += float(loss.item()) * n
            total_ce += float(ce_loss.item()) * n
            total_supcon += float(metric_loss.item()) * n
            total_correct += int((prediction == user_y).sum().item())
            total_n += n

    denominator = max(total_n, 1)
    return (
        total_loss / denominator,
        total_ce / denominator,
        total_supcon / denominator,
        total_correct / denominator,
    )


def argument_value(name: str, default: str) -> str:
    if name in sys.argv:
        position = sys.argv.index(name)
        if position + 1 < len(sys.argv):
            return sys.argv[position + 1]
    return default


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    default_output = str(project_dir / "output" / "augmented_supcon_auth")
    if "--output-dir" not in sys.argv:
        sys.argv.extend(["--output-dir", default_output])

    # Reuse the already verified training/evaluation protocol, changing only
    # how training batches are transformed and contrasted.
    supcon.run_epoch = augmented_run_epoch
    supcon.main()

    output_dir = Path(argument_value("--output-dir", default_output))
    checkpoint_path = output_dir / "embedding_1dcnn_supcon.pt"
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    checkpoint.update(
        {
            "training_loss": "cross_entropy+two_view_augmented_supcon",
            "augmentation": {
                "coordinate_velocity_noise_std": NOISE_STD,
                "max_temporal_shift_frames": MAX_SHIFT,
                "duration_noise_std": DURATION_NOISE_STD,
                "valid_mask_augmented": False,
            },
            "date_usage": "split_only",
        }
    )
    torch.save(checkpoint, checkpoint_path)

    summary_path = output_dir / "summary.txt"
    original = summary_path.read_text(encoding="utf-8")
    header = (
        "Two-view augmented SupCon basic 1D-CNN authentication\n"
        f"augmentation_noise_std={NOISE_STD}\n"
        f"augmentation_max_shift_frames={MAX_SHIFT}\n"
        f"augmentation_duration_noise_std={DURATION_NOISE_STD}\n"
        "valid_mask_augmented=false\n"
    )
    first_newline = original.find("\n")
    remainder = original[first_newline + 1 :] if first_newline >= 0 else original
    summary_path.write_text(header + remainder, encoding="utf-8")


if __name__ == "__main__":
    main()
