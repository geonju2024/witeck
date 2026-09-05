import numpy as np

from witeck_auth.augmentation import SequenceAugmenter
from witeck_auth.data import PairDataset


def test_augmentation_is_deterministic_and_does_not_mutate_input():
    sequence = np.arange(32 * 169, dtype=np.float32).reshape(32, 169)
    original = sequence.copy()
    augmenter = SequenceAugmenter()

    first = augmenter(sequence, np.random.default_rng(7))
    second = augmenter(sequence, np.random.default_rng(7))

    np.testing.assert_array_equal(sequence, original)
    np.testing.assert_allclose(first, second)
    assert first.shape == sequence.shape
    assert first.dtype == np.float32


def test_frame_mask_sets_continuous_features_and_valid_mask():
    sequence = np.ones((8, 169), dtype=np.float32)
    augmenter = SequenceAugmenter(
        noise_std=0.0,
        frame_mask_probability=1.0,
        max_mask_frames=8,
        temporal_crop_probability=0.0,
        max_shift_frames=0,
        invalid_mask_value=-3.0,
    )

    augmented = augmenter(sequence, np.random.default_rng(3))
    masked = np.flatnonzero(augmented[:, 168] == -3.0)

    assert len(masked) >= 1
    assert np.all(augmented[masked, :168] == 0.0)


def test_invalid_configuration_is_rejected():
    try:
        SequenceAugmenter(min_crop_ratio=0.0)
    except ValueError as exc:
        assert "min_crop_ratio" in str(exc)
    else:
        raise AssertionError("expected invalid crop ratio to be rejected")


def test_pair_dataset_augments_train_pairs_deterministically_by_epoch():
    x = np.ones((8, 8, 169), dtype=np.float32)
    users = np.array(["P1", "P1", "P2", "P2"] * 2)
    gestures = np.array(["G1"] * 4 + ["G2"] * 4)
    augmenter = SequenceAugmenter(
        noise_std=0.1,
        frame_mask_probability=0.0,
        temporal_crop_probability=0.0,
        max_shift_frames=0,
    )
    dataset = PairDataset(
        x,
        users,
        gestures,
        pairs_per_epoch=8,
        seed=11,
        augmenter=augmenter,
    )

    first_left, first_right, first_target = dataset[0]
    repeated_left, repeated_right, repeated_target = dataset[0]
    np.testing.assert_allclose(first_left.numpy(), repeated_left.numpy())
    np.testing.assert_allclose(first_right.numpy(), repeated_right.numpy())
    assert first_target.item() == repeated_target.item()

    dataset.set_epoch(1)
    next_left, _, _ = dataset[0]
    assert not np.allclose(first_left.numpy(), next_left.numpy())