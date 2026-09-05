# Multi-Stream Dilated Siamese 1D CNN

`MultiStreamDilatedSiamese` combines two architecture changes for the WITECK
feature layout `[N, 32, 169]`:

1. Four semantic streams encode hand position (63), hand velocity (63), pose
   position (21), and pose velocity (21) independently before feature fusion.
2. Every stream uses compact multi-kernel temporal blocks with kernels 3/5 and
   dilations 1/2/4. This avoids dilation 8 on the short 32-frame sequence.

The valid-mask feature at index 168 is retained as an auxiliary fusion channel.
The encoder returns a normalized 128-dimensional embedding and uses the same
Siamese verification interface as the existing InceptionTime and TCN models.

## Train

```bash
python train_siamese.py \
  --data train.npz \
  --validation-data validation.npz \
  --model multistream \
  --output checkpoints/multistream_dilated_siamese.pt
```

## Evaluate

```bash
python evaluate_pairs.py \
  --data final_test.npz \
  --checkpoint checkpoints/multistream_dilated_siamese.pt
```

The checkpoint stores `model_name=multistream`, so `evaluate_pairs.py` selects
the correct architecture automatically. As with the other Siamese models, the
locked authentication threshold comes only from the validation split.

## Train-time landmark augmentation

train_siamese.py now enables lightweight train-only augmentation by default.
Validation and final-test sequences are never modified. The defaults combine
standardized coordinate jitter, short contiguous frame masking, temporal crop
and resampling, and a shift of at most two frames. For WITECK's 169-feature
layout, masked frames also receive the standardized invalid value at valid-mask
index 168.

To reproduce the previous behavior, add the --no-augmentation flag.

Useful tuning flags are --noise-std, --frame-mask-probability,
--max-mask-frames, --temporal-crop-probability, --min-crop-ratio, and
--max-shift-frames. The selected configuration is stored in the checkpoint.
Use the same seed and split files for augmentation/no-augmentation comparisons.
