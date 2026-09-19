# WITECK dataset — 1,955 samples

`WITECK_P07_results_recovered_1955.zip` contains the complete feature dataset
and its matching labels after recovering the eight previously missing P01
samples.

## Primary files inside the archive

- `updated/dataset.npz`: `X.shape == (1955, 5408)`
- `updated/labels.csv`: 1,955 unique label rows
- `updated/WITECK_recovered_missing8_landmarks.zip`: eight recovered landmark caches
- `updated/recovery_validation.json`: numeric and merge validation
- `updated/recovery_detection_report.json`: per-video recovery details
- `updated/recovery_contacts.jpg`: visual landmark overlay check

All 1,947 previously accepted dataset rows are unchanged. The eight recovered
samples were processed with the repository's original `02_build_features.py`
feature schema. No label rows were removed.

SHA-256:

```text
1CD4C5AD33B17A9EBCA8E5B3D6CFA4A886D40822AE8B6C21DCD6EEB65D011040
```

The same archive is also available from the `dataset-v1955` GitHub Release.
