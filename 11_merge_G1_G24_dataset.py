import numpy as np
from pathlib import Path
from collections import Counter

BASE_PATH = Path("dataset_1955_p09_retake_hand_only_20260923.npz")
AH_PATH = Path("dataset_AH650_G6_G24_hand_only_th03.npz")
OUTPUT_PATH = Path("dataset_G1_G24_hand_only_2902_20260923.npz")


def main():
    base = np.load(BASE_PATH, allow_pickle=True)
    ah = np.load(AH_PATH, allow_pickle=True)

    # 공통 feature 설정 확인
    for key in [
        "T",
        "D",
        "feature_layout",
        "velocity_mode",
        "canonical_hand",
        "hand_only",
    ]:
        a = base[key].item()
        b = ah[key].item()

        if a != b:
            raise RuntimeError(
                f"{key} mismatch: base={a!r}, AH650={b!r}"
            )

    if base["X"].shape != (1955, 4064):
        raise RuntimeError(
            f"Unexpected base X shape: {base['X'].shape}"
        )

    if ah["X"].shape != (947, 4064):
        raise RuntimeError(
            f"Unexpected AH650 X shape: {ah['X'].shape}"
        )

    # Feature
    X = np.concatenate(
        [base["X"], ah["X"]],
        axis=0,
    ).astype(np.float32)

    # Sample metadata
    gesture = np.concatenate([
        base["gesture"].astype(str),
        ah["gesture"].astype(str),
    ])

    performer = np.concatenate([
        base["performer"].astype(str),
        ah["performer"].astype(str),
    ])

    session = np.concatenate([
        base["session"].astype(str),
        ah["session"].astype(str),
    ])

    role = np.concatenate([
        base["role"].astype(str),
        ah["role"].astype(str),
    ])

    hand = np.concatenate([
        base["hand"].astype(str),
        ah["hand"].astype(str),
    ])

    name = np.concatenate([
        base["name"].astype(str),
        ah["name"].astype(str),
    ])

    duration_sec = np.concatenate([
        base["duration_sec"],
        ah["duration_sec"],
    ]).astype(np.float32)

    fps = np.concatenate([
        base["fps"],
        ah["fps"],
    ]).astype(np.float32)

    truncated = np.concatenate([
        base["truncated"],
        ah["truncated"],
    ]).astype(bool)

    # WITECK G1~G5에는 실제 사용자 identity가 있음.
    # AH650 G6~G24에는 사용자 identity를 사용하지 않음.
    identity_available = np.concatenate([
        np.ones(1955, dtype=bool),
        np.zeros(947, dtype=bool),
    ])

    # 기존 WITECK에는 detection_rate가 저장되어 있지 않으므로 NaN.
    detection_rate = np.concatenate([
        np.full(
            1955,
            np.nan,
            dtype=np.float32,
        ),
        ah["detection_rate"].astype(np.float32),
    ])

    source_group = np.concatenate([
        np.full(
            1955,
            "WITECK_G1_G5",
        ),
        np.full(
            947,
            "AH650_G6_G24",
        ),
    ])

    N = len(X)

    if N != 2902:
        raise RuntimeError(
            f"Expected 2902 samples, got {N}"
        )

    # 모든 sample-level metadata 길이 확인
    arrays = {
        "gesture": gesture,
        "performer": performer,
        "session": session,
        "role": role,
        "hand": hand,
        "name": name,
        "duration_sec": duration_sec,
        "fps": fps,
        "detection_rate": detection_rate,
        "truncated": truncated,
        "identity_available": identity_available,
        "source_group": source_group,
    }

    for key, value in arrays.items():
        if len(value) != N:
            raise RuntimeError(
                f"{key} length mismatch: {len(value)} != {N}"
            )

    if np.isnan(X).any():
        raise RuntimeError("X contains NaN")

    if np.isinf(X).any():
        raise RuntimeError("X contains Inf")

    # identity mask 검증
    base_mask = source_group == "WITECK_G1_G5"
    ah_mask = source_group == "AH650_G6_G24"

    if not np.all(identity_available[base_mask]):
        raise RuntimeError(
            "WITECK identity mask is incorrect"
        )

    if np.any(identity_available[ah_mask]):
        raise RuntimeError(
            "AH650 identity mask is incorrect"
        )

    # 저장
    np.savez_compressed(
        OUTPUT_PATH,

        gesture=gesture,
        performer=performer,
        session=session,
        role=role,
        hand=hand,
        name=name,

        duration_sec=duration_sec,
        fps=fps,
        detection_rate=detection_rate,
        truncated=truncated,

        identity_available=identity_available,
        source_group=source_group,

        canonical_hand=np.array(True, dtype=bool),
        T=np.array(32, dtype=np.int64),

        velocity_mode=np.array(
            "real_time_units_per_second"
        ),

        X=X,

        D=np.array(127, dtype=np.int64),

        feature_layout=np.array(
            "hand_xyz_63+hand_velocity_63+valid_mask_1"
        ),

        hand_only=np.array(True, dtype=bool),

        source_dataset=np.array(
            "WITECK_G1_G5_1955 + AH650_G6_G24_947"
        ),
    )

    # 결과 출력
    print()
    print("=" * 60)
    print(" G1-G24 UNIFIED DATASET COMPLETE")
    print("=" * 60)

    print("output             :", OUTPUT_PATH.resolve())
    print("X                  :", X.shape)
    print("T                  :", 32)
    print("D                  :", 127)
    print("samples            :", N)

    print(
        "identity available :",
        int(identity_available.sum()),
    )

    print(
        "gesture-only       :",
        int((~identity_available).sum()),
    )

    print("X NaN              :", int(np.isnan(X).sum()))
    print("X Inf              :", int(np.isinf(X).sum()))

    print()
    print("Gesture counts:")

    counts = Counter(gesture.tolist())

    for g in sorted(
        counts,
        key=lambda x: int(x.replace("G", "")),
    ):
        print(f"{g:>3s}: {counts[g]}")

    print()
    print("Source counts:")

    for source, count in Counter(
        source_group.tolist()
    ).items():
        print(f"{source}: {count}")

    print()
    print(
        "AH650 detection rate mean:",
        f"{np.nanmean(detection_rate[ah_mask]):.4f}",
    )

    print("=" * 60)


if __name__ == "__main__":
    main()
