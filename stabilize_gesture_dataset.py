import numpy as np
from pathlib import Path
from collections import Counter


# =========================================================
# Paths
# =========================================================

BASE_PATH = Path(
    "data/processed/witeck_g1_g5_hand_only_v1.npz"
)

AH_PATH = Path(
    "data/processed/ah650_g6_g24_hand_only_v1.npz"
)

OUTPUT_PATH = Path(
    "data/processed/witeck_g1_g24_mobile_v1.npz"
)


# =========================================================
# Stabilization settings
# =========================================================

T = 32
D = 127

POSITION_END = 63
VELOCITY_END = 126

POSITION_THRESHOLD = 10.0
VELOCITY_THRESHOLD = 100.0

POSITION_CLIP = 10.0
VELOCITY_CLIP = 100.0


def main():

    # =====================================================
    # Load datasets
    # =====================================================

    base = np.load(
        BASE_PATH,
        allow_pickle=True,
    )

    ah = np.load(
        AH_PATH,
        allow_pickle=True,
    )

    # =====================================================
    # 공통 feature 설정 확인
    # =====================================================

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
                f"{key} mismatch: "
                f"base={a!r}, AH650={b!r}"
            )

    if base["X"].shape != (1955, 4064):
        raise RuntimeError(
            f"Unexpected base X shape: "
            f"{base['X'].shape}"
        )

    if ah["X"].shape != (947, 4064):
        raise RuntimeError(
            f"Unexpected AH650 X shape: "
            f"{ah['X'].shape}"
        )

    # =====================================================
    # Feature merge
    #
    # WITECK G1~G5 : 1955
    # AH650  G6~G24: 947
    #
    # Total        : 2902
    # =====================================================

    X = np.concatenate(
        [
            base["X"],
            ah["X"],
        ],
        axis=0,
    ).astype(np.float32)

    # 원본 보존용
    X_original = X.copy()

    # =====================================================
    # Collapse-aware stabilization
    #
    # D=127 layout
    #
    # 0:63     hand position
    # 63:126   hand velocity
    # 126      valid mask
    #
    # 한 T=32 frame에서
    #
    #   position 중 하나라도 |value| > 10
    #
    # AND
    #
    #   velocity 중 하나라도 |value| > 100
    #
    # 인 경우에만 collapse 의심 frame으로 판단한다.
    #
    # 해당 frame에 대해서만:
    #
    #   position -> [-10, 10]
    #   velocity -> [-100, 100]
    #
    # valid mask는 수정하지 않는다.
    # =====================================================

    X_3d = X.reshape(
        -1,
        T,
        D,
    )

    position_abs = np.abs(
        X_3d[
            :,
            :,
            :POSITION_END,
        ]
    )

    velocity_abs = np.abs(
        X_3d[
            :,
            :,
            POSITION_END:VELOCITY_END,
        ]
    )

    # ---------------------------------------------
    # collapse frame 탐지
    # ---------------------------------------------

    position_bad = np.any(
        position_abs > POSITION_THRESHOLD,
        axis=2,
    )

    velocity_bad = np.any(
        velocity_abs > VELOCITY_THRESHOLD,
        axis=2,
    )

    collapse_mask = (
        position_bad
        &
        velocity_bad
    )

    affected_samples = np.any(
        collapse_mask,
        axis=1,
    )

    # ---------------------------------------------
    # source별 탐지 통계
    # ---------------------------------------------

    base_collapse = collapse_mask[:1955]
    ah_collapse = collapse_mask[1955:]

    base_affected = affected_samples[:1955]
    ah_affected = affected_samples[1955:]

    print()
    print("=" * 70)
    print(" COLLAPSE-AWARE STABILIZATION")
    print("=" * 70)

    print()
    print(
        "Rule: "
        "|Position| > 10 AND "
        "|Velocity| > 100"
    )

    print()
    print("[WITECK G1-G5]")

    print(
        "detected frames  :",
        int(base_collapse.sum()),
        "/",
        1955 * T,
        f"({base_collapse.mean() * 100:.4f}%)",
    )

    print(
        "affected samples :",
        int(base_affected.sum()),
        "/",
        1955,
        f"({base_affected.mean() * 100:.4f}%)",
    )

    print()
    print("[AH650 G6-G24]")

    print(
        "detected frames  :",
        int(ah_collapse.sum()),
        "/",
        947 * T,
        f"({ah_collapse.mean() * 100:.4f}%)",
    )

    print(
        "affected samples :",
        int(ah_affected.sum()),
        "/",
        947,
        f"({ah_affected.mean() * 100:.4f}%)",
    )

    # =====================================================
    # Stabilization 적용
    # =====================================================

    X_3d[
        collapse_mask,
        :POSITION_END,
    ] = np.clip(
        X_3d[
            collapse_mask,
            :POSITION_END,
        ],
        -POSITION_CLIP,
        POSITION_CLIP,
    )

    X_3d[
        collapse_mask,
        POSITION_END:VELOCITY_END,
    ] = np.clip(
        X_3d[
            collapse_mask,
            POSITION_END:VELOCITY_END,
        ],
        -VELOCITY_CLIP,
        VELOCITY_CLIP,
    )

    # 다시 [N, 4064]
    X = X_3d.reshape(
        -1,
        T * D,
    ).astype(np.float32)

    # =====================================================
    # 실제 변경 결과 검증
    # =====================================================

    original_3d = X_original.reshape(
        -1,
        T,
        D,
    )

    final_3d = X.reshape(
        -1,
        T,
        D,
    )

    changed_values = ~np.isclose(
        original_3d[:, :, :VELOCITY_END],
        final_3d[:, :, :VELOCITY_END],
    )

    changed_frames = np.any(
        changed_values,
        axis=2,
    )

    changed_samples = np.any(
        changed_frames,
        axis=1,
    )

    # valid mask가 절대 바뀌지 않았는지 검증
    if not np.array_equal(
        original_3d[:, :, 126],
        final_3d[:, :, 126],
    ):
        raise RuntimeError(
            "Valid mask was unexpectedly modified"
        )

    print()
    print("[Actual modifications]")

    print(
        "changed frames   :",
        int(changed_frames.sum()),
        "/",
        len(X) * T,
        f"({changed_frames.mean() * 100:.4f}%)",
    )

    print(
        "changed samples  :",
        int(changed_samples.sum()),
        "/",
        len(X),
        f"({changed_samples.mean() * 100:.4f}%)",
    )

    # =====================================================
    # 변경 후 source별 분포 확인
    # =====================================================

    print()
    print("=" * 70)
    print(" DISTRIBUTION AFTER STABILIZATION")
    print("=" * 70)

    for label, subset in [
        (
            "WITECK",
            final_3d[:1955],
        ),
        (
            "AH650",
            final_3d[1955:],
        ),
    ]:

        P = np.abs(
            subset[
                :,
                :,
                :POSITION_END,
            ]
        ).reshape(-1)

        V = np.abs(
            subset[
                :,
                :,
                POSITION_END:VELOCITY_END,
            ]
        ).reshape(-1)

        print()
        print(f"[{label}]")

        print(
            "Position : "
            f"P99={np.percentile(P, 99):.3f} | "
            f"P99.9={np.percentile(P, 99.9):.3f} | "
            f"Max={P.max():.3f}"
        )

        print(
            "Velocity : "
            f"P99={np.percentile(V, 99):.3f} | "
            f"P99.9={np.percentile(V, 99.9):.3f} | "
            f"Max={V.max():.3f}"
        )

    # =====================================================
    # Sample metadata
    # =====================================================

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

    # =====================================================
    # Identity availability
    # =====================================================

    identity_available = np.concatenate([
        np.ones(
            1955,
            dtype=bool,
        ),
        np.zeros(
            947,
            dtype=bool,
        ),
    ])

    # =====================================================
    # Detection rate
    #
    # 기존 WITECK에는 detection_rate가 없으므로 NaN
    # =====================================================

    detection_rate = np.concatenate([
        np.full(
            1955,
            np.nan,
            dtype=np.float32,
        ),
        ah["detection_rate"].astype(
            np.float32
        ),
    ])

    # =====================================================
    # Source group
    # =====================================================

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

    # =====================================================
    # Metadata 길이 검증
    # =====================================================

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
                f"{key} length mismatch: "
                f"{len(value)} != {N}"
            )

    # =====================================================
    # Feature sanity check
    # =====================================================

    if np.isnan(X).any():
        raise RuntimeError(
            "X contains NaN"
        )

    if np.isinf(X).any():
        raise RuntimeError(
            "X contains Inf"
        )

    # =====================================================
    # Identity mask 검증
    # =====================================================

    base_mask = (
        source_group
        ==
        "WITECK_G1_G5"
    )

    ah_mask = (
        source_group
        ==
        "AH650_G6_G24"
    )

    if not np.all(
        identity_available[
            base_mask
        ]
    ):
        raise RuntimeError(
            "WITECK identity mask is incorrect"
        )

    if np.any(
        identity_available[
            ah_mask
        ]
    ):
        raise RuntimeError(
            "AH650 identity mask is incorrect"
        )

    # =====================================================
    # 저장
    # =====================================================

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

        canonical_hand=np.array(
            True,
            dtype=bool,
        ),

        T=np.array(
            T,
            dtype=np.int64,
        ),

        velocity_mode=np.array(
            "real_time_units_per_second"
        ),

        X=X,

        D=np.array(
            D,
            dtype=np.int64,
        ),

        feature_layout=np.array(
            "hand_xyz_63+hand_velocity_63+valid_mask_1"
        ),

        hand_only=np.array(
            True,
            dtype=bool,
        ),

        source_dataset=np.array(
            "WITECK_G1_G5_1955 + "
            "AH650_G6_G24_947 + "
            "collapse_aware_P10_V100"
        ),

        # 이번 stabilization 정보도 저장
        stabilization=np.array(
            "collapse_aware"
        ),

        stabilization_position_threshold=np.float32(
            POSITION_THRESHOLD
        ),

        stabilization_velocity_threshold=np.float32(
            VELOCITY_THRESHOLD
        ),

        stabilization_position_clip=np.float32(
            POSITION_CLIP
        ),

        stabilization_velocity_clip=np.float32(
            VELOCITY_CLIP
        ),
    )

    # =====================================================
    # 최종 결과
    # =====================================================

    print()
    print("=" * 70)
    print(" G1-G24 STABILIZED DATASET COMPLETE")
    print("=" * 70)

    print(
        "output             :",
        OUTPUT_PATH.resolve(),
    )

    print(
        "X                  :",
        X.shape,
    )

    print(
        "T                  :",
        T,
    )

    print(
        "D                  :",
        D,
    )

    print(
        "samples            :",
        N,
    )

    print(
        "identity available :",
        int(
            identity_available.sum()
        ),
    )

    print(
        "gesture-only       :",
        int(
            (~identity_available).sum()
        ),
    )

    print(
        "X NaN              :",
        int(
            np.isnan(X).sum()
        ),
    )

    print(
        "X Inf              :",
        int(
            np.isinf(X).sum()
        ),
    )

    print()
    print("Gesture counts:")

    counts = Counter(
        gesture.tolist()
    )

    for g in sorted(
        counts.keys(),
        key=lambda x: int(
            x.replace("G", "")
        ),
    ):
        print(
            f"{g:>4} : "
            f"{counts[g]}"
        )


if __name__ == "__main__":
    main()
