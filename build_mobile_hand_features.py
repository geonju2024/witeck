from pathlib import Path
import json

import numpy as np


INPUT_ROOT = Path("data/AH650_landmarks_th03_cpu")
OUTPUT_PATH = Path("data/processed/ah650_g6_g24_hand_only_v1.npz")

T_OUT = 32
WRIST = 0
MID_MCP = 9


# =========================================================
# Time helpers
# =========================================================
def resample_time(seq, frame_times, t_out=T_OUT):
    """
    실제 시간축의 sequence를 T_OUT개 시점으로 선형 보간한다.
    기존 G1~G5 feature builder와 동일한 방식.
    """
    seq = np.asarray(seq, dtype=np.float32)
    frame_times = np.asarray(frame_times, dtype=np.float64)

    T = seq.shape[0]

    if T == 0:
        raise ValueError("empty sequence")

    if T == 1:
        return np.repeat(
            seq,
            t_out,
            axis=0,
        ).astype(np.float32)

    dst = np.linspace(
        float(frame_times[0]),
        float(frame_times[-1]),
        t_out,
    )

    flat = seq.reshape(T, -1)

    out = np.stack(
        [
            np.interp(
                dst,
                frame_times,
                flat[:, d],
            )
            for d in range(flat.shape[1])
        ],
        axis=1,
    )

    return (
        out.astype(np.float32)
        .reshape(
            t_out,
            *seq.shape[1:],
        )
    )


def velocity_from_time(seq, frame_times):
    """
    실제 timestamp 기준 velocity 계산.
    단위: normalized coordinate / second
    """
    seq = np.asarray(
        seq,
        dtype=np.float32,
    )

    frame_times = np.asarray(
        frame_times,
        dtype=np.float64,
    )

    T = seq.shape[0]

    if T <= 1:
        return np.zeros_like(
            seq,
            dtype=np.float32,
        )

    dt = np.diff(frame_times)

    if (
        not np.all(np.isfinite(dt))
        or np.any(dt <= 1e-8)
    ):
        raise ValueError(
            "timestamps are not strictly increasing"
        )

    flat = seq.reshape(T, -1)

    vel = np.empty_like(
        flat,
        dtype=np.float32,
    )

    for d in range(flat.shape[1]):
        vel[:, d] = np.gradient(
            flat[:, d],
            frame_times,
            edge_order=1,
        ).astype(np.float32)

    return vel.reshape(seq.shape)


# =========================================================
# Spatial normalization
# =========================================================
def apply_aspect(seq, width, height):
    """
    MediaPipe normalized coordinate의 aspect ratio 보정.

    기존 G1~G5와 동일하게:
        x *= width / height
        z *= width / height
    """
    seq = np.asarray(
        seq,
        dtype=np.float32,
    ).copy()

    ratio = (
        float(width)
        / float(height)
    )

    seq[..., 0] *= ratio
    seq[..., 2] *= ratio

    return seq


def normalize_hand(seq):
    """
    기존 G1~G5와 동일한 손 공간 정규화.

    1. wrist를 원점으로 이동
    2. wrist -> middle MCP(9)의 XY 거리로 scale 정규화
    """
    seq = np.asarray(
        seq,
        dtype=np.float32,
    ).copy()

    seq -= seq[
        :,
        WRIST:WRIST + 1,
        :
    ]

    span = np.linalg.norm(
        seq[
            :,
            MID_MCP,
            :2,
        ],
        axis=1,
    )

    span = np.maximum(
        span,
        1e-6,
    )[
        :,
        None,
        None,
    ]

    return seq / span


# =========================================================
# One JSON -> D=127
# =========================================================
def build_one(path):
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        d = json.load(f)

    frames = d["frames"]

    if len(frames) == 0:
        raise ValueError(
            "no detected frames"
        )

    video = d["video"]

    width = float(
        video["width"]
    )

    height = float(
        video["height"]
    )

    fps = float(
        video["fps"]
    )

    total_frames = int(
        video["total_frames"]
    )

    detected_frames = int(
        video["detected_frames"]
    )

    if (
        not np.isfinite(fps)
        or fps <= 1e-6
    ):
        fps = 30.0

    if total_frames <= 0:
        raise ValueError(
            f"invalid total_frames={total_frames}"
        )

    # -----------------------------------------------------
    # 검출된 landmark의 실제 timestamp
    # -----------------------------------------------------
    frame_times = np.asarray(
        [
            float(frame["tMs"]) / 1000.0
            for frame in frames
        ],
        dtype=np.float64,
    )

    # 첫 검출 frame을 0초 기준으로 맞춤
    frame_times -= frame_times[0]

    if (
        len(frame_times) >= 2
        and np.any(
            np.diff(frame_times) <= 1e-8
        )
    ):
        raise ValueError(
            "non-increasing detected timestamps"
        )

    # -----------------------------------------------------
    # [detected T, 21, 3]
    # -----------------------------------------------------
    hand = np.asarray(
        [
            [
                [
                    lm["x"],
                    lm["y"],
                    lm["z"],
                ]
                for lm in frame["landmarks"]
            ]
            for frame in frames
        ],
        dtype=np.float32,
    )

    if hand.shape[1:] != (21, 3):
        raise ValueError(
            f"invalid landmark shape: {hand.shape}"
        )

    # -----------------------------------------------------
    # G1~G5와 동일한 spatial preprocessing
    # -----------------------------------------------------
    hand = apply_aspect(
        hand,
        width,
        height,
    )

    hand = normalize_hand(
        hand
    )

    # -----------------------------------------------------
    # 영상 전체 majority handedness 기준
    # Left -> Right canonicalization
    # -----------------------------------------------------
    majority = str(
        d.get(
            "majority_handedness",
            "Unknown",
        )
    ).lower()

    if majority == "left":
        hand[..., 0] *= -1.0

    # -----------------------------------------------------
    # T=32로 줄이기 전에 실제 시간 기준 velocity 계산
    # -----------------------------------------------------
    velocity = velocity_from_time(
        hand,
        frame_times,
    )

    # -----------------------------------------------------
    # 위치 / velocity 각각 T=32 resampling
    # -----------------------------------------------------
    hand32 = resample_time(
        hand,
        frame_times,
    )

    velocity32 = resample_time(
        velocity,
        frame_times,
    )

    # -----------------------------------------------------
    # 전체 원본 video timeline에서 valid mask 복원
    #
    # 검출 성공 = 1
    # 검출 실패 = 0
    # -----------------------------------------------------
    full_valid = np.zeros(
        (total_frames, 1),
        dtype=np.float32,
    )

    for frame in frames:
        idx = int(
            frame["frame_index"]
        )

        if 0 <= idx < total_frames:
            full_valid[
                idx,
                0,
            ] = 1.0

    # 전체 video frame의 시간축
    full_frame_times = (
        np.arange(
            total_frames,
            dtype=np.float64,
        )
        / float(fps)
    )

    # 기존 G1~G5와 동일하게 mask도 T=32 resampling
    valid32 = resample_time(
        full_valid,
        full_frame_times,
    )

    # -----------------------------------------------------
    # 최종 D=127
    #
    # hand xyz       63
    # hand velocity  63
    # valid mask       1
    # -------------------
    #                 127
    # -----------------------------------------------------
    feat = np.concatenate(
        [
            hand32.reshape(
                T_OUT,
                63,
            ),

            velocity32.reshape(
                T_OUT,
                63,
            ),

            valid32.reshape(
                T_OUT,
                1,
            ),
        ],
        axis=1,
    ).astype(np.float32)

    if feat.shape != (
        T_OUT,
        127,
    ):
        raise AssertionError(
            feat.shape
        )

    detection_rate = (
        float(detected_frames)
        / float(total_frames)
    )

    # 기존 dataset의 duration_sec 의미에 맞춰
    # 전체 영상 길이 사용
    duration_sec = (
        float(total_frames)
        / float(fps)
    )

    return (
        feat.reshape(-1),
        fps,
        duration_sec,
        detection_rate,
        majority,
    )


# =========================================================
# Main
# =========================================================
def main():

    paths = []

    for g in range(6, 25):

        gesture_dir = (
            INPUT_ROOT
            / f"G{g}"
        )

        files = sorted(
            gesture_dir.glob(
                "*.json"
            )
        )

        if len(files) != 50:
            raise RuntimeError(
                f"G{g}: expected 50 JSONs, "
                f"found {len(files)}"
            )

        # MediaPipe가 단 한 프레임도 검출하지 못한
        # JSON은 feature를 만들 수 없으므로 제외한다.
        valid_files = []

        for path in files:
            with open(path, "r", encoding="utf-8") as f:
                info = json.load(f)

            if len(info.get("frames", [])) == 0:
                print(
                    "SKIP zero-detection:",
                    path,
                )
                continue

            valid_files.append(path)

        paths.extend(valid_files)

    print(
        "JSON files:",
        len(paths),
    )

    if len(paths) != 947:
        raise RuntimeError(
            f"Expected 947 usable JSON files, "
            f"found {len(paths)}"
        )

    X = []

    gestures = []
    performers = []
    sessions = []
    roles = []
    hands = []
    names = []

    fps_list = []
    duration_list = []
    detection_rates = []

    for i, path in enumerate(
        paths,
        1,
    ):

        gesture = (
            path.parent.name
        )

        (
            feat,
            fps,
            duration,
            det_rate,
            hand,
        ) = build_one(path)

        X.append(feat)

        gestures.append(
            gesture
        )

        # AH650에는 실제 사용자 identity가 없음.
        # 기존 P01~P10과 혼동되지 않도록 별도 표기.
        performers.append(
            "AH"
        )

        sessions.append(
            "AH650"
        )

        roles.append(
            "gesture_only"
        )

        if hand == "right":
            hands.append("R")

        elif hand == "left":
            hands.append("L")

        else:
            hands.append("U")

        names.append(
            f"{gesture}/{path.stem}.mp4"
        )

        fps_list.append(
            fps
        )

        duration_list.append(
            duration
        )

        detection_rates.append(
            det_rate
        )

        if (
            i % 50 == 0
            or i == len(paths)
        ):
            print(
                f"[{i:3d}/{len(paths)}] "
                f"{gesture}"
            )

    X = np.stack(
        X,
        axis=0,
    ).astype(np.float32)

    # -----------------------------------------------------
    # Final validation
    # -----------------------------------------------------
    expected_shape = (
        947,
        T_OUT * 127,
    )

    if X.shape != expected_shape:
        raise AssertionError(
            f"Expected {expected_shape}, "
            f"got {X.shape}"
        )

    if not np.all(
        np.isfinite(X)
    ):
        raise RuntimeError(
            "X contains NaN or Inf"
        )

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------
    np.savez_compressed(
        OUTPUT_PATH,

        gesture=np.asarray(
            gestures
        ),

        performer=np.asarray(
            performers
        ),

        session=np.asarray(
            sessions
        ),

        role=np.asarray(
            roles
        ),

        hand=np.asarray(
            hands
        ),

        name=np.asarray(
            names
        ),

        duration_sec=np.asarray(
            duration_list,
            dtype=np.float32,
        ),

        fps=np.asarray(
            fps_list,
            dtype=np.float32,
        ),

        detection_rate=np.asarray(
            detection_rates,
            dtype=np.float32,
        ),

        truncated=np.zeros(
            len(paths),
            dtype=bool,
        ),

        canonical_hand=np.bool_(
            True
        ),

        T=np.int64(
            T_OUT
        ),

        velocity_mode=np.array(
            "real_time_units_per_second"
        ),

        X=X,

        D=np.int64(
            127
        ),

        feature_layout=np.array(
            "hand_xyz_63+hand_velocity_63+valid_mask_1"
        ),

        hand_only=np.bool_(
            True
        ),

        source_dataset=np.array(
            str(
                INPUT_ROOT.resolve()
            )
        ),

        # AH650은 사용자 인증용 identity dataset이 아님
        identity_available=np.bool_(
            False
        ),
    )

    # -----------------------------------------------------
    # Report
    # -----------------------------------------------------
    print()
    print(
        "============================================="
    )
    print(
        " AH650 HAND-ONLY DATASET COMPLETE"
    )
    print(
        "============================================="
    )

    print(
        "output       :",
        OUTPUT_PATH.resolve(),
    )

    print(
        "X            :",
        X.shape,
    )

    print(
        "T            :",
        T_OUT,
    )

    print(
        "D            :",
        127,
    )

    print(
        "samples      :",
        len(X),
    )

    print(
        "NaN          :",
        int(
            np.isnan(X).sum()
        ),
    )

    print(
        "Inf          :",
        int(
            np.isinf(X).sum()
        ),
    )

    print(
        "det rate mean:",
        f"{np.mean(detection_rates):.4f}",
    )

    unique, counts = np.unique(
        gestures,
        return_counts=True,
    )

    print()
    print(
        "Gesture counts:"
    )

    for gesture, count in zip(
        unique,
        counts,
    ):
        print(
            f"{gesture}: {count}"
        )


if __name__ == "__main__":
    main()
