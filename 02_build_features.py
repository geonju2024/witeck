"""
02_build_features.py

MediaPipe 랜드마크(.npz)
-> 고정 길이 시계열 특징
-> dataset.npz

핵심 변경
---------
기존 코드는 모든 영상을 먼저 T=32로 리샘플한 뒤 단순 좌표 차분을 계산했다.
그 방식은 1초짜리 영상과 3초짜리 영상의 절대 수행속도 차이를 상당 부분 제거한다.

이번 버전에서는 다음 순서로 처리한다.

    raw landmark
    -> 종횡비 보정
    -> 결측 프레임 시간기반 보간
    -> 손/pose 공간 정규화
    -> 실제 frame_times 기준 velocity 계산 (Δposition / Δtime)
    -> 위치와 velocity를 각각 T=32로 리샘플

따라서 최종 feature 크기는 기존과 동일하다.

    Hand xyz        63
    Hand velocity   63   # 실제 시간 기준
    Pose xyz        21
    Pose velocity   21   # 실제 시간 기준
    Valid mask       1
    -------------------
    Total           169 / frame

영상 하나:
    [32, 169]

추가 metadata
-------------
dataset.npz에 다음 값을 함께 저장한다.

    duration_sec
    fps

duration_sec는 이후 ML / GRU / LSTM / 1D CNN / Transformer가
절대적인 수행시간 정보를 사용할 수 있도록 별도 global feature로 사용한다.

fps 자체는 사람의 특징으로 학습시키지 않고,
시간 계산 및 품질 확인용 metadata로만 저장한다.

사용법:
    python 02_build_features.py
"""

import argparse
import csv
import os
import sys

import numpy as np

import paths


T_OUT = 32

WRIST = 0
IDX_MCP = 5
MID_MCP = 9
PINKY_MCP = 17

# 양 어깨, 양 팔꿈치, 양 손목, 코
POSE_UPPER = [
    11, 12,
    13, 14,
    15, 16,
    0,
]


# =========================================================
# Time helpers
# =========================================================
def validate_frame_times(
    frame_times: np.ndarray,
    n_frames: int,
    fps: float,
) -> np.ndarray:
    """
    frame_times를 검증하고 필요하면 fps 기반 시간축으로 복구한다.

    반환:
        [T] seconds, 첫 프레임 = 0
    """
    frame_times = np.asarray(
        frame_times,
        dtype=np.float64,
    ).reshape(-1)

    valid = (
        len(frame_times) == n_frames
        and np.all(np.isfinite(frame_times))
        and (
            n_frames <= 1
            or np.all(np.diff(frame_times) > 1e-8)
        )
    )

    if not valid:
        if not np.isfinite(fps) or fps <= 1e-6:
            fps = 30.0

        frame_times = (
            np.arange(
                n_frames,
                dtype=np.float64,
            )
            / float(fps)
        )

    frame_times = (
        frame_times
        - frame_times[0]
    )

    return frame_times.astype(
        np.float32
    )


def resample_time(
    seq: np.ndarray,
    frame_times: np.ndarray,
    t_out: int = T_OUT,
) -> np.ndarray:
    """
    실제 시간축 seq(T, ...)을
    시작~끝 사이 T_OUT개 시점으로 선형보간한다.

    주의:
        출력 길이는 항상 32지만,
        velocity 자체가 units/sec이므로 절대 속도 크기는 유지된다.
    """
    seq = np.asarray(seq)
    frame_times = np.asarray(
        frame_times,
        dtype=np.float64,
    )

    T = seq.shape[0]

    if T == 0:
        raise ValueError(
            "빈 sequence는 resample할 수 없습니다."
        )

    if T == 1:
        return np.repeat(
            seq.astype(np.float32),
            t_out,
            axis=0,
        )

    src = frame_times

    # 마지막 실제 timestamp까지 동일하게 32개 지점 생성
    dst = np.linspace(
        float(src[0]),
        float(src[-1]),
        t_out,
    )

    flat = seq.reshape(
        T,
        -1,
    )

    out = np.stack(
        [
            np.interp(
                dst,
                src,
                flat[:, d],
            )
            for d in range(
                flat.shape[1]
            )
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


def interp_missing_time(
    seq: np.ndarray,
    valid: np.ndarray,
    frame_times: np.ndarray,
):
    """
    미검출 프레임을 실제 시간축 기준으로 선형보간한다.

    valid == 0 인 위치를
    앞뒤 유효 frame의 실제 timestamp를 이용해 채운다.

    전부 결측이면:
        (원본 seq, False)
    """
    seq = np.asarray(seq)
    valid = np.asarray(valid)
    frame_times = np.asarray(
        frame_times,
        dtype=np.float64,
    )

    T = seq.shape[0]

    idx = np.where(
        valid > 0
    )[0]

    if len(idx) == 0:
        return seq, False

    if len(idx) == T:
        return seq, True

    flat = (
        seq.reshape(T, -1)
        .copy()
    )

    for d in range(
        flat.shape[1]
    ):
        flat[:, d] = np.interp(
            frame_times,
            frame_times[idx],
            flat[idx, d],
        )

    return (
        flat.reshape(
            seq.shape
        ),
        True,
    )


def velocity_from_time(
    seq: np.ndarray,
    frame_times: np.ndarray,
) -> np.ndarray:
    """
    실제 시간 기준 velocity 계산.

        velocity ~= Δposition / Δtime

    seq:
        [T, ...]

    frame_times:
        [T] seconds

    반환:
        seq와 같은 shape
        단위는 '정규화 좌표 / second'

    내부 frame에서는 np.gradient를 사용하여
    앞뒤 프레임을 함께 고려한다.
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

    dt = np.diff(
        frame_times
    )

    # 비정상 timestamp 방어
    if (
        not np.all(np.isfinite(dt))
        or np.any(dt <= 1e-8)
    ):
        raise ValueError(
            "frame_times가 strictly increasing이 아닙니다."
        )

    flat = seq.reshape(
        T,
        -1,
    )

    vel = np.empty_like(
        flat,
        dtype=np.float32,
    )

    # 각 feature dimension별 시간 미분
    for d in range(
        flat.shape[1]
    ):
        vel[:, d] = np.gradient(
            flat[:, d],
            frame_times,
            edge_order=1,
        ).astype(
            np.float32
        )

    return vel.reshape(
        seq.shape
    )


# =========================================================
# Hand helpers
# =========================================================
def pick_primary_hand(
    hand,
    hand_valid,
):
    """
    (T,2,21,3) 중 주 손 하나를 선택한다.

    두 손이 잡힌 경우
    화면에서 더 크게 검출된 손을 선택한다.
    """
    T = hand.shape[0]

    out = np.zeros(
        (T, 21, 3),
        np.float32,
    )

    valid = np.zeros(
        (T,),
        np.uint8,
    )

    chosen = np.full(
        (T,),
        -1,
        np.int8,
    )

    for t in range(T):

        best = -1
        best_size = -1.0

        for k in range(
            hand.shape[1]
        ):

            if not hand_valid[t, k]:
                continue

            pts = hand[
                t,
                k,
                :,
                :2,
            ]

            size = float(
                np.ptp(
                    pts[:, 0]
                )
                + np.ptp(
                    pts[:, 1]
                )
            )

            if size > best_size:
                best = k
                best_size = size

        if best >= 0:

            out[t] = hand[
                t,
                best,
            ]

            valid[t] = 1
            chosen[t] = best

    return (
        out,
        valid,
        chosen,
    )


def primary_hand_side(
    chosen,
    handedness,
):
    """
    주 손 좌우를 다수결로 판정.

    1 = Right
    0 = Left
   -1 = unknown
    """
    sides = [
        int(
            handedness[t, k]
        )
        for t, k in enumerate(
            chosen
        )
        if (
            k >= 0
            and handedness[t, k] >= 0
        )
    ]

    if not sides:
        return -1

    return (
        1
        if float(
            np.mean(sides)
        ) >= 0.5
        else 0
    )


def mirror_to_right(
    h,
    p,
):
    """
    왼손 수행분을 오른손 기준으로 좌우 반전한다.
    """
    h = h.copy()
    h[..., 0] *= -1.0

    p = p.copy()
    p[..., 0] *= -1.0

    # [L shoulder, R shoulder,
    #  L elbow,    R elbow,
    #  L wrist,    R wrist,
    #  nose]
    p = p[
        :,
        [1, 0, 3, 2, 5, 4, 6],
        :
    ]

    return h, p


# =========================================================
# Spatial normalization
# =========================================================
def apply_aspect(
    seq,
    width,
    height,
):
    """
    MediaPipe normalized coordinates를
    x/y가 동일한 물리적 배율을 갖는 좌표로 보정한다.
    """
    seq = seq.astype(
        np.float32,
        copy=True,
    )

    ratio = (
        float(width)
        / float(height)
    )

    seq[..., 0] *= ratio

    if seq.shape[-1] > 2:
        seq[..., 2] *= ratio

    return seq


def normalize_hand(
    seq,
):
    """
    손목 원점 이동 + 손 크기 정규화.

    위치와 카메라 거리에 대한 영향을 줄인다.
    """
    seq = seq.copy()

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


def normalize_pose(
    pose_seq,
):
    """
    어깨 중심 원점 이동 + 어깨너비 scale 정규화.
    """
    xyz = pose_seq[
        :,
        :,
        :3,
    ]

    center = (
        xyz[:, 11, :]
        + xyz[:, 12, :]
    ) / 2.0

    width = np.linalg.norm(
        xyz[:, 11, :2]
        - xyz[:, 12, :2],
        axis=1,
    )

    width = np.maximum(
        width,
        1e-6,
    )[
        :,
        None,
        None,
    ]

    sub = (
        xyz[
            :,
            POSE_UPPER,
            :
        ]
        - center[
            :,
            None,
            :
        ]
    )

    return (
        sub / width
    ).astype(
        np.float32
    )


# =========================================================
# Feature builder
# =========================================================
def features_from_npz(
    path,
    canonical_hand=True,
):
    """
    landmark npz 1개
    ->
    feature vector [32*169],
    hand detection rate,
    hand side,
    duration_sec,
    fps,
    truncated
    """
    z = np.load(
        path
    )

    W = float(
        z["width"]
    )

    H = float(
        z["height"]
    )

    hand = apply_aspect(
        z["hand"],
        W,
        H,
    )

    hand_valid = z[
        "hand_valid"
    ]

    pose = z[
        "pose"
    ].astype(
        np.float32,
        copy=True,
    )

    pose_valid = z[
        "pose_valid"
    ]

    pose[
        :,
        :,
        :3,
    ] = apply_aspect(
        pose[
            :,
            :,
            :3,
        ],
        W,
        H,
    )

    T = hand.shape[0]

    # -----------------------------------------------------
    # 시간 metadata
    # -----------------------------------------------------
    fps = (
        float(z["fps"])
        if "fps" in z.files
        else 30.0
    )

    if (
        not np.isfinite(fps)
        or fps <= 1e-6
    ):
        fps = 30.0

    if "frame_times" in z.files:
        frame_times = z[
            "frame_times"
        ]
    else:
        # 구버전 landmark 대응용 fallback
        frame_times = (
            np.arange(
                T,
                dtype=np.float32,
            )
            / fps
        )

    frame_times = validate_frame_times(
        frame_times,
        T,
        fps,
    )

    if "duration_sec" in z.files:
        duration_sec = float(
            z["duration_sec"]
        )
    else:
        if T >= 2:
            duration_sec = float(
                frame_times[-1]
                + np.median(
                    np.diff(
                        frame_times
                    )
                )
            )
        else:
            duration_sec = (
                1.0 / fps
            )

    truncated = (
        bool(z["truncated"])
        if "truncated" in z.files
        else False
    )

    # -----------------------------------------------------
    # primary hand
    # -----------------------------------------------------
    h, hv, chosen = pick_primary_hand(
        hand,
        hand_valid,
    )

    side = primary_hand_side(
        chosen,
        z["handedness"],
    )

    det_rate = float(
        hv.mean()
    )

    # -----------------------------------------------------
    # 결측 보간 - 실제 시간축 기준
    # -----------------------------------------------------
    h, hand_ok = interp_missing_time(
        h,
        hv,
        frame_times,
    )

    if not hand_ok:
        return (
            None,
            det_rate,
            side,
            duration_sec,
            fps,
            truncated,
        )

    # 손 공간 정규화
    h = normalize_hand(
        h
    )

    # Pose 결측 보간
    p_raw, pose_ok = interp_missing_time(
        pose,
        pose_valid,
        frame_times,
    )

    if pose_ok:
        p = normalize_pose(
            p_raw
        )
    else:
        p = np.zeros(
            (
                T,
                len(POSE_UPPER),
                3,
            ),
            np.float32,
        )

    # 왼손 -> 오른손 기준 통일
    if (
        canonical_hand
        and side == 0
    ):
        h, p = mirror_to_right(
            h,
            p,
        )

    # -----------------------------------------------------
    # 핵심 변경:
    # 32 frame으로 줄이기 전에 실제 시간기준 velocity 계산
    # -----------------------------------------------------
    vh = velocity_from_time(
        h,
        frame_times,
    )

    vp = velocity_from_time(
        p,
        frame_times,
    )

    # -----------------------------------------------------
    # 모든 시계열을 같은 32개 실제 시간 위치로 리샘플
    # -----------------------------------------------------
    h32 = resample_time(
        h,
        frame_times,
    )

    vh32 = resample_time(
        vh,
        frame_times,
    )

    p32 = resample_time(
        p,
        frame_times,
    )

    vp32 = resample_time(
        vp,
        frame_times,
    )

    v32 = resample_time(
        hv.astype(
            np.float32
        )[:, None],
        frame_times,
    )

    # -----------------------------------------------------
    # 기존과 동일한 169 features / frame
    # -----------------------------------------------------
    feat = np.concatenate(
        [
            h32.reshape(
                T_OUT,
                -1,
            ),       # 63

            vh32.reshape(
                T_OUT,
                -1,
            ),       # 63, units/sec

            p32.reshape(
                T_OUT,
                -1,
            ),       # 21

            vp32.reshape(
                T_OUT,
                -1,
            ),       # 21, units/sec

            v32,     # 1
        ],
        axis=1,
    )

    assert feat.shape == (
        T_OUT,
        169,
    ), feat.shape

    return (
        feat.reshape(-1).astype(
            np.float32
        ),
        det_rate,
        side,
        duration_sec,
        fps,
        truncated,
    )


# =========================================================
# Main
# =========================================================
def main():
    try:
        sys.stdout.reconfigure(
            encoding="utf-8"
        )
    except Exception:
        pass

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--landmarks",
        default=str(
            paths.LANDMARKS_DIR
        ),
        help=(
            "랜드마크 캐시 폴더. "
            f"기본값: {paths.LANDMARKS_DIR}"
        ),
    )

    ap.add_argument(
        "--labels",
        default=str(
            paths.LABELS_CSV
        ),
        help=(
            "CSV: "
            "relative_path,gesture,performer,session,role "
            "(헤더 포함). "
            f"기본값: {paths.LABELS_CSV}"
        ),
    )

    ap.add_argument(
        "--out",
        default=str(
            paths.DATASET_NPZ
        ),
        help=(
            "출력 dataset.npz. "
            f"기본값: {paths.DATASET_NPZ}"
        ),
    )

    ap.add_argument(
        "--min-det",
        type=float,
        default=0.15,
        help=(
            "손 검출률이 이 값 미만인 영상은 제외"
        ),
    )

    ap.add_argument(
        "--keep-hand-side",
        action="store_true",
        help=(
            "좌우 정규화를 끈다. "
            "기본값은 오른손 기준 통일."
        ),
    )

    ap.add_argument(
        "--drop-truncated",
        action="store_true",
        help=(
            "01번에서 max_frames 때문에 잘린 영상을 제외한다."
        ),
    )

    args = ap.parse_args()

    landmarks_dir = str(
        paths.assert_external(
            args.landmarks,
            "랜드마크 캐시",
        )
    )

    labels_path = str(
        paths.assert_external(
            args.labels,
            "labels.csv",
        )
    )

    out_path = str(
        paths.assert_external(
            args.out,
            "dataset.npz",
        )
    )

    print(
        f"랜드마크: {landmarks_dir}"
    )

    print(
        f"라벨    : {labels_path}"
    )

    print(
        f"출력    : {out_path}\n"
    )

    os.makedirs(
        os.path.dirname(
            out_path
        ),
        exist_ok=True,
    )

    # -----------------------------------------------------
    # labels.csv
    # -----------------------------------------------------
    meta = {}

    with open(
        labels_path,
        encoding="utf-8-sig",
        newline="",
    ) as f:

        for row in csv.DictReader(
            f
        ):
            rel = (
                row[
                    "relative_path"
                ]
                .strip()
                .replace(
                    "\\",
                    "/",
                )
            )

            meta[rel] = row

    # -----------------------------------------------------
    # build dataset
    # -----------------------------------------------------
    X = []

    gestures = []
    performers = []
    sessions = []
    roles = []
    hands = []
    names = []

    durations = []
    fps_values = []
    truncated_flags = []

    dropped = []

    for rel, row in sorted(
        meta.items()
    ):

        path = os.path.join(
            landmarks_dir,
            os.path.splitext(
                rel
            )[0] + ".npz",
        )

        if not os.path.exists(
            path
        ):
            dropped.append(
                (
                    rel,
                    "랜드마크 npz 없음 "
                    "(01번을 먼저 실행)",
                )
            )
            continue

        (
            feat,
            det,
            side,
            duration_sec,
            fps,
            truncated,
        ) = features_from_npz(
            path,
            canonical_hand=(
                not args.keep_hand_side
            ),
        )

        if feat is None:
            dropped.append(
                (
                    rel,
                    "손 미검출 100%",
                )
            )
            continue

        if det < args.min_det:
            dropped.append(
                (
                    rel,
                    f"손 검출률 {det:.0%}",
                )
            )
            continue

        if (
            args.drop_truncated
            and truncated
        ):
            dropped.append(
                (
                    rel,
                    "max_frames로 영상 일부가 잘림",
                )
            )
            continue

        X.append(
            feat
        )

        gestures.append(
            row["gesture"].strip()
        )

        performers.append(
            row["performer"].strip()
        )

        sessions.append(
            row["session"].strip()
        )

        roles.append(
            row["role"].strip()
        )

        hands.append(
            {
                1: "R",
                0: "L",
            }.get(
                side,
                "?",
            )
        )

        names.append(
            rel
        )

        durations.append(
            float(duration_sec)
        )

        fps_values.append(
            float(fps)
        )

        truncated_flags.append(
            bool(truncated)
        )

    if not X:
        print(
            "사용 가능한 샘플이 0개입니다."
        )

        print(
            f"labels.csv 행 수: {len(meta)}"
        )

        if not meta:
            print(
                "-> labels.csv에 데이터 행이 없습니다."
            )
        else:
            print(
                "-> 제외 사유:"
            )

            for sample, reason in dropped[
                :20
            ]:
                print(
                    f"   {sample}: {reason}"
                )

        raise SystemExit(1)

    X = np.stack(
        X
    )

    durations = np.asarray(
        durations,
        dtype=np.float32,
    )

    fps_values = np.asarray(
        fps_values,
        dtype=np.float32,
    )

    truncated_flags = np.asarray(
        truncated_flags,
        dtype=np.bool_,
    )

    # -----------------------------------------------------
    # 최종 dataset.npz
    # -----------------------------------------------------
    np.savez_compressed(
        out_path,

        X=X,

        gesture=np.array(
            gestures
        ),

        performer=np.array(
            performers
        ),

        session=np.array(
            sessions
        ),

        role=np.array(
            roles
        ),

        hand=np.array(
            hands
        ),

        name=np.array(
            names
        ),

        # 새 global metadata
        duration_sec=durations,
        fps=fps_values,
        truncated=truncated_flags,

        canonical_hand=(
            not args.keep_hand_side
        ),

        T=T_OUT,

        D=(
            X.shape[1]
            // T_OUT
        ),

        velocity_mode=np.array(
            "real_time_units_per_second"
        ),
    )

    # -----------------------------------------------------
    # summary
    # -----------------------------------------------------
    print(
        f"X = {X.shape} "
        f"(영상 {X.shape[0]}개 "
        f"x {X.shape[1]}차원)"
    )

    print(
        f"시계열 = "
        f"[N={len(X)}, "
        f"T={T_OUT}, "
        f"D={X.shape[1] // T_OUT}]"
    )

    print(
        "velocity = 실제 시간 기준 "
        "(정규화 좌표 / second)"
    )

    print(
        f"제스처 종류: "
        f"{sorted(set(gestures))}"
    )

    print(
        f"수행자: "
        f"{sorted(set(performers))}"
    )

    print(
        f"세션: "
        f"{sorted(set(sessions))}"
    )

    hc = {
        side: hands.count(
            side
        )
        for side in sorted(
            set(hands)
        )
    }

    mode = (
        "끔(왼손/오른손 구분 유지)"
        if args.keep_hand_side
        else "켬(오른손 기준 통일)"
    )

    print(
        f"주 손 분포: {hc} "
        f"좌우 정규화: {mode}"
    )

    print(
        "\nDuration(sec)"
    )

    print(
        f"  min    = {durations.min():.3f}"
    )

    print(
        f"  median = {np.median(durations):.3f}"
    )

    print(
        f"  mean   = {durations.mean():.3f}"
    )

    print(
        f"  max    = {durations.max():.3f}"
    )

    unique_fps = sorted(
        set(
            np.round(
                fps_values,
                3,
            )
        )
    )

    print(
        f"\nFPS 종류: {unique_fps}"
    )

    n_truncated = int(
        truncated_flags.sum()
    )

    print(
        f"truncated: "
        f"{n_truncated}/{len(X)}"
    )

    if dropped:
        print(
            f"\n제외 {len(dropped)}개:"
        )

        for sample, reason in dropped[
            :20
        ]:
            print(
                f"  {sample}: {reason}"
            )

        if len(dropped) > 20:
            print(
                f"  ... 나머지 "
                f"{len(dropped) - 20}개"
            )

    print(
        f"\n저장 -> {out_path}"
    )


if __name__ == "__main__":
    main()