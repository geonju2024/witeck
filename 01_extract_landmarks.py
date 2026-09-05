"""
01_extract_landmarks.py

영상 -> MediaPipe 랜드마크(.npz) 캐시.

추가된 시간 정보
----------------
절대적인 수행 속도(tempo)를 보존하기 위해 영상별로 다음 값을 함께 저장한다.

    fps
    frame_times   # 각 프레임의 시간(초)
    duration_sec  # 실제 처리된 영상 구간의 길이

이 값들은 02_build_features.py에서
실제 시간 기준 velocity = Δposition / Δtime
를 계산할 때 사용한다.

기존 MediaPipe 처리 방식은 유지한다.

경로는 paths.py 가 정한다.
인자 없이 실행하면

    WITECH/videos
    ->
    WITECH/derived/landmarks

로 동작한다.

사용법:
    python 01_extract_landmarks.py
    python 01_extract_landmarks.py --workers 8
"""

import argparse
import glob
import os
import sys
import time

import numpy as np

import paths


# ---- 프로젝트 공통 상수 ----
N_HAND = 21
N_POSE = 33
MAX_HANDS = 2


def read_frames(path, max_frames=300):
    """
    영상을 프레임과 시간정보로 읽는다.

    Returns
    -------
    frames : list[np.ndarray]
        OpenCV BGR frame 목록

    fps : float
        OpenCV가 읽은 영상 FPS.
        잘못된 값이면 frame timestamp를 이용해 추정한다.

    frame_times : np.ndarray, shape [T]
        각 처리 프레임의 시간 위치(초).

    duration_sec : float
        실제 처리된 T개 프레임 구간의 길이.

    source_frame_count : int
        컨테이너 metadata의 전체 프레임 수.

    truncated : bool
        max_frames 때문에 원본 영상 일부가 잘렸는지 여부.

    Notes
    -----
    CAP_PROP_ORIENTATION_AUTO가 중요하다.

    스마트폰 세로 영상은 픽셀 자체는 가로 방향으로 저장되고
    회전 metadata만 붙어 있는 경우가 있다.
    이 옵션이 없으면 MediaPipe가 회전되지 않은 프레임을 볼 수 있다.
    """
    import cv2

    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)

    if not cap.isOpened():
        cap.release()
        return [], 0.0, np.empty(0, np.float32), 0.0, 0, False

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    source_frame_count = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

    frames = []
    times = []

    while len(frames) < max_frames:
        ok, frame = cap.read()

        if not ok:
            break

        frames.append(frame)

        # 현재 프레임의 재생 위치(ms).
        # MP4에 따라 0 또는 중복값이 나올 수 있어서 아래에서 검증 후 fallback한다.
        pos_msec = float(cap.get(cv2.CAP_PROP_POS_MSEC))
        times.append(pos_msec / 1000.0)

    cap.release()

    T = len(frames)

    if T == 0:
        return [], fps, np.empty(0, np.float32), 0.0, source_frame_count, False

    frame_times = np.asarray(times, dtype=np.float64)

    # ---------------------------------------------------------
    # timestamp 검증
    # ---------------------------------------------------------
    # 실제 timestamp가 정상이라면 strictly increasing이어야 한다.
    # OpenCV/backend에 따라 전부 0이거나 같은 값이 나오는 경우가 있어
    # 그때는 FPS 기반 일정 간격 timestamp로 대체한다.
    # ---------------------------------------------------------
    timestamps_valid = (
        len(frame_times) == T
        and np.all(np.isfinite(frame_times))
        and (
            T == 1
            or np.all(np.diff(frame_times) > 1e-6)
        )
    )

    # fps도 비정상일 수 있다.
    fps_valid = np.isfinite(fps) and fps > 1e-6

    if not timestamps_valid:
        if fps_valid:
            frame_times = np.arange(T, dtype=np.float64) / fps
        else:
            # 둘 다 없으면 30 fps를 fallback으로 사용.
            # 이 경우 절대속도 신뢰도가 떨어지므로 출력에 경고를 남긴다.
            fps = 30.0
            frame_times = np.arange(T, dtype=np.float64) / fps

    else:
        # timestamp는 정상인데 fps metadata가 이상하면 timestamp로 추정
        if not fps_valid and T >= 2:
            dt = np.diff(frame_times)
            median_dt = float(np.median(dt))

            if median_dt > 1e-6:
                fps = 1.0 / median_dt
            else:
                fps = 30.0

    # 첫 프레임을 0초로 맞춘다.
    frame_times = frame_times - frame_times[0]

    # 처리된 프레임 구간의 실제 길이.
    # 마지막 프레임 timestamp에 한 프레임 간격을 더해 전체 구간 길이로 정의한다.
    if T >= 2:
        dt = np.diff(frame_times)
        valid_dt = dt[np.isfinite(dt) & (dt > 1e-6)]

        if len(valid_dt) > 0:
            last_dt = float(np.median(valid_dt))
        else:
            last_dt = 1.0 / max(fps, 1e-6)

        duration_sec = float(frame_times[-1] + last_dt)

    else:
        duration_sec = 1.0 / max(fps, 1e-6)

    truncated = (
        source_frame_count > 0
        and source_frame_count > T
        and T >= max_frames
    )

    return (
        frames,
        float(fps),
        frame_times.astype(np.float32),
        float(duration_sec),
        source_frame_count,
        truncated,
    )


def extract_one(args):
    video_path, out_dir, videos_root = args

    import cv2
    import mediapipe as mp

    # 출력 경로는 videos/ 하위 구조를 그대로 미러링한다.
    #
    # videos/G1/P01/20260725/G1_P01_20260725_own_001.mp4
    # ->
    # landmarks/G1/P01/20260725/G1_P01_20260725_own_001.npz
    #
    # basename만 사용하면 다른 세션의 동명 파일이 충돌할 수 있으므로
    # 반드시 상대경로 전체를 유지한다.
    rel = os.path.relpath(
        video_path,
        videos_root,
    )

    stem = os.path.splitext(rel)[0].replace(
        os.sep,
        "/",
    )

    out_path = os.path.join(
        out_dir,
        os.path.splitext(rel)[0] + ".npz",
    )

    if os.path.exists(out_path):
        return stem, "cached", 0.0

    os.makedirs(
        os.path.dirname(out_path),
        exist_ok=True,
    )

    t0 = time.time()

    (
        frames,
        fps,
        frame_times,
        duration_sec,
        source_frame_count,
        truncated,
    ) = read_frames(
        video_path
    )

    if not frames:
        return stem, "EMPTY", 0.0

    H, W = frames[0].shape[:2]
    T = len(frames)

    hand = np.zeros(
        (T, MAX_HANDS, N_HAND, 3),
        np.float32,
    )

    hand_valid = np.zeros(
        (T, MAX_HANDS),
        np.uint8,
    )

    handedness = np.full(
        (T, MAX_HANDS),
        -1,
        np.int8,
    )

    pose = np.zeros(
        (T, N_POSE, 4),
        np.float32,
    )

    pose_valid = np.zeros(
        (T,),
        np.uint8,
    )

    mp_hands = mp.solutions.hands
    mp_pose = mp.solutions.pose

    # tracking mode 유지
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=MAX_HANDS,
        model_complexity=1,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3,
    ) as hands, mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3,
    ) as poser:

        for t, frame in enumerate(frames):

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            rgb.flags.writeable = False

            # -------------------------
            # Hands
            # -------------------------
            rh = hands.process(rgb)

            if rh.multi_hand_landmarks:

                for k, hl in enumerate(
                    rh.multi_hand_landmarks[:MAX_HANDS]
                ):

                    hand[t, k] = [
                        [p.x, p.y, p.z]
                        for p in hl.landmark
                    ]

                    hand_valid[t, k] = 1

                    if (
                        rh.multi_handedness
                        and k < len(rh.multi_handedness)
                    ):
                        label = (
                            rh.multi_handedness[k]
                            .classification[0]
                            .label
                        )

                        handedness[t, k] = (
                            1
                            if label == "Right"
                            else 0
                        )

            # -------------------------
            # Pose
            # -------------------------
            rp = poser.process(rgb)

            if rp.pose_landmarks:

                pose[t] = [
                    [
                        p.x,
                        p.y,
                        p.z,
                        p.visibility,
                    ]
                    for p in rp.pose_landmarks.landmark
                ]

                pose_valid[t] = 1

    # ---------------------------------------------------------
    # 영상별 raw landmark + 시간 metadata 저장
    # ---------------------------------------------------------
    np.savez_compressed(
        out_path,

        hand=hand,
        hand_valid=hand_valid,
        handedness=handedness,

        pose=pose,
        pose_valid=pose_valid,

        width=W,
        height=H,

        n_frames=T,

        # 새로 추가
        fps=np.float32(fps),
        frame_times=frame_times,
        duration_sec=np.float32(duration_sec),
        source_frame_count=np.int32(source_frame_count),
        truncated=np.bool_(truncated),

        relative_path=(
            stem
            + os.path.splitext(rel)[1]
        ),
    )

    hand_rate = (
        hand_valid
        .max(axis=1)
        .mean()
    )

    pose_rate = (
        pose_valid.mean()
    )

    trunc_msg = " TRUNCATED" if truncated else ""

    return (
        stem,
        (
            f"T={T} "
            f"fps={fps:.2f} "
            f"duration={duration_sec:.2f}s "
            f"hand={hand_rate:.0%} "
            f"pose={pose_rate:.0%}"
            f"{trunc_msg}"
        ),
        time.time() - t0,
    )


def main():
    try:
        sys.stdout.reconfigure(
            encoding="utf-8"
        )
    except Exception:
        pass

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--videos",
        default=str(paths.VIDEOS_DIR),
        help=(
            "영상 폴더 "
            "(하위 폴더까지 재귀 탐색). "
            f"기본값: {paths.VIDEOS_DIR}"
        ),
    )

    ap.add_argument(
        "--out",
        default=str(paths.LANDMARKS_DIR),
        help=(
            "랜드마크 캐시 폴더. "
            f"기본값: {paths.LANDMARKS_DIR}"
        ),
    )

    ap.add_argument(
        "--workers",
        type=int,
        default=max(
            1,
            (os.cpu_count() or 4) // 2,
        ),
    )

    args = ap.parse_args()

    videos_dir = str(
        paths.assert_external(
            args.videos,
            "영상",
        )
    )

    out_dir = str(
        paths.assert_external(
            args.out,
            "랜드마크 캐시",
        )
    )

    print(
        f"영상   : {videos_dir}"
    )

    print(
        f"출력   : {out_dir}"
    )

    os.makedirs(
        out_dir,
        exist_ok=True,
    )

    exts = (
        "mp4",
        "MP4",
        "mov",
        "MOV",
        "avi",
        "AVI",
        "mkv",
    )

    files = sorted(
        {
            f
            for ext in exts
            for f in glob.glob(
                os.path.join(
                    videos_dir,
                    "**",
                    f"*.{ext}",
                ),
                recursive=True,
            )
        }
    )

    print(
        f"영상 {len(files)}개 발견, "
        f"workers={args.workers}"
    )

    jobs = [
        (
            f,
            out_dir,
            videos_dir,
        )
        for f in files
    ]

    t0 = time.time()

    if args.workers > 1:

        from multiprocessing import Pool

        with Pool(args.workers) as pool:

            for i, (
                stem,
                msg,
                dt,
            ) in enumerate(
                pool.imap_unordered(
                    extract_one,
                    jobs,
                ),
                1,
            ):

                print(
                    f"[{i}/{len(files)}] "
                    f"{stem}: "
                    f"{msg} "
                    f"({dt:.1f}s)",
                    flush=True,
                )

    else:

        for i, job in enumerate(
            jobs,
            1,
        ):

            stem, msg, dt = extract_one(
                job
            )

            print(
                f"[{i}/{len(files)}] "
                f"{stem}: "
                f"{msg} "
                f"({dt:.1f}s)",
                flush=True,
            )

    print(
        f"\n완료. "
        f"총 {time.time() - t0:.0f}초 "
        f"-> {out_dir}"
    )


if __name__ == "__main__":
    main()
