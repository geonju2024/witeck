import cv2
import mediapipe as mp
import os
import json
import time
import argparse
from collections import Counter


# =========================================================
# 기본 설정
# =========================================================

VIDEO_ROOT = "data/AH650_full"
OUTPUT_ROOT = "data/AH650_landmarks_th03_cpu"
MODEL_PATH = "models/hand_landmarker.task"

DETECTION_THRESHOLD = 0.3
PRESENCE_THRESHOLD = 0.5
TRACKING_THRESHOLD = 0.5

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


# =========================================================
# 명령행 인자
# =========================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--start-gesture",
    type=int,
    required=True,
    help="시작 gesture 번호"
)

parser.add_argument(
    "--end-gesture",
    type=int,
    required=True,
    help="끝 gesture 번호"
)

args = parser.parse_args()

START_GESTURE = args.start_gesture
END_GESTURE = args.end_gesture


# =========================================================
# 한 영상에서 landmark 추출
# =========================================================

def extract_video(video_path, output_path):

    # 각 영상은 독립적인 VIDEO stream.
    # CPU delegate를 사용하는 Landmarker를 영상마다 새로 생성.
    options = HandLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=MODEL_PATH,
            delegate=BaseOptions.Delegate.CPU
        ),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=DETECTION_THRESHOLD,
        min_hand_presence_confidence=PRESENCE_THRESHOLD,
        min_tracking_confidence=TRACKING_THRESHOLD,
    )

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(
            f"[ERROR] Cannot open: {video_path}",
            flush=True
        )
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30.0

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    total_frames = 0
    detected_frames = 0

    frames = []
    handedness_list = []

    last_timestamp = -1

    with HandLandmarker.create_from_options(options) as landmarker:

        while True:

            ok, frame = cap.read()

            if not ok:
                break

            frame_index = total_frames

            # 원본 영상 timestamp
            pos_ms = cap.get(
                cv2.CAP_PROP_POS_MSEC
            )

            if pos_ms > 0:
                timestamp_ms = int(
                    round(pos_ms)
                )
            else:
                timestamp_ms = int(
                    round(
                        frame_index
                        * 1000.0
                        / fps
                    )
                )

            # MediaPipe VIDEO mode 요구사항:
            # timestamp는 반드시 증가해야 함.
            if timestamp_ms <= last_timestamp:
                timestamp_ms = last_timestamp + 1

            last_timestamp = timestamp_ms

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            result = landmarker.detect_for_video(
                mp_image,
                timestamp_ms
            )

            total_frames += 1

            # 모바일 앱처럼 손 검출에 성공한 프레임만 저장
            if not result.hand_landmarks:
                continue

            detected_frames += 1

            hand = result.hand_landmarks[0]

            landmarks = [
                {
                    "x": float(lm.x),
                    "y": float(lm.y),
                    "z": float(lm.z),
                }
                for lm in hand
            ]

            handedness = None
            handedness_score = None

            if (
                result.handedness
                and result.handedness[0]
            ):

                category = (
                    result.handedness[0][0]
                )

                handedness = (
                    category.category_name
                )

                handedness_score = float(
                    category.score
                )

                handedness_list.append(
                    handedness
                )

            frames.append({
                "frame_index": frame_index,
                "tMs": timestamp_ms,
                "width": width,
                "height": height,
                "handedness": handedness,
                "handedness_score": handedness_score,
                "landmarks": landmarks,
            })

    cap.release()

    # =====================================================
    # 영상 전체 handedness
    # =====================================================

    handedness_counts = dict(
        Counter(handedness_list)
    )

    if handedness_list:

        majority_handedness = (
            Counter(
                handedness_list
            ).most_common(1)[0][0]
        )

    else:

        majority_handedness = None

    if total_frames > 0:

        detection_rate = (
            detected_frames
            / total_frames
        )

    else:

        detection_rate = 0.0

    # =====================================================
    # 저장 데이터
    # =====================================================

    output = {

        "source_video": os.path.basename(
            video_path
        ),

        "extractor":
            "MediaPipe Tasks HandLandmarker",

        "delegate": "CPU",

        "settings": {

            "num_hands": 1,

            "min_hand_detection_confidence":
                DETECTION_THRESHOLD,

            "min_hand_presence_confidence":
                PRESENCE_THRESHOLD,

            "min_tracking_confidence":
                TRACKING_THRESHOLD,

            "running_mode": "VIDEO",
        },

        "video": {

            "fps": float(fps),

            "width": width,

            "height": height,

            "total_frames":
                total_frames,

            "detected_frames":
                detected_frames,

            "detection_rate":
                detection_rate,
        },

        "handedness_counts":
            handedness_counts,

        "majority_handedness":
            majority_handedness,

        "frames": frames,
    }

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    # 먼저 임시 파일 저장
    # 중간 종료 시 깨진 JSON 방지
    temp_path = (
        output_path + ".tmp"
    )

    with open(
        temp_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output,
            f,
            ensure_ascii=False
        )

    os.replace(
        temp_path,
        output_path
    )

    return True


# =========================================================
# 담당 Gesture의 영상 목록 생성
# =========================================================

videos = []

for g in range(
    START_GESTURE,
    END_GESTURE + 1
):

    gesture = f"G{g}"

    input_dir = os.path.join(
        VIDEO_ROOT,
        gesture
    )

    if not os.path.isdir(input_dir):

        print(
            f"[WARNING] Missing: {input_dir}",
            flush=True
        )

        continue

    for filename in sorted(
        os.listdir(input_dir)
    ):

        if not filename.lower().endswith(
            ".mp4"
        ):
            continue

        video_path = os.path.join(
            input_dir,
            filename
        )

        output_name = (
            os.path.splitext(filename)[0]
            + ".json"
        )

        output_path = os.path.join(
            OUTPUT_ROOT,
            gesture,
            output_name
        )

        videos.append(
            (
                gesture,
                filename,
                video_path,
                output_path
            )
        )


# =========================================================
# 시작 정보
# =========================================================

print()
print(
    "=============================================="
)
print(
    " AH650 GPU LANDMARK EXTRACTION"
)
print(
    "=============================================="
)

print(
    f"Gesture range       : "
    f"G{START_GESTURE} ~ G{END_GESTURE}"
)

print(
    f"Videos              : {len(videos)}"
)

print(
    f"Detection threshold : "
    f"{DETECTION_THRESHOLD}"
)

print(
    f"Delegate            : CPU"
)

print(
    f"Output              : {OUTPUT_ROOT}"
)

print()


# =========================================================
# 추출
# =========================================================

start_time = time.time()

processed = 0
skipped = 0
failed = 0

for index, (
    gesture,
    filename,
    video_path,
    output_path
) in enumerate(
    videos,
    start=1
):

    # =====================================================
    # Resume
    # =====================================================

    if os.path.isfile(output_path):

        try:

            with open(
                output_path,
                "r",
                encoding="utf-8"
            ) as f:

                existing = json.load(f)

            if (
                "frames" in existing
                and "video" in existing
            ):

                skipped += 1

                if (
                    index % 10 == 0
                    or index == len(videos)
                ):

                    elapsed = (
                        time.time()
                        - start_time
                    ) / 60.0

                    print(
                        f"[{index:3d}/{len(videos)}] "
                        f"{gesture} | "
                        f"processed={processed} "
                        f"skipped={skipped} "
                        f"failed={failed} "
                        f"| elapsed={elapsed:.1f} min",
                        flush=True
                    )

                continue

        except Exception:
            # 손상된 JSON이면 다시 추출
            pass

    # =====================================================
    # 실제 landmark 추출
    # =====================================================

    success = extract_video(
        video_path,
        output_path
    )

    if success:
        processed += 1
    else:
        failed += 1

    if (
        index % 10 == 0
        or index == len(videos)
    ):

        elapsed = (
            time.time()
            - start_time
        ) / 60.0

        print(
            f"[{index:3d}/{len(videos)}] "
            f"{gesture} | "
            f"processed={processed} "
            f"skipped={skipped} "
            f"failed={failed} "
            f"| elapsed={elapsed:.1f} min",
            flush=True
        )


# =========================================================
# 완료
# =========================================================

elapsed = (
    time.time()
    - start_time
) / 60.0

print()
print(
    "=============================================="
)
print(
    " EXTRACTION COMPLETE"
)
print(
    "=============================================="
)

print(
    f"Gesture range : "
    f"G{START_GESTURE} ~ G{END_GESTURE}"
)

print(
    f"Total         : {len(videos)}"
)

print(
    f"Processed     : {processed}"
)

print(
    f"Skipped       : {skipped}"
)

print(
    f"Failed        : {failed}"
)

print(
    f"Elapsed       : {elapsed:.1f} min"
)

print(
    f"Output        : {OUTPUT_ROOT}"
)

print("Done.")

