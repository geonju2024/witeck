"""
01_extract_landmarks.py
영상 -> MediaPipe 랜드마크(.npz) 캐시.

이 단계가 전체 파이프라인에서 유일하게 느린 부분입니다(영상 1개당 약 3~5초).
한 번 뽑아두면 이후 실험은 전부 초 단위로 반복할 수 있습니다.

사용법:
    python 01_extract_landmarks.py --videos ./videos --out ./landmarks --workers 4
"""
import argparse
import os
import glob
import time
import numpy as np

# ---- 중요: 이 값들은 프로젝트 전체에서 공유되는 상수 ----
N_HAND = 21          # MediaPipe Hands 랜드마크 수
N_POSE = 33          # MediaPipe Pose 랜드마크 수
MAX_HANDS = 2


def read_frames(path, max_frames=300):
    """
    영상을 프레임 리스트로 읽는다.

    CAP_PROP_ORIENTATION_AUTO 가 핵심이다.
    스마트폰으로 세로 촬영한 mp4 는 실제 픽셀은 가로(1920x1080)로 저장되고
    '재생할 때 90도 돌려라'는 회전 메타데이터(display matrix)만 붙는다.
    ffmpeg/윈도우 재생기는 이걸 적용하지만 OpenCV 는 기본적으로 무시한다.
    그래서 이 플래그가 없으면 MediaPipe 는 옆으로 누운 사람을 보게 되고,
    Pose 는 완전히 망가진 좌표를 반환한다(에러는 안 나서 알아채기 어렵다).
    """
    import cv2
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    frames = []
    while len(frames) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


def extract_one(args):
    video_path, out_dir = args
    import cv2
    import mediapipe as mp

    stem = os.path.splitext(os.path.basename(video_path))[0]
    out_path = os.path.join(out_dir, stem + ".npz")
    if os.path.exists(out_path):
        return stem, "cached", 0.0

    t0 = time.time()
    frames = read_frames(video_path)
    if not frames:
        return stem, "EMPTY", 0.0
    H, W = frames[0].shape[:2]
    T = len(frames)

    hand = np.zeros((T, MAX_HANDS, N_HAND, 3), np.float32)
    hand_valid = np.zeros((T, MAX_HANDS), np.uint8)   # 0=미검출, 1=검출
    handedness = np.zeros((T, MAX_HANDS), np.int8)    # 0=Left, 1=Right, -1=없음
    handedness[:] = -1
    pose = np.zeros((T, N_POSE, 4), np.float32)       # x,y,z,visibility
    pose_valid = np.zeros((T,), np.uint8)

    mp_hands = mp.solutions.hands
    mp_pose = mp.solutions.pose

    # 실측 결과 tracking 모드(static_image_mode=False)가 프레임별 static 모드보다
    # 검출률이 높았다(최악 영상 기준 38% vs 3%). 손이 빠르게 움직여 모션 블러가
    # 생기는 구간에서 이전 프레임 트래킹이 버텨주기 때문.
    with mp_hands.Hands(static_image_mode=False,
                        max_num_hands=MAX_HANDS,
                        model_complexity=1,
                        min_detection_confidence=0.3,
                        min_tracking_confidence=0.3) as hands, \
         mp_pose.Pose(static_image_mode=False,
                      model_complexity=1,
                      min_detection_confidence=0.3,
                      min_tracking_confidence=0.3) as poser:

        for t, frame in enumerate(frames):
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False

            rh = hands.process(rgb)
            if rh.multi_hand_landmarks:
                for k, hl in enumerate(rh.multi_hand_landmarks[:MAX_HANDS]):
                    hand[t, k] = [[p.x, p.y, p.z] for p in hl.landmark]
                    hand_valid[t, k] = 1
                    if rh.multi_handedness and k < len(rh.multi_handedness):
                        lab = rh.multi_handedness[k].classification[0].label
                        handedness[t, k] = 1 if lab == "Right" else 0

            rp = poser.process(rgb)
            if rp.pose_landmarks:
                pose[t] = [[p.x, p.y, p.z, p.visibility] for p in rp.pose_landmarks.landmark]
                pose_valid[t] = 1

    np.savez_compressed(out_path,
                        hand=hand, hand_valid=hand_valid, handedness=handedness,
                        pose=pose, pose_valid=pose_valid,
                        width=W, height=H, n_frames=T)

    rate = hand_valid.max(axis=1).mean()
    return stem, f"T={T} hand={rate:.0%} pose={pose_valid.mean():.0%}", time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True, help="영상 폴더 (하위 폴더까지 재귀 탐색)")
    ap.add_argument("--out", default="./landmarks")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    exts = ("mp4", "MP4", "mov", "MOV", "avi", "AVI", "mkv")
    files = sorted({f for e in exts
                    for f in glob.glob(os.path.join(a.videos, "**", f"*.{e}"), recursive=True)})
    print(f"영상 {len(files)}개 발견, workers={a.workers}")

    jobs = [(f, a.out) for f in files]
    t0 = time.time()
    if a.workers > 1:
        # MediaPipe 그래프는 프로세스마다 새로 만들어야 안전하다 (스레드 공유 금지)
        from multiprocessing import Pool
        with Pool(a.workers) as p:
            for i, (stem, msg, dt) in enumerate(p.imap_unordered(extract_one, jobs), 1):
                print(f"[{i}/{len(files)}] {stem}: {msg} ({dt:.1f}s)", flush=True)
    else:
        for i, job in enumerate(jobs, 1):
            stem, msg, dt = extract_one(job)
            print(f"[{i}/{len(files)}] {stem}: {msg} ({dt:.1f}s)", flush=True)

    print(f"\n완료. 총 {time.time()-t0:.0f}초 -> {a.out}")


if __name__ == "__main__":
    main()
