import cv2 as cv
import mediapipe as mp
import numpy as np
import os
import csv

# =========================
# 설정 부분
# =========================
DATASET_DIR = "gesture_dataset"
SEQUENCE_LENGTH = 60   # 한 번 제스처를 수행할 때 저장할 프레임 수

user_id = input("사용자 ID를 입력하세요: ")          # 예: user1
gesture_label = input("제스처 이름을 입력하세요: ")  # 예: gesture_A
auth_label = input("인증 라벨을 입력하세요(success/fail): ").strip()
if auth_label == "": auth_label = "success"

save_dir = os.path.join(DATASET_DIR, user_id, gesture_label)
os.makedirs(save_dir, exist_ok=True)

metadata_path = os.path.join(DATASET_DIR, "metadata.csv")

# metadata.csv 파일이 없으면 헤더 생성
if not os.path.exists(metadata_path):
    with open(metadata_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file_path", "user_id", "gesture_label", "sample_id", "sequence_length", "feature_size", "auth_label"])


def get_next_sample_id(save_dir):
    files = [f for f in os.listdir(save_dir) if f.endswith(".npy")]
    return len(files) + 1


def extract_landmarks(hand_landmarks):
    """
    MediaPipe 손 랜드마크 21개를 추출한다.
    손목(landmark 0)을 기준으로 좌표를 정규화한다.
    """
    landmarks = hand_landmarks.landmark

    base_x = landmarks[0].x
    base_y = landmarks[0].y
    base_z = landmarks[0].z

    data = []

    for lm in landmarks:
        # 손목 기준 상대 좌표
        x = lm.x - base_x
        y = lm.y - base_y
        z = lm.z - base_z

        data.extend([x, y, z])

    return data   # 총 21 * 3 = 63개 값


def save_sequence(sequence, sample_id):
    file_name = f"sample_{sample_id:03d}.npy"
    file_path = os.path.join(save_dir, file_name)

    sequence_array = np.array(sequence, dtype=np.float32)

    # shape: (60, 63)
    np.save(file_path, sequence_array)

    with open(metadata_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            file_path,
            user_id,
            gesture_label,
            sample_id,
            SEQUENCE_LENGTH,
            63,
            auth_label
        ])

    print(f"저장 완료: {file_path}, shape={sequence_array.shape}")


# =========================
# MediaPipe 설정
# =========================
mp_hand = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

hand = mp_hand.Hands(
    max_num_hands=2,
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv.VideoCapture(0, cv.CAP_DSHOW)

if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

recording = False
sequence = []
sample_id = get_next_sample_id(save_dir)

print("====================================")
print("r 키: 제스처 데이터 녹화 시작")
print("q 키: 프로그램 종료")
print("====================================")

while True:
    ret, frame = cap.read()

    if not ret:
        print("프레임 획득에 실패하여 루프를 나갑니다.")
        break

    frame = cv.flip(frame, 1)
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    res = hand.process(rgb_frame)

    current_landmarks = None

    if res.multi_hand_landmarks:
        for landmarks in res.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                landmarks,
                mp_hand.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style()
            )

            current_landmarks = extract_landmarks(landmarks)

    # 녹화 중이면 landmark 저장
    if recording:
        if current_landmarks is not None:
            sequence.append(current_landmarks)

        progress_text = f"Recording: {len(sequence)}/{SEQUENCE_LENGTH}"
        cv.putText(frame, progress_text, (20, 50),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # 지정한 프레임 수만큼 저장되면 파일로 저장
        if len(sequence) >= SEQUENCE_LENGTH:
            save_sequence(sequence, sample_id)
            sample_id += 1
            sequence = []
            recording = False

    else:
        cv.putText(frame, "Press 'r' to record gesture", (20, 50),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv.putText(frame, f"user: {user_id}, gesture: {gesture_label}", (20, 90),
               cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv.imshow("MediaPipe Dataset Collector", frame)

    key = cv.waitKey(1) & 0xFF

    if key == ord('r') and not recording:
        print("녹화를 시작합니다.")
        sequence = []
        recording = True

    elif key == ord('q'):
        break

cap.release()
hand.close()
cv.destroyAllWindows()