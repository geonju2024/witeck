import argparse
import csv
import os

import numpy as np

from gesture_auth_config import FEATURE_SIZE, METADATA_PATH, SEQUENCE_LENGTH

DATASET_DIR = "gesture_dataset"


def parse_args():
    parser = argparse.ArgumentParser(description="Collect hand-gesture samples for person recognition.")
    parser.add_argument("--user-id", help="Person label, for example user2.")
    parser.add_argument("--gesture-label", help="Gesture label, for example ges1.")
    parser.add_argument("--count", type=int, default=10, help="Number of samples to record.")
    parser.add_argument("--camera-index", type=int, default=0, help="OpenCV camera index.")
    return parser.parse_args()


def prompt_required(name, value=None):
    if value is None:
        value = input(f"{name}: ").strip()
    else:
        value = value.strip()
    if not value:
        raise ValueError(f"{name} is required.")
    return value


def ensure_metadata():
    os.makedirs(DATASET_DIR, exist_ok=True)
    if not os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "file_path",
                    "user_id",
                    "gesture_label",
                    "sample_id",
                    "sequence_length",
                    "feature_size",
                    "auth_label",
                ]
            )


def get_next_sample_id(save_dir):
    if not os.path.exists(save_dir):
        return 1
    files = [name for name in os.listdir(save_dir) if name.endswith(".npy")]
    return len(files) + 1


def extract_landmarks(hand_landmarks):
    landmarks = hand_landmarks.landmark
    base_x = landmarks[0].x
    base_y = landmarks[0].y
    base_z = landmarks[0].z

    data = []
    for lm in landmarks:
        data.extend([lm.x - base_x, lm.y - base_y, lm.z - base_z])
    return data


def save_sequence(sequence, save_dir, user_id, gesture_label, sample_id):
    file_name = f"sample_{sample_id:03d}.npy"
    file_path = os.path.join(save_dir, file_name)
    sequence_array = np.asarray(sequence, dtype=np.float32)

    if sequence_array.shape != (SEQUENCE_LENGTH, FEATURE_SIZE):
        raise ValueError(f"Invalid sequence shape: {sequence_array.shape}")

    np.save(file_path, sequence_array)
    with open(METADATA_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                file_path,
                user_id,
                gesture_label,
                sample_id,
                SEQUENCE_LENGTH,
                FEATURE_SIZE,
                "success",
            ]
        )

    print(f"Saved: {file_path}")


def main():
    args = parse_args()

    import cv2 as cv
    import mediapipe as mp
    ensure_metadata()
    user_id = prompt_required("user_id, for example user2", args.user_id)
    gesture_label = prompt_required("gesture_label, for example ges1", args.gesture_label)
    target_count = args.count
    if target_count <= 0:
        raise ValueError("--count must be greater than 0.")

    save_dir = os.path.join(DATASET_DIR, user_id, gesture_label)
    os.makedirs(save_dir, exist_ok=True)
    sample_id = get_next_sample_id(save_dir)
    saved_count = 0

    mp_hand = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles
    hand = mp_hand.Hands(
        max_num_hands=1,
        static_image_mode=False,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    cap = cv.VideoCapture(args.camera_index, cv.CAP_DSHOW)
    if not cap.isOpened():
        hand.close()
        raise RuntimeError(f"Could not open camera {args.camera_index}.")

    recording = False
    sequence = []

    print("Press r to record one sample. Press q to quit.")
    print(f"Saving {target_count} samples for user={user_id}, gesture={gesture_label}.")

    try:
        while saved_count < target_count:
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError("Failed to read camera frame.")

            frame = cv.flip(frame, 1)
            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            result = hand.process(rgb_frame)

            current_landmarks = None
            if result.multi_hand_landmarks:
                landmarks = result.multi_hand_landmarks[0]
                current_landmarks = extract_landmarks(landmarks)
                mp_drawing.draw_landmarks(
                    frame,
                    landmarks,
                    mp_hand.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style(),
                )

            if recording:
                if current_landmarks is not None:
                    sequence.append(current_landmarks)
                status = f"Recording {len(sequence)}/{SEQUENCE_LENGTH}"
                color = (0, 0, 255)
                if len(sequence) >= SEQUENCE_LENGTH:
                    save_sequence(sequence, save_dir, user_id, gesture_label, sample_id)
                    sample_id += 1
                    saved_count += 1
                    sequence = []
                    recording = False
            else:
                status = "Press r to record"
                color = (0, 255, 0)

            cv.putText(frame, status, (20, 45), cv.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv.putText(
                frame,
                f"saved: {saved_count}/{target_count} | user: {user_id} | gesture: {gesture_label}",
                (20, 85),
                cv.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv.imshow("Person Gesture Sample Collector", frame)

            key = cv.waitKey(1) & 0xFF
            if key == ord("r") and not recording:
                sequence = []
                recording = True
                print("Recording started.")
            elif key == ord("q"):
                break
    finally:
        cap.release()
        hand.close()
        cv.destroyAllWindows()

    print(f"Done. Saved {saved_count} sample(s).")


if __name__ == "__main__":
    main()
