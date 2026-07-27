import cv2 as cv
import mediapipe as mp
import numpy as np
import os
import csv
import time
from datetime import datetime
from tensorflow.keras.models import load_model

from gesture_auth_config import DEFAULT_AUTH_THRESHOLD, FEATURE_SIZE, MODEL_PATH, SEQUENCE_LENGTH


# =========================
# Settings
# =========================
THRESHOLD = DEFAULT_AUTH_THRESHOLD
MAX_ATTEMPTS = 3         # Lock temporarily after 3 failed attempts
LOCKOUT_SECONDS = 30     # Lock for 30 seconds
UNLOCK_SECONDS = 5       # Simulated unlock duration in seconds

MIN_HAND_SIZE = 0.12     # Hand must be large enough
MIN_MOTION = 0.002       # Hand must have enough motion

LOG_PATH = "auth_log.csv"


# =========================
# Load model
# =========================
model = load_model(MODEL_PATH)
print("Model loaded:", MODEL_PATH)


# =========================
# Save log function
# =========================
def save_auth_log(result, probability, motion_score):
    file_exists = os.path.exists(LOG_PATH)

    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "time",
                "result",
                "probability",
                "motion_score"
            ])

        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            result,
            f"{probability:.4f}",
            f"{motion_score:.6f}"
        ])


# =========================
# Extract landmarks
# =========================
def extract_landmarks(hand_landmarks):
    landmarks = hand_landmarks.landmark

    base_x = landmarks[0].x
    base_y = landmarks[0].y
    base_z = landmarks[0].z

    data = []

    for lm in landmarks:
        x = lm.x - base_x
        y = lm.y - base_y
        z = lm.z - base_z
        data.extend([x, y, z])

    return data


# =========================
# Check hand size
# =========================
def is_hand_large_enough(hand_landmarks, min_size=MIN_HAND_SIZE):
    xs = [lm.x for lm in hand_landmarks.landmark]
    ys = [lm.y for lm in hand_landmarks.landmark]

    width = max(xs) - min(xs)
    height = max(ys) - min(ys)

    return width > min_size and height > min_size


# =========================
# Check motion
# =========================
def check_motion(sequence):
    sequence = np.array(sequence, dtype=np.float32)

    # Calculate the amount of change between frames
    diffs = np.abs(np.diff(sequence, axis=0))
    motion_score = np.mean(diffs)

    motion_ok = motion_score >= MIN_MOTION

    return motion_ok, motion_score


# =========================
# MediaPipe settings
# =========================
mp_hand = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

hand = mp_hand.Hands(
    max_num_hands=2,
    static_image_mode=False,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)

cap = cv.VideoCapture(0, cv.CAP_DSHOW)

if not cap.isOpened():
    print("Cannot open camera.")
    exit()


# =========================
# Security state variables
# =========================
recording = False
sequence = []

failed_attempts = 0
locked_until = 0
door_unlocked_until = 0

result_text = "Press 'r' to authenticate"
prob_text = ""
security_text = "SYSTEM READY"

print("====================================")
print("r key: start authentication")
print("q key: quit")
print("====================================")


while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to read frame")
        break

    frame = cv.flip(frame, 1)
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    res = hand.process(rgb_frame)

    current_landmarks = None
    hand_detected = False
    hand_size_ok = False

    # Current time
    now = time.time()

    # Check lock state
    is_locked = now < locked_until
    is_unlocked = now < door_unlocked_until

    # Detect hand
    if res.multi_hand_landmarks:
        for landmarks in res.multi_hand_landmarks:
            if is_hand_large_enough(landmarks):
                hand_detected = True
                hand_size_ok = True

                mp_drawing.draw_landmarks(
                    frame,
                    landmarks,
                    mp_hand.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style()
                )

                current_landmarks = extract_landmarks(landmarks)
            else:
                hand_detected = True
                hand_size_ok = False

    # Display system status
    if is_locked:
        remain = int(locked_until - now)
        security_text = f"SYSTEM LOCKED: wait {remain}s"

    elif is_unlocked:
        remain = int(door_unlocked_until - now)
        security_text = f"DOOR UNLOCKED: {remain}s"

    else:
        security_text = f"READY | failed attempts: {failed_attempts}/{MAX_ATTEMPTS}"

    # Recording
    if recording and not is_locked:
        if current_landmarks is not None:
            sequence.append(current_landmarks)

        result_text = f"Recording: {len(sequence)}/{SEQUENCE_LENGTH}"

        if len(sequence) >= SEQUENCE_LENGTH:
            input_data = np.array(sequence, dtype=np.float32)

            if input_data.shape != (SEQUENCE_LENGTH, FEATURE_SIZE):
                print("shape error:", input_data.shape)
                sequence = []
                recording = False
                continue

            # liveness check
            motion_ok, motion_score = check_motion(input_data)

            # model prediction
            input_data = np.expand_dims(input_data, axis=0)
            pred = model.predict(input_data, verbose=0)[0][0]

            # security decision
            if pred >= THRESHOLD and motion_ok:
                result_text = "AUTH SUCCESS"
                prob_text = f"probability: {pred:.4f}, motion: {motion_score:.6f}"

                failed_attempts = 0
                door_unlocked_until = time.time() + UNLOCK_SECONDS

                save_auth_log("success", pred, motion_score)

                print("AUTH SUCCESS")
                print("probability:", pred)
                print("motion_score:", motion_score)

            else:
                result_text = "AUTH FAIL"
                prob_text = f"probability: {pred:.4f}, motion: {motion_score:.6f}"

                failed_attempts += 1

                save_auth_log("fail", pred, motion_score)

                print("AUTH FAIL")
                print("probability:", pred)
                print("motion_score:", motion_score)

                if failed_attempts >= MAX_ATTEMPTS:
                    locked_until = time.time() + LOCKOUT_SECONDS
                    failed_attempts = 0
                    result_text = "SYSTEM LOCKED"

            sequence = []
            recording = False

    # Draw output
    if hand_detected and hand_size_ok:
        cv.putText(frame, "Hand: YES", (20, 40),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    elif hand_detected and not hand_size_ok:
        cv.putText(frame, "Hand: TOO SMALL", (20, 40),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
    else:
        cv.putText(frame, "Hand: NO", (20, 40),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv.putText(frame, result_text, (20, 90),
               cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv.putText(frame, prob_text, (20, 130),
               cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv.putText(frame, security_text, (20, 170),
               cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv.imshow("Security Simulation", frame)

    key = cv.waitKey(1) & 0xFF

    if key == ord('r') and not recording:
        if is_locked:
            print("System is currently locked.")
        else:
            print("Authentication started")
            sequence = []
            recording = True
            result_text = "Recording..."
            prob_text = ""

    elif key == ord('q'):
        break


cap.release()
hand.close()
cv.destroyAllWindows()
