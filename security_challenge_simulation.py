import cv2 as cv
import mediapipe as mp
import numpy as np
import random
import time
import csv
import os
from datetime import datetime
from tensorflow.keras.models import load_model

from gesture_auth_config import DEFAULT_AUTH_THRESHOLD, FEATURE_SIZE, MODEL_PATH, SEQUENCE_LENGTH


# =========================
# Settings
# =========================
THRESHOLD = DEFAULT_AUTH_THRESHOLD
MIN_MOTION = 0.002           # liveness check
MIN_HAND_SIZE = 0.12         # Reject hands that are too small

MAX_ATTEMPTS = 3
LOCKOUT_SECONDS = 30
UNLOCK_SECONDS = 5

CHALLENGE_GESTURES = ["ges1", "ges2", "ges3"]
CHALLENGE_LENGTH = 3
TIME_LIMIT_PER_STEP = 8      # Each gesture must be completed within 8 seconds

LOG_PATH = "auth_log.csv"


# =========================
# Load model
# =========================
model = load_model(MODEL_PATH)
print("Model loaded:", MODEL_PATH)


# =========================
# Save log
# =========================
def save_auth_log(result, probability, motion_score, challenge_text, step_gesture):
    file_exists = os.path.exists(LOG_PATH)

    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "time",
                "result",
                "probability",
                "motion_score",
                "challenge",
                "step_gesture"
            ])

        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            result,
            f"{probability:.4f}",
            f"{motion_score:.6f}",
            challenge_text,
            step_gesture
        ])


# =========================
# Generate challenge
# =========================
def generate_challenge():
    return random.choices(CHALLENGE_GESTURES, k=CHALLENGE_LENGTH)


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
# Check motion: liveness check
# =========================
def check_motion(sequence):
    sequence = np.array(sequence, dtype=np.float32)

    diffs = np.abs(np.diff(sequence, axis=0))
    motion_score = np.mean(diffs)

    motion_ok = motion_score >= MIN_MOTION

    return motion_ok, motion_score


# =========================
# Handle authentication failure
# =========================
def handle_fail(reason):
    global failed_attempts, locked_until
    global result_text, prob_text, recording, sequence

    failed_attempts += 1
    result_text = f"AUTH FAIL: {reason}"
    prob_text = ""

    print(result_text)

    if failed_attempts >= MAX_ATTEMPTS:
        locked_until = time.time() + LOCKOUT_SECONDS
        failed_attempts = 0
        result_text = "SYSTEM LOCKED"

    sequence = []
    recording = False


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
# State variables
# =========================
challenge = generate_challenge()
current_step = 0
step_deadline = time.time() + TIME_LIMIT_PER_STEP

recording = False
sequence = []

failed_attempts = 0
locked_until = 0
door_unlocked_until = 0

result_text = "Press 'r' to start step"
prob_text = ""

print("====================================")
print("r key: record the current step gesture")
print("n key: generate a new challenge")
print("q key: quit")
print("====================================")
print("Challenge:", " -> ".join(challenge))


while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to read frame")
        break

    now = time.time()

    frame = cv.flip(frame, 1)
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    res = hand.process(rgb_frame)

    current_landmarks = None
    hand_detected = False
    hand_size_ok = False

    # Lock / unlock state
    is_locked = now < locked_until
    is_unlocked = now < door_unlocked_until

    # Detect hand
    if res.multi_hand_landmarks:
        for landmarks in res.multi_hand_landmarks:
            hand_detected = True

            if is_hand_large_enough(landmarks):
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
                hand_size_ok = False

    # Display current challenge
    challenge_text = " -> ".join(challenge)

    if current_step < CHALLENGE_LENGTH:
        current_gesture = challenge[current_step]
    else:
        current_gesture = "-"

    # Check time limit
    if not is_locked and not is_unlocked and not recording:
        if now > step_deadline:
            handle_fail("TIME OUT")
            challenge = generate_challenge()
            current_step = 0
            step_deadline = time.time() + TIME_LIMIT_PER_STEP
            print("New Challenge:", " -> ".join(challenge))

    # Recording
    if recording and not is_locked and not is_unlocked:
        if now > step_deadline:
            handle_fail("TIME OUT")
            challenge = generate_challenge()
            current_step = 0
            step_deadline = time.time() + TIME_LIMIT_PER_STEP
            print("New Challenge:", " -> ".join(challenge))

        else:
            if current_landmarks is not None:
                sequence.append(current_landmarks)

            result_text = f"Recording {current_gesture}: {len(sequence)}/{SEQUENCE_LENGTH}"

            if len(sequence) >= SEQUENCE_LENGTH:
                input_data = np.array(sequence, dtype=np.float32)

                if input_data.shape != (SEQUENCE_LENGTH, FEATURE_SIZE):
                    handle_fail("SHAPE ERROR")
                    continue

                # liveness check
                motion_ok, motion_score = check_motion(input_data)

                # model prediction
                input_data = np.expand_dims(input_data, axis=0)
                pred = model.predict(input_data, verbose=0)[0][0]

                # The current binary model only predicts success or fail
                if pred >= THRESHOLD and motion_ok:
                    result_text = f"STEP SUCCESS: {current_gesture}"
                    prob_text = f"probability: {pred:.4f}, motion: {motion_score:.6f}"

                    print(result_text)
                    print(prob_text)

                    save_auth_log(
                        "step_success",
                        pred,
                        motion_score,
                        challenge_text,
                        current_gesture
                    )

                    current_step += 1

                    # All challenge steps completed
                    if current_step >= CHALLENGE_LENGTH:
                        result_text = "AUTH SUCCESS - DOOR UNLOCKED"
                        prob_text = "Challenge completed"

                        door_unlocked_until = time.time() + UNLOCK_SECONDS
                        failed_attempts = 0

                        save_auth_log(
                            "auth_success",
                            pred,
                            motion_score,
                            challenge_text,
                            "ALL"
                        )

                        print("AUTH SUCCESS - DOOR UNLOCKED")

                        # Prepare a new challenge for the next authentication
                        challenge = generate_challenge()
                        current_step = 0
                        step_deadline = time.time() + TIME_LIMIT_PER_STEP

                    else:
                        # Move to the next step
                        step_deadline = time.time() + TIME_LIMIT_PER_STEP

                else:
                    result_text = "AUTH FAIL"
                    prob_text = f"probability: {pred:.4f}, motion: {motion_score:.6f}"

                    print(result_text)
                    print(prob_text)

                    save_auth_log(
                        "auth_fail",
                        pred,
                        motion_score,
                        challenge_text,
                        current_gesture
                    )

                    handle_fail("MODEL OR LIVENESS FAIL")

                    # Generate a new challenge after failure
                    challenge = generate_challenge()
                    current_step = 0
                    step_deadline = time.time() + TIME_LIMIT_PER_STEP
                    print("New Challenge:", " -> ".join(challenge))

                sequence = []
                recording = False

    # Display status
    if is_locked:
        remain = int(locked_until - now)
        status_text = f"SYSTEM LOCKED: wait {remain}s"

    elif is_unlocked:
        remain = int(door_unlocked_until - now)
        status_text = f"DOOR UNLOCKED: {remain}s"

    else:
        remain = max(0, int(step_deadline - now))
        status_text = f"STEP {current_step + 1}/{CHALLENGE_LENGTH} | Do: {current_gesture} | {remain}s"

    # Display hand status
    if hand_detected and hand_size_ok:
        cv.putText(frame, "Hand: YES", (20, 40),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    elif hand_detected and not hand_size_ok:
        cv.putText(frame, "Hand: TOO SMALL", (20, 40),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
    else:
        cv.putText(frame, "Hand: NO", (20, 40),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    #cv.putText(frame, f"Challenge: {challenge_text}", (20, 90),
               #cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv.putText(frame, status_text, (20, 90),
               cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv.putText(frame, result_text, (20, 130),
               cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv.putText(frame, prob_text, (20, 170),
               cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv.imshow("Challenge Response Security Simulation", frame)

    key = cv.waitKey(1) & 0xFF

    if key == ord('r') and not recording:
        if is_locked:
            print("System is locked.")
        elif is_unlocked:
            print("System is already unlocked.")
        else:
            print(f"Recording started: {current_gesture}")
            sequence = []
            recording = True
            result_text = f"Recording {current_gesture}..."
            prob_text = ""

    elif key == ord('n'):
        challenge = generate_challenge()
        current_step = 0
        step_deadline = time.time() + TIME_LIMIT_PER_STEP
        sequence = []
        recording = False
        result_text = "New challenge generated"
        prob_text = ""
        print("New Challenge:", " -> ".join(challenge))

    elif key == ord('q'):
        break


cap.release()
hand.close()
cv.destroyAllWindows()
