import cv2 as cv
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

from gesture_auth_config import DEFAULT_AUTH_THRESHOLD, FEATURE_SIZE, MODEL_PATH, SEQUENCE_LENGTH

# =========================
# Settings
# =========================
THRESHOLD = DEFAULT_AUTH_THRESHOLD

# =========================
# Load model
# =========================
model = load_model(MODEL_PATH)
print("Model loaded:", MODEL_PATH)


def extract_landmarks(hand_landmarks):
    """
    Normalize landmarks relative to wrist landmark 0, matching data_collect.py.
    """
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

    return data  # 21 * 3 = 63


# =========================
# MediaPipe settings
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
    print("Cannot open camera.")
    exit()

recording = False
sequence = []

result_text = "Press 'r' to test"
prob_text = ""

print("====================================")
print("r key: start authentication test")
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

    if res.multi_hand_landmarks:
        hand_detected = True

        for landmarks in res.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                landmarks,
                mp_hand.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style()
            )

            current_landmarks = extract_landmarks(landmarks)

    # Save landmarks while recording
    if recording:
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

            # model input shape: (1, 60, 63)
            input_data = np.expand_dims(input_data, axis=0)

            pred = model.predict(input_data, verbose=0)[0][0]

            if pred >= THRESHOLD:
                result_text = "AUTH SUCCESS"
            else:
                result_text = "AUTH FAIL"

            prob_text = f"probability: {pred:.4f}"

            print("prediction probability:", pred)
            print("result:", result_text)

            sequence = []
            recording = False

    # Display output
    if hand_detected:
        cv.putText(frame, "Hand: YES", (20, 40),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv.putText(frame, "Hand: NO", (20, 40),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv.putText(frame, result_text, (20, 90),
               cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv.putText(frame, prob_text, (20, 130),
               cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv.imshow("Gesture Authentication Camera Test", frame)

    key = cv.waitKey(1) & 0xFF

    if key == ord('r') and not recording:
        print("Authentication test started")
        sequence = []
        result_text = "Recording..."
        prob_text = ""
        recording = True

    elif key == ord('q'):
        break

cap.release()
hand.close()
cv.destroyAllWindows()
