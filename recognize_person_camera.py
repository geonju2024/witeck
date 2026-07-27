import argparse
import json
import os

import cv2 as cv
import mediapipe as mp
import numpy as np

from gesture_auth_config import FEATURE_SIZE, SEQUENCE_LENGTH
from person_model_registry import (
    DEFAULT_PERSON_MODEL_TYPE,
    PERSON_MODEL_TYPES,
    load_person_model,
    person_labels_path,
    person_model_path,
    predict_probabilities,
)

CONFIDENCE_THRESHOLD = 0.60


def parse_args():
    parser = argparse.ArgumentParser(
        description="Recognize a person with one selected model."
    )
    parser.add_argument(
        "--model",
        choices=PERSON_MODEL_TYPES,
        default=DEFAULT_PERSON_MODEL_TYPE,
        help=f"Model to load (default: {DEFAULT_PERSON_MODEL_TYPE}).",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="OpenCV camera index.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=CONFIDENCE_THRESHOLD,
        help=f"Minimum confidence for a known user (default: {CONFIDENCE_THRESHOLD}).",
    )
    return parser.parse_args()


def load_labels(path, expected_model_type):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing {path}. Train {expected_model_type} first."
        )
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    saved_model_type = payload.get("model_type")
    if saved_model_type and saved_model_type != expected_model_type:
        raise ValueError(
            f"Labels belong to {saved_model_type}, not {expected_model_type}."
        )

    classes = payload.get("classes")
    if not classes:
        raise ValueError(f"{path} does not contain a non-empty classes list.")
    return classes


def extract_landmarks(hand_landmarks):
    landmarks = hand_landmarks.landmark
    base_x = landmarks[0].x
    base_y = landmarks[0].y
    base_z = landmarks[0].z

    data = []
    for lm in landmarks:
        data.extend([lm.x - base_x, lm.y - base_y, lm.z - base_z])
    return data


def predict_person(model_type, model, classes, sequence, confidence_threshold):
    input_data = np.asarray(sequence, dtype=np.float32)
    if input_data.shape != (SEQUENCE_LENGTH, FEATURE_SIZE):
        raise ValueError(f"Invalid sequence shape: {input_data.shape}")

    probabilities = predict_probabilities(model_type, model, input_data)[0]
    index = int(np.argmax(probabilities))
    confidence = float(probabilities[index])
    label = classes[index]
    if confidence < confidence_threshold:
        return "unknown", confidence, probabilities
    return label, confidence, probabilities


def main():
    args = parse_args()
    model_path = person_model_path(args.model)
    labels_path = person_labels_path(args.model)

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Missing {model_path}. Run train_person_recognition.py "
            f"--model {args.model} after collecting multi-user data."
        )

    classes = load_labels(labels_path, args.model)
    model = load_person_model(args.model, model_path)
    print(f"Loaded model: {args.model} ({model_path})")

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
    result_text = "Press r to recognize"
    detail_text = ""

    print("Press r to record a gesture for person recognition. Press q to quit.")

    try:
        while True:
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
                result_text = f"Recording {len(sequence)}/{SEQUENCE_LENGTH}"
                if len(sequence) >= SEQUENCE_LENGTH:
                    label, confidence, probabilities = predict_person(
                        args.model,
                        model,
                        classes,
                        sequence,
                        args.confidence_threshold,
                    )
                    result_text = f"Person: {label}"
                    detail_text = f"confidence: {confidence:.3f}"
                    print(result_text)
                    print(detail_text)
                    print(
                        {
                            cls: float(prob)
                            for cls, prob in zip(classes, probabilities)
                        }
                    )
                    sequence = []
                    recording = False

            cv.putText(
                frame,
                result_text,
                (20, 45),
                cv.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )
            cv.putText(
                frame,
                detail_text,
                (20, 85),
                cv.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )
            cv.imshow(f"Person Recognition - {args.model}", frame)

            key = cv.waitKey(1) & 0xFF
            if key == ord("r") and not recording:
                sequence = []
                recording = True
                detail_text = ""
            elif key == ord("q"):
                break
    finally:
        cap.release()
        hand.close()
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()
