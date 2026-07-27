import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

from gesture_auth_config import (DEFAULT_AUTH_THRESHOLD, FEATURE_SIZE, METADATA_PATH, MODEL_PATH, SEQUENCE_LENGTH)

THRESHOLD = DEFAULT_AUTH_THRESHOLD

model = load_model(MODEL_PATH)
df = pd.read_csv(METADATA_PATH)

correct = 0
total = 0

for idx, row in df.iterrows():
    file_path = str(row["file_path"]).replace("\\", os.sep).replace("/", os.sep)
    true_label = str(row["auth_label"]).strip().lower()

    if not os.path.exists(file_path):
        print("Missing file:", file_path)
        continue

    data = np.load(file_path)

    if data.shape != (SEQUENCE_LENGTH, FEATURE_SIZE):
        print("shape mismatch:", file_path, data.shape)
        continue

    input_data = np.expand_dims(data, axis=0)
    pred_prob = model.predict(input_data, verbose=0)[0][0]

    pred_label = "success" if pred_prob >= THRESHOLD else "fail"

    print(f"{file_path}")
    print(f"truth: {true_label}, prediction: {pred_label}, probability: {pred_prob:.4f}")
    print("-" * 50)

    if pred_label == true_label:
        correct += 1

    total += 1

print("total:", total)
print("correct:", correct)

if total > 0:
    print("Accuracy:", correct / total)
