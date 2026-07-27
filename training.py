import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from gesture_auth_config import FEATURE_SIZE, METADATA_PATH, MODEL_PATH, SEQUENCE_LENGTH


# =========================
# Settings
# =========================
MODEL_SAVE_PATH = MODEL_PATH


# =========================
# Read metadata.csv
# =========================
df = pd.read_csv(METADATA_PATH)

print("metadata columns:", df.columns.tolist())
print(df.head())

if "auth_label" not in df.columns:
    raise ValueError("metadata.csv is missing the auth_label column. Add success/fail labels.")


# =========================
# Load data
# =========================
X = []
y = []

for idx, row in df.iterrows():
    file_path = str(row["file_path"])

    # Normalize paths across Windows / Mac / Linux
    file_path = file_path.replace("\\", os.sep).replace("/", os.sep)

    if not os.path.exists(file_path):
        print(f"Missing file, skipped: {file_path}")
        continue

    data = np.load(file_path)

    # Validate data shape
    if data.shape != (SEQUENCE_LENGTH, FEATURE_SIZE):
        print(f"Shape mismatch, skipped: {file_path}, shape={data.shape}")
        continue

    auth_label = str(row["auth_label"]).strip().lower()

    if auth_label == "success":
        label = 1
    elif auth_label == "fail":
        label = 0
    else:
        print(f"Invalid label, skipped: {auth_label}")
        continue

    X.append(data)
    y.append(label)


X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int32)

print("X shape:", X.shape)
print("y shape:", y.shape)
print("success count:", np.sum(y == 1))
print("fail count:", np.sum(y == 0))


# =========================
# Validate data
# =========================
if len(X) == 0:
    raise ValueError("No training data found.")

if len(np.unique(y)) < 2:
    raise ValueError("Both success and fail data are required. Only one label type is currently present.")


# =========================
# Split train / test data
# =========================
# If the dataset is too small, stratify can fail, so fall back to a normal split.
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
except ValueError:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)


# =========================
# Build GRU model
# =========================
model = Sequential([
    GRU(64, input_shape=(SEQUENCE_LENGTH, FEATURE_SIZE)),
    Dropout(0.3),
    Dense(32, activation="relu"),
    Dropout(0.2),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# =========================
# Train model
# =========================
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=8,
    validation_data=(X_test, y_test),
    callbacks=[early_stop]
)


# =========================
# Evaluate model
# =========================
loss, acc = model.evaluate(X_test, y_test)
print("Test loss:", loss)
print("Test accuracy:", acc)

y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob >= 0.5).astype(int).reshape(-1)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

print("Classification Report:")
print(classification_report(
    y_test,
    y_pred,
    labels=[0, 1],
    target_names=["fail", "success"],
    zero_division=0
))


# =========================
# Save model
# =========================
model.save(MODEL_SAVE_PATH)
print(f"Model saved: {MODEL_SAVE_PATH}")
