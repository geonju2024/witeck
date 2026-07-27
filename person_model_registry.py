import os

import numpy as np


PERSON_MODEL_TYPES = ("bilstm", "tcn", "transformer", "random_forest")
DEFAULT_PERSON_MODEL_TYPE = "tcn"


def validate_model_type(model_type):
    normalized = str(model_type).strip().lower().replace("-", "_")
    if normalized not in PERSON_MODEL_TYPES:
        raise ValueError(
            f"Unsupported model '{model_type}'. Choose one of: {', '.join(PERSON_MODEL_TYPES)}"
        )
    return normalized


def is_keras_model(model_type):
    return validate_model_type(model_type) != "random_forest"


def person_model_path(model_type):
    model_type = validate_model_type(model_type)
    extension = "keras" if is_keras_model(model_type) else "joblib"
    return f"gesture_person_{model_type}_model.{extension}"


def person_labels_path(model_type):
    model_type = validate_model_type(model_type)
    return f"gesture_person_{model_type}_labels.json"


def person_report_path(model_type):
    model_type = validate_model_type(model_type)
    return f"gesture_person_{model_type}_evaluation_report.json"


def extract_random_forest_features(sequences):
    sequences = np.asarray(sequences, dtype=np.float32)
    if sequences.ndim == 2:
        sequences = np.expand_dims(sequences, axis=0)
    if sequences.ndim != 3:
        raise ValueError(f"Expected sequence array with 3 dimensions, got {sequences.shape}")

    velocity = np.diff(sequences, axis=1)
    acceleration = np.diff(velocity, axis=1)

    def summarize(values):
        return np.concatenate(
            [
                np.mean(values, axis=1),
                np.std(values, axis=1),
                np.min(values, axis=1),
                np.max(values, axis=1),
            ],
            axis=1,
        )

    trajectory = np.sum(np.abs(velocity), axis=1)
    return np.concatenate(
        [summarize(sequences), summarize(velocity), summarize(acceleration), trajectory],
        axis=1,
    ).astype(np.float32)


def build_random_forest(random_state=42):
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(
        n_estimators=400,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=-1,
        random_state=random_state,
    )


def _build_bilstm(sequence_length, feature_size, class_count):
    from tensorflow.keras.layers import Bidirectional, Dense, Dropout, Input, LSTM
    from tensorflow.keras.models import Model

    inputs = Input(shape=(sequence_length, feature_size))
    x = Bidirectional(LSTM(64))(inputs)
    x = Dropout(0.35)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.25)(x)
    outputs = Dense(class_count, activation="softmax")(x)
    return Model(inputs, outputs, name="person_bilstm")


def _tcn_block(x, filters, dilation_rate, dropout_rate=0.2):
    from tensorflow.keras.layers import Add, Conv1D, Dropout, LayerNormalization, ReLU

    residual = x
    x = Conv1D(filters, 3, padding="causal", dilation_rate=dilation_rate)(x)
    x = LayerNormalization()(x)
    x = ReLU()(x)
    x = Dropout(dropout_rate)(x)
    x = Conv1D(filters, 3, padding="causal", dilation_rate=dilation_rate)(x)
    x = LayerNormalization()(x)
    return ReLU()(Add()([residual, x]))


def _build_tcn(sequence_length, feature_size, class_count):
    from tensorflow.keras.layers import Conv1D, Dense, Dropout, GlobalAveragePooling1D, Input
    from tensorflow.keras.models import Model

    inputs = Input(shape=(sequence_length, feature_size))
    x = Conv1D(64, 1, padding="same")(inputs)
    for dilation_rate in (1, 2, 4, 8):
        x = _tcn_block(x, filters=64, dilation_rate=dilation_rate)
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.25)(x)
    outputs = Dense(class_count, activation="softmax")(x)
    return Model(inputs, outputs, name="person_tcn")


_POSITION_ENCODING_CLASS = None


def _get_position_encoding_class():
    global _POSITION_ENCODING_CLASS
    if _POSITION_ENCODING_CLASS is not None:
        return _POSITION_ENCODING_CLASS

    import tensorflow as tf

    @tf.keras.utils.register_keras_serializable(package="gesture_auth")
    class AddPositionEncoding(tf.keras.layers.Layer):
        def call(self, inputs):
            length = tf.shape(inputs)[1]
            depth = tf.shape(inputs)[2]
            positions = tf.cast(tf.range(length)[:, tf.newaxis], tf.float32)
            dimensions = tf.cast(tf.range(depth)[tf.newaxis, :], tf.float32)
            rates = 1.0 / tf.pow(
                10000.0,
                (2.0 * tf.floor(dimensions / 2.0)) / tf.cast(depth, tf.float32),
            )
            angles = positions * rates
            parity = tf.cast(
                tf.math.floormod(tf.range(depth), 2), tf.float32
            )[tf.newaxis, :]
            encoding = (
                tf.sin(angles) * (1.0 - parity) + tf.cos(angles) * parity
            )
            return inputs + encoding[tf.newaxis, :, :]

        def get_config(self):
            return super().get_config()

    _POSITION_ENCODING_CLASS = AddPositionEncoding
    return _POSITION_ENCODING_CLASS


def _build_transformer(sequence_length, feature_size, class_count):
    from tensorflow.keras.layers import (
        Add,
        Dense,
        Dropout,
        GlobalAveragePooling1D,
        Input,
        LayerNormalization,
        MultiHeadAttention,
    )
    from tensorflow.keras.models import Model

    AddPositionEncoding = _get_position_encoding_class()
    model_dim = 64
    inputs = Input(shape=(sequence_length, feature_size))
    x = Dense(model_dim)(inputs)
    x = AddPositionEncoding(name="position_encoding")(x)

    for block_index in range(3):
        attention = MultiHeadAttention(
            num_heads=4,
            key_dim=model_dim // 4,
            dropout=0.1,
            name=f"attention_{block_index + 1}",
        )(x, x)
        x = LayerNormalization()(Add()([x, Dropout(0.1)(attention)]))
        feed_forward = Dense(128, activation="gelu")(x)
        feed_forward = Dense(model_dim)(feed_forward)
        x = LayerNormalization()(Add()([x, Dropout(0.1)(feed_forward)]))

    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.25)(x)
    outputs = Dense(class_count, activation="softmax")(x)
    return Model(inputs, outputs, name="person_transformer")


def build_keras_model(model_type, sequence_length, feature_size, class_count):
    from tensorflow.keras.optimizers import Adam

    model_type = validate_model_type(model_type)
    builders = {
        "bilstm": _build_bilstm,
        "tcn": _build_tcn,
        "transformer": _build_transformer,
    }
    if model_type not in builders:
        raise ValueError(f"{model_type} is not a Keras model")

    model = builders[model_type](sequence_length, feature_size, class_count)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_person_model(model_type, model, path):
    model_type = validate_model_type(model_type)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if is_keras_model(model_type):
        model.save(path)
    else:
        import joblib

        joblib.dump(model, path)


def load_person_model(model_type, path):
    model_type = validate_model_type(model_type)
    if is_keras_model(model_type):
        from tensorflow.keras.models import load_model

        custom_objects = {}
        if model_type == "transformer":
            position_class = _get_position_encoding_class()
            custom_objects = {
                "AddPositionEncoding": position_class,
                "gesture_auth>AddPositionEncoding": position_class,
            }
        return load_model(path, compile=False, custom_objects=custom_objects)

    import joblib

    return joblib.load(path)


def predict_probabilities(model_type, model, sequences):
    model_type = validate_model_type(model_type)
    sequences = np.asarray(sequences, dtype=np.float32)
    if sequences.ndim == 2:
        sequences = np.expand_dims(sequences, axis=0)

    if is_keras_model(model_type):
        return np.asarray(model.predict(sequences, verbose=0), dtype=np.float32)
    return np.asarray(model.predict_proba(extract_random_forest_features(sequences)), dtype=np.float32)
