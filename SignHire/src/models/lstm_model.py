"""
src/models/lstm_model.py
------------------------
Primary LSTM model for ISL sign recognition.

Architecture
------------
    Input  : (sequence_frames, feature_dim)   e.g. (30, 231)
    → LSTM(64, return_sequences=True)
    → Dropout(0.3)
    → LSTM(32)
    → Dropout(0.3)
    → Dense(num_classes, activation='softmax')

Design rationale
----------------
- Two stacked LSTM layers capture both short-term (hand shape) and
  longer-term (motion trajectory) temporal patterns.
- Units are intentionally small (64/32) to avoid overfitting on a
  small dataset (~40 clips).
- Dropout reduces co-adaptation between neurons.
- Softmax output gives a probability distribution over classes, which
  we use directly as a confidence score at inference time.

Usage
-----
    from src.models.lstm_model import build_lstm_model
    model = build_lstm_model(sequence_frames=30, feature_dim=231,
                             num_classes=8, settings=cfg["model"])
    model.summary()
"""

import tensorflow as tf
from tensorflow import keras


def build_lstm_model(sequence_frames: int, feature_dim: int,
                     num_classes: int, settings: dict) -> keras.Model:
    """
    Build and compile the LSTM model.

    Parameters
    ----------
    sequence_frames : int
        Number of time steps (frames) per input sequence.
    feature_dim : int
        Number of features per frame (landmark coordinates).
    num_classes : int
        Number of sign classes to predict.
    settings : dict
        The 'model' section from settings.yaml.  Keys used:
            lstm_units_1, lstm_units_2, dropout, learning_rate

    Returns
    -------
    keras.Model  (compiled, ready to train)
    """
    units_1  = settings["lstm_units_1"]   # e.g. 64
    units_2  = settings["lstm_units_2"]   # e.g. 32
    dropout  = settings["dropout"]         # e.g. 0.3
    lr       = settings["learning_rate"]   # e.g. 0.001

    # ── Model definition ──────────────────────────────────────────────────
    inputs = keras.Input(shape=(sequence_frames, feature_dim),
                         name="sequence_input")

    # First LSTM layer — return_sequences=True passes the full sequence
    # to the second LSTM layer so it can see the temporal context.
    x = keras.layers.LSTM(units_1, return_sequences=True,
                           name="lstm_1")(inputs)
    x = keras.layers.Dropout(dropout, name="dropout_1")(x)

    # Second LSTM layer — return_sequences=False collapses the time axis
    # into a single context vector.
    x = keras.layers.LSTM(units_2, return_sequences=False,
                           name="lstm_2")(x)
    x = keras.layers.Dropout(dropout, name="dropout_2")(x)

    # Output layer — softmax gives per-class probabilities
    outputs = keras.layers.Dense(num_classes, activation="softmax",
                                  name="output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="SignHire_LSTM")

    # ── Compile ───────────────────────────────────────────────────────────
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model
