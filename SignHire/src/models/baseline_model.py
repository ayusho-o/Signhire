"""
src/models/baseline_model.py
-----------------------------
GRU baseline model for ISL sign recognition.

Architecture
------------
    Input  : (sequence_frames, feature_dim)   e.g. (30, 231)
    → GRU(32)
    → Dropout(0.3)
    → Dense(num_classes, activation='softmax')

Design rationale
----------------
- Single GRU layer with fewer units than the LSTM — a simpler model.
- GRU has fewer parameters than LSTM (no separate cell state) so it
  trains faster and is a fair "lite" baseline comparison.
- Trained and evaluated identically to the LSTM so the comparison is fair.
- If the baseline matches the LSTM, that tells us the extra LSTM
  complexity isn't justified for this dataset size.

Usage
-----
    from src.models.baseline_model import build_baseline_model
    model = build_baseline_model(sequence_frames=30, feature_dim=231,
                                  num_classes=8, settings=cfg["model"])
    model.summary()
"""

import tensorflow as tf
from tensorflow import keras


def build_baseline_model(sequence_frames: int, feature_dim: int,
                          num_classes: int, settings: dict) -> keras.Model:
    """
    Build and compile the GRU baseline model.

    Parameters
    ----------
    sequence_frames : int
        Number of time steps (frames) per input sequence.
    feature_dim : int
        Number of features per frame.
    num_classes : int
        Number of sign classes to predict.
    settings : dict
        The 'model' section from settings.yaml.  Keys used:
            baseline_gru_units, dropout, learning_rate

    Returns
    -------
    keras.Model  (compiled, ready to train)
    """
    gru_units = settings["baseline_gru_units"]  # e.g. 32
    dropout   = settings["dropout"]              # e.g. 0.3
    lr        = settings["learning_rate"]        # e.g. 0.001

    # ── Model definition ──────────────────────────────────────────────────
    inputs = keras.Input(shape=(sequence_frames, feature_dim),
                         name="sequence_input")

    # Single GRU layer — collapses time axis directly
    x = keras.layers.GRU(gru_units, return_sequences=False,
                          name="gru_1")(inputs)
    x = keras.layers.Dropout(dropout, name="dropout_1")(x)

    outputs = keras.layers.Dense(num_classes, activation="softmax",
                                  name="output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs,
                        name="SignHire_GRU_Baseline")

    # ── Compile ───────────────────────────────────────────────────────────
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model
