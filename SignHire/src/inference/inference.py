"""
src/inference/inference.py
--------------------------
Phase 4 — real-time inference engine.

How it works
------------
1. A rolling frame buffer holds the last `sequence_frames` landmark
   vectors (one per webcam frame).
2. Every `predict_every_n_frames` frames, the buffer is copied and fed
   to the model in a background thread so the webcam feed never stalls.
3. The softmax output is compared against `confidence_threshold`:
     max_prob >= threshold  →  return predicted label + confidence
     max_prob <  threshold  →  return UNSURE + confidence
4. The latest prediction is stored in a thread-safe result object that
   the caller can read at any time.

Usage
-----
    from src.inference.inference import InferenceEngine

    engine = InferenceEngine(
        model_path="data/models/baseline_fold_1.keras",
        class_names=["hello", "no", ...],
        settings_path="config/settings.yaml",
    )
    engine.start()

    # In your frame loop:
    engine.push_frame(feature_vector)   # call once per webcam frame
    result = engine.latest_result()     # non-blocking read

    engine.stop()

Result dict keys
----------------
    label       : str   — predicted sign label or "Unsure, please repeat"
    confidence  : float — max softmax probability (0-1)
    is_unsure   : bool  — True when below threshold
    raw_probs   : list  — full softmax vector
"""

import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import yaml


class InferenceEngine:
    """
    Thread-safe, non-blocking inference engine.

    Parameters
    ----------
    model_path : str or Path
        Path to a saved Keras model (.keras file).
    class_names : list of str
        Ordered list of sign labels matching model output indices.
    settings_path : str or Path
        Path to config/settings.yaml.
    """

    # Sentinel result shown before the buffer is full or a model is loaded
    _EMPTY_RESULT = {
        "label":      "Waiting…",
        "confidence": 0.0,
        "is_unsure":  True,
        "raw_probs":  [],
    }

    def __init__(self, model_path, class_names: list,
                 settings_path: str = "config/settings.yaml"):

        settings_path = Path(settings_path)
        with open(settings_path) as f:
            cfg = yaml.safe_load(f)

        self.class_names  = class_names
        self.seq_frames   = cfg["recording"]["sequence_frames"]
        self.threshold    = cfg["inference"]["confidence_threshold"]
        self.predict_every = cfg["inference"]["predict_every_n_frames"]

        # Rolling buffer — deque with maxlen auto-discards oldest frames
        self._buffer = deque(maxlen=self.seq_frames)
        self._frame_count = 0  # frames pushed since last prediction

        # Latest result — written by background thread, read by main thread
        self._result      = dict(self._EMPTY_RESULT)
        self._result_lock = threading.Lock()

        # Background prediction thread
        self._pending_seq  = None   # sequence waiting to be predicted
        self._seq_lock     = threading.Lock()
        self._stop_event   = threading.Event()
        self._thread       = None

        # Load model (import TensorFlow here so startup is not penalised
        # if the module is imported but the engine never started)
        self._model = None
        self._model_path = Path(model_path)

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self):
        """Load the model and start the background prediction thread."""
        self._load_model()
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._prediction_loop, daemon=True, name="inference"
        )
        self._thread.start()

    def stop(self):
        """Signal the background thread to stop and wait for it."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def push_frame(self, feature_vector: np.ndarray):
        """
        Add one frame's landmark vector to the rolling buffer.

        Call this once per webcam frame.  When the buffer is full and
        `predict_every_n_frames` frames have elapsed since the last
        prediction, a copy of the buffer is queued for the background
        thread.

        Parameters
        ----------
        feature_vector : np.ndarray, shape (feature_dim,)
        """
        self._buffer.append(feature_vector.astype(np.float32))
        self._frame_count += 1

        # Queue a prediction when buffer is full and interval has elapsed
        if (len(self._buffer) == self.seq_frames and
                self._frame_count >= self.predict_every):
            self._frame_count = 0
            seq = np.array(self._buffer, dtype=np.float32)  # (T, F)
            with self._seq_lock:
                self._pending_seq = seq  # overwrite any unprocessed seq

    def latest_result(self) -> dict:
        """
        Return the most recent prediction result (non-blocking).

        Returns a copy of the internal result dict so the caller can
        safely read it without holding the lock.
        """
        with self._result_lock:
            return dict(self._result)

    def buffer_fill(self) -> float:
        """Return buffer fill fraction (0.0 – 1.0) for the HUD."""
        return len(self._buffer) / self.seq_frames

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _load_model(self):
        """Load the Keras model from disk."""
        import tensorflow as tf  # deferred import
        if not self._model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {self._model_path}\n"
                "Run src/train/train.py first."
            )
        self._model = tf.keras.models.load_model(str(self._model_path))

    def _prediction_loop(self):
        """
        Background thread: poll for pending sequences and predict.

        Runs until stop() is called.
        """
        while not self._stop_event.is_set():
            seq = None
            with self._seq_lock:
                if self._pending_seq is not None:
                    seq = self._pending_seq
                    self._pending_seq = None

            if seq is not None:
                result = self._predict(seq)
                with self._result_lock:
                    self._result = result
            else:
                # Nothing to do — sleep briefly to avoid busy-waiting
                time.sleep(0.01)

    def _predict(self, seq: np.ndarray) -> dict:
        """
        Run the model on one sequence and apply the confidence gate.

        Parameters
        ----------
        seq : np.ndarray, shape (seq_frames, feature_dim)

        Returns
        -------
        dict with label, confidence, is_unsure, raw_probs
        """
        # Model expects batch dimension → (1, T, F)
        batch = seq[np.newaxis, ...]
        probs = self._model.predict(batch, verbose=0)[0]  # shape (n_classes,)

        best_idx  = int(np.argmax(probs))
        confidence = float(probs[best_idx])
        is_unsure  = confidence < self.threshold

        label = (
            "Unsure, please repeat"
            if is_unsure
            else self.class_names[best_idx]
        )

        return {
            "label":      label,
            "confidence": round(confidence, 4),
            "is_unsure":  is_unsure,
            "raw_probs":  [round(float(p), 4) for p in probs],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Model discovery helper
# ─────────────────────────────────────────────────────────────────────────────

def find_best_model(models_dir: Path, preference: str = "baseline") -> Path:
    """
    Find the best saved model file to use for live inference.

    Strategy
    --------
    1. Look for <preference>_fold_1.keras (first fold of preferred model).
    2. Fall back to any .keras file in models_dir.
    3. Raise FileNotFoundError if nothing is found.

    Parameters
    ----------
    models_dir : Path
    preference : "baseline" or "lstm"

    Returns
    -------
    Path to the chosen .keras file.
    """
    models_dir = Path(models_dir)

    # Preferred: first fold of preferred model type
    preferred = models_dir / f"{preference}_fold_1.keras"
    if preferred.exists():
        return preferred

    # Also try LOSO naming convention
    for p in sorted(models_dir.glob(f"{preference}_loso_*.keras")):
        return p

    # Fall back to any model
    all_models = sorted(models_dir.glob("*.keras"))
    if all_models:
        return all_models[0]

    raise FileNotFoundError(
        f"No trained model found in {models_dir}.\n"
        "Run src/train/train.py first."
    )
