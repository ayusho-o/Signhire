"""
tests/test_inference.py
------------------------
Smoke test for the InferenceEngine — verifies it loads a model,
accepts frames, and returns a well-formed result dict.

Run from SignHire/:
    pytest tests/test_inference.py -v
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.inference.inference import InferenceEngine, find_best_model


def get_engine():
    """Build an InferenceEngine using the first available model."""
    import yaml
    with open(_ROOT / "config" / "settings.yaml") as f:
        cfg = yaml.safe_load(f)
    with open(_ROOT / "data" / "processed" / "meta.json") as f:
        meta = json.load(f)

    class_names = meta["class_names"]
    models_dir  = _ROOT / cfg["paths"]["models_dir"]
    model_path  = find_best_model(models_dir, preference="baseline")
    engine = InferenceEngine(
        model_path=str(model_path),
        class_names=class_names,
        settings_path=str(_ROOT / "config" / "settings.yaml"),
    )
    return engine, meta["feature_dim"], cfg["recording"]["sequence_frames"]


def test_find_best_model():
    """find_best_model should return a path that exists."""
    import yaml
    with open(_ROOT / "config" / "settings.yaml") as f:
        cfg = yaml.safe_load(f)
    models_dir = _ROOT / cfg["paths"]["models_dir"]
    path = find_best_model(models_dir, preference="baseline")
    assert path.exists(), f"Model file not found: {path}"


def test_engine_starts_and_stops():
    """Engine should start and stop without errors."""
    engine, F, T = get_engine()
    engine.start()
    engine.stop()


def test_empty_buffer_returns_waiting():
    """Before any frames are pushed, result label should be 'Waiting...'."""
    engine, F, T = get_engine()
    engine.start()
    result = engine.latest_result()
    assert result["label"] == "Waiting\u2026" or result["is_unsure"] is True
    engine.stop()


def test_full_buffer_returns_result():
    """After pushing a full buffer of frames, result should have a valid label."""
    engine, F, T = get_engine()
    engine.start()

    import yaml
    with open(_ROOT / "config" / "settings.yaml") as f:
        cfg = yaml.safe_load(f)
    with open(_ROOT / "data" / "processed" / "meta.json") as f:
        meta = json.load(f)
    class_names = meta["class_names"]

    rng = np.random.default_rng(42)
    for _ in range(T):
        engine.push_frame(rng.random(F).astype(np.float32))

    # Wait for background thread to process — TF first-call can be slow
    # Retry up to 5 seconds before failing
    deadline = time.time() + 5.0
    result = engine.latest_result()
    while result["label"] in ("Waiting\u2026", "Waiting...") and time.time() < deadline:
        time.sleep(0.2)
        result = engine.latest_result()
    assert "label" in result
    assert "confidence" in result
    assert "is_unsure" in result
    assert isinstance(result["confidence"], float)
    assert 0.0 <= result["confidence"] <= 1.0

    # Label must be either a class name or the unsure message
    valid_labels = set(class_names) | {"Unsure, please repeat"}
    assert result["label"] in valid_labels, (
        f"Unexpected label: {result['label']}"
    )
    engine.stop()


def test_buffer_fill_increases():
    """buffer_fill() should go from 0 to 1 as frames are pushed."""
    engine, F, T = get_engine()
    engine.start()

    assert engine.buffer_fill() == 0.0

    rng = np.random.default_rng(0)
    for _ in range(T):
        engine.push_frame(rng.random(F).astype(np.float32))

    assert engine.buffer_fill() == 1.0
    engine.stop()
