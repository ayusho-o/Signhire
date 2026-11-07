"""
tests/test_extractor.py
------------------------
Basic sanity checks for LandmarkExtractor (Phase 1).

Run from the SignHire/ directory:
    pytest tests/test_extractor.py -v
"""

import sys
from pathlib import Path
import numpy as np

# Add SignHire root to path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.landmarks.extractor import LandmarkExtractor


def make_extractor():
    return LandmarkExtractor(
        settings_path=str(_ROOT / "config" / "settings.yaml"),
        draw_landmarks=False,  # no drawing needed in tests
    )


def test_feature_dim_is_positive():
    """Extractor should report a positive feature dimension."""
    ext = make_extractor()
    assert ext.feature_dim > 0, "feature_dim must be > 0"
    ext.close()


def test_process_blank_frame_returns_correct_shape():
    """
    A blank (black) frame has no detectable pose, so the extractor should
    return a zero vector of the correct length.
    """
    ext = make_extractor()
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    vec, annotated = ext.process_frame(blank)

    assert vec.shape == (ext.feature_dim,), (
        f"Expected shape ({ext.feature_dim},), got {vec.shape}"
    )
    # All zeros because no landmarks are detected in a blank frame
    assert np.all(vec == 0), "Blank frame should produce an all-zero vector"
    ext.close()


def test_annotated_frame_same_size_as_input():
    """The annotated frame must have the same spatial dimensions as the input."""
    ext = make_extractor()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    _, annotated = ext.process_frame(frame)
    assert annotated.shape == frame.shape, (
        "Annotated frame shape must match input frame shape"
    )
    ext.close()


def test_feature_dim_matches_config():
    """
    Verify the computed feature_dim matches the sum implied by config:
      pose_indices * 3 + face_indices * 3 + hand_landmarks * 3 * 2
    """
    import yaml
    ext = make_extractor()
    with open(_ROOT / "config" / "settings.yaml") as f:
        cfg = yaml.safe_load(f)
    lm = cfg["landmarks"]
    expected = (
        len(lm["pose_indices"]) * 3
        + len(lm["face_indices"]) * 3
        + lm["hand_landmarks"] * 3 * 2
    )
    assert ext.feature_dim == expected, (
        f"feature_dim {ext.feature_dim} != expected {expected}"
    )
    ext.close()
