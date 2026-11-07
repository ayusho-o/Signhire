"""
src/capture/landmark_viewer.py
-------------------------------
Phase 1 deliverable: live webcam feed with MediaPipe skeleton overlay.

Run from the SignHire/ directory:
    python src/capture/landmark_viewer.py

What it shows
-------------
- Live webcam feed with pose skeleton, both hand skeletons, and selected
  face landmark dots drawn on top.
- Top-left HUD: feature vector dimension, FPS, and whether each landmark
  group (pose, left hand, right hand, face) was detected this frame.
- The raw normalised feature vector is NOT displayed (it's 231 numbers),
  but the script prints a one-line summary to the terminal every second
  so you can confirm data is flowing.

Controls
--------
  Q  — quit

Purpose
-------
This script is for visual verification only.  It lets you confirm that
MediaPipe is running, the skeleton looks correct, and normalisation is
working before you record any data.  No data is saved here.
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np

# ── Make sure the SignHire root is on sys.path so relative imports work
# when running the script directly (python src/capture/landmark_viewer.py).
_ROOT = Path(__file__).resolve().parents[2]  # SignHire/
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.landmarks.extractor import LandmarkExtractor


# ── Colours (BGR) ─────────────────────────────────────────────────────────────
GREEN  = (0, 220, 0)
RED    = (0, 0, 220)
WHITE  = (255, 255, 255)
BLACK  = (0, 0, 0)
YELLOW = (0, 220, 220)


def draw_hud(frame: np.ndarray, fps: float, feature_dim: int,
             pose_ok: bool, left_ok: bool, right_ok: bool, face_ok: bool):
    """
    Draws a heads-up display in the top-left corner of the frame.

    Shows FPS, feature vector dimension, and a coloured indicator for each
    detected landmark group (green = detected, red = missing).
    """
    h, w = frame.shape[:2]

    # Semi-transparent dark background strip for readability
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (320, 130), BLACK, -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    def status(detected: bool) -> tuple:
        return (GREEN if detected else RED, "OK" if detected else "--")

    lines = [
        f"SignHire  |  Phase 1  |  FPS: {fps:5.1f}",
        f"Feature dim: {feature_dim}",
        "",   # spacer
        "",   # pose line (drawn separately for colour)
        "",   # left hand
        "",   # right hand
        "",   # face
    ]

    # First two plain lines
    for i, text in enumerate(lines[:2]):
        cv2.putText(frame, text, (10, 22 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1, cv2.LINE_AA)

    # Coloured status indicators
    statuses = [
        ("Pose      ", pose_ok),
        ("Left hand ", left_ok),
        ("Right hand", right_ok),
        ("Face      ", face_ok),
    ]
    y0 = 22 + 2 * 22 + 8  # start after the first two lines + spacer
    for i, (label, detected) in enumerate(statuses):
        colour, tag = status(detected)
        text = f"{label}: {tag}"
        cv2.putText(frame, text, (10, y0 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, colour, 1, cv2.LINE_AA)

    # Quit hint at bottom
    cv2.putText(frame, "Press Q to quit", (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, YELLOW, 1, cv2.LINE_AA)


def main():
    settings_path = _ROOT / "config" / "settings.yaml"

    print("SignHire — Landmark Viewer")
    print(f"  Settings : {settings_path}")
    print("  Controls : Q = quit")
    print()

    # Initialise extractor (loads config, creates MediaPipe models)
    extractor = LandmarkExtractor(
        settings_path=str(settings_path),
        draw_landmarks=True,
    )
    print(f"  Feature vector dimension: {extractor.feature_dim}")
    print("  Opening webcam …")

    # Try the default camera (index 0).  Change to 1 or 2 if needed.
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam. Is one connected and not in use?")
        sys.exit(1)

    # Request 30 fps — the OS may not honour this, but it sets intent.
    cap.set(cv2.CAP_PROP_FPS, 30)

    window_name = "SignHire — Landmark Viewer  (Q to quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 540)

    # FPS tracking
    fps = 0.0
    frame_count = 0
    fps_timer = time.perf_counter()

    # Terminal summary timer (print once per second)
    summary_timer = time.perf_counter()

    print("  Webcam open. Window should appear shortly.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("WARNING: Failed to read frame from webcam.")
            break

        # Flip horizontally so it feels like a mirror (more natural for signing)
        frame = cv2.flip(frame, 1)

        # Run MediaPipe extraction
        feature_vector, annotated = extractor.process_frame(frame)

        # Determine which groups were detected (non-zero sub-vectors)
        n_pose  = len(extractor.pose_indices) * 3
        n_face  = len(extractor.face_indices) * 3
        n_hand  = extractor.hand_lm_count * 3

        pose_vec  = feature_vector[:n_pose]
        face_vec  = feature_vector[n_pose: n_pose + n_face]
        left_vec  = feature_vector[n_pose + n_face: n_pose + n_face + n_hand]
        right_vec = feature_vector[n_pose + n_face + n_hand:]

        pose_ok  = np.any(pose_vec  != 0)
        face_ok  = np.any(face_vec  != 0)
        left_ok  = np.any(left_vec  != 0)
        right_ok = np.any(right_vec != 0)

        # FPS calculation (rolling average over 1 second)
        frame_count += 1
        elapsed = time.perf_counter() - fps_timer
        if elapsed >= 1.0:
            fps = frame_count / elapsed
            frame_count = 0
            fps_timer = time.perf_counter()

        # Draw HUD on annotated frame
        draw_hud(annotated, fps, extractor.feature_dim,
                 pose_ok, left_ok, right_ok, face_ok)

        # Show frame
        cv2.imshow(window_name, annotated)

        # Terminal summary once per second
        if time.perf_counter() - summary_timer >= 1.0:
            non_zero = int(np.count_nonzero(feature_vector))
            print(
                f"FPS: {fps:5.1f}  |  "
                f"pose={'Y' if pose_ok else 'N'}  "
                f"left={'Y' if left_ok else 'N'}  "
                f"right={'Y' if right_ok else 'N'}  "
                f"face={'Y' if face_ok else 'N'}  |  "
                f"non-zero values: {non_zero}/{extractor.feature_dim}"
            )
            summary_timer = time.perf_counter()

        # Q or Escape to quit
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break

    # Cleanup
    cap.release()
    extractor.close()
    cv2.destroyAllWindows()
    print("\nLandmark viewer closed.")


if __name__ == "__main__":
    main()
