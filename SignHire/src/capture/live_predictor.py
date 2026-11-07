"""
src/capture/live_predictor.py
------------------------------
Phase 4 deliverable — standalone live prediction viewer.

Shows the webcam feed with the MediaPipe skeleton overlay and a
real-time prediction HUD.  This is the Phase 4 test script; the same
InferenceEngine is reused inside the Flask app in Phase 5.

Run from SignHire/:
    python src/capture/live_predictor.py
    python src/capture/live_predictor.py --camera 1
    python src/capture/live_predictor.py --model lstm

What the HUD shows
------------------
- Predicted sign label (large, centred)
- Confidence bar (green = confident, orange = below threshold)
- "Unsure, please repeat" when confidence < threshold
- Buffer fill indicator (how full the 30-frame window is)
- Current FPS

Controls
--------
  Q  — quit
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.landmarks.extractor import LandmarkExtractor
from src.inference.inference  import InferenceEngine, find_best_model


# ── Colours (BGR) ─────────────────────────────────────────────────────────────
WHITE   = (255, 255, 255)
BLACK   = (0,   0,   0)
GREEN   = (0,   210, 0)
ORANGE  = (0,   140, 255)
RED     = (0,   0,   220)
YELLOW  = (0,   210, 210)
GREY    = (120, 120, 120)
BLUE    = (200, 100, 0)


def draw_prediction_hud(frame: np.ndarray, result: dict,
                         buffer_fill: float, fps: float,
                         threshold: float):
    """
    Draw the prediction HUD on the frame in-place.

    Layout
    ------
    Top strip    : FPS + buffer fill bar
    Centre       : predicted label (large)
    Below label  : confidence bar + percentage
    Bottom strip : "Unsure, please repeat" warning if needed
    """
    h, w = frame.shape[:2]

    label      = result["label"]
    confidence = result["confidence"]
    is_unsure  = result["is_unsure"]

    # ── Top strip background ──────────────────────────────────────────────
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 44), BLACK, -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    # FPS
    cv2.putText(frame, f"FPS: {fps:5.1f}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1, cv2.LINE_AA)

    # Buffer fill bar (right side of top strip)
    bar_w = 160
    bar_x = w - bar_w - 10
    bar_y = 14
    cv2.rectangle(frame, (bar_x, bar_y),
                  (bar_x + bar_w, bar_y + 16), GREY, -1)
    filled = int(bar_w * buffer_fill)
    fill_colour = GREEN if buffer_fill >= 1.0 else YELLOW
    cv2.rectangle(frame, (bar_x, bar_y),
                  (bar_x + filled, bar_y + 16), fill_colour, -1)
    cv2.putText(frame, "Buffer", (bar_x - 55, bar_y + 13),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1, cv2.LINE_AA)

    # ── Prediction label (centred, large) ─────────────────────────────────
    label_display = label.upper()
    font_scale = 1.8 if not is_unsure else 0.85
    label_colour = GREEN if not is_unsure else ORANGE

    (tw, th), _ = cv2.getTextSize(
        label_display, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 3)
    lx = (w - tw) // 2
    ly = h // 2 + 20

    # Shadow for readability
    cv2.putText(frame, label_display, (lx + 2, ly + 2),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, BLACK, 4, cv2.LINE_AA)
    cv2.putText(frame, label_display, (lx, ly),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, label_colour, 3,
                cv2.LINE_AA)

    # ── Confidence bar ────────────────────────────────────────────────────
    conf_bar_w = 300
    conf_bar_x = (w - conf_bar_w) // 2
    conf_bar_y = ly + 30
    conf_bar_h = 18

    # Background
    cv2.rectangle(frame,
                  (conf_bar_x, conf_bar_y),
                  (conf_bar_x + conf_bar_w, conf_bar_y + conf_bar_h),
                  GREY, -1)
    # Threshold marker
    thresh_x = conf_bar_x + int(conf_bar_w * threshold)
    cv2.line(frame,
             (thresh_x, conf_bar_y - 4),
             (thresh_x, conf_bar_y + conf_bar_h + 4),
             YELLOW, 2)
    # Fill
    filled_w = int(conf_bar_w * confidence)
    bar_colour = GREEN if confidence >= threshold else ORANGE
    cv2.rectangle(frame,
                  (conf_bar_x, conf_bar_y),
                  (conf_bar_x + filled_w, conf_bar_y + conf_bar_h),
                  bar_colour, -1)
    # Confidence percentage text
    conf_text = f"{confidence * 100:.1f}%"
    cv2.putText(frame, conf_text,
                (conf_bar_x + conf_bar_w + 8, conf_bar_y + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1, cv2.LINE_AA)
    # Threshold label
    cv2.putText(frame, f"thresh={threshold:.0%}",
                (thresh_x - 45, conf_bar_y - 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, YELLOW, 1, cv2.LINE_AA)

    # ── Bottom quit hint ──────────────────────────────────────────────────
    cv2.putText(frame, "Q to quit", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, GREY, 1, cv2.LINE_AA)


def main(camera_index: int = 1, model_preference: str = "baseline"):
    """Open webcam and run live prediction loop."""

    settings_path = _ROOT / "config" / "settings.yaml"
    with open(settings_path) as f:
        cfg = yaml.safe_load(f)

    # Load class names from processed dataset metadata
    meta_path = _ROOT / cfg["paths"]["processed_data"] / "meta.json"
    if not meta_path.exists():
        print("ERROR: Processed dataset not found.")
        print("  Run: python src/preprocess/preprocess.py")
        sys.exit(1)
    with open(meta_path) as f:
        meta = json.load(f)
    class_names = meta["class_names"]

    # Find best model
    models_dir = _ROOT / cfg["paths"]["models_dir"]
    try:
        model_path = find_best_model(models_dir, preference=model_preference)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    threshold = cfg["inference"]["confidence_threshold"]

    print("SignHire — Live Predictor")
    print(f"  Model      : {model_path.name}")
    print(f"  Classes    : {class_names}")
    print(f"  Threshold  : {threshold}")
    print(f"  Camera     : {camera_index}")
    print()

    # Initialise components
    extractor = LandmarkExtractor(
        settings_path=str(settings_path), draw_landmarks=True
    )
    engine = InferenceEngine(
        model_path=str(model_path),
        class_names=class_names,
        settings_path=str(settings_path),
    )

    print("  Loading model…")
    engine.start()
    print("  Model loaded. Opening webcam…")

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"ERROR: Could not open camera {camera_index}.")
        engine.stop()
        extractor.close()
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FPS, cfg["recording"]["fps"])

    window = "SignHire — Live Predictor  (Q to quit)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 960, 540)

    # FPS tracking
    fps = 0.0
    frame_count = 0
    fps_timer = time.perf_counter()

    print("  Running. Press Q in the window to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("WARNING: Failed to read frame.")
            break
        frame = cv2.flip(frame, 1)

        # Extract landmarks + draw skeleton
        feature_vec, annotated = extractor.process_frame(frame)

        # Push to inference engine
        engine.push_frame(feature_vec)

        # Read latest prediction (non-blocking)
        result = engine.latest_result()

        # Update FPS
        frame_count += 1
        elapsed = time.perf_counter() - fps_timer
        if elapsed >= 1.0:
            fps = frame_count / elapsed
            frame_count = 0
            fps_timer = time.perf_counter()

        # Draw HUD
        draw_prediction_hud(
            annotated,
            result=result,
            buffer_fill=engine.buffer_fill(),
            fps=fps,
            threshold=threshold,
        )

        cv2.imshow(window, annotated)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break

    # Cleanup
    cap.release()
    engine.stop()
    extractor.close()
    cv2.destroyAllWindows()
    print("Live predictor closed.")


def parse_args():
    p = argparse.ArgumentParser(description="SignHire live predictor")
    p.add_argument("--camera", type=int, default=1,
                   help="Camera index (default: 1)")
    p.add_argument("--model", choices=["lstm", "baseline"],
                   default="baseline",
                   help="Which model to use (default: baseline)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(camera_index=args.camera, model_preference=args.model)
