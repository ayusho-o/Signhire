"""
src/app/app.py
--------------
Phase 5 — Flask web server for the SignHire interview aid.

Run from SignHire/:
    python src/app/app.py
    python src/app/app.py --camera 1 --model baseline --port 5000

Endpoints
---------
GET  /                     Serve the main UI (web/index.html)
GET  /video_feed           MJPEG stream: webcam + skeleton overlay
GET  /status               JSON: latest prediction + confidence
POST /interviewer_speech   Receive transcribed speech text
GET  /transcript           JSON: full shared transcript
POST /clear_transcript     Clear the transcript
GET  /translate            JSON: translate a label to target_lang
GET  /config               JSON: class names + target languages for UI
"""

import json
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import yaml
from flask import Flask, Response, jsonify, request, send_from_directory

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.landmarks.extractor  import LandmarkExtractor
from src.inference.inference  import InferenceEngine, find_best_model
from src.translate.translator import get_translator


# ─────────────────────────────────────────────────────────────────────────────
# Application state (module-level singletons)
# ─────────────────────────────────────────────────────────────────────────────

app = Flask(
    __name__,
    static_folder=str(_ROOT / "web" / "static"),
    template_folder=str(_ROOT / "web"),
)

# Loaded at startup
_cfg          = None
_class_names  = None
_extractor    = None
_engine       = None
_translator   = None
_camera_index = 1   # default; overridden by CLI

# Shared transcript — list of dicts
# Each entry: {speaker: "applicant"|"interviewer", text: str, time: str}
_transcript      = []
_transcript_lock = threading.Lock()

# Latest MJPEG frame (bytes) — written by capture thread, read by /video_feed
_latest_frame      = None
_latest_frame_lock = threading.Lock()

# Capture thread handle
_capture_thread = None
_stop_capture   = threading.Event()


# ─────────────────────────────────────────────────────────────────────────────
# Startup
# ─────────────────────────────────────────────────────────────────────────────

def init_app(camera_index: int = 1, model_preference: str = "baseline"):
    """Load config, models, and start the webcam capture thread."""
    global _cfg, _class_names, _extractor, _engine, _translator
    global _camera_index, _capture_thread

    _camera_index = camera_index

    # Load config
    with open(_ROOT / "config" / "settings.yaml") as f:
        _cfg = yaml.safe_load(f)
    with open(_ROOT / "data" / "processed" / "meta.json") as f:
        meta = json.load(f)

    _class_names = meta["class_names"]

    # Translator
    _translator = get_translator(_cfg["translation"])
    provider = _cfg["translation"]["provider"]
    print(f"  Translation  : {provider}"
          + ("" if _translator.is_configured()
             else " (not configured — English only)"))

    # Landmark extractor
    _extractor = LandmarkExtractor(
        settings_path=str(_ROOT / "config" / "settings.yaml"),
        draw_landmarks=True,
    )

    # Inference engine
    models_dir = _ROOT / _cfg["paths"]["models_dir"]
    model_path = find_best_model(models_dir, preference=model_preference)
    print(f"  Model        : {model_path.name}")
    _engine = InferenceEngine(
        model_path=str(model_path),
        class_names=_class_names,
        settings_path=str(_ROOT / "config" / "settings.yaml"),
    )
    _engine.start()

    # Start webcam capture thread
    _stop_capture.clear()
    _capture_thread = threading.Thread(
        target=_capture_loop,
        args=(camera_index,),
        daemon=True,
        name="capture",
    )
    _capture_thread.start()
    print(f"  Camera       : {camera_index}")
    print(f"  Classes      : {_class_names}")


# ─────────────────────────────────────────────────────────────────────────────
# Webcam capture loop (background thread)
# ─────────────────────────────────────────────────────────────────────────────

def _capture_loop(camera_index: int):
    """
    Continuously read frames, extract landmarks, push to inference engine,
    and encode the annotated frame as JPEG for the MJPEG stream.
    """
    global _latest_frame

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"ERROR: Could not open camera {camera_index}")
        return

    cap.set(cv2.CAP_PROP_FPS, _cfg["recording"]["fps"])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while not _stop_capture.is_set():
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        frame = cv2.flip(frame, 1)

        # Extract landmarks + draw skeleton
        feature_vec, annotated = _extractor.process_frame(frame)

        # Push to inference engine
        _engine.push_frame(feature_vec)

        # Overlay prediction on the frame
        result = _engine.latest_result()
        _draw_inference_overlay(annotated, result,
                                _engine.buffer_fill(),
                                _cfg["inference"]["confidence_threshold"])

        # Encode as JPEG
        ok, jpeg = cv2.imencode(
            ".jpg", annotated,
            [cv2.IMWRITE_JPEG_QUALITY, 80],
        )
        if ok:
            with _latest_frame_lock:
                _latest_frame = jpeg.tobytes()

    cap.release()


def _draw_inference_overlay(frame: np.ndarray, result: dict,
                             buffer_fill: float, threshold: float):
    """Draw a compact prediction overlay onto the frame in-place."""
    h, w = frame.shape[:2]
    WHITE  = (255, 255, 255)
    BLACK  = (0,   0,   0)
    GREEN  = (0,   210, 0)
    ORANGE = (0,   140, 255)
    YELLOW = (0,   210, 210)
    GREY   = (120, 120, 120)

    label      = result.get("label", "Waiting\u2026")
    confidence = result.get("confidence", 0.0)
    is_unsure  = result.get("is_unsure", True)

    # Dark top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 48), BLACK, -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    # Label
    label_colour = GREEN if not is_unsure else ORANGE
    font_scale   = 0.9 if not is_unsure else 0.65
    cv2.putText(frame, label.upper(), (12, 32),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                label_colour, 2, cv2.LINE_AA)

    # Confidence bar (right side)
    bar_w = 120
    bar_x = w - bar_w - 10
    bar_y = 16
    cv2.rectangle(frame, (bar_x, bar_y),
                  (bar_x + bar_w, bar_y + 16), GREY, -1)
    thresh_x = bar_x + int(bar_w * threshold)
    cv2.line(frame, (thresh_x, bar_y - 2),
             (thresh_x, bar_y + 18), YELLOW, 2)
    fill_w = int(bar_w * confidence)
    cv2.rectangle(frame, (bar_x, bar_y),
                  (bar_x + fill_w, bar_y + 16),
                  GREEN if not is_unsure else ORANGE, -1)
    cv2.putText(frame, f"{confidence*100:.0f}%",
                (bar_x + bar_w + 4, bar_y + 13),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1, cv2.LINE_AA)

    # Buffer fill dot row (bottom-left)
    dot_y = h - 12
    for i in range(10):
        filled = (i / 10) < buffer_fill
        cv2.circle(frame,
                   (12 + i * 14, dot_y), 4,
                   GREEN if filled else GREY, -1)


# ─────────────────────────────────────────────────────────────────────────────
# Flask routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the main UI."""
    return send_from_directory(str(_ROOT / "web"), "index.html")


@app.route("/video_feed")
def video_feed():
    """MJPEG stream of the annotated webcam feed."""
    return Response(
        _mjpeg_generator(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


def _mjpeg_generator():
    """Yield JPEG frames in MJPEG format."""
    while True:
        with _latest_frame_lock:
            frame = _latest_frame
        if frame is None:
            time.sleep(0.03)
            continue
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame
            + b"\r\n"
        )
        time.sleep(0.033)  # ~30 fps ceiling for the stream


@app.route("/status")
def status():
    """Return the latest prediction as JSON."""
    if _engine is None:
        return jsonify({"error": "Engine not initialised"}), 503

    result = _engine.latest_result()
    return jsonify({
        "label":       result.get("label", "Waiting\u2026"),
        "confidence":  result.get("confidence", 0.0),
        "is_unsure":   result.get("is_unsure", True),
        "buffer_fill": _engine.buffer_fill(),
    })


@app.route("/interviewer_speech", methods=["POST"])
def interviewer_speech():
    """
    Receive transcribed speech from the browser Web Speech API.
    Appends the text to the shared transcript.
    """
    data = request.get_json(silent=True) or {}
    text = str(data.get("text", "")).strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400

    entry = {
        "speaker": "interviewer",
        "text":    text,
        "time":    _timestamp(),
    }
    with _transcript_lock:
        _transcript.append(entry)

    return jsonify({"ok": True, "entry": entry})


@app.route("/applicant_sign", methods=["POST"])
def applicant_sign():
    """
    Called by the frontend when it wants to commit the current prediction
    to the transcript (e.g. user presses a 'Send' button).
    """
    if _engine is None:
        return jsonify({"error": "Engine not initialised"}), 503

    result = _engine.latest_result()
    label  = result.get("label", "")

    if result.get("is_unsure") or not label or label == "Waiting\u2026":
        return jsonify({"error": "No confident prediction to commit"}), 400

    # Translate to the requested target language
    target_lang = request.get_json(silent=True or {}).get("target_lang", "en")
    translated  = _translator.translate(label, target_lang)

    entry = {
        "speaker":    "applicant",
        "text":       label,
        "translated": translated,
        "time":       _timestamp(),
    }
    with _transcript_lock:
        _transcript.append(entry)

    return jsonify({"ok": True, "entry": entry})


@app.route("/transcript")
def transcript():
    """Return the full shared transcript as JSON."""
    with _transcript_lock:
        return jsonify({"transcript": list(_transcript)})


@app.route("/clear_transcript", methods=["POST"])
def clear_transcript():
    """Clear the shared transcript."""
    with _transcript_lock:
        _transcript.clear()
    return jsonify({"ok": True})


@app.route("/translate")
def translate_label():
    """
    Translate a label to a target language.
    Query params: ?text=hello&target=hi
    """
    text        = request.args.get("text", "")
    target_lang = request.args.get("target", "en")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    translated = _translator.translate(text, target_lang)
    configured = _translator.is_configured() or target_lang == "en"
    return jsonify({
        "original":   text,
        "translated": translated,
        "target":     target_lang,
        "configured": configured,
    })


@app.route("/config")
def get_config():
    """Return UI configuration: class names and target languages."""
    target_langs = []
    if _cfg:
        target_langs = _cfg["translation"].get("target_languages", [])
        # Mark Hindi as unavailable if no provider configured
        provider = _cfg["translation"].get("provider", "none")
        if provider == "none":
            target_langs = [
                dict(lang, disabled=(lang["code"] != "en"))
                for lang in target_langs
            ]

    return jsonify({
        "class_names":      _class_names or [],
        "target_languages": target_langs,
        "threshold":        (_cfg or {}).get("inference", {})
                             .get("confidence_threshold", 0.7),
    })


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _timestamp() -> str:
    """Return current time as HH:MM:SS string."""
    return time.strftime("%H:%M:%S")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    import argparse
    p = argparse.ArgumentParser(description="SignHire Flask app")
    p.add_argument("--camera", type=int, default=1,
                   help="Camera index (default: 1)")
    p.add_argument("--model", choices=["lstm", "baseline"],
                   default="baseline",
                   help="Model type to use (default: baseline)")
    p.add_argument("--port", type=int, default=5000,
                   help="Port to listen on (default: 5000)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("SignHire — Web App")
    print(f"  Port    : {args.port}")
    init_app(camera_index=args.camera, model_preference=args.model)
    print(f"\n  Open http://localhost:{args.port} in your browser")
    print("  Press Ctrl+C to stop\n")

    # threaded=False: Flask's dev server + our own threads is sufficient;
    # threaded=True can cause issues with TF on some platforms.
    app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)
