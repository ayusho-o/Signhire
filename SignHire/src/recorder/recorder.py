"""
src/recorder/recorder.py
------------------------
Phase 2 — Guided clip recorder for ISL sign data collection.

Run from the SignHire/ directory:
    python src/recorder/recorder.py --signer signer_01

What it does
------------
For each sign in the vocabulary, the recorder:
  1. Shows an instruction screen with the sign name.
  2. Counts down (3-2-1) with on-screen text and audio-free visual cue.
  3. Records exactly `sequence_frames` frames while showing a red border.
  4. Shows the clip back and asks: Keep (K) or Re-record (R)?
  5. On Keep: saves the landmark sequence as a .npy file and writes a
     .json metadata file next to it.
  6. Repeats until `clips_per_sign` accepted clips exist for this sign.

Data is saved under:
    data/raw/<signer_id>/<sign>/<signer_id>_<sign>_<clip_index>.npy
    data/raw/<signer_id>/<sign>/<signer_id>_<sign>_<clip_index>.json

Each .npy file has shape (sequence_frames, feature_dim).
Each .json file contains:
    signer_id, label, source, clip_index, timestamp,
    recorded_fps, recorded_duration_s, frame_count_raw

At the end of each session a summary table is printed showing how many
clips were recorded per sign and the mean recorded FPS/duration.

Controls (keyboard)
-------------------
  K  — keep the last clip and move on
  R  — discard and re-record
  Q  — quit the session (progress is saved; resume later)

Usage
-----
    python src/recorder/recorder.py --signer signer_01
    python src/recorder/recorder.py --signer signer_01 --camera 1
    python src/recorder/recorder.py --signer signer_01 --sign hello
        (record only one specific sign)
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml

# ── Make SignHire root importable when running as a script ───────────────────
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.landmarks.extractor import LandmarkExtractor


# ── Colours (BGR) ─────────────────────────────────────────────────────────────
WHITE  = (255, 255, 255)
BLACK  = (0,   0,   0)
RED    = (0,   0,   220)
GREEN  = (0,   220, 0)
YELLOW = (0,   220, 220)
BLUE   = (220, 100, 0)


# ─────────────────────────────────────────────────────────────────────────────
# Helper drawing functions
# ─────────────────────────────────────────────────────────────────────────────

def put_centred(frame: np.ndarray, text: str, y: int,
                scale: float, colour: tuple, thickness: int = 2):
    """Draw text centred horizontally on the frame at vertical position y."""
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x = (frame.shape[1] - tw) // 2
    cv2.putText(frame, text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thickness, cv2.LINE_AA)


def draw_border(frame: np.ndarray, colour: tuple, thickness: int = 12):
    """Draw a solid border rectangle around the entire frame."""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), colour, thickness)


def dark_overlay(frame: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """Return a copy of frame with a dark semi-transparent overlay."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), BLACK, -1)
    return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)


# ─────────────────────────────────────────────────────────────────────────────
# Recorder class
# ─────────────────────────────────────────────────────────────────────────────

class Recorder:
    """
    Manages the guided recording session for one signer.

    Parameters
    ----------
    signer_id : str
        Identifier written into metadata and used as folder name.
    config_dir : Path
        Directory containing vocabulary.yaml and settings.yaml.
    camera_index : int
        OpenCV camera index (0 = default webcam).
    target_sign : str or None
        If given, record only this one sign (must be in the vocabulary).
    """

    def __init__(self, signer_id: str, config_dir: Path,
                 camera_index: int = 0, target_sign: str = None):

        self.signer_id = signer_id
        self.config_dir = config_dir
        self.camera_index = camera_index

        # ── Load config ───────────────────────────────────────────────────
        with open(config_dir / "vocabulary.yaml") as f:
            vocab_cfg = yaml.safe_load(f)
        with open(config_dir / "settings.yaml") as f:
            self.cfg = yaml.safe_load(f)

        self.signs: list[str] = vocab_cfg["signs"]
        if target_sign:
            if target_sign not in self.signs:
                raise ValueError(
                    f"Sign '{target_sign}' is not in the vocabulary. "
                    f"Available: {self.signs}"
                )
            self.signs = [target_sign]

        rec = self.cfg["recording"]
        self.target_fps:     int = rec["fps"]
        self.seq_frames:     int = rec["sequence_frames"]
        self.countdown_secs: int = rec["countdown_seconds"]
        self.clips_per_sign: int = rec["clips_per_sign"]

        # ── Data output directory ─────────────────────────────────────────
        raw_dir = _ROOT / self.cfg["paths"]["raw_data"]
        self.signer_dir = raw_dir / signer_id
        self.signer_dir.mkdir(parents=True, exist_ok=True)

        # ── MediaPipe extractor ───────────────────────────────────────────
        self.extractor = LandmarkExtractor(
            settings_path=str(config_dir / "settings.yaml"),
            draw_landmarks=True,
        )

        # ── Session statistics (for end-of-session summary) ───────────────
        # {sign: [{"fps": float, "duration": float, "frames": int}, ...]}
        self.session_stats: dict[str, list] = {s: [] for s in self.signs}

    # ── Public entry point ────────────────────────────────────────────────────

    def run(self):
        """Open the webcam and start the guided recording session."""
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            print("ERROR: Could not open webcam.")
            sys.exit(1)
        cap.set(cv2.CAP_PROP_FPS, self.target_fps)

        window = "SignHire — Recorder  (K=keep  R=re-record  Q=quit)"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, 960, 540)

        try:
            for sign in self.signs:
                if not self._record_sign(cap, window, sign):
                    break  # user pressed Q
        finally:
            cap.release()
            self.extractor.close()
            cv2.destroyAllWindows()
            self._print_summary()

    # ── Per-sign recording loop ───────────────────────────────────────────────

    def _record_sign(self, cap, window: str, sign: str) -> bool:
        """
        Record `clips_per_sign` accepted clips for one sign.
        Returns False if the user quit mid-session.
        """
        sign = str(sign)  # guard: YAML parses bare yes/no as booleans
        sign_dir = self.signer_dir / sign
        sign_dir.mkdir(parents=True, exist_ok=True)

        # Count already-accepted clips so we can resume a partial session.
        existing = len(list(sign_dir.glob("*.npy")))
        clip_idx = existing  # next clip index to write

        print(f"\n── Sign: {sign.upper()}  "
              f"(need {self.clips_per_sign}, have {existing}) ──")

        while clip_idx < self.clips_per_sign:
            remaining = self.clips_per_sign - clip_idx
            print(f"  Clip {clip_idx + 1}/{self.clips_per_sign}  "
                  f"({remaining} remaining)")

            # ── Instruction screen ────────────────────────────────────────
            if not self._show_instruction(cap, window, sign, clip_idx):
                return False  # Q pressed

            # ── Countdown ────────────────────────────────────────────────
            if not self._show_countdown(cap, window, sign):
                return False

            # ── Record ───────────────────────────────────────────────────
            sequence, meta = self._capture_clip(cap, window, sign)
            if sequence is None:
                return False  # Q pressed during recording

            # ── Playback + decision ───────────────────────────────────────
            decision = self._show_review(cap, window, sign, sequence)
            if decision == "quit":
                return False
            if decision == "rerecord":
                print("    → Re-recording.")
                continue

            # ── Save ──────────────────────────────────────────────────────
            self._save_clip(sign_dir, sign, clip_idx, sequence, meta)
            self.session_stats[sign].append(meta)
            clip_idx += 1
            print(f"    → Saved clip {clip_idx}/{self.clips_per_sign}  "
                  f"({meta['recorded_fps']:.1f} fps, "
                  f"{meta['recorded_duration_s']:.2f} s, "
                  f"{meta['frame_count_raw']} raw frames)")

        print(f"  ✓ All {self.clips_per_sign} clips recorded for '{sign}'.")
        return True

    # ── Screen: instruction ───────────────────────────────────────────────────

    def _show_instruction(self, cap, window: str, sign: str,
                          clip_idx: int) -> bool:
        """
        Show a still instruction screen until the user presses SPACE or K.
        Returns False if Q is pressed.
        """
        label = sign.upper().replace("_", " ")
        progress = f"Clip {clip_idx + 1} of {self.clips_per_sign}"

        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            frame = dark_overlay(frame, alpha=0.5)

            put_centred(frame, "Get ready to sign:", frame.shape[0] // 2 - 90,
                        1.0, WHITE, 2)
            put_centred(frame, label, frame.shape[0] // 2,
                        2.2, YELLOW, 3)
            put_centred(frame, progress, frame.shape[0] // 2 + 80,
                        0.75, WHITE, 1)
            put_centred(frame, "Press SPACE to start  |  Q to quit",
                        frame.shape[0] - 30, 0.55, WHITE, 1)

            cv2.imshow(window, frame)
            key = cv2.waitKey(30) & 0xFF
            if key in (ord(" "), ord("k"), ord("K")):
                return True
            if key in (ord("q"), ord("Q"), 27):
                return False

    # ── Screen: countdown ────────────────────────────────────────────────────

    def _show_countdown(self, cap, window: str, sign: str) -> bool:
        """
        Show a live countdown (3-2-1-GO!).
        Returns False if Q is pressed.
        """
        label = sign.upper().replace("_", " ")

        for count in range(self.countdown_secs, 0, -1):
            deadline = time.perf_counter() + 1.0
            while time.perf_counter() < deadline:
                ret, frame = cap.read()
                if not ret:
                    continue
                frame = cv2.flip(frame, 1)
                # Dim but keep the live feed visible
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0),
                              (frame.shape[1], frame.shape[0]), BLACK, -1)
                cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

                put_centred(frame, label,
                            frame.shape[0] // 2 - 80, 1.2, YELLOW, 2)
                put_centred(frame, str(count),
                            frame.shape[0] // 2 + 40, 4.0, WHITE, 5)

                cv2.imshow(window, frame)
                key = cv2.waitKey(30) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    return False

        # "GO!" flash for half a second
        deadline = time.perf_counter() + 0.5
        while time.perf_counter() < deadline:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            put_centred(frame, "GO!", frame.shape[0] // 2, 4.0, GREEN, 6)
            draw_border(frame, GREEN, 14)
            cv2.imshow(window, frame)
            cv2.waitKey(30)

        return True

    # ── Screen: record ────────────────────────────────────────────────────────

    def _capture_clip(self, cap, window: str, sign: str):
        """
        Capture exactly `seq_frames` processed frames.

        Returns
        -------
        sequence : np.ndarray, shape (seq_frames, feature_dim)  or  None
        meta     : dict with timing info  or  None
        """
        label = sign.upper().replace("_", " ")
        sequence_raw = []      # landmark vectors as they arrive
        timestamps   = []      # wall-clock time of each captured frame

        t_start = time.perf_counter()

        while len(sequence_raw) < self.seq_frames:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)

            vec, annotated = self.extractor.process_frame(frame)
            sequence_raw.append(vec)
            timestamps.append(time.perf_counter())

            # Progress bar
            progress = len(sequence_raw) / self.seq_frames
            bar_w = int(frame.shape[1] * 0.6)
            bar_x = (frame.shape[1] - bar_w) // 2
            bar_y = frame.shape[0] - 50
            cv2.rectangle(annotated, (bar_x, bar_y),
                          (bar_x + bar_w, bar_y + 18), (80, 80, 80), -1)
            cv2.rectangle(annotated, (bar_x, bar_y),
                          (bar_x + int(bar_w * progress), bar_y + 18),
                          RED, -1)
            put_centred(annotated, f"RECORDING  {label}",
                        50, 1.0, RED, 2)
            put_centred(annotated,
                        f"Frame {len(sequence_raw)}/{self.seq_frames}",
                        frame.shape[0] - 65, 0.6, WHITE, 1)
            draw_border(annotated, RED, 12)

            cv2.imshow(window, annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                return None, None

        t_end = timestamps[-1]
        duration = t_end - t_start
        recorded_fps = len(sequence_raw) / duration if duration > 0 else 0.0

        meta = {
            "recorded_fps":      round(recorded_fps, 2),
            "recorded_duration_s": round(duration, 3),
            "frame_count_raw":   len(sequence_raw),
        }

        sequence = np.array(sequence_raw, dtype=np.float32)
        return sequence, meta

    # ── Screen: review ────────────────────────────────────────────────────────

    def _show_review(self, cap, window: str, sign: str,
                     sequence: np.ndarray) -> str:
        """
        Play back the recorded landmark overlay (from the live feed, not
        stored video — we just show a "review" prompt) and let the user decide.

        Returns "keep", "rerecord", or "quit".
        """
        label = sign.upper().replace("_", " ")

        # We don't store video, so show a static review screen instead.
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            frame = dark_overlay(frame, 0.5)

            put_centred(frame, f"Clip recorded: {label}",
                        frame.shape[0] // 2 - 80, 1.0, YELLOW, 2)
            put_centred(frame, f"{sequence.shape[0]} frames captured",
                        frame.shape[0] // 2 - 30, 0.7, WHITE, 1)
            put_centred(frame, "K = Keep    R = Re-record    Q = Quit",
                        frame.shape[0] // 2 + 40, 0.85, WHITE, 2)

            cv2.imshow(window, frame)
            key = cv2.waitKey(30) & 0xFF

            if key in (ord("k"), ord("K")):
                return "keep"
            if key in (ord("r"), ord("R")):
                return "rerecord"
            if key in (ord("q"), ord("Q"), 27):
                return "quit"

    # ── Save clip ─────────────────────────────────────────────────────────────

    def _save_clip(self, sign_dir: Path, sign: str, clip_idx: int,
                   sequence: np.ndarray, timing_meta: dict):
        """
        Save the landmark sequence and its metadata.

        Files written:
            <signer_id>_<sign>_<clip_idx:03d>.npy   — shape (frames, features)
            <signer_id>_<sign>_<clip_idx:03d>.json  — metadata
        """
        stem = f"{self.signer_id}_{sign}_{clip_idx:03d}"
        npy_path  = sign_dir / f"{stem}.npy"
        json_path = sign_dir / f"{stem}.json"

        np.save(str(npy_path), sequence)

        metadata = {
            "signer_id":           self.signer_id,
            "label":               sign,
            "source":              "webcam",
            "clip_index":          clip_idx,
            "timestamp":           datetime.now().isoformat(),
            "feature_dim":         int(sequence.shape[1]),
            "frame_count_saved":   int(sequence.shape[0]),
            "recorded_fps":        timing_meta["recorded_fps"],
            "recorded_duration_s": timing_meta["recorded_duration_s"],
            "frame_count_raw":     timing_meta["frame_count_raw"],
            "configured_seq_frames": self.seq_frames,
            "npy_file":            npy_path.name,
        }
        with open(json_path, "w") as f:
            json.dump(metadata, f, indent=2)

    # ── Session summary ───────────────────────────────────────────────────────

    def _print_summary(self):
        """Print a per-sign table of clip counts and timing stats."""
        print("\n" + "═" * 60)
        print("SESSION SUMMARY")
        print("═" * 60)
        print(f"  Signer : {self.signer_id}")
        print(f"  Target : {self.clips_per_sign} clips per sign")
        print()
        print(f"  {'Sign':<14} {'Clips':>6}  {'Mean FPS':>9}  {'Mean dur (s)':>13}")
        print("  " + "-" * 46)

        for sign in self.signs:
            sign = str(sign)  # guard: YAML parses bare yes/no as booleans
            stats = self.session_stats.get(sign, [])
            # Count total on disk (includes clips from previous sessions)
            sign_dir = self.signer_dir / sign
            on_disk = len(list(sign_dir.glob("*.npy")))

            if stats:
                mean_fps = sum(s["recorded_fps"] for s in stats) / len(stats)
                mean_dur = sum(s["recorded_duration_s"] for s in stats) / len(stats)
                fps_str  = f"{mean_fps:9.1f}"
                dur_str  = f"{mean_dur:13.2f}"
            else:
                fps_str  = "        —"
                dur_str  = "           —"

            # Flag if real duration differs a lot from configured window
            configured_dur = self.seq_frames / self.target_fps
            warn = ""
            if stats:
                mean_dur_val = sum(s["recorded_duration_s"] for s in stats) / len(stats)
                if abs(mean_dur_val - configured_dur) > 0.3:
                    warn = "  ← check fps/window size"

            print(f"  {sign:<14} {on_disk:>6}  {fps_str}  {dur_str}{warn}")

        print()
        configured_dur = self.seq_frames / self.target_fps
        print(f"  Configured window : {self.seq_frames} frames "
              f"@ {self.target_fps} fps = {configured_dur:.2f} s")
        print("  If 'Mean dur' differs significantly from the configured")
        print("  window, consider adjusting sequence_frames in settings.yaml.")
        print("═" * 60 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="SignHire guided ISL data recorder"
    )
    p.add_argument(
        "--signer", required=True,
        help="Unique signer identifier, e.g. signer_01"
    )
    p.add_argument(
        "--camera", type=int, default=0,
        help="OpenCV camera index (default: 0)"
    )
    p.add_argument(
        "--sign", default=None,
        help="Record only this one sign (must be in vocabulary)"
    )
    return p.parse_args()


def main():
    args = parse_args()
    config_dir = _ROOT / "config"

    print("SignHire — Guided Recorder")
    print(f"  Signer   : {args.signer}")
    print(f"  Camera   : {args.camera}")
    print(f"  Config   : {config_dir}")
    if args.sign:
        print(f"  Sign     : {args.sign} (single-sign mode)")
    print()

    recorder = Recorder(
        signer_id=args.signer,
        config_dir=config_dir,
        camera_index=args.camera,
        target_sign=args.sign,
    )
    recorder.run()


if __name__ == "__main__":
    main()
