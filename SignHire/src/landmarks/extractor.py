"""
src/landmarks/extractor.py
--------------------------
Extracts and normalises MediaPipe landmarks from a single BGR video frame.

What this module does
---------------------
1. Runs MediaPipe Hands, Pose, and FaceMesh on a frame.
2. Selects only the landmark subsets defined in config/settings.yaml.
3. Normalises all coordinates so that the shoulder midpoint is the origin
   and distances are scaled by the shoulder width.  This makes the feature
   vector body-size invariant and camera-distance invariant.
4. Returns a flat 1-D numpy array (one feature vector per frame) ready to
   be stacked into a sequence.

Coordinate convention
---------------------
MediaPipe returns (x, y, z) in normalised image space (0-1).  After our
normalisation step the values are in units of "shoulder widths" and centred
on the shoulder midpoint, so they are typically in the range [-2, 2].

Missing landmarks (hand not visible) are filled with zeros so every frame
always has the same vector length.

Usage
-----
    extractor = LandmarkExtractor(settings_path="config/settings.yaml")
    frame_vector, annotated_frame = extractor.process_frame(bgr_frame)
    # frame_vector: np.ndarray, shape (FEATURE_DIM,)
    # annotated_frame: BGR frame with skeleton drawn on it
"""

import numpy as np
import cv2
import mediapipe as mp
import yaml
from pathlib import Path


# ── MediaPipe solution handles (created once, reused) ────────────────────────
_mp_hands = mp.solutions.hands
_mp_pose = mp.solutions.pose
_mp_face = mp.solutions.face_mesh
_mp_draw = mp.solutions.drawing_utils
_mp_draw_styles = mp.solutions.drawing_styles


class LandmarkExtractor:
    """
    Wraps MediaPipe Hands + Pose + FaceMesh and exposes a single
    process_frame() method.

    Parameters
    ----------
    settings_path : str or Path
        Path to config/settings.yaml.
    draw_landmarks : bool
        Whether to draw the skeleton overlay onto the returned frame.
        Set False during batch preprocessing to save time.
    """

    def __init__(self, settings_path: str = "config/settings.yaml",
                 draw_landmarks: bool = True):

        settings_path = Path(settings_path)
        with open(settings_path, "r") as f:
            cfg = yaml.safe_load(f)

        lm_cfg = cfg["landmarks"]
        self.pose_indices = lm_cfg["pose_indices"]          # list[int]
        self.face_indices = lm_cfg["face_indices"]          # list[int]
        self.hand_lm_count = lm_cfg["hand_landmarks"]       # 21

        self.draw_landmarks = draw_landmarks

        # Precompute expected feature dimension so callers can query it.
        # Each landmark contributes 3 values (x, y, z).
        n_pose = len(self.pose_indices) * 3         # e.g. 25 × 3 = 75
        n_face = len(self.face_indices) * 3         # e.g. 10 × 3 = 30
        n_hands = self.hand_lm_count * 3 * 2       # 21 × 3 × 2 hands = 126
        self.feature_dim = n_pose + n_face + n_hands  # total ~231

        # Initialise MediaPipe models.
        # min_detection_confidence / min_tracking_confidence balance speed vs
        # robustness; 0.5 is a reasonable default for a live viewer.
        self._hands = _mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._pose = _mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,          # 0 = fastest, good enough for CPU
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._face = _mp_face.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,      # keep False for speed
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def process_frame(self, bgr_frame: np.ndarray):
        """
        Run MediaPipe on one BGR frame.

        Returns
        -------
        feature_vector : np.ndarray, shape (self.feature_dim,)
            Normalised landmark coordinates.  All zeros if pose shoulders
            are not detected (can't normalise without a reference point).
        annotated_frame : np.ndarray
            Copy of bgr_frame with skeleton drawn (or the original frame if
            draw_landmarks=False).
        """
        # MediaPipe needs RGB
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False  # small speed gain

        hand_result = self._hands.process(rgb)
        pose_result = self._pose.process(rgb)
        face_result = self._face.process(rgb)

        rgb.flags.writeable = True

        # Draw overlays on a copy so we don't mutate the caller's frame
        if self.draw_landmarks:
            annotated = bgr_frame.copy()
            self._draw_overlays(annotated, hand_result, pose_result, face_result)
        else:
            annotated = bgr_frame

        # Build feature vector
        feature_vector = self._build_feature_vector(
            hand_result, pose_result, face_result, bgr_frame.shape
        )

        return feature_vector, annotated

    def close(self):
        """Release MediaPipe resources."""
        self._hands.close()
        self._pose.close()
        self._face.close()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_feature_vector(self, hand_result, pose_result, face_result,
                               frame_shape) -> np.ndarray:
        """
        Assemble and normalise the feature vector for one frame.

        Layout (all flat, in this order):
            [pose_selected | face_selected | left_hand | right_hand]

        If shoulders are not detected we cannot normalise, so we return zeros.
        """
        # ── Step 1: get normalisation reference from pose ─────────────────
        origin, scale = self._get_normalisation_reference(pose_result)
        if origin is None:
            # Shoulders not detected — return zero vector
            return np.zeros(self.feature_dim, dtype=np.float32)

        # ── Step 2: extract each landmark group ───────────────────────────
        pose_vec = self._extract_pose(pose_result, origin, scale)
        face_vec = self._extract_face(face_result, origin, scale)
        left_vec, right_vec = self._extract_hands(hand_result, origin, scale)

        return np.concatenate([pose_vec, face_vec, left_vec, right_vec])

    def _get_normalisation_reference(self, pose_result):
        """
        Compute shoulder midpoint (origin) and shoulder width (scale).

        MediaPipe Pose landmark indices:
            11 = left shoulder,  12 = right shoulder

        Returns (origin, scale) or (None, None) if shoulders not found.
        origin : np.ndarray shape (3,) — [x, y, z] midpoint
        scale  : float — Euclidean distance between shoulders (in image space)
        """
        if not pose_result.pose_landmarks:
            return None, None

        lms = pose_result.pose_landmarks.landmark
        # Visibility check — mediapipe sets visibility < 0.5 when uncertain
        if lms[11].visibility < 0.5 or lms[12].visibility < 0.5:
            return None, None

        left = np.array([lms[11].x, lms[11].y, lms[11].z], dtype=np.float32)
        right = np.array([lms[12].x, lms[12].y, lms[12].z], dtype=np.float32)

        origin = (left + right) / 2.0
        scale = float(np.linalg.norm(left - right))

        # Guard against degenerate scale (shoulders too close / overlapping)
        if scale < 1e-6:
            return None, None

        return origin, scale

    def _normalise(self, coords: np.ndarray, origin: np.ndarray,
                   scale: float) -> np.ndarray:
        """Translate to origin, divide by scale. Works on shape (..., 3)."""
        return (coords - origin) / scale

    def _extract_pose(self, pose_result, origin, scale) -> np.ndarray:
        """
        Extract selected pose landmarks and normalise.
        Returns flat array of shape (len(pose_indices) * 3,).
        """
        n = len(self.pose_indices)
        out = np.zeros(n * 3, dtype=np.float32)

        if not pose_result.pose_landmarks:
            return out

        lms = pose_result.pose_landmarks.landmark
        for i, idx in enumerate(self.pose_indices):
            lm = lms[idx]
            coords = np.array([lm.x, lm.y, lm.z], dtype=np.float32)
            normed = self._normalise(coords, origin, scale)
            out[i * 3: i * 3 + 3] = normed

        return out

    def _extract_face(self, face_result, origin, scale) -> np.ndarray:
        """
        Extract selected face mesh landmarks and normalise.
        Returns flat array of shape (len(face_indices) * 3,).
        """
        n = len(self.face_indices)
        out = np.zeros(n * 3, dtype=np.float32)

        if not face_result.multi_face_landmarks:
            return out

        lms = face_result.multi_face_landmarks[0].landmark
        for i, idx in enumerate(self.face_indices):
            lm = lms[idx]
            coords = np.array([lm.x, lm.y, lm.z], dtype=np.float32)
            normed = self._normalise(coords, origin, scale)
            out[i * 3: i * 3 + 3] = normed

        return out

    def _extract_hands(self, hand_result, origin, scale):
        """
        Extract left and right hand landmarks.

        MediaPipe labels each detected hand as 'Left' or 'Right' from the
        perspective of the person (i.e. mirror-flipped vs. the camera view).
        We place left-hand features before right-hand features consistently
        so the model always sees the same layout.

        Returns two flat arrays, each of shape (hand_lm_count * 3,).
        Missing hand → zero array.
        """
        dim = self.hand_lm_count * 3
        left_vec = np.zeros(dim, dtype=np.float32)
        right_vec = np.zeros(dim, dtype=np.float32)

        if not hand_result.multi_hand_landmarks:
            return left_vec, right_vec

        for hand_landmarks, handedness in zip(
            hand_result.multi_hand_landmarks,
            hand_result.multi_handedness,
        ):
            label = handedness.classification[0].label  # "Left" or "Right"
            coords = np.array(
                [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
                dtype=np.float32,
            )  # shape (21, 3)
            normed = self._normalise(coords, origin, scale).flatten()

            if label == "Left":
                left_vec = normed
            else:
                right_vec = normed

        return left_vec, right_vec

    # ── Drawing helpers ───────────────────────────────────────────────────────

    def _draw_overlays(self, frame, hand_result, pose_result, face_result):
        """Draw MediaPipe skeleton on frame in-place."""
        h, w = frame.shape[:2]

        # Pose skeleton (upper body connections look better with UPPER_BODY_ONLY
        # style but that was removed in newer MediaPipe; we use default here).
        if pose_result.pose_landmarks:
            _mp_draw.draw_landmarks(
                frame,
                pose_result.pose_landmarks,
                _mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=_mp_draw_styles.get_default_pose_landmarks_style(),
            )

        # Both hands
        if hand_result.multi_hand_landmarks:
            for hand_lms in hand_result.multi_hand_landmarks:
                _mp_draw.draw_landmarks(
                    frame,
                    hand_lms,
                    _mp_hands.HAND_CONNECTIONS,
                    _mp_draw_styles.get_default_hand_landmarks_style(),
                    _mp_draw_styles.get_default_hand_connections_style(),
                )

        # Face mesh — draw only the selected landmark indices as dots
        # (drawing the full mesh is visually noisy and slow)
        if face_result.multi_face_landmarks:
            face_lms = face_result.multi_face_landmarks[0].landmark
            for idx in self.face_indices:
                lm = face_lms[idx]
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 3, (0, 255, 255), -1)  # cyan dots
