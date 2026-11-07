"""
src/preprocess/preprocess.py
-----------------------------
Phase 2 — sequence resampling, keypoint augmentation, and dataset builder.

What this module does
---------------------
1. Loads all .npy landmark sequences from data/raw/.
2. Resamples every sequence to exactly `sequence_frames` frames (the value
   from settings.yaml) using linear interpolation.  This handles clips that
   were captured at a slightly different rate than the target.
3. Builds train/test splits for leave-one-signer-out (LOSO) cross-validation.
4. Applies keypoint augmentation to training sequences only:
     - Scale: uniform stretch/shrink around the origin
     - Shift: translate all points
     - Rotation: 2-D rotation in the XY plane
     - Noise: add Gaussian noise to every coordinate
     - Time-warp: stretch/compress the time axis then resample back
5. Saves the processed dataset (X.npy, y.npy, signers.npy) to data/processed/.

Usage
-----
    python src/preprocess/preprocess.py

    # Or import and call directly from a training script:
    from src.preprocess.preprocess import build_dataset, loso_splits
"""

import json
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.interpolate import interp1d

# ── Make SignHire root importable ────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# Config loading
# ─────────────────────────────────────────────────────────────────────────────

def load_config(config_dir: Path = None):
    """Load vocabulary and settings. Returns (vocab_cfg, settings_cfg)."""
    if config_dir is None:
        config_dir = _ROOT / "config"
    with open(config_dir / "vocabulary.yaml") as f:
        vocab = yaml.safe_load(f)
    with open(config_dir / "settings.yaml") as f:
        settings = yaml.safe_load(f)
    return vocab, settings


# ─────────────────────────────────────────────────────────────────────────────
# Resampling
# ─────────────────────────────────────────────────────────────────────────────

def resample_sequence(seq: np.ndarray, target_len: int) -> np.ndarray:
    """
    Resample a landmark sequence to exactly `target_len` frames using
    linear interpolation along the time axis.

    Parameters
    ----------
    seq : np.ndarray, shape (T, F)
        Raw sequence with T frames and F features.
    target_len : int
        Desired number of output frames.

    Returns
    -------
    np.ndarray, shape (target_len, F)
    """
    T, F = seq.shape
    if T == target_len:
        return seq.astype(np.float32)

    # Original time axis (0 to 1) and target axis
    t_orig   = np.linspace(0.0, 1.0, T)
    t_target = np.linspace(0.0, 1.0, target_len)

    # Interpolate each feature independently
    # interp1d with kind='linear' is fast and good enough for smooth motion
    interpolator = interp1d(t_orig, seq, axis=0, kind="linear",
                            fill_value="extrapolate")
    return interpolator(t_target).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Augmentation functions
# ─────────────────────────────────────────────────────────────────────────────
# Every function takes a sequence of shape (T, F) and returns (T, F).
# F is assumed to be a flat concatenation of (x, y, z) triplets, so we
# reshape to (T, F//3, 3) to apply spatial transforms, then flatten back.

def _as_points(seq: np.ndarray):
    """Reshape (T, F) → (T, F//3, 3). Assumes F is divisible by 3."""
    T, F = seq.shape
    assert F % 3 == 0, f"Feature dim {F} is not divisible by 3"
    return seq.reshape(T, F // 3, 3)


def _as_flat(pts: np.ndarray) -> np.ndarray:
    """Reshape (T, N, 3) → (T, F)."""
    T, N, _ = pts.shape
    return pts.reshape(T, N * 3)


def aug_scale(seq: np.ndarray, scale_range: float,
              rng: np.random.Generator) -> np.ndarray:
    """
    Uniformly scale all landmark coordinates by a random factor.

    A scale > 1 makes the signing look larger; < 1 makes it smaller.
    Applied identically to all frames so the motion trajectory is preserved.
    """
    factor = 1.0 + rng.uniform(-scale_range, scale_range)
    return (seq * factor).astype(np.float32)


def aug_shift(seq: np.ndarray, shift_range: float,
              rng: np.random.Generator) -> np.ndarray:
    """
    Shift all landmark coordinates by a random constant offset.

    The same offset is applied to every frame (rigid body translation).
    shift_range is in the same units as the normalised landmarks
    (fractions of shoulder width).
    """
    # Shift is applied only to x and y (every 3rd value starting at 0 and 1)
    pts = _as_points(seq).copy()
    dx = rng.uniform(-shift_range, shift_range)
    dy = rng.uniform(-shift_range, shift_range)
    pts[:, :, 0] += dx
    pts[:, :, 1] += dy
    return _as_flat(pts).astype(np.float32)


def aug_rotate(seq: np.ndarray, rotation_deg: float,
               rng: np.random.Generator) -> np.ndarray:
    """
    Rotate all landmarks around the origin in the XY plane by a random angle.

    The z coordinate (depth) is left unchanged.
    Rotation is constant across all frames.
    """
    angle = rng.uniform(-rotation_deg, rotation_deg)
    theta = np.radians(angle)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    pts = _as_points(seq).copy()
    x = pts[:, :, 0].copy()
    y = pts[:, :, 1].copy()
    pts[:, :, 0] = cos_t * x - sin_t * y
    pts[:, :, 1] = sin_t * x + cos_t * y
    return _as_flat(pts).astype(np.float32)


def aug_noise(seq: np.ndarray, noise_sigma: float,
              rng: np.random.Generator) -> np.ndarray:
    """
    Add independent Gaussian noise to every coordinate in every frame.

    This simulates small landmark detection jitter.
    """
    noise = rng.normal(0.0, noise_sigma, size=seq.shape).astype(np.float32)
    return (seq + noise).astype(np.float32)


def aug_time_warp(seq: np.ndarray, warp_range: float,
                  rng: np.random.Generator) -> np.ndarray:
    """
    Warp the time axis by a smooth random curve, then resample back to
    the original length.

    This makes the model less sensitive to signing speed variations.

    Implementation: generate a random warp amount w ∈ [-warp_range, +warp_range].
    A positive w stretches the middle of the clip (slow down the peak of the
    motion) and compresses the ends.  We use a simple cubic warp via a
    quadratic displacement on the normalised time axis.
    """
    T = seq.shape[0]
    t_orig = np.linspace(0.0, 1.0, T)

    # Random warp: displace the midpoint by w, keep endpoints fixed.
    w = rng.uniform(-warp_range, warp_range)
    # Quadratic displacement: peak at t=0.5, zero at t=0 and t=1.
    warp_curve = w * 4.0 * t_orig * (1.0 - t_orig)
    t_warped = np.clip(t_orig + warp_curve, 0.0, 1.0)

    # t_warped may not be strictly increasing after clipping — sort to fix.
    t_warped = np.sort(t_warped)

    # Resample: interpolate original frames at the warped time points,
    # then resample back to the original uniform grid.
    interp = interp1d(t_warped, seq, axis=0, kind="linear",
                      fill_value="extrapolate")
    warped = interp(t_orig).astype(np.float32)
    return warped


def augment_sequence(seq: np.ndarray, aug_cfg: dict,
                     rng: np.random.Generator) -> np.ndarray:
    """
    Apply all augmentations in sequence to one training clip.

    Parameters
    ----------
    seq : np.ndarray, shape (T, F)
    aug_cfg : dict
        The 'augmentation' section from settings.yaml.
    rng : np.random.Generator
        Seeded random generator for reproducibility.

    Returns
    -------
    np.ndarray, shape (T, F) — augmented sequence
    """
    seq = aug_scale(seq,      aug_cfg["scale_range"],    rng)
    seq = aug_shift(seq,      aug_cfg["shift_range"],    rng)
    seq = aug_rotate(seq,     aug_cfg["rotation_deg"],   rng)
    seq = aug_noise(seq,      aug_cfg["noise_sigma"],    rng)
    seq = aug_time_warp(seq,  aug_cfg["time_warp_range"], rng)
    return seq


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loading
# ─────────────────────────────────────────────────────────────────────────────

def load_raw_dataset(raw_dir: Path, signs: list[str], target_frames: int):
    """
    Walk data/raw/<signer_id>/<sign>/*.npy and load every clip.

    Each clip is resampled to `target_frames` frames.

    Returns
    -------
    sequences : list of np.ndarray, each shape (target_frames, F)
    labels    : list of str
    signers   : list of str
    """
    sequences = []
    labels    = []
    signers   = []

    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")

    # Walk signer directories
    signer_dirs = sorted([d for d in raw_dir.iterdir() if d.is_dir()])
    if not signer_dirs:
        raise ValueError(f"No signer directories found under {raw_dir}")

    for signer_dir in signer_dirs:
        signer_id = signer_dir.name
        for sign in signs:
            sign_dir = signer_dir / sign
            if not sign_dir.exists():
                continue
            npy_files = sorted(sign_dir.glob("*.npy"))
            for npy_path in npy_files:
                seq = np.load(str(npy_path)).astype(np.float32)
                if seq.ndim != 2:
                    print(f"  WARNING: skipping {npy_path.name} "
                          f"(unexpected shape {seq.shape})")
                    continue
                seq = resample_sequence(seq, target_frames)
                sequences.append(seq)
                labels.append(sign)
                signers.append(signer_id)

    print(f"  Loaded {len(sequences)} sequences from {len(signer_dirs)} signer(s).")
    return sequences, labels, signers


# ─────────────────────────────────────────────────────────────────────────────
# LOSO split builder
# ─────────────────────────────────────────────────────────────────────────────

def loso_splits(sequences: list, labels: list, signers: list,
                aug_cfg: dict, copies_per_clip: int,
                rng_seed: int = 42):
    """
    Generate leave-one-signer-out (LOSO) train/test splits.

    For each unique signer S:
      - Test  : all clips from S (no augmentation)
      - Train : all clips from every other signer, plus `copies_per_clip`
                augmented copies of each training clip

    Parameters
    ----------
    sequences : list of np.ndarray
    labels    : list of str
    signers   : list of str
    aug_cfg   : dict  (augmentation section of settings.yaml)
    copies_per_clip : int
    rng_seed  : int

    Yields
    ------
    fold_id : str   (the held-out signer ID)
    X_train : np.ndarray, shape (N_train, T, F)
    y_train : np.ndarray, shape (N_train,)   — integer class indices
    X_test  : np.ndarray, shape (N_test,  T, F)
    y_test  : np.ndarray, shape (N_test,)
    class_names : list of str  — maps integer index → sign label
    """
    unique_signers = sorted(set(signers))
    unique_labels  = sorted(set(labels))
    label_to_idx   = {lbl: i for i, lbl in enumerate(unique_labels)}

    sequences_arr = np.array(sequences, dtype=np.float32)  # (N, T, F)
    labels_arr    = np.array(labels)
    signers_arr   = np.array(signers)

    rng = np.random.default_rng(rng_seed)

    for held_out in unique_signers:
        test_mask  = signers_arr == held_out
        train_mask = ~test_mask

        X_test_raw = sequences_arr[test_mask]
        y_test_raw = labels_arr[test_mask]

        X_train_raw = sequences_arr[train_mask]
        y_train_raw = labels_arr[train_mask]

        # Augment training data
        X_train_aug = [X_train_raw]
        y_train_aug = [y_train_raw]

        for _ in range(copies_per_clip):
            aug_batch = np.stack([
                augment_sequence(seq, aug_cfg, rng)
                for seq in X_train_raw
            ], axis=0)
            X_train_aug.append(aug_batch)
            y_train_aug.append(y_train_raw)

        X_train = np.concatenate(X_train_aug, axis=0)
        y_train_str = np.concatenate(y_train_aug, axis=0)

        # Shuffle training data
        perm = rng.permutation(len(X_train))
        X_train = X_train[perm]
        y_train_str = y_train_str[perm]

        # Convert labels to integer indices
        y_train = np.array([label_to_idx[l] for l in y_train_str], dtype=np.int32)
        y_test  = np.array([label_to_idx[l] for l in y_test_raw],  dtype=np.int32)

        yield held_out, X_train, y_train, X_test_raw, y_test, unique_labels


# ─────────────────────────────────────────────────────────────────────────────
# Build and save processed dataset
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset(config_dir: Path = None, raw_dir: Path = None,
                  processed_dir: Path = None):
    """
    Load raw clips, resample, and save the full (unaugmented) dataset to
    data/processed/ as:
        X.npy       — shape (N, target_frames, F)
        y.npy       — shape (N,) integer class indices
        signers.npy — shape (N,) signer IDs as object array
        meta.json   — class_names, feature_dim, target_frames, signer list

    Augmentation is NOT applied here — it is applied on-the-fly per LOSO
    fold during training so test data is never contaminated.

    Returns
    -------
    X, y, signers, class_names
    """
    vocab, settings = load_config(config_dir)

    if raw_dir is None:
        raw_dir = _ROOT / settings["paths"]["raw_data"]
    if processed_dir is None:
        processed_dir = _ROOT / settings["paths"]["processed_data"]

    target_frames = settings["recording"]["sequence_frames"]
    signs = vocab["signs"]

    print(f"\nBuilding dataset")
    print(f"  Raw data     : {raw_dir}")
    print(f"  Target frames: {target_frames}")
    print(f"  Signs        : {signs}")

    sequences, labels, signers = load_raw_dataset(raw_dir, signs, target_frames)

    if not sequences:
        print("  ERROR: No sequences found. Record some data first.")
        sys.exit(1)

    unique_labels = sorted(set(labels))
    label_to_idx  = {lbl: i for i, lbl in enumerate(unique_labels)}

    X       = np.stack(sequences, axis=0).astype(np.float32)
    y       = np.array([label_to_idx[l] for l in labels], dtype=np.int32)
    signers_arr = np.array(signers, dtype=object)

    # Print per-class counts
    print("\n  Samples per sign:")
    for sign in unique_labels:
        count = int(np.sum(np.array(labels) == sign))
        print(f"    {sign:<16} {count:>4} clips")

    print("\n  Samples per signer:")
    for s in sorted(set(signers)):
        count = int(np.sum(signers_arr == s))
        print(f"    {s:<16} {count:>4} clips")

    # Save
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    np.save(str(processed_dir / "X.npy"), X)
    np.save(str(processed_dir / "y.npy"), y)
    np.save(str(processed_dir / "signers.npy"), signers_arr)

    meta = {
        "class_names":    unique_labels,
        "feature_dim":    int(X.shape[2]),
        "target_frames":  int(X.shape[1]),
        "num_samples":    int(X.shape[0]),
        "signers":        sorted(set(signers)),
    }
    with open(processed_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  Saved to {processed_dir}")
    print(f"  X shape  : {X.shape}  (samples, frames, features)")
    print(f"  Classes  : {unique_labels}")

    return X, y, signers_arr, unique_labels


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    build_dataset()
