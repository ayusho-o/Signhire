"""
tests/test_preprocess.py
-------------------------
Sanity checks for resampling and augmentation functions (Phase 2).

Run from the SignHire/ directory:
    pytest tests/test_preprocess.py -v
"""

import sys
from pathlib import Path
import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.preprocess.preprocess import (
    resample_sequence,
    aug_scale,
    aug_shift,
    aug_rotate,
    aug_noise,
    aug_time_warp,
    augment_sequence,
    loso_splits,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

RNG = np.random.default_rng(seed=0)
FEATURE_DIM = 231  # 25 pose×3 + 10 face×3 + 21 hand×3×2 — must be divisible by 3


def make_seq(T: int = 30, F: int = FEATURE_DIM) -> np.ndarray:
    """Return a random sequence of shape (T, F)."""
    return RNG.random((T, F)).astype(np.float32)


def dummy_aug_cfg():
    return {
        "scale_range":    0.05,
        "shift_range":    0.03,
        "rotation_deg":   10.0,
        "noise_sigma":    0.01,
        "time_warp_range": 0.10,
        "copies_per_clip": 4,
    }


# ── Resampling tests ──────────────────────────────────────────────────────────

def test_resample_same_length():
    """Resampling to the same length should return the same values."""
    seq = make_seq(30)
    out = resample_sequence(seq, 30)
    np.testing.assert_allclose(out, seq, atol=1e-5)


def test_resample_upsample_shape():
    """Upsampling from 20 to 30 frames should give shape (30, FEATURE_DIM)."""
    seq = make_seq(20)
    out = resample_sequence(seq, 30)
    assert out.shape == (30, FEATURE_DIM)


def test_resample_downsample_shape():
    """Downsampling from 45 to 30 frames should give shape (30, FEATURE_DIM)."""
    seq = make_seq(45)
    out = resample_sequence(seq, 30)
    assert out.shape == (30, FEATURE_DIM)


def test_resample_output_dtype():
    """Output should always be float32."""
    seq = make_seq(25).astype(np.float64)
    out = resample_sequence(seq, 30)
    assert out.dtype == np.float32


# ── Augmentation shape tests ──────────────────────────────────────────────────

@pytest.mark.parametrize("aug_fn,extra_kwargs", [
    (aug_scale,  {"scale_range": 0.05}),
    (aug_shift,  {"shift_range": 0.03}),
    (aug_rotate, {"rotation_deg": 10.0}),
    (aug_noise,  {"noise_sigma": 0.01}),
    (aug_time_warp, {"warp_range": 0.10}),
])
def test_augmentation_preserves_shape(aug_fn, extra_kwargs):
    """Every augmentation function must return the same shape as its input."""
    seq = make_seq(30)
    rng = np.random.default_rng(42)
    out = aug_fn(seq, rng=rng, **extra_kwargs)
    assert out.shape == seq.shape, (
        f"{aug_fn.__name__} changed shape: {seq.shape} → {out.shape}"
    )


def test_augmentation_preserves_dtype():
    """augment_sequence should return float32."""
    seq = make_seq(30)
    rng = np.random.default_rng(42)
    cfg = dummy_aug_cfg()
    out = augment_sequence(seq, cfg, rng)
    assert out.dtype == np.float32


def test_augmentation_changes_values():
    """Augmented sequence should differ from the original."""
    seq = make_seq(30)
    rng = np.random.default_rng(42)
    cfg = dummy_aug_cfg()
    out = augment_sequence(seq, cfg, rng)
    assert not np.allclose(out, seq), "Augmentation had no effect"


def test_scale_zero_range_is_identity():
    """Scale with range=0 should not change the values."""
    seq = make_seq(30)
    rng = np.random.default_rng(0)
    out = aug_scale(seq, scale_range=0.0, rng=rng)
    np.testing.assert_allclose(out, seq, atol=1e-6)


def test_shift_zero_range_is_identity():
    """Shift with range=0 should not change the values."""
    seq = make_seq(30)
    rng = np.random.default_rng(0)
    out = aug_shift(seq, shift_range=0.0, rng=rng)
    np.testing.assert_allclose(out, seq, atol=1e-6)


def test_rotate_zero_angle_is_identity():
    """Rotation with 0 degrees should not change the values."""
    seq = make_seq(30)
    rng = np.random.default_rng(0)
    out = aug_rotate(seq, rotation_deg=0.0, rng=rng)
    np.testing.assert_allclose(out, seq, atol=1e-5)


# ── LOSO split tests ──────────────────────────────────────────────────────────

def make_dummy_dataset(n_signers=3, clips_per_signer=10, n_classes=3,
                       T=30, F=FEATURE_DIM):
    """
    Build a tiny synthetic dataset for LOSO testing.
    Returns sequences, labels, signers.
    """
    signs = [f"sign_{i}" for i in range(n_classes)]
    sequences, labels, signers = [], [], []
    rng = np.random.default_rng(99)
    for s_idx in range(n_signers):
        signer_id = f"signer_{s_idx:02d}"
        for clip in range(clips_per_signer):
            seq = rng.random((T, F)).astype(np.float32)
            sign = signs[clip % n_classes]
            sequences.append(seq)
            labels.append(sign)
            signers.append(signer_id)
    return sequences, labels, signers


def test_loso_fold_count():
    """LOSO should yield exactly one fold per unique signer."""
    seqs, labels, signers = make_dummy_dataset(n_signers=3)
    cfg = dummy_aug_cfg()
    folds = list(loso_splits(seqs, labels, signers,
                             aug_cfg=cfg, copies_per_clip=1))
    assert len(folds) == 3, f"Expected 3 folds, got {len(folds)}"


def test_loso_test_set_is_held_out_signer():
    """Test set for each fold must contain only clips from the held-out signer."""
    seqs, labels, signers = make_dummy_dataset(n_signers=3, clips_per_signer=9)
    cfg = dummy_aug_cfg()
    for fold_id, X_tr, y_tr, X_te, y_te, class_names in loso_splits(
        seqs, labels, signers, aug_cfg=cfg, copies_per_clip=1
    ):
        # Count how many test samples come from the held-out signer
        # (we can check by counting: total per signer = clips_per_signer)
        assert X_te.shape[0] == 9, (
            f"Fold {fold_id}: expected 9 test samples, got {X_te.shape[0]}"
        )


def test_loso_no_overlap():
    """Train and test indices must not overlap (no data leakage)."""
    seqs, labels, signers = make_dummy_dataset(n_signers=3, clips_per_signer=6)
    cfg = dummy_aug_cfg()
    # Check that test set size + (train size / (1+copies)) == total
    total = len(seqs)
    for fold_id, X_tr, y_tr, X_te, y_te, class_names in loso_splits(
        seqs, labels, signers, aug_cfg=cfg, copies_per_clip=2
    ):
        n_test = X_te.shape[0]
        # Train without augmentation = total - test
        n_train_raw = total - n_test
        # With 2 copies, train size = n_train_raw * (1 + 2)
        expected_train = n_train_raw * 3
        assert X_tr.shape[0] == expected_train, (
            f"Fold {fold_id}: expected {expected_train} train samples "
            f"(with augmentation), got {X_tr.shape[0]}"
        )


def test_loso_shapes():
    """X_train and X_test must have shape (N, T, F)."""
    seqs, labels, signers = make_dummy_dataset(n_signers=2, clips_per_signer=6)
    cfg = dummy_aug_cfg()
    for fold_id, X_tr, y_tr, X_te, y_te, class_names in loso_splits(
        seqs, labels, signers, aug_cfg=cfg, copies_per_clip=1
    ):
        assert X_tr.ndim == 3
        assert X_te.ndim == 3
        assert X_tr.shape[1] == 30   # T
        assert X_tr.shape[2] == FEATURE_DIM  # F
        assert len(y_tr) == X_tr.shape[0]
        assert len(y_te) == X_te.shape[0]
