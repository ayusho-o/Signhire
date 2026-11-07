"""
src/train/train.py
------------------
Phase 3 — training loop with leave-one-signer-out (LOSO) cross-validation.

How it works
------------
1. Loads the processed dataset from data/processed/.
2. Decides the evaluation strategy:
     ≥ 2 signers → LOSO cross-validation (one fold per signer)
     1 signer    → Stratified k-fold (k=5) clearly labelled as
                   "single-signer CV" — NOT equivalent to LOSO.
3. For each fold:
   a. Applies augmentation to training data only.
   b. Trains both the LSTM model and the GRU baseline.
   c. Saves the best weights (by val_accuracy) via ModelCheckpoint.
   d. Passes results to evaluate.py to compute metrics and plots.
4. Saves a final JSON report and prints a summary table.

Run from SignHire/:
    python src/train/train.py
    python src/train/train.py --model lstm      (train only LSTM)
    python src/train/train.py --model baseline  (train only baseline)
    python src/train/train.py --epochs 80       (override epochs)
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.models.lstm_model     import build_lstm_model
from src.models.baseline_model import build_baseline_model
from src.preprocess.preprocess import load_config, augment_sequence
from src.evaluate.evaluate     import (
    compute_fold_metrics,
    save_confusion_matrix,
    save_training_curves,
    save_final_report,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fold generators
# ─────────────────────────────────────────────────────────────────────────────

def loso_folds(X, y, signers_arr):
    """
    Yield (fold_label, X_train_raw, y_train, X_test, y_test) for each
    unique signer held out in turn.
    """
    for held_out in sorted(set(signers_arr)):
        test_mask  = signers_arr == held_out
        train_mask = ~test_mask
        yield (
            f"loso_{held_out}",
            X[train_mask], y[train_mask],
            X[test_mask],  y[test_mask],
        )


def stratified_kfold_folds(X, y, k: int = 5, rng_seed: int = 42):
    """
    Yield (fold_label, X_train_raw, y_train, X_test, y_test) using
    stratified k-fold splitting.

    Stratified means each fold has roughly equal class proportions —
    important when there are only 5 clips per class.
    """
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=rng_seed)
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        yield (
            f"fold_{fold_idx + 1}",
            X[train_idx], y[train_idx],
            X[test_idx],  y[test_idx],
        )


# ─────────────────────────────────────────────────────────────────────────────
# Augmentation helper
# ─────────────────────────────────────────────────────────────────────────────

def augment_train_split(X_train: np.ndarray, y_train: np.ndarray,
                         aug_cfg: dict, copies: int,
                         rng: np.random.Generator):
    """
    Expand the training set by generating `copies` augmented versions
    of every training clip.  Returns concatenated (X_aug, y_aug).
    """
    if copies == 0:
        return X_train, y_train

    batches_X = [X_train]
    batches_y = [y_train]

    for _ in range(copies):
        aug_batch = np.stack(
            [augment_sequence(seq, aug_cfg, rng) for seq in X_train],
            axis=0,
        )
        batches_X.append(aug_batch)
        batches_y.append(y_train)

    X_aug = np.concatenate(batches_X, axis=0)
    y_aug = np.concatenate(batches_y, axis=0)

    # Shuffle so augmented copies are interleaved with originals
    perm = rng.permutation(len(X_aug))
    return X_aug[perm], y_aug[perm]


# ─────────────────────────────────────────────────────────────────────────────
# Single-fold trainer
# ─────────────────────────────────────────────────────────────────────────────

def train_one_fold(model_name: str, fold_label: str,
                   X_train_raw, y_train_raw,
                   X_test, y_test,
                   class_names, cfg, aug_cfg,
                   models_dir: Path, reports_dir: Path,
                   rng: np.random.Generator):
    """
    Train one model on one fold and return a metrics dict.

    Parameters
    ----------
    model_name  : "lstm" or "baseline"
    fold_label  : e.g. "loso_signer_01" or "fold_1"
    X_train_raw : np.ndarray (N_train, T, F) — before augmentation
    y_train_raw : np.ndarray (N_train,)
    X_test      : np.ndarray (N_test,  T, F)
    y_test      : np.ndarray (N_test,)
    class_names : list of str
    cfg         : full settings dict
    aug_cfg     : cfg["augmentation"]
    models_dir  : Path to save .keras weights
    reports_dir : Path to save plots and metrics
    rng         : shared numpy random generator

    Returns
    -------
    dict with keys: fold, model, accuracy, per_class, n_test,
                    history, inference_latency_ms
    """
    import tensorflow as tf

    model_cfg  = cfg["model"]
    train_cfg  = cfg["recording"]
    n_classes  = len(class_names)
    seq_frames = int(X_train_raw.shape[1])
    feat_dim   = int(X_train_raw.shape[2])
    epochs     = model_cfg["epochs"]
    batch_size = model_cfg["batch_size"]
    copies     = aug_cfg["copies_per_clip"]

    print(f"\n  ── {model_name.upper()}  |  {fold_label} ──────────────────")
    print(f"     Train (raw): {len(X_train_raw)}  Test: {len(X_test)}")

    # ── Augment training data ─────────────────────────────────────────────
    X_train, y_train = augment_train_split(
        X_train_raw, y_train_raw, aug_cfg, copies, rng
    )
    print(f"     Train (aug): {len(X_train)}")

    # ── Build model ───────────────────────────────────────────────────────
    if model_name == "lstm":
        model = build_lstm_model(seq_frames, feat_dim, n_classes, model_cfg)
    else:
        model = build_baseline_model(seq_frames, feat_dim, n_classes, model_cfg)

    # ── Callbacks ─────────────────────────────────────────────────────────
    model_path = models_dir / f"{model_name}_{fold_label}.keras"
    callbacks = [
        # Save the best weights by validation accuracy
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=0,
        ),
        # Stop early if val_accuracy doesn't improve for 15 epochs
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=15,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    # ── Train ─────────────────────────────────────────────────────────────
    # Use 20% of training data as validation (for early stopping / curves).
    # This validation split comes from training data only — never from test.
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=0,   # suppress per-epoch output; summary printed below
    )
    actual_epochs = len(history.history["loss"])
    best_val_acc  = max(history.history.get("val_accuracy", [0]))
    print(f"     Epochs run : {actual_epochs}  |  "
          f"Best val_acc: {best_val_acc:.3f}")

    # ── Measure inference latency ─────────────────────────────────────────
    # Run 50 single-sample predictions and report mean latency in ms.
    sample = X_test[:1]  # shape (1, T, F)
    # Warm-up run (first call is slower due to TF graph compilation)
    _ = model.predict(sample, verbose=0)
    latencies = []
    for _ in range(50):
        t0 = time.perf_counter()
        model.predict(sample, verbose=0)
        latencies.append((time.perf_counter() - t0) * 1000)
    latency_ms = float(np.mean(latencies))
    print(f"     Inference   : {latency_ms:.1f} ms/sample (mean of 50 runs)")

    # ── Evaluate on test set ──────────────────────────────────────────────
    metrics = compute_fold_metrics(
        model, X_test, y_test, class_names,
        fold_label=fold_label, model_name=model_name,
    )
    metrics["inference_latency_ms"] = round(latency_ms, 2)
    metrics["n_test"]    = int(len(X_test))
    metrics["n_train_raw"] = int(len(X_train_raw))
    metrics["n_train_aug"] = int(len(X_train))
    metrics["epochs_run"]  = actual_epochs
    metrics["best_val_acc"] = round(float(best_val_acc), 4)
    metrics["history"] = {k: [float(v) for v in vals]
                          for k, vals in history.history.items()}

    print(f"     Test acc    : {metrics['accuracy']:.3f}  "
          f"(n_test={metrics['n_test']})")

    # ── Save training curves ──────────────────────────────────────────────
    save_training_curves(
        history.history,
        fold_label=fold_label,
        model_name=model_name,
        reports_dir=reports_dir,
    )

    # ── Save confusion matrix ─────────────────────────────────────────────
    save_confusion_matrix(
        model, X_test, y_test, class_names,
        fold_label=fold_label,
        model_name=model_name,
        reports_dir=reports_dir,
    )

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Main training orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def run_training(which_models: list[str] = None, epochs_override: int = None):
    """
    Full training run: load data → choose CV strategy → train all folds
    → save final report.

    Parameters
    ----------
    which_models : list of "lstm" and/or "baseline".  None = both.
    epochs_override : if given, overrides cfg["model"]["epochs"].
    """
    vocab_cfg, cfg = load_config(_ROOT / "config")

    if which_models is None:
        which_models = ["lstm", "baseline"]
    if epochs_override:
        cfg["model"]["epochs"] = epochs_override

    aug_cfg    = cfg["augmentation"]
    models_dir = _ROOT / cfg["paths"]["models_dir"]
    reports_dir = _ROOT / cfg["paths"]["reports_dir"]
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # ── Load processed dataset ────────────────────────────────────────────
    processed_dir = _ROOT / cfg["paths"]["processed_data"]
    X_path = processed_dir / "X.npy"
    y_path = processed_dir / "y.npy"
    s_path = processed_dir / "signers.npy"
    meta_path = processed_dir / "meta.json"

    if not X_path.exists():
        print("ERROR: Processed dataset not found. Run src/preprocess/preprocess.py first.")
        sys.exit(1)

    X       = np.load(str(X_path))
    y       = np.load(str(y_path))
    signers = np.load(str(s_path), allow_pickle=True)

    with open(meta_path) as f:
        meta = json.load(f)
    class_names = meta["class_names"]

    print(f"\nSignHire — Training")
    print(f"  Dataset      : {X.shape}  (samples, frames, features)")
    print(f"  Classes      : {class_names}")
    print(f"  Signers      : {sorted(set(signers))}")
    print(f"  Models       : {which_models}")
    print(f"  Epochs       : {cfg['model']['epochs']}")
    print(f"  Aug copies   : {aug_cfg['copies_per_clip']}")

    # ── Choose CV strategy ────────────────────────────────────────────────
    unique_signers = sorted(set(signers))
    n_signers = len(unique_signers)

    if n_signers >= 2:
        cv_mode = "loso"
        fold_gen = loso_folds(X, y, signers)
        n_folds  = n_signers
        print(f"\n  CV strategy  : Leave-One-Signer-Out ({n_folds} folds)")
    else:
        cv_mode = "stratified_kfold"
        k = 5
        fold_gen = stratified_kfold_folds(X, y, k=k)
        n_folds  = k
        print(f"\n  CV strategy  : Stratified {k}-fold (single signer)")
        print(f"  *** NOTE: Only 1 signer present. This is NOT equivalent")
        print(f"  *** to LOSO. Results reflect within-signer generalisation")
        print(f"  *** only. Record more signers for a proper evaluation.")

    # ── Training loop ─────────────────────────────────────────────────────
    rng = np.random.default_rng(42)
    all_results = {m: [] for m in which_models}

    for fold_data in fold_gen:
        fold_label, X_tr, y_tr, X_te, y_te = fold_data

        for model_name in which_models:
            result = train_one_fold(
                model_name=model_name,
                fold_label=fold_label,
                X_train_raw=X_tr, y_train_raw=y_tr,
                X_test=X_te,      y_test=y_te,
                class_names=class_names,
                cfg=cfg,
                aug_cfg=aug_cfg,
                models_dir=models_dir,
                reports_dir=reports_dir,
                rng=rng,
            )
            all_results[model_name].append(result)

    # ── Final summary ─────────────────────────────────────────────────────
    print("\n" + "═" * 65)
    print("TRAINING COMPLETE — SUMMARY")
    print("═" * 65)
    print(f"  CV mode : {cv_mode}")
    print(f"  Folds   : {n_folds}")
    print()

    for model_name in which_models:
        folds = all_results[model_name]
        accs  = [f["accuracy"] for f in folds]
        lats  = [f["inference_latency_ms"] for f in folds]
        unsure_pcts = [f.get("unsure_pct", 0.0) for f in folds]

        print(f"  {model_name.upper()}")
        print(f"    Accuracy per fold : "
              + "  ".join(f"{a:.3f}" for a in accs))
        print(f"    Mean ± std        : "
              f"{np.mean(accs):.3f} ± {np.std(accs):.3f}  "
              f"(n_folds={len(accs)})")
        print(f"    Avg latency       : {np.mean(lats):.1f} ms/sample")
        print(f"    Avg unsure %      : {np.mean(unsure_pcts):.1f}%")
        print()

    # ── Save final report ─────────────────────────────────────────────────
    save_final_report(
        all_results=all_results,
        cv_mode=cv_mode,
        class_names=class_names,
        cfg=cfg,
        reports_dir=reports_dir,
    )

    print(f"  Reports saved to : {reports_dir}")
    print("═" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="SignHire model training")
    p.add_argument("--model", choices=["lstm", "baseline", "both"],
                   default="both",
                   help="Which model(s) to train (default: both)")
    p.add_argument("--epochs", type=int, default=None,
                   help="Override the epoch count from settings.yaml")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    models = ["lstm", "baseline"] if args.model == "both" else [args.model]
    run_training(which_models=models, epochs_override=args.epochs)
