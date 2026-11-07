"""
src/evaluate/evaluate.py
------------------------
Phase 3 — metrics, confusion matrix, training curves, and final report.

All numbers produced here come from actual model predictions on held-out
test data.  Nothing is hardcoded or estimated.

Functions used by train.py
--------------------------
  compute_fold_metrics()   — accuracy, per-class P/R/F1, unsure %
  save_confusion_matrix()  — saves a PNG heatmap
  save_training_curves()   — saves loss + accuracy vs epoch PNG
  save_final_report()      — saves reports/final_report.json

Important note on dataset size
-------------------------------
All reports include a caveat that results come from a small self-recorded
dataset and should be treated as indicative only.  See reports/CAVEAT.md.
"""

import json
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe on all platforms
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)


# ── Dataset size caveat — appended to every report ────────────────────────────
_CAVEAT = (
    "IMPORTANT: These results are based on a small self-recorded dataset "
    "(1 signer, 8 signs, 5 clips per sign). They reflect within-dataset "
    "performance only and should NOT be treated as conclusive. "
    "See reports/CAVEAT.md for full details."
)


# ─────────────────────────────────────────────────────────────────────────────
# Core metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_fold_metrics(model, X_test: np.ndarray, y_test: np.ndarray,
                          class_names: list, fold_label: str,
                          model_name: str,
                          confidence_threshold: float = None) -> dict:
    """
    Run model predictions on the test set and compute all metrics.

    Parameters
    ----------
    model              : trained Keras model
    X_test             : np.ndarray (N, T, F)
    y_test             : np.ndarray (N,) integer class indices
    class_names        : list of str
    fold_label         : e.g. "fold_1" or "loso_signer_01"
    model_name         : "lstm" or "baseline"
    confidence_threshold : float or None.  If given, predictions below
                           this threshold are counted as "unsure".
                           If None, no unsure gating is applied.

    Returns
    -------
    dict with keys:
        fold, model, accuracy, n_test,
        per_class (dict per sign: precision, recall, f1, support),
        unsure_pct (% of predictions below threshold),
        y_pred (list of int), y_prob_max (list of float)
    """
    # Get softmax probabilities for all test samples
    probs = model.predict(X_test, verbose=0)  # shape (N, n_classes)
    y_pred_all  = np.argmax(probs, axis=1)
    y_prob_max  = np.max(probs, axis=1)

    # Apply confidence threshold if given
    if confidence_threshold is not None:
        unsure_mask = y_prob_max < confidence_threshold
        unsure_pct  = float(np.mean(unsure_mask) * 100)
        # For metrics, only count confident predictions;
        # unsure samples are excluded from accuracy (counted as wrong)
        y_pred_gated = y_pred_all.copy()
        # Mark unsure as wrong by setting to an invalid class index
        y_pred_gated[unsure_mask] = -1
    else:
        unsure_pct   = 0.0
        y_pred_gated = y_pred_all

    # Accuracy (unsure = wrong)
    accuracy = float(accuracy_score(y_test, y_pred_gated))

    # Per-class precision, recall, F1
    # zero_division=0 prevents warnings when a class has no predictions
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred_gated,
        labels=list(range(len(class_names))),
        zero_division=0,
    )

    per_class = {}
    for i, name in enumerate(class_names):
        per_class[name] = {
            "precision": round(float(precision[i]), 4),
            "recall":    round(float(recall[i]),    4),
            "f1":        round(float(f1[i]),        4),
            "support":   int(support[i]),
        }

    return {
        "fold":       fold_label,
        "model":      model_name,
        "accuracy":   round(accuracy, 4),
        "per_class":  per_class,
        "unsure_pct": round(unsure_pct, 2),
        "y_pred":     y_pred_all.tolist(),
        "y_prob_max": [round(float(p), 4) for p in y_prob_max],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Confusion matrix
# ─────────────────────────────────────────────────────────────────────────────

def save_confusion_matrix(model, X_test: np.ndarray, y_test: np.ndarray,
                           class_names: list, fold_label: str,
                           model_name: str, reports_dir: Path):
    """
    Compute and save a confusion matrix heatmap as a PNG.

    File: reports/<model_name>_<fold_label>_confusion.png
    """
    probs  = model.predict(X_test, verbose=0)
    y_pred = np.argmax(probs, axis=1)
    cm     = confusion_matrix(y_test, y_pred,
                               labels=list(range(len(class_names))))

    fig, ax = plt.subplots(figsize=(max(6, len(class_names)),
                                    max(5, len(class_names))))

    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(
        f"Confusion Matrix\n{model_name.upper()}  |  {fold_label}\n"
        f"(n_test={len(y_test)})"
    )

    # Annotate cells with counts
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center", fontsize=10,
                    color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    out_path = Path(reports_dir) / f"{model_name}_{fold_label}_confusion.png"
    fig.savefig(str(out_path), dpi=100)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Training curves
# ─────────────────────────────────────────────────────────────────────────────

def save_training_curves(history: dict, fold_label: str,
                          model_name: str, reports_dir: Path):
    """
    Save training and validation loss + accuracy curves as a PNG.

    File: reports/<model_name>_<fold_label>_curves.png
    """
    epochs = range(1, len(history["loss"]) + 1)

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    ax_loss.plot(epochs, history["loss"],     label="Train loss")
    ax_loss.plot(epochs, history["val_loss"], label="Val loss",
                 linestyle="--")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title(f"Loss — {model_name.upper()} | {fold_label}")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    # Accuracy
    ax_acc.plot(epochs, history["accuracy"],     label="Train acc")
    ax_acc.plot(epochs, history["val_accuracy"], label="Val acc",
                linestyle="--")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_title(f"Accuracy — {model_name.upper()} | {fold_label}")
    ax_acc.set_ylim(0, 1.05)
    ax_acc.legend()
    ax_acc.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = (Path(reports_dir) /
                f"{model_name}_{fold_label}_curves.png")
    fig.savefig(str(out_path), dpi=100)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Final report
# ─────────────────────────────────────────────────────────────────────────────

def save_final_report(all_results: dict, cv_mode: str,
                       class_names: list, cfg: dict,
                       reports_dir: Path):
    """
    Aggregate all fold results into one JSON report and print a
    per-class summary table.

    File: reports/final_report.json
    """
    reports_dir = Path(reports_dir)

    summary = {
        "caveat":      _CAVEAT,
        "cv_mode":     cv_mode,
        "class_names": class_names,
        "models":      {},
    }

    print("\n  PER-CLASS METRICS (averaged across folds)")
    print("  " + "─" * 70)

    for model_name, folds in all_results.items():
        if not folds:
            continue

        accs  = [f["accuracy"]            for f in folds]
        lats  = [f["inference_latency_ms"] for f in folds]
        unsure = [f["unsure_pct"]           for f in folds]

        # Aggregate per-class metrics across folds (macro average)
        agg_per_class = {}
        for sign in class_names:
            p_vals = [f["per_class"][sign]["precision"] for f in folds
                      if sign in f["per_class"]]
            r_vals = [f["per_class"][sign]["recall"]    for f in folds
                      if sign in f["per_class"]]
            f_vals = [f["per_class"][sign]["f1"]        for f in folds
                      if sign in f["per_class"]]
            s_vals = [f["per_class"][sign]["support"]   for f in folds
                      if sign in f["per_class"]]
            agg_per_class[sign] = {
                "mean_precision": round(float(np.mean(p_vals)), 4) if p_vals else 0.0,
                "mean_recall":    round(float(np.mean(r_vals)), 4) if r_vals else 0.0,
                "mean_f1":        round(float(np.mean(f_vals)), 4) if f_vals else 0.0,
                "total_support":  int(sum(s_vals)),
            }

        model_summary = {
            "accuracy_per_fold": [round(a, 4) for a in accs],
            "mean_accuracy":     round(float(np.mean(accs)), 4),
            "std_accuracy":      round(float(np.std(accs)),  4),
            "n_folds":           len(folds),
            "mean_latency_ms":   round(float(np.mean(lats)), 2),
            "mean_unsure_pct":   round(float(np.mean(unsure)), 2),
            "per_class":         agg_per_class,
            "folds":             [
                {k: v for k, v in f.items() if k != "history"}
                for f in folds
            ],
        }
        summary["models"][model_name] = model_summary

        # Print per-class table for this model
        print(f"\n  {model_name.upper()}")
        print(f"  {'Sign':<16} {'Prec':>6} {'Rec':>6} {'F1':>6} "
              f"{'Support':>8}")
        print(f"  {'─'*16} {'─'*6} {'─'*6} {'─'*6} {'─'*8}")
        for sign, vals in agg_per_class.items():
            print(f"  {sign:<16} "
                  f"{vals['mean_precision']:6.3f} "
                  f"{vals['mean_recall']:6.3f} "
                  f"{vals['mean_f1']:6.3f} "
                  f"{vals['total_support']:8d}")
        print(f"\n  Mean accuracy : {np.mean(accs):.3f} ± {np.std(accs):.3f}"
              f"  (n_test total = {sum(f['n_test'] for f in folds)})")
        print(f"  Avg latency   : {np.mean(lats):.1f} ms/sample")

    # Write JSON
    report_path = reports_dir / "final_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Also write a human-readable text summary
    _write_text_summary(summary, reports_dir)


def _write_text_summary(summary: dict, reports_dir: Path):
    """Write a plain-text summary file alongside the JSON report."""
    lines = [
        "SignHire — Evaluation Summary",
        "=" * 60,
        "",
        f"CV mode : {summary['cv_mode']}",
        f"Classes : {', '.join(summary['class_names'])}",
        "",
        _CAVEAT,
        "",
    ]

    for model_name, data in summary["models"].items():
        lines += [
            f"Model: {model_name.upper()}",
            f"  Accuracy per fold : "
            + "  ".join(f"{a:.3f}" for a in data["accuracy_per_fold"]),
            f"  Mean ± std        : "
            f"{data['mean_accuracy']:.3f} ± {data['std_accuracy']:.3f}",
            f"  Avg latency       : {data['mean_latency_ms']:.1f} ms",
            f"  Avg unsure %      : {data['mean_unsure_pct']:.1f}%",
            "",
            f"  {'Sign':<16} {'Prec':>6} {'Rec':>6} {'F1':>6}",
            f"  {'─'*16} {'─'*6} {'─'*6} {'─'*6}",
        ]
        for sign, vals in data["per_class"].items():
            lines.append(
                f"  {sign:<16} "
                f"{vals['mean_precision']:6.3f} "
                f"{vals['mean_recall']:6.3f} "
                f"{vals['mean_f1']:6.3f}"
            )
        lines.append("")

    out_path = reports_dir / "summary.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
