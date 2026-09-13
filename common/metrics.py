"""Metric helpers shared by baselines and GNN so results are computed identically."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)

from preprocessing.labels import ATTACK_CLASSES

# Attacks whose signature is spread across many flows/hosts: where topology
# should help most. Used for the "coordinated-attack subset" comparison.
COORDINATED_CLASSES = ["DDoS", "Probe", "Botnet"]


def binary_metrics(y_true, scores, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    y_pred = (scores >= threshold).astype(int)
    negatives = y_true == 0
    out = {
        "threshold": float(threshold),
        "support": int(len(y_true)),
        "positives": int(y_true.sum()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "false_positive_rate": float(y_pred[negatives].mean()) if negatives.any() else None,
        "roc_auc": None,
        "pr_auc": None,
    }
    if len(np.unique(y_true)) == 2:
        out["roc_auc"] = float(roc_auc_score(y_true, scores))
        out["pr_auc"] = float(average_precision_score(y_true, scores))
    return out


def best_threshold(y_true, scores, grid: np.ndarray | None = None) -> float:
    """Threshold maximising F1 (ties -> higher recall, then lower threshold)."""
    grid = np.linspace(0.05, 0.95, 91) if grid is None else grid
    best, best_key = 0.5, (-1.0, -1.0)
    for t in grid:
        m = binary_metrics(y_true, scores, float(t))
        key = (m["f1"], m["recall"])
        if key > best_key:
            best, best_key = float(t), key
    return best


def multiclass_metrics(y_true, y_pred, class_names: list[str] | None = None, min_support: int = 20) -> dict:
    """Macro/weighted F1 over classes present in ``y_true`` plus per-class scores.

    ``major_macro_f1`` averages only classes with at least ``min_support``
    samples: an F1 from one or two test windows is noise, and would otherwise
    dominate the macro average.
    """
    class_names = class_names or ATTACK_CLASSES
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    present = sorted(np.unique(y_true).tolist())
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=present, zero_division=0)
    per_class = {
        class_names[c]: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
        for i, c in enumerate(present)
    }
    major = [i for i in range(len(present)) if s[i] >= min_support]
    all_labels = list(range(len(class_names)))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "major_macro_f1": float(np.mean(f[major])) if major else None,
        "major_classes": [class_names[present[i]] for i in major],
        "macro_f1": float(f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=present, average="weighted", zero_division=0)),
        "per_class": per_class,
        "confusion_matrix": {
            "labels": class_names,
            "matrix": confusion_matrix(y_true, y_pred, labels=all_labels).tolist(),
        },
    }


def detection_rate_by_class(y_multi, binary_pred, class_names: list[str] | None = None) -> dict:
    """Share of windows of each true class flagged as attack (FPR for Benign)."""
    class_names = class_names or ATTACK_CLASSES
    y_multi = np.asarray(y_multi).astype(int)
    binary_pred = np.asarray(binary_pred).astype(int)
    return {
        class_names[c]: {"flagged_rate": float(binary_pred[y_multi == c].mean()), "support": int((y_multi == c).sum())}
        for c in sorted(np.unique(y_multi).tolist())
    }


def coordinated_subset_metrics(y_multi, scores, threshold: float, class_names: list[str] | None = None) -> dict:
    """Binary metrics on benign + coordinated-attack windows only."""
    class_names = class_names or ATTACK_CLASSES
    y_multi = np.asarray(y_multi).astype(int)
    keep_ids = {class_names.index("Benign")} | {class_names.index(c) for c in COORDINATED_CLASSES}
    mask = np.isin(y_multi, list(keep_ids))
    y_bin = (y_multi[mask] != class_names.index("Benign")).astype(int)
    return binary_metrics(y_bin, np.asarray(scores)[mask], threshold)
