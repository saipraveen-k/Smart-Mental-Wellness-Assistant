"""Shared classification metrics — computed from real predictions only."""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def compute_classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    label_names: List[str],
) -> Dict:
    """Accuracy / precision / recall / F1 / macro-F1 / weighted-F1 / CM."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape or y_true.size == 0:
        raise ValueError("y_true/y_pred must be non-empty and the same shape")

    indices = list(range(len(label_names)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=indices, zero_division=0
    )
    per_class = {
        name: {
            "precision": round(float(precision[i]), 4),
            "recall": round(float(recall[i]), 4),
            "f1": round(float(f1[i]), 4),
            "support": int(support[i]),
        }
        for i, name in enumerate(label_names)
    }
    cm = confusion_matrix(y_true, y_pred, labels=indices)
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, labels=indices, average="macro", zero_division=0)), 4),
        "weighted_f1": round(
            float(f1_score(y_true, y_pred, labels=indices, average="weighted", zero_division=0)), 4
        ),
        "micro_f1": round(float(f1_score(y_true, y_pred, labels=indices, average="micro", zero_division=0)), 4),
        "per_class": per_class,
        "confusion_matrix": {
            "labels": list(label_names),
            "matrix": cm.tolist(),
        },
        "num_samples": int(y_true.size),
        "classification_report": classification_report(
            y_true, y_pred, labels=indices, target_names=label_names, zero_division=0, output_dict=True
        ),
    }
