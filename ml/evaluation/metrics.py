"""Classification Metrics Calculator."""
from __future__ import annotations

from typing import Dict, List, Any
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

def compute_classification_metrics(y_true: List[int], y_pred: List[int], labels: List[str]) -> Dict[str, Any]:
    """Computes comprehensive classification metrics:
      - Accuracy
      - Precision (macro & weighted)
      - Recall (macro & weighted)
      - F1 Score (macro & weighted)
      - Per-class metrics
      - Confusion matrix
    """
    acc = float(accuracy_score(y_true, y_pred))
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    
    p_class, r_class, f1_class, support_class = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred).tolist()
    
    per_class_report = {}
    for i, label in enumerate(labels):
        if i < len(p_class):
            per_class_report[label] = {
                "precision": float(p_class[i]),
                "recall": float(r_class[i]),
                "f1": float(f1_class[i]),
                "support": int(support_class[i])
            }

    return {
        "accuracy": acc,
        "macro_precision": float(p_macro),
        "macro_recall": float(r_macro),
        "macro_f1": float(f1_macro),
        "weighted_precision": float(p_weighted),
        "weighted_recall": float(r_weighted),
        "weighted_f1": float(f1_weighted),
        "confusion_matrix": cm,
        "per_class": per_class_report
    }
