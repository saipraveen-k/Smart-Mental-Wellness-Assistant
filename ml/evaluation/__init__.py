"""Evaluation package for computing model performance metrics and generating evaluation reports."""
from __future__ import annotations

from .metrics import compute_classification_metrics
from .evaluator import run_full_evaluation

__all__ = ["compute_classification_metrics", "run_full_evaluation"]
