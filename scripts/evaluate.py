#!/usr/bin/env python
"""Model evaluation script for Smart Mental Wellness Assistant.

Evaluates trained model artifacts stored in `ml/artifacts/`:
  - `text_emotion`: DistilBERT text classifier on GoEmotions test split
  - `cv_emotion`: FacialEmotionCNN on FER-2013 test split
  - `crisis`: C-SSRS Crisis classifier on held-out crisis test split

Computes:
  - Accuracy
  - Precision (macro & weighted)
  - Recall (macro & weighted)
  - F1 Score (macro & weighted)
  - Confusion Matrix
  - Per-class metrics
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "backend"))

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

def evaluate_text_model() -> Dict[str, Any]:
    print("[1/3] Evaluating Text Emotion Model (DistilBERT) ...")
    artifact_dir = REPO_ROOT / "ml" / "artifacts" / "text_emotion"
    metrics_path = artifact_dir / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        print(f"      ✓ Text Model Metrics Found (Evaluated at {metrics.get('evaluated_at', 'N/A')})")
        print(f"        Accuracy: {metrics.get('accuracy', 0.0):.4f} | Macro F1: {metrics.get('macro_f1', 0.0):.4f}")
        return metrics
    else:
        print("      ⚠ Text model metrics file not found in ml/artifacts/text_emotion/")
        return {}

def evaluate_cv_model() -> Dict[str, Any]:
    print("[2/3] Evaluating CV Emotion Model (FacialEmotionCNN) ...")
    artifact_dir = REPO_ROOT / "ml" / "artifacts" / "cv_emotion"
    metrics_path = artifact_dir / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        print(f"      ✓ CV Model Metrics Found (Evaluated at {metrics.get('evaluated_at', 'N/A')})")
        print(f"        Accuracy: {metrics.get('accuracy', 0.0):.4f} | Macro F1: {metrics.get('macro_f1', 0.0):.4f}")
        return metrics
    else:
        print("      ⚠ CV model metrics file not found in ml/artifacts/cv_emotion/")
        return {}

def evaluate_crisis_model() -> Dict[str, Any]:
    print("[3/3] Evaluating Crisis Classifier Subsystem ...")
    artifact_dir = REPO_ROOT / "ml" / "artifacts" / "crisis"
    metrics_path = artifact_dir / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        print(f"      ✓ Crisis Model Metrics Found (Evaluated at {metrics.get('evaluated_at', 'N/A')})")
        print(f"        Crisis Recall: {metrics.get('crisis_recall', 0.0):.4f} | Accuracy: {metrics.get('accuracy', 0.0):.4f}")
        return metrics
    else:
        print("      ⚠ Crisis metrics file not found in ml/artifacts/crisis/")
        return {}

def main() -> int:
    from ml.evaluation.evaluator import run_full_evaluation
    run_full_evaluation()
    return 0

if __name__ == "__main__":
    sys.exit(main())
