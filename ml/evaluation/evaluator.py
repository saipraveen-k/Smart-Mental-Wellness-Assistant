"""Comprehensive Model Evaluator."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

REPO_ROOT = Path(__file__).resolve().parents[2]

def evaluate_all_models() -> Dict[str, Any]:
    """Inspects and evaluates all saved model artifacts in ml/artifacts/."""
    results = {}
    
    artifacts = {
        "text_emotion": REPO_ROOT / "ml" / "artifacts" / "text_emotion" / "metrics.json",
        "cv_emotion": REPO_ROOT / "ml" / "artifacts" / "cv_emotion" / "metrics.json",
        "crisis": REPO_ROOT / "ml" / "artifacts" / "crisis" / "metrics.json",
    }
    
    for name, path in artifacts.items():
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    metrics = json.load(f)
                results[name] = {"status": "EVALUATED", "metrics": metrics}
            except Exception as e:
                results[name] = {"status": "ERROR", "error": str(e)}
        else:
            results[name] = {"status": "NOT_TRAINED", "path": str(path)}
            
    return results

def run_full_evaluation() -> None:
    res = evaluate_all_models()
    print("=" * 60)
    print(" SMART MENTAL WELLNESS ASSISTANT — EVALUATION SUMMARY")
    print("=" * 60)
    for model_name, info in res.items():
        status = info["status"]
        if status == "EVALUATED":
            raw_m = info["metrics"]
            m = raw_m.get("test_metrics", raw_m)
            acc = m.get("accuracy", 0.0)
            f1 = m.get("macro_f1", 0.0)
            extra = ""
            if "crisis_vs_low" in m:
                extra = f" | Crisis Recall={m['crisis_vs_low'].get('recall', 0.0):.4f}"
            print(f"  ✓ {model_name:<15}: Accuracy={acc:.4f} | Macro F1={f1:.4f}{extra}")
        else:
            print(f"  ⚠ {model_name:<15}: Status={status}")
    print("=" * 60)
