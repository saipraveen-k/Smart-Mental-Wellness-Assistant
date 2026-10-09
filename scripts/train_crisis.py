#!/usr/bin/env python
"""Train the crisis-text classifier (separate from the emotion classifier).

Dataset:
    CRISIS_DATASET_ID (default: av9ash/CSSR-S_labelled_suicidewatch_posts_reddit)
        - r/SuicideWatch posts with C-SSRS-style severity labels (0-6)
    CRISIS_CONTROL_DATASET_ID (default: SetFit/go_emotions)
        - general, non-crisis everyday text used as the LOW class, so the
          model learns that ordinary distress (sadness, anxiety) is NOT a
          crisis. Sampled from the dataset's *test* split so it does not
          overlap the text-emotion model's training data.

Labels (conservative mapping, documented in docs/SAFETY.md):
    LOW      <- severity 0 (no suicidal ideation) + control everyday text
    MODERATE <- severity 1-3 (ideation, with or without intent)
    HIGH     <- severity 4-6 (specific plan/intent or suicidal behaviour)

The decision rule prioritises RECALL for MODERATE/HIGH: thresholds are tuned
on a validation split to maximise crisis recall subject to a precision floor.

Usage:
    python scripts/train_crisis.py
"""
from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))

import numpy as np  # noqa: E402

SEED = 42
CRISIS_CLASSES = ["LOW", "MODERATE", "HIGH"]

# C-SSRS severity (0-6) -> crisis class
SEVERITY_TO_CLASS = {
    0: "LOW",
    1: "MODERATE", 2: "MODERATE", 3: "MODERATE",
    4: "HIGH", 5: "HIGH", 6: "HIGH",
}

PRECISION_FLOOR = 0.70  # crisis recall maximised subject to precision >= floor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the crisis text classifier")
    p.add_argument("--output", type=str, default="ml/artifacts/crisis")
    p.add_argument("--control-samples", type=int, default=2500)
    p.add_argument("--max-features", type=int, default=30000)
    p.add_argument("--test-size", type=float, default=0.15)
    p.add_argument("--val-size", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args()


def load_crisis_data() -> Tuple[List[str], List[str], Dict]:
    """Download + parse the CSSR-S dataset dynamically from the Hub."""
    import urllib.request

    from app.config import settings
    from app.ml.dataset_manager import DatasetUnavailableError

    repo = settings.crisis_dataset_id
    url = f"https://huggingface.co/datasets/{repo}/resolve/main/labeled_rSuicidewatch_posts.csv"
    print(f"[1/5] Downloading crisis dataset {repo} ...")
    for attempt in range(1, 4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "smwa-crisis-trainer"})
            raw = urllib.request.urlopen(req, timeout=60).read().decode("utf-8", "replace")
            break
        except Exception as exc:  # noqa: BLE001
            if attempt == 3:
                raise DatasetUnavailableError(
                    repo, f"could not download crisis CSV ({exc})"
                ) from exc
            time.sleep(2 * attempt)

    import csv
    import io

    reader = csv.DictReader(io.StringIO(raw))
    texts: List[str] = []
    labels: List[str] = []
    skipped = 0
    for row in reader:
        content = (row.get("content") or "").strip()
        sev_raw = (row.get("severity") or "").strip()
        if not content or not sev_raw:
            skipped += 1
            continue
        try:
            sev = int(float(sev_raw))
        except ValueError:
            skipped += 1
            continue
        if sev not in SEVERITY_TO_CLASS:
            skipped += 1
            continue
        texts.append(content)
        labels.append(SEVERITY_TO_CLASS[sev])

    from huggingface_hub import HfApi

    try:
        sha = HfApi().dataset_info(repo).sha
    except Exception:
        sha = None

    prov = {
        "dataset_id": repo,
        "revision": sha,
        "rows": len(texts),
        "skipped": skipped,
        "severity_distribution_source": "column 'severity' of the dataset",
        "class_distribution": dict(Counter(labels)),
    }
    print(f"      rows={len(texts)} skipped={skipped} dist={prov['class_distribution']} revision={sha}")
    return texts, labels, prov


def load_control_texts(n_samples: int, seed: int) -> Tuple[List[str], List[str]]:
    """Everyday non-crisis text from the configured control dataset's test split."""
    from app.config import settings
    from app.ml.dataset_manager import dataset_manager
    from app.ml.text_preprocessing import clean_text, is_meaningful

    print(f"[2/5] Sampling {n_samples} control (non-crisis) texts from "
          f"{settings.crisis_control_dataset_id} test split ...")
    ds = dataset_manager.load_text_dataset()
    pool = []
    for split in ("test", "validation"):
        if split in ds:
            for row in ds[split]:
                t = clean_text(row.get("text", ""))
                if is_meaningful(t):
                    pool.append(t)
    rng = random.Random(seed)
    rng.shuffle(pool)
    control = pool[:n_samples]
    print(f"      sampled {len(control)} control texts")
    return control, ["LOW"] * len(control)


def stratified_split(texts, labels, test_size, val_size, seed):
    from sklearn.model_selection import train_test_split

    idx = np.arange(len(labels))
    tr_i, te_i = train_test_split(idx, test_size=test_size, random_state=seed, stratify=labels)
    tr_lab = [labels[i] for i in tr_i]
    rel_val = val_size / (1.0 - test_size)
    tr_i, va_i = train_test_split(tr_i, test_size=rel_val, random_state=seed, stratify=tr_lab)

    def take(indices):
        return [texts[i] for i in indices], [labels[i] for i in indices]

    return take(tr_i), take(va_i), take(te_i)


def _class_columns(estimator_classes) -> Dict[str, int]:
    """Map class name -> predict_proba column (sklearn sorts classes)."""
    return {str(c): i for i, c in enumerate(estimator_classes)}


def tune_thresholds(proba: np.ndarray, y_true: List[str], cols: Dict[str, int]) -> Dict[str, float]:
    """Choose MODERATE/HIGH thresholds maximising crisis recall subject to a
    precision floor (recall-prioritised, conservative)."""
    from sklearn.metrics import precision_score, recall_score

    crisis_true = np.array([l != "LOW" for l in y_true])
    best = {"moderate": 0.35, "high": 0.35}
    best_score = (-1.0, -1.0)
    col_m = cols["MODERATE"]
    col_h = cols["HIGH"]
    for tm in (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50):
        for th in (0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50):
            pred = []
            for row in proba:
                if row[col_h] >= th:
                    pred.append("HIGH")
                elif row[col_m] >= tm:
                    pred.append("MODERATE")
                else:
                    pred.append("LOW")
            crisis_pred = np.array([l != "LOW" for l in pred])
            if crisis_pred.sum() == 0:
                continue
            prec = precision_score(crisis_true, crisis_pred, zero_division=0)
            rec = recall_score(crisis_true, crisis_pred, zero_division=0)
            if prec < PRECISION_FLOOR:
                continue
            score = (rec, prec)
            if score > best_score:
                best_score = score
                best = {"moderate": tm, "high": th}
    return best


def predict_with_thresholds(proba: np.ndarray, thresholds: Dict[str, float],
                            cols: Dict[str, int]) -> List[str]:
    col_m = cols["MODERATE"]
    col_h = cols["HIGH"]
    out = []
    for row in proba:
        if row[col_h] >= thresholds["high"]:
            out.append("HIGH")
        elif row[col_m] >= thresholds["moderate"]:
            out.append("MODERATE")
        else:
            out.append("LOW")
    return out


def main() -> int:
    args = parse_args()

    from app.ml.evaluation import compute_classification_metrics

    texts, labels, prov = load_crisis_data()
    control_texts, control_labels = load_control_texts(args.control_samples, args.seed)
    texts = texts + control_texts
    labels = labels + control_labels

    print("[3/5] Splitting + featurizing ...")
    (tr_x, tr_y), (va_x, va_y), (te_x, te_y) = stratified_split(
        texts, labels, args.test_size, args.val_size, args.seed
    )
    print(f"      train={len(tr_x)} val={len(va_x)} test={len(te_x)} "
          f"train_dist={dict(Counter(tr_y))}")

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2), min_df=2, max_features=args.max_features,
                sublinear_tf=True, strip_accents="unicode",
            )),
            ("clf", LogisticRegression(
                max_iter=2000, class_weight="balanced", C=2.0, solver="lbfgs", random_state=args.seed,
            )),
        ]
    )

    print("[4/5] Training (TF-IDF + LogisticRegression, class-balanced) ...")
    t0 = time.time()
    pipeline.fit(tr_x, tr_y)
    train_time = time.time() - t0
    print(f"      trained in {train_time:.1f}s")

    va_proba = pipeline.predict_proba(va_x)
    cols = _class_columns(pipeline.classes_)
    thresholds = tune_thresholds(va_proba, va_y, cols)
    print(f"      tuned thresholds: {thresholds} (crisis recall-max, precision floor {PRECISION_FLOOR})")

    val_pred = predict_with_thresholds(va_proba, thresholds, cols)
    val_metrics = compute_classification_metrics(
        [CRISIS_CLASSES.index(l) for l in va_y],
        [CRISIS_CLASSES.index(l) for l in val_pred],
        CRISIS_CLASSES,
    )

    print("[5/5] Final evaluation on held-out test split ...")
    te_proba = pipeline.predict_proba(te_x)
    te_pred = predict_with_thresholds(te_proba, thresholds, cols)
    y_true_idx = [CRISIS_CLASSES.index(l) for l in te_y]
    y_pred_idx = [CRISIS_CLASSES.index(l) for l in te_pred]
    test_metrics = compute_classification_metrics(y_true_idx, y_pred_idx, CRISIS_CLASSES)

    # Binary crisis-vs-low view + false negatives (the safety-critical number)
    import sklearn

    le = sklearn.preprocessing.LabelEncoder().fit(CRISIS_CLASSES)
    crisis_true = np.array([l != "LOW" for l in te_y])
    crisis_pred = np.array([l != "LOW" for l in te_pred])
    false_negatives = [
        {"text_excerpt": t[:140], "true": y}
        for t, y, ct, cp in zip(te_x, te_y, crisis_true, crisis_pred)
        if ct and not cp
    ]
    from sklearn.metrics import precision_score, recall_score

    test_metrics["crisis_vs_low"] = {
        "recall": round(float(recall_score(crisis_true, crisis_pred, zero_division=0)), 4),
        "precision": round(float(precision_score(crisis_true, crisis_pred, zero_division=0)), 4),
        "false_negatives": len(false_negatives),
        "false_negative_examples": false_negatives[:10],
        "note": "False negatives are safety-critical; thresholds prioritise recall.",
    }

    # --- Save artifacts -------------------------------------------------
    out_dir = REPO_ROOT / args.output
    model_dir = out_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    with open(model_dir / "crisis_model.pkl", "wb") as fh:
        pickle.dump({"pipeline": pipeline, "classes": CRISIS_CLASSES,
                     "thresholds": thresholds, "class_columns": cols}, fh)

    labels_doc = {
        "classes": CRISIS_CLASSES,
        "id2label": {str(i): c for i, c in enumerate(CRISIS_CLASSES)},
        "severity_mapping": {str(k): v for k, v in SEVERITY_TO_CLASS.items()},
        "dataset": prov,
        "control_dataset": "SetFit/go_emotions test+validation splits (everyday non-crisis text)",
        "decision_rule": (
            "argmax-style thresholding: HIGH if P(HIGH)>=high_thr; else MODERATE if "
            "P(MODERATE)>=moderate_thr; else LOW. Thresholds tuned on a validation "
            f"split to maximise crisis recall subject to precision >= {PRECISION_FLOOR}."
        ),
        "thresholds": thresholds,
    }
    (out_dir / "labels.json").write_text(json.dumps(labels_doc, indent=2), encoding="utf-8")

    metrics = {
        "model": "TF-IDF + LogisticRegression (class-balanced)",
        "trained_on": prov,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "training": {
            "train_samples": len(tr_x),
            "val_samples": len(va_x),
            "test_samples": len(te_x),
            "max_features": args.max_features,
            "seed": args.seed,
            "train_seconds": round(train_time, 1),
            "thresholds": thresholds,
            "precision_floor": PRECISION_FLOOR,
        },
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    tm = test_metrics
    print("\n===== CRISIS MODEL RESULTS (test split) =====")
    print(f"accuracy     : {tm['accuracy']:.4f}   macro F1: {tm['macro_f1']:.4f}")
    for name, m in tm["per_class"].items():
        print(f"  {name:<9} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} n={m['support']}")
    print(f"crisis vs low: recall={tm['crisis_vs_low']['recall']:.4f} "
          f"precision={tm['crisis_vs_low']['precision']:.4f} "
          f"FN={tm['crisis_vs_low']['false_negatives']}")
    print(f"artifacts -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
