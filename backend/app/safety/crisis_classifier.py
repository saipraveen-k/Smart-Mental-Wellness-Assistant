"""Crisis classifier — a dedicated risk screen, separate from the emotion model.

Loads the model trained by ``scripts/train_crisis.py``. If the artifacts are
missing, every call fails with MODEL_UNAVAILABLE: we NEVER substitute
heuristics or pretend a screening happened.

Conservative safety language applies everywhere this module's output is shown:
the model is a screening signal, not a guarantee, not a diagnosis, and not a
replacement for professional or emergency support.
"""
from __future__ import annotations

import logging
import pickle
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.config import settings

logger = logging.getLogger("app.crisis")

CRISIS_CLASSES = ["LOW", "MODERATE", "HIGH"]


class ModelUnavailableError(Exception):
    code = "MODEL_UNAVAILABLE"


class CrisisClassifier:
    """Lazy, thread-safe loader for the trained crisis model."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._payload: Optional[Dict[str, Any]] = None
        self._load_error: Optional[str] = None

    # ------------------------------------------------------------------ #
    def _model_path(self) -> Path:
        return settings.crisis_model_dir / "crisis_model.pkl"

    def available(self) -> bool:
        return self._model_path().exists()

    def load(self) -> Dict[str, Any]:
        with self._lock:
            if self._payload is not None:
                return self._payload
            path = self._model_path()
            if not path.exists():
                raise ModelUnavailableError(
                    "Crisis model is not currently available. "
                    "Train it with: python scripts/train_crisis.py"
                )
            try:
                with open(path, "rb") as fh:
                    payload = pickle.load(fh)
            except Exception as exc:  # noqa: BLE001
                self._load_error = str(exc)
                logger.error("crisis model load failed: %s", exc)
                raise ModelUnavailableError(
                    "Crisis model could not be loaded. Re-train with scripts/train_crisis.py"
                ) from exc
            self._payload = payload
            return payload

    def status(self) -> Dict[str, Any]:
        return {
            "name": "crisis_classifier",
            "type": "TF-IDF + LogisticRegression (class-balanced)",
            "path": str(self._model_path()),
            "available": self.available(),
            "load_error": self._load_error,
            "classes": CRISIS_CLASSES,
        }

    # ------------------------------------------------------------------ #
    def assess(self, text: str) -> Dict[str, Any]:
        """Screen text for crisis risk. Returns a conservative risk signal.

        Raises ModelUnavailableError when the trained model is absent — the
        caller must surface that error, never a fabricated "LOW".
        """
        payload = self.load()
        pipeline = payload["pipeline"]
        thresholds = payload["thresholds"]
        cols: Dict[str, int] = payload["class_columns"]

        t0 = time.perf_counter()
        proba = pipeline.predict_proba([text])[0]
        latency_ms = (time.perf_counter() - t0) * 1000.0

        p_high = float(proba[cols["HIGH"]])
        p_moderate = float(proba[cols["MODERATE"]])
        p_low = float(proba[cols["LOW"]])
        if p_high >= thresholds["high"]:
            level = "HIGH"
        elif p_moderate >= thresholds["moderate"]:
            level = "MODERATE"
        else:
            level = "LOW"
        confidence = {"HIGH": p_high, "MODERATE": p_moderate, "LOW": p_low}[level]

        return {
            "level": level,
            "confidence": round(float(confidence), 4),
            "probabilities": {
                "LOW": round(p_low, 4),
                "MODERATE": round(p_moderate, 4),
                "HIGH": round(p_high, 4),
            },
            "thresholds": thresholds,
            "signals": self._top_signals(pipeline, text, level),
            "model": "crisis_classifier",
            "latency_ms": round(latency_ms, 2),
            "screened_at": datetime.now(timezone.utc).isoformat(),
            "disclaimer": (
                "This is an AI screening signal, not a diagnosis or a guarantee. "
                "If you are in danger, contact local emergency services or a crisis line."
            ),
        }

    # ------------------------------------------------------------------ #
    @staticmethod
    def _top_signals(pipeline, text: str, level: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Real contributing features: TF-IDF value x logistic coefficient."""
        try:
            vectorizer = pipeline.named_steps["tfidf"]
            clf = pipeline.named_steps["clf"]
            feats = vectorizer.transform([text])
            row = feats.toarray()[0]
            class_idx = list(clf.classes_).index(level)
            coefs = clf.coef_[class_idx]
            contribution = row * coefs
            nz = feats.nonzero()[1]
            pairs = [
                (vectorizer.get_feature_names_out()[i], float(contribution[i]))
                for i in nz
            ]
            pairs.sort(key=lambda x: abs(x[1]), reverse=True)
            return [{"signal": f, "weight": round(w, 4)} for f, w in pairs[:top_k]]
        except Exception as exc:  # explanation must never break the safety path
            logger.warning("crisis signal explanation failed: %s", exc)
            return []

    predict = assess
    predict_text = assess


crisis_classifier = CrisisClassifier()

