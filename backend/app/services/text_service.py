"""Text emotion inference service.

Loads the fine-tuned DistilBERT (artifacts/text_emotion) once and reuses it.
If the trained model is missing, every call raises MODEL_UNAVAILABLE — we
never substitute random or heuristic predictions and present them as the
trained model.
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from app.config import settings
from app.ml.emotion_mapping import WELLNESS_LABELS, top_label
from app.ml.text_preprocessing import clean_text, is_meaningful

logger = logging.getLogger("app.text_service")


class ModelUnavailableError(Exception):
    code = "MODEL_UNAVAILABLE"


class InvalidInputError(Exception):
    code = "INVALID_INPUT"


class TextEmotionAnalyzer:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._model = None
        self._tokenizer = None
        self._load_error: Optional[str] = None

    # ------------------------------------------------------------------ #
    def model_dir(self) -> Path:
        return settings.artifacts_dir / "text_emotion"

    def available(self) -> bool:
        model_dir = self.model_dir() / "model"
        tok_dir = self.model_dir() / "tokenizer"
        return (model_dir / "config.json").exists() and (
            (tok_dir / "tokenizer_config.json").exists() or (tok_dir / "config.json").exists()
        )

    def load(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            if not self.available():
                raise ModelUnavailableError(
                    "Text emotion model is not currently available. "
                    "Train it with: python scripts/train_text.py"
                )
            try:
                import torch
                from transformers import AutoModelForSequenceClassification, AutoTokenizer

                base = self.model_dir()
                self._tokenizer = AutoTokenizer.from_pretrained(base / "tokenizer")
                self._model = AutoModelForSequenceClassification.from_pretrained(base / "model")
                self._model.eval()
                self._device = torch.device("cpu")
                logger.info("text emotion model loaded from %s on %s", base, self._device)
            except Exception as exc:  # noqa: BLE001
                self._load_error = str(exc)
                logger.error("failed to load text emotion model: %s", exc)
                raise ModelUnavailableError(
                    "Text emotion model could not be loaded. Re-train with scripts/train_text.py"
                ) from exc

    def status(self) -> Dict[str, Any]:
        return {
            "name": "text_emotion",
            "type": "DistilBERT (fine-tuned, 5-class wellness taxonomy)",
            "path": str(self.model_dir()),
            "available": self.available(),
            "loaded": self._model is not None,
            "load_error": self._load_error,
            "labels": list(WELLNESS_LABELS),
        }

    # ------------------------------------------------------------------ #
    def analyze(self, text: str, explain: bool = True, top_k: int = 5) -> Dict[str, Any]:
        """Predict the wellness emotion distribution of a text."""
        self.load()
        cleaned = clean_text(text)
        if not is_meaningful(cleaned):
            raise InvalidInputError("Text is empty or too short to analyze.")
        if len(cleaned) > settings.max_text_length:
            raise InvalidInputError(
                f"Text exceeds the maximum length of {settings.max_text_length} characters."
            )

        import torch

        t0 = time.perf_counter()
        enc = self._tokenizer(
            cleaned, return_tensors="pt", truncation=True, max_length=64
        )
        tokenize_ms = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        with torch.no_grad():
            out = self._model(
                input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]
            )
        inference_ms = (time.perf_counter() - t1) * 1000.0

        probs = torch.softmax(out.logits, dim=-1)[0]
        distribution = {
            label: round(float(probs[i]), 4) for i, label in enumerate(WELLNESS_LABELS)
        }
        emotion, confidence = top_label(distribution)

        explanation = None
        explain_ms = 0.0
        if explain:
            from app.ml.explainability import text_token_attribution

            t2 = time.perf_counter()
            try:
                attribution = text_token_attribution(
                    self._model, self._tokenizer, cleaned, max_length=64, top_k=top_k
                )
                explanation = {
                    "method": attribution["method"],
                    "contributing_tokens": attribution["attributions"],
                    "note": (
                        "Tokens with the largest gradient-based attribution for the "
                        "predicted class. Attribution indicates which parts of your "
                        "text influenced the model, not why you feel that way."
                    ),
                }
            except Exception as exc:  # noqa: BLE001 - explanation is optional
                logger.warning("text attribution failed: %s", exc)
                explanation = {"method": None, "contributing_tokens": [],
                               "error": "EXPLANATION_UNAVAILABLE"}
            explain_ms = (time.perf_counter() - t2) * 1000.0

        return {
            "emotion": emotion,
            "confidence": confidence,
            "distribution": distribution,
            "explanation": explanation,
            "latency_ms": {
                "tokenize": round(tokenize_ms, 2),
                "inference": round(inference_ms, 2),
                "explanation": round(explain_ms, 2),
                "total": round(tokenize_ms + inference_ms + explain_ms, 2),
            },
            "model": "distilbert_text_emotion",
            "analyzed_at": datetime.now(timezone.utc).isoformat(),
        }


    predict_emotion = analyze


text_analyzer = TextEmotionAnalyzer()
text_service = text_analyzer

