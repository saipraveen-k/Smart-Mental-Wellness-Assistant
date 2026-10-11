"""Facial-expression analysis service (computer vision).

Honest framing enforced here and in the UI: this module estimates a visible
FACIAL EXPRESSION. It cannot determine a person's true internal emotional
state. Raw camera frames are processed transiently and discarded — never
stored.
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.config import settings
from app.cv.image_utils import crop_and_normalize, decode_image, detect_faces
from app.ml.emotion_mapping import (
    CV_DISPLAY_LABELS,
    FER2013_LABELS,
    WELLNESS_LABELS,
    map_cv_distribution,
    top_label,
)

logger = logging.getLogger("app.vision_service")


class ModelUnavailableError(Exception):
    code = "MODEL_UNAVAILABLE"


class VisionAnalysisError(Exception):
    def __init__(self, reason: str, message: str, details: Optional[Dict] = None):
        super().__init__(message)
        self.reason = reason
        self.code = reason
        self.message = message
        self.details = details or {}


class FacialExpressionAnalyzer:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._model = None
        self._load_error: Optional[str] = None
        self._labels: List[str] = list(FER2013_LABELS)

    # ------------------------------------------------------------------ #
    def available(self) -> bool:
        return Path(settings.cv_model_file).exists()

    def load(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            if not self.available():
                raise ModelUnavailableError(
                    "Facial-expression model is not currently available. "
                    "Train it with: python scripts/train_cv.py"
                )
            try:
                import torch

                from app.cv.model import FacialEmotionCNN

                payload = torch.load(settings.cv_model_file, map_location="cpu", weights_only=True)
                model = FacialEmotionCNN.from_config(payload["arch"])
                model.load_state_dict(payload["model_state_dict"])
                model.eval()
                self._model = model
                labels_path = settings.cv_model_file.parent / "labels.json"
                if labels_path.exists():
                    import json

                    doc = json.loads(labels_path.read_text(encoding="utf-8"))
                    self._labels = [doc["id2label"][str(i)] for i in range(len(doc["id2label"]))]
                logger.info("CV model loaded from %s", settings.cv_model_file)
            except Exception as exc:  # noqa: BLE001
                self._load_error = str(exc)
                logger.error("failed to load CV model: %s", exc)
                raise ModelUnavailableError(
                    "Facial-expression model could not be loaded. Re-train with scripts/train_cv.py"
                ) from exc

    def status(self) -> Dict[str, Any]:
        return {
            "name": "cv_emotion",
            "type": "FacialEmotionCNN (FER-2013, 7 expression classes)",
            "path": str(settings.cv_model_file),
            "available": self.available(),
            "loaded": self._model is not None,
            "load_error": self._load_error,
            "labels": list(self._labels),
            "framing": "facial-expression analysis — not true emotion detection",
        }

    # ------------------------------------------------------------------ #
    def analyze_image(
        self,
        image_bytes: bytes,
        face_index: Optional[int] = None,
        explain: bool = True,
    ) -> Dict[str, Any]:
        """Analyze the facial expression in an uploaded/captured image."""
        self.load()

        if not image_bytes:
            raise VisionAnalysisError("EMPTY_IMAGE", "No image data received.")
        if len(image_bytes) > settings.max_image_bytes:
            raise VisionAnalysisError(
                "IMAGE_TOO_LARGE",
                f"Image exceeds the {settings.max_image_bytes // (1024 * 1024)} MB limit.",
            )

        t0 = time.perf_counter()
        image = decode_image(image_bytes)
        if image is None:
            raise VisionAnalysisError("INVALID_IMAGE", "The file is not a decodable image.")

        faces = detect_faces(image)
        decode_ms = (time.perf_counter() - t0) * 1000.0
        if not faces:
            raise VisionAnalysisError(
                "NO_FACE_DETECTED",
                "No face was detected in the image. Try a clearer, front-facing photo.",
                {"faces_found": 0},
            )

        # Multi-face policy (documented): we never silently pick a random
        # person. By default the request is rejected with the face count; the
        # caller may explicitly allow the primary face (largest by area) or
        # pass face_index.
        if len(faces) > 1 and face_index is None:
            raise VisionAnalysisError(
                "MULTIPLE_FACES_DETECTED",
                f"{len(faces)} faces were detected. Please use a single-person image, "
                "or explicitly select the primary face (largest detected face).",
                {"faces_found": len(faces), "faces": faces[:5],
                 "primary_face_policy": "largest face by pixel area"},
            )
        if face_index is not None:
            if face_index < 0 or face_index >= len(faces):
                raise VisionAnalysisError(
                    "FACE_INDEX_OUT_OF_RANGE",
                    f"face_index {face_index} out of range for {len(faces)} detected faces.",
                    {"faces_found": len(faces)},
                )
            box = faces[face_index]
        else:
            box = faces[0]  # documented primary-face policy: largest by area

        face_crop = crop_and_normalize(image, box)

        import torch

        t1 = time.perf_counter()
        tensor = torch.from_numpy(face_crop.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            logits = self._model(tensor)
        probs_t = torch.softmax(logits, dim=-1)[0]
        inference_ms = (time.perf_counter() - t1) * 1000.0

        expr_distribution = {
            self._labels[i]: round(float(probs_t[i]), 4) for i in range(len(self._labels))
        }
        expr_label, expr_conf = top_label(expr_distribution)
        wellness_distribution = map_cv_distribution(expr_distribution)
        wellness_label, wellness_conf = top_label(wellness_distribution)

        gradcam = None
        explain_ms = 0.0
        if explain:
            from app.ml.explainability import grad_cam

            t2 = time.perf_counter()
            try:
                cam = grad_cam(self._model, tensor, target_class=self._labels.index(expr_label))
                flat = cam.reshape(-1)
                top_pixels = np.argsort(flat)[::-1][:5]
                gradcam = {
                    "method": "grad_cam",
                    "heatmap_shape": list(cam.shape),
                    "peak_intensity": round(float(flat.max()), 4),
                    "mean_intensity": round(float(flat.mean()), 4),
                    "note": (
                        "Grad-CAM heatmap over the face crop. Brighter regions contributed "
                        "more to the predicted expression class."
                    ),
                }
            except Exception as exc:  # noqa: BLE001
                logger.warning("grad-cam failed: %s", exc)
                gradcam = {"method": None, "error": "EXPLANATION_UNAVAILABLE"}
            explain_ms = (time.perf_counter() - t2) * 1000.0

        return {
            "face": {
                "detected": True,
                "faces_found": len(faces),
                "selected_face": box,
                "selection_policy": (
                    "user-selected" if face_index is not None
                    else ("primary face: largest detected face by pixel area" if len(faces) > 1
                          else "single detected face")
                ),
            },
            "expression": {
                "label": expr_label,
                "display_label": CV_DISPLAY_LABELS.get(expr_label, expr_label.title()),
                "confidence": expr_conf,
                "distribution": expr_distribution,
            },
            "wellness_mapping": {
                "emotion": wellness_label,
                "confidence": wellness_conf,
                "distribution": wellness_distribution,
            },
            "explanation": gradcam,
            "latency_ms": {
                "decode_and_detect": round(decode_ms, 2),
                "inference": round(inference_ms, 2),
                "explanation": round(explain_ms, 2),
                "total": round(decode_ms + inference_ms + explain_ms, 2),
            },
            "image_retention": "processed transiently and discarded (not stored)",
            "model": "facial_expression_cnn",
            "analyzed_at": datetime.now(timezone.utc).isoformat(),
            "framing": (
                "Facial-expression estimate only: expressions do not reliably "
                "indicate a person's true internal emotional state."
            ),
        }


    analyze_image_bytes = analyze_image


vision_analyzer = FacialExpressionAnalyzer()
vision_service = vision_analyzer

