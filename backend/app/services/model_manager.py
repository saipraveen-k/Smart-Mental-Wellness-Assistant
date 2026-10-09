"""Model Manager — loads once, reuses everywhere.

Responsibilities:
  * load_text_model / load_cv_model / load_crisis_model
  * unload_models
  * get_model_status
  * device detection (CUDA when available, else CPU)
"""
from __future__ import annotations

import gc
import logging
from datetime import datetime, timezone
from typing import Any, Dict

from app.config import settings
from app.safety.crisis_classifier import crisis_classifier
from app.services.text_service import text_analyzer
from app.services.vision_service import vision_analyzer

logger = logging.getLogger("app.model_manager")


def detect_device() -> str:
    """CUDA when available (and not disabled), otherwise CPU."""
    if settings.model_device == "cpu":
        return "cpu"
    try:
        import torch

        if settings.model_device == "cuda":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001
        return "cpu"


def _torch_info() -> Dict[str, Any]:
    """Torch/device info that never crashes the API (import failures reported)."""
    for attempt in (1, 2):
        try:
            import torch

            cuda_available = torch.cuda.is_available()
            selected = detect_device()
            return {
                "selected": selected,
                "cuda_available": cuda_available,
                "cuda_device_name": torch.cuda.get_device_name(0) if cuda_available else None,
                "note": ("CUDA unavailable — using CPU fallback." if not cuda_available else None),
                "torch_version": torch.__version__,
            }
        except Exception as exc:  # noqa: BLE001
            if attempt == 2:
                logger.error("torch unavailable: %s", exc)
                return {
                    "selected": "unknown",
                    "cuda_available": False,
                    "cuda_device_name": None,
                    "torch_version": None,
                    "error": f"torch unavailable: {exc}",
                }


class ModelManager:
    def load_text_model(self) -> None:
        text_analyzer.load()

    def load_cv_model(self) -> None:
        vision_analyzer.load()

    def load_crisis_model(self) -> None:
        crisis_classifier.load()

    def load_stressor_model(self) -> Dict[str, Any]:
        """Stressor detection is rule-based (documented); nothing to load."""
        from app.safety.stressor import STRESSOR_CATEGORIES

        return {"name": "stressor_detector", "type": "rule_based",
                "categories": list(STRESSOR_CATEGORIES)}

    def unload_models(self) -> None:
        text_analyzer._model = None
        vision_analyzer._model = None
        crisis_classifier._payload = None
        gc.collect()
        logger.info("all models unloaded")

    def get_model_status(self) -> Dict[str, Any]:
        return {
            "device": _torch_info(),
            "models": {
                "text_emotion": text_analyzer.status(),
                "cv_emotion": vision_analyzer.status(),
                "crisis": crisis_classifier.status(),
                "stressor": self.load_stressor_model(),
            },
            "checked_at": datetime.now(timezone.utc).isoformat(),
        }

    def warm_up(self) -> Dict[str, Any]:
        """Load every available model at startup; record per-model outcome."""
        results: Dict[str, Any] = {}
        for name, loader in (
            ("text_emotion", self.load_text_model),
            ("cv_emotion", self.load_cv_model),
            ("crisis", self.load_crisis_model),
        ):
            try:
                loader()
                results[name] = "loaded"
            except Exception as exc:  # noqa: BLE001 - reported, never hidden
                results[name] = f"unavailable: {exc}"
                logger.warning("model warm-up failed for %s: %s", name, exc)
        logger.info("model warm-up complete: %s", results)
        return results


model_manager = ModelManager()
