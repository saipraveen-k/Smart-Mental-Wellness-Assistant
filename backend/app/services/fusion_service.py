"""Multimodal fusion: text emotion signal + facial-expression signal.

Fusion formula (weights configurable via TEXT_MODALITY_WEIGHT /
VISION_MODALITY_WEIGHT, enforced to sum to 1.0 at config load):

    final[class] = w_text * text_dist[class] + w_vision * vision_dist[class]

then renormalized. Both distributions are mapped into the same 5-class
wellness taxonomy before fusion. Missing modalities are never fabricated:
with one modality present, that modality's (renormalized) distribution is the
result and the response says which modality was used.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from app.config import settings
from app.ml.emotion_mapping import WELLNESS_LABELS, top_label

logger = logging.getLogger("app.fusion")


def fuse_distributions(
    text_distribution: Optional[Dict[str, float]],
    vision_distribution: Optional[Dict[str, float]],
    text_weight: Optional[float] = None,
    vision_weight: Optional[float] = None,
) -> Dict[str, Any]:
    """Fuse two wellness-taxonomy distributions with configurable weights."""
    t0 = time.perf_counter()

    if text_distribution is None and vision_distribution is None:
        raise ValueError("At least one modality distribution is required for fusion")

    w_t = settings.text_modality_weight if text_weight is None else float(text_weight)
    w_v = settings.vision_modality_weight if vision_weight is None else float(vision_weight)
    total_w = w_t + w_v
    if total_w <= 0:
        raise ValueError("Fusion weights must be positive")
    w_t, w_v = w_t / total_w, w_v / total_w

    def _norm(dist: Dict[str, float]) -> Dict[str, float]:
        missing = [l for l in WELLNESS_LABELS if l not in dist]
        if missing:
            raise ValueError(f"Distribution missing wellness labels: {missing}")
        s = sum(dist[l] for l in WELLNESS_LABELS)
        if s <= 0:
            raise ValueError("Distribution sums to zero")
        return {l: dist[l] / s for l in WELLNESS_LABELS}

    if text_distribution is not None and vision_distribution is not None:
        mode = "multimodal"
        t_dist, v_dist = _norm(text_distribution), _norm(vision_distribution)
        fused = {
            label: w_t * t_dist[label] + w_v * v_dist[label] for label in WELLNESS_LABELS
        }
        fused = {k: v / sum(fused.values()) for k, v in fused.items()}
    elif text_distribution is not None:
        mode = "text_only"
        fused = _norm(text_distribution)
        w_t, w_v = 1.0, 0.0
    else:
        mode = "vision_only"
        fused = _norm(vision_distribution)  # type: ignore[arg-type]
        w_t, w_v = 0.0, 1.0

    emotion, confidence = top_label(fused)
    fusion_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "emotion": emotion,
        "confidence": round(float(confidence), 4),
        "distribution": {k: round(v, 4) for k, v in fused.items()},
        "fusion": {
            "mode": mode,
            "text_weight": round(w_t, 4),
            "vision_weight": round(w_v, 4),
            "formula": "final = w_text*text_dist + w_vision*vision_dist (renormalized)",
            "weights_source": "TEXT_MODALITY_WEIGHT / VISION_MODALITY_WEIGHT env config",
            "latency_ms": round(fusion_ms, 3),
        },
        "fused_at": datetime.now(timezone.utc).isoformat(),
    }


compute_fusion = fuse_distributions

