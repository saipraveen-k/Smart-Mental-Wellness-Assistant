"""Multimodal analysis: text + facial expression -> fused wellness signal."""
import base64
import binascii
import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Request

from app.api._common import map_service_errors
from app.models.schemas import MultimodalRequest
from app.services.fusion_service import fuse_distributions
from app.services.text_service import text_analyzer
from app.services.vision_service import vision_analyzer
from app.utils.errors import ApiError, success
from app.utils.rate_limit import limiter

logger = logging.getLogger("app.multimodal")

router = APIRouter(prefix="/api/multimodal", tags=["multimodal"])


def _decode_base64_image(image_base64: str) -> bytes:
    raw = image_base64.strip()
    if raw.startswith("data:"):
        try:
            raw = raw.split(",", 1)[1]
        except IndexError:
            raise ApiError("INVALID_IMAGE", "Malformed data URL.", status_code=400)
    try:
        return base64.b64decode(raw, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ApiError("INVALID_IMAGE", "Image is not valid base64.", status_code=400,
                       suggestion="Re-capture the image and try again.") from exc


@router.post("/analyze")
@limiter.limit("30/minute")
def analyze_multimodal(payload: MultimodalRequest, request: Request):
    """Fuse text and/or facial-expression signals into one estimate.

    Missing modalities are never fabricated: with one modality available the
    response says mode=text_only or mode=vision_only.
    """
    if not payload.text and not payload.image_base64:
        raise ApiError(
            "NO_MODALITY_PROVIDED",
            "Provide at least a text message or an image to analyze.",
            status_code=400,
            suggestion="Type a message, upload an image, or both.",
        )

    timings: Dict[str, float] = {}
    text_signal: Optional[Dict[str, Any]] = None
    vision_signal: Optional[Dict[str, Any]] = None

    with map_service_errors("Multimodal analysis failed."):
        if payload.text:
            try:
                text_result = text_analyzer.analyze(payload.text, explain=payload.explain)
                text_signal = {
                    "emotion": text_result["emotion"],
                    "confidence": text_result["confidence"],
                    "distribution": text_result["distribution"],
                    "explanation": text_result["explanation"],
                    "latency_ms": text_result["latency_ms"]["total"],
                }
                timings["text_ms"] = text_result["latency_ms"]["total"]
            except Exception as exc:
                if payload.image_base64 is None:
                    raise  # with no fallback modality the error must surface
                logger.warning("text modality failed, continuing with vision only: %s", exc)
                text_signal = {"error": getattr(exc, "code", "TEXT_ANALYSIS_FAILED")}

        if payload.image_base64:
            image_bytes = _decode_base64_image(payload.image_base64)
            try:
                vision_result = vision_analyzer.analyze_image(
                    image_bytes, face_index=payload.face_index, explain=payload.explain
                )
                vision_signal = {
                    "expression": vision_result["expression"],
                    "emotion": vision_result["wellness_mapping"]["emotion"],
                    "confidence": vision_result["wellness_mapping"]["confidence"],
                    "distribution": vision_result["wellness_mapping"]["distribution"],
                    "explanation": vision_result["explanation"],
                    "latency_ms": vision_result["latency_ms"]["total"],
                    "framing": vision_result["framing"],
                }
                timings["vision_ms"] = vision_result["latency_ms"]["total"]
            except Exception as exc:
                if payload.text is None:
                    raise
                code = getattr(exc, "code", None)
                logger.warning("vision modality failed, continuing with text only: %s", exc)
                vision_signal = {
                    "error": code or "VISION_ANALYSIS_FAILED",
                    "message": getattr(exc, "message", None),
                }

    text_dist = text_signal.get("distribution") if text_signal and "distribution" in text_signal else None
    vision_dist = vision_signal.get("distribution") if vision_signal and "distribution" in vision_signal else None
    if text_dist is None and vision_dist is None:
        raise ApiError("NO_MODALITY_AVAILABLE",
                       "Neither modality produced a result.", status_code=400,
                       suggestion="Check the image and text and try again.")

    fused = fuse_distributions(text_dist, vision_dist)
    timings["fusion_ms"] = fused["fusion"]["latency_ms"]
    timings["total_ms"] = round(sum(timings.values()), 2)

    from app.services.wellness_service import compute_wellness_state
    from app.services.intervention_service import recommend_interventions

    wellness = compute_wellness_state(
        emotion=fused["emotion"],
        confidence=fused["confidence"],
    )
    rec_interventions = recommend_interventions(
        emotion=fused["emotion"],
        wellness_state=wellness["state"],
    )

    return success(
        {
            "emotion": fused["emotion"],
            "fused_emotion": fused["emotion"],
            "confidence": fused["confidence"],
            "fused_confidence": fused["confidence"],
            "distribution": fused["distribution"],
            "text_signal": text_signal,
            "vision_signal": vision_signal,
            "fusion": fused["fusion"],
            "wellness_state": wellness["state"],
            "wellness": wellness,
            "recommended_intervention": (
                rec_interventions["interventions"][0]
                if rec_interventions.get("interventions")
                else None
            ),
            "interventions": rec_interventions.get("interventions", []),
            "latency_ms": timings,
            "disclaimer": (
                "AI-estimated wellness signal, not a medical diagnosis. "
                "Facial-expression analysis estimates visible expressions only."
            ),
        }
    )
