"""AI Assistant endpoint — the full check-in pipeline in one call.

    text (+ optional image)
      -> emotion analysis (text and/or facial expression)
      -> multimodal fusion
      -> crisis screening (safety override when MODERATE/HIGH)
      -> stressor detection
      -> wellness indicator
      -> personalized interventions
      -> optional consent-gated save
"""
import base64
import binascii
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, Request

from app.api._common import map_service_errors
from app.database.mongo import database
from app.services.fusion_service import fuse_distributions
from app.services.intervention_service import recommend_interventions
from app.safety.crisis_classifier import crisis_classifier
from app.safety.stressor import detect_stressor
from app.services.text_service import text_analyzer
from app.services.vision_service import vision_analyzer
from app.services.wellness_service import compute_wellness_state, emotion_valence
from app.utils.errors import ApiError, success
from app.utils.rate_limit import limiter
from app.utils.security import get_current_user

logger = logging.getLogger("app.chat")

router = APIRouter(prefix="/api/chat", tags=["chat"])

CRISIS_RESOURCES_MESSAGE = (
    "We're concerned that you may be going through a serious situation. "
    "You don't have to handle this alone."
)


def _decode_base64_image(image_base64: str) -> bytes:
    raw = image_base64.strip()
    if raw.startswith("data:"):
        raw = raw.split(",", 1)[1]
    try:
        return base64.b64decode(raw, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ApiError("INVALID_IMAGE", "Image is not valid base64.", status_code=400) from exc


@router.post("/message")
@limiter.limit("30/minute")
def chat_message(payload: dict, request: Request, user: dict = Depends(get_current_user)):
    """Run the complete check-in analysis pipeline.

    Body: {"text": str, "image_base64": str?, "save_checkin": bool?, "explain": bool?}
    """
    text = (payload.get("text") or "").strip() or None
    image_b64 = payload.get("image_base64")
    save_checkin = bool(payload.get("save_checkin", False))
    explain = bool(payload.get("explain", True))
    if not text and not image_b64:
        raise ApiError("NO_INPUT", "Provide a message or an image to analyze.", status_code=400)

    timings: Dict[str, float] = {}
    text_signal: Optional[Dict[str, Any]] = None
    vision_signal: Optional[Dict[str, Any]] = None
    crisis_result: Optional[Dict[str, Any]] = None
    stressor_result: Optional[Dict[str, Any]] = None

    with map_service_errors("The assistant could not complete the analysis."):
        # --- 1. Crisis screening first (safety priority) -----------------
        if text:
            try:
                crisis_result = crisis_classifier.assess(text)
            except Exception as exc:  # noqa: BLE001 — surface, never fake
                logger.error("crisis screening failed: %s", exc)
                raise ApiError(
                    "CRISIS_MODEL_UNAVAILABLE",
                    "Crisis screening is unavailable right now, so we can't complete "
                    "your check-in safely.",
                    status_code=503,
                    suggestion="Try again shortly, or contact support resources directly.",
                ) from exc

        # --- 2. Emotion analysis ----------------------------------------
        if text:
            try:
                t0 = text_analyzer.analyze(text, explain=explain)
                text_signal = {
                    "emotion": t0["emotion"], "confidence": t0["confidence"],
                    "distribution": t0["distribution"], "explanation": t0["explanation"],
                    "latency_ms": t0["latency_ms"]["total"],
                }
                timings["text_ms"] = t0["latency_ms"]["total"]
            except Exception as exc:
                if not image_b64:
                    raise
                logger.warning("text modality failed: %s", exc)

        if image_b64:
            image_bytes = _decode_base64_image(image_b64)
            try:
                v0 = vision_analyzer.analyze_image(image_bytes, explain=explain)
                vision_signal = {
                    "expression": v0["expression"],
                    "emotion": v0["wellness_mapping"]["emotion"],
                    "confidence": v0["wellness_mapping"]["confidence"],
                    "distribution": v0["wellness_mapping"]["distribution"],
                    "explanation": v0["explanation"],
                    "latency_ms": v0["latency_ms"]["total"],
                    "framing": v0["framing"],
                }
                timings["vision_ms"] = v0["latency_ms"]["total"]
            except Exception as exc:
                if not text:
                    raise
                logger.warning("vision modality failed: %s", exc)
                vision_signal = {"error": getattr(exc, "code", "VISION_ANALYSIS_FAILED"),
                                 "message": getattr(exc, "message", None)}

        text_dist = text_signal.get("distribution") if text_signal else None
        vision_dist = vision_signal.get("distribution") if vision_signal else None
        fused = fuse_distributions(text_dist, vision_dist)
        timings["fusion_ms"] = fused["fusion"]["latency_ms"]

        # --- 3. Stressors ------------------------------------------------
        if text:
            stressor_result = detect_stressor(text)

        # --- 4. Wellness indicator ---------------------------------------
        from app.forecasting.forecaster import analyze_trend, build_daily_series

        history = database.mood_history(user["id"], limit=100)
        trend = analyze_trend(build_daily_series(history)) if history else {"available": False}
        declining = bool(trend.get("available") and trend.get("declining"))

        wellness = compute_wellness_state(
            emotion=fused["emotion"],
            confidence=fused["confidence"],
            crisis_level=crisis_result["level"] if crisis_result else "LOW",
            stressor_intensity=stressor_result.get("intensity") if stressor_result else None,
            trend_declining=declining,
        )

        # --- 5. Interventions --------------------------------------------
        stats = database.feedback_stats(user["id"])
        interventions = recommend_interventions(
            emotion=fused["emotion"],
            wellness_state=wellness["state"],
            crisis_level=crisis_result["level"] if crisis_result else "LOW",
            stressor=stressor_result.get("category") if stressor_result else None,
            stressor_intensity=stressor_result.get("intensity") if stressor_result else None,
            trend_declining=declining,
            feedback_stats=stats,
        )

    timings["total_ms"] = round(sum(timings.values()), 2)

    # --- 6. Safety resources when crisis is indicated --------------------
    crisis_resources = None
    if crisis_result and crisis_result["level"] in ("MODERATE", "HIGH"):
        from app.safety.resources import get_crisis_resources

        crisis_resources = get_crisis_resources()
        try:
            database.insert_crisis_event(
                {"user_id": user["id"], "level": crisis_result["level"],
                 "probabilities": crisis_result["probabilities"], "source": "chat"}
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("crisis event not stored: %s", exc)

    # --- 7. Optional consent-gated save ----------------------------------
    saved = None
    if save_checkin:
        consent = user.get("consent", {})
        if consent.get("save_mood_history"):
            record_id = database.insert_mood_record(
                {
                    "user_id": user["id"],
                    "timestamp": datetime.now(timezone.utc),
                    "text_emotion": text_signal.get("emotion") if text_signal else None,
                    "text_confidence": text_signal.get("confidence") if text_signal else None,
                    "vision_emotion": vision_signal.get("emotion") if vision_signal else None,
                    "vision_confidence": vision_signal.get("confidence") if vision_signal else None,
                    "fused_emotion": fused["emotion"],
                    "fused_confidence": fused["confidence"],
                    "valence": round(
                        emotion_valence(fused["emotion"], fused["confidence"]), 4
                    ),
                    "stressor": stressor_result.get("category") if stressor_result else None,
                    "severity": stressor_result.get("intensity") if stressor_result else None,
                    "wellness_state": wellness["state"],
                    "crisis_level": crisis_result["level"] if crisis_result else None,
                    "intervention_id": interventions["interventions"][0]["id"]
                    if interventions["interventions"] else None,
                }
            )
            saved = {"saved": True, "id": record_id}
        else:
            saved = {
                "saved": False,
                "reason": "CONSENT_NOT_GRANTED",
                "message": "Mood history is disabled in your privacy settings.",
            }

    return success(
        {
            "crisis": crisis_result,
            "crisis_resources": crisis_resources,
            "emotion": fused["emotion"],
            "confidence": fused["confidence"],
            "distribution": fused["distribution"],
            "text_signal": text_signal,
            "vision_signal": vision_signal,
            "fusion": fused["fusion"],
            "stressor": stressor_result,
            "wellness": wellness,
            "interventions": interventions,
            "trend_context": {"declining": declining} if trend.get("available") else None,
            "saved": saved,
            "latency_ms": timings,
            "disclaimer": (
                "AI-assisted wellness support — not a medical diagnostic system, "
                "therapist, or emergency service."
            ),
        }
    )
