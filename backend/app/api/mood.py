"""Mood history, trends and check-in saving (consent-gated)."""
import logging
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any, Dict

from fastapi import APIRouter, Depends, Query, Request

from app.database.mongo import database
from app.models.schemas import SaveCheckinRequest
from app.services.wellness_service import emotion_valence
from app.utils.errors import ApiError, success
from app.utils.rate_limit import limiter
from app.utils.security import get_current_user

logger = logging.getLogger("app.mood")

router = APIRouter(prefix="/api/mood", tags=["mood"])


@router.post("/save")
@limiter.limit("30/minute")
def save_checkin(
    payload: SaveCheckinRequest,
    request: Request,
    user: dict = Depends(get_current_user),
):
    """Store a check-in. Requires consent for mood-history storage."""
    if not user.get("consent", {}).get("save_mood_history", False):
        raise ApiError(
            "CONSENT_REQUIRED",
            "You have not allowed mood-history storage.",
            status_code=403,
            suggestion="Enable 'Save mood history' on the Privacy page, then retry.",
        )
    record = {
        "user_id": user["id"],
        "timestamp": datetime.now(timezone.utc),
        "text_emotion": payload.text_emotion,
        "text_confidence": payload.text_confidence,
        "vision_emotion": payload.vision_emotion,
        "vision_confidence": payload.vision_confidence,
        "fused_emotion": payload.fused_emotion,
        "fused_confidence": payload.fused_confidence,
        "valence": round(emotion_valence(payload.fused_emotion, payload.fused_confidence), 4),
        "stressor": payload.stressor,
        "severity": payload.severity,
        "wellness_state": payload.wellness_state,
        "crisis_level": payload.crisis_level,
        "intervention_id": payload.intervention_id,
    }
    record_id = database.insert_mood_record(record)
    return success({"id": record_id, "saved": True,
                    "timestamp": record["timestamp"].isoformat()})


@router.get("/history")
@limiter.limit("120/minute")
def mood_history(
    request: Request,
    limit: int = Query(default=200, ge=1, le=1000),
    user: dict = Depends(get_current_user),
):
    """Recent mood records for this user (empty list when none stored)."""
    records = database.mood_history(user["id"], limit=limit)
    return success(
        {
            "records": records,
            "count": len(records),
            "empty_message": (
                None if records else "No mood history yet. Start your first check-in."
            ),
        }
    )


@router.get("/trends")
@limiter.limit("60/minute")
def mood_trends(request: Request, user: dict = Depends(get_current_user)):
    """Real trend statistics computed from stored mood records."""
    records = database.mood_history(user["id"], limit=1000)
    if not records:
        return success(
            {
                "available": False,
                "message": "No mood history yet. Start your first check-in.",
                "records": 0,
            }
        )

    from app.forecasting.forecaster import analyze_trend, build_daily_series

    daily = build_daily_series(records)
    trend = analyze_trend(daily)

    emotion_counts = Counter(r["fused_emotion"] for r in records if r.get("fused_emotion"))
    stressor_counts = Counter(r["stressor"] for r in records if r.get("stressor"))

    now = datetime.now(timezone.utc)
    week_ago = (now - timedelta(days=7)).isoformat()
    two_weeks_ago = (now - timedelta(days=14)).isoformat()
    recent = [r for r in records if r["timestamp"] >= week_ago]
    previous = [r for r in records if two_weeks_ago <= r["timestamp"] < week_ago]

    def _mean_val(rs):
        vals = [r.get("valence") for r in rs if r.get("valence") is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    # Intervention effectiveness from real feedback
    feedback_stats = database.feedback_stats(user["id"])
    effectiveness = [
        {
            "intervention_id": iid,
            "ratings": s["ratings"],
            "total": s["total"],
            "mean_score": s["mean_score"],
        }
        for iid, s in feedback_stats.items()
    ]

    return success(
        {
            "available": True,
            "records": len(records),
            "daily_series": daily,
            "trend": trend,
            "emotion_distribution": dict(emotion_counts),
            "stressor_distribution": dict(stressor_counts),
            "weekly": {
                "recent_7d_mean_valence": _mean_val(recent),
                "previous_7d_mean_valence": _mean_val(previous),
                "recent_checkins": len(recent),
                "previous_checkins": len(previous),
            },
            "intervention_effectiveness": effectiveness,
        }
    )
