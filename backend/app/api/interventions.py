"""Intervention recommendations + feedback learning endpoints."""
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, Query, Request

from app.database.mongo import database
from app.models.schemas import FeedbackRequest
from app.services.intervention_service import get_catalog, recommend_interventions
from app.utils.errors import ApiError, success
from app.utils.rate_limit import limiter
from app.utils.security import get_current_user

logger = logging.getLogger("app.interventions")

router = APIRouter(prefix="/api/interventions", tags=["interventions"])


@router.get("")
@limiter.limit("60/minute")
def list_interventions(
    request: Request,
    emotion: Optional[str] = Query(default=None),
    wellness_state: Optional[str] = Query(default=None, pattern="^(STABLE|NEEDS_ATTENTION|HIGH_CONCERN)$"),
    crisis_level: Optional[str] = Query(default="LOW", pattern="^(LOW|MODERATE|HIGH)$"),
    stressor: Optional[str] = Query(default=None),
    stressor_intensity: Optional[str] = Query(default=None, pattern="^(low|medium|high)$"),
    user: dict = Depends(get_current_user),
):
    """Personalized, feedback-adjusted intervention recommendations."""
    stats = database.feedback_stats(user["id"])
    result = recommend_interventions(
        emotion=emotion or "Neutral",
        wellness_state=wellness_state or "STABLE",
        crisis_level=crisis_level or "LOW",
        stressor=stressor,
        stressor_intensity=stressor_intensity,
        feedback_stats=stats,
    )
    # Record what was recommended (for auditability of the feedback loop).
    try:
        for item in result["interventions"]:
            database.insert_intervention(
                {
                    "user_id": user["id"],
                    "intervention_id": item["id"],
                    "priority": item["priority"],
                    "context": {
                        "emotion": emotion,
                        "wellness_state": wellness_state,
                        "crisis_level": crisis_level,
                        "stressor": stressor,
                    },
                }
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("intervention record not stored: %s", exc)
    return success(result)


@router.get("/catalog")
@limiter.limit("60/minute")
def catalog(request: Request, user: dict = Depends(get_current_user)):
    """The full intervention catalog (exercises and their content)."""
    return success({"interventions": get_catalog()})


@router.post("/feedback")
@limiter.limit("60/minute")
def submit_feedback(payload: FeedbackRequest, request: Request,
                    user: dict = Depends(get_current_user)):
    """Store intervention feedback; used to re-rank future recommendations."""
    doc_id = database.insert_feedback(
        {
            "user_id": user["id"],
            "intervention_id": payload.intervention_id,
            "rating": payload.rating,
            "context": payload.context or {},
        }
    )
    stats = database.feedback_stats(user["id"])
    return success(
        {
            "id": doc_id,
            "stored": True,
            "intervention_stats": stats.get(payload.intervention_id),
            "learning_note": (
                "Your feedback adjusts future recommendation priorities via a "
                "documented additive formula (not reinforcement learning)."
            ),
        }
    )
