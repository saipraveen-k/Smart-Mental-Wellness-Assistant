"""Crisis screening endpoint (dedicated safety subsystem)."""
import logging

from fastapi import APIRouter, Depends, Request

from app.api._common import map_service_errors
from app.database.mongo import database
from app.models.schemas import CrisisRequest
from app.safety.crisis_classifier import crisis_classifier
from app.utils.errors import success
from app.utils.rate_limit import limiter
from app.utils.security import get_current_user

logger = logging.getLogger("app.crisis_api")

router = APIRouter(prefix="/api/crisis", tags=["crisis"])


@router.post("/analyze")
@limiter.limit("30/minute")
def analyze_crisis(
    payload: CrisisRequest,
    request: Request,
    user: dict = Depends(get_current_user),
):
    """Screen text for crisis risk with the dedicated crisis model.

    Conservative by design: when MODERATE/HIGH risk is indicated the client
    must surface crisis resources and stop casual recommendations. The app
    never claims guaranteed detection or suicide prevention.
    """
    with map_service_errors("Crisis screening failed."):
        result = crisis_classifier.assess(payload.text)

    # Record the event (level + probabilities only — never the raw text).
    try:
        database.insert_crisis_event(
            {
                "user_id": user["id"],
                "level": result["level"],
                "probabilities": result["probabilities"],
                "model": result["model"],
            }
        )
    except Exception as exc:  # noqa: BLE001 - never block the safety response
        logger.warning("crisis event not stored: %s", exc)

    return success(result)
