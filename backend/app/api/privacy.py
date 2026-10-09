"""Privacy controls: consent management, data export, data deletion."""
import logging

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from app.database.mongo import database
from app.models.schemas import ConsentRequest
from app.utils.errors import ApiError, success
from app.utils.rate_limit import limiter
from app.utils.security import get_current_user

logger = logging.getLogger("app.privacy")

router = APIRouter(prefix="/api/privacy", tags=["privacy"])


@router.get("/consent")
@limiter.limit("60/minute")
def get_consent(request: Request, user: dict = Depends(get_current_user)):
    """Current consent status for this account."""
    return success(
        {
            "consent": user.get("consent", {}),
            "data_we_store": [
                "account email and name",
                "mood check-in records (only when mood-history consent is enabled)",
                "intervention feedback",
                "crisis screening events (level and probabilities only — never your text)",
            ],
            "data_we_never_store": [
                "camera frames or uploaded images (processed transiently, then discarded)",
                "raw conversation text in logs or crisis events",
            ],
        }
    )


@router.put("/consent")
@limiter.limit("30/minute")
def update_consent(payload: ConsentRequest, request: Request,
                   user: dict = Depends(get_current_user)):
    """Update consent flags. Disabling mood history stops new writes."""
    updates = {k: v for k, v in payload.model_dump().items() if v is not None}
    if not updates:
        raise ApiError("NO_CONSENT_FIELDS", "Provide at least one consent field.", status_code=400)
    consent = database.update_consent(user["id"], {**user.get("consent", {}), **updates})
    logger.info("consent updated user_id=%s fields=%s", user["id"], sorted(updates))
    return success({"consent": consent})


@router.get("/export")
@limiter.limit("5/minute")
def export_data(request: Request, user: dict = Depends(get_current_user)):
    """Download everything stored about this account (JSON)."""
    data = database.export_user_data(user["id"])
    return JSONResponse(
        content=success(data),
        headers={"content-disposition": 'attachment; filename="my_wellness_data.json"'},
    )


@router.delete("/data")
@limiter.limit("5/minute")
def delete_data(request: Request, user: dict = Depends(get_current_user)):
    """Permanently delete this account and all associated records."""
    counts = database.delete_user_data(user["id"])
    logger.info("user data deleted user_id=%s", user["id"])
    return success({"deleted": True, "counts": counts,
                    "note": "Your account and all associated records were permanently deleted."})
