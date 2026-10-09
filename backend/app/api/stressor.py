"""Stressor detection endpoint."""
from fastapi import APIRouter, Request

from app.models.schemas import StressorRequest
from app.safety.stressor import STRESSOR_CATEGORIES, detect_stressor
from app.utils.errors import success
from app.utils.rate_limit import limiter

router = APIRouter(prefix="/api/stressor", tags=["stressor"])


@router.post("/analyze")
@limiter.limit("60/minute")
def analyze_stressor(payload: StressorRequest, request: Request):
    """Rule-based stressor detection with matched-cue explanations."""
    result = detect_stressor(payload.text)
    result["categories"] = list(STRESSOR_CATEGORIES)
    return success(result)
