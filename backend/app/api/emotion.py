"""Text emotion analysis endpoint."""
import logging
from typing import Any, Dict

from fastapi import APIRouter, Request

from app.api._common import map_service_errors
from app.models.schemas import TextEmotionRequest
from app.services.text_service import text_analyzer
from app.utils.errors import success
from app.utils.rate_limit import limiter

logger = logging.getLogger("app.emotion")

router = APIRouter(prefix="/api/emotion", tags=["emotion"])


@router.post("/text")
@limiter.limit("60/minute")
def analyze_text(payload: TextEmotionRequest, request: Request):
    """Fine-tuned DistilBERT emotion analysis of a text (5-class taxonomy)."""
    with map_service_errors("Text analysis failed."):
        result = text_analyzer.analyze(payload.text, explain=payload.explain)
    return success(result)
