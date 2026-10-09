"""Facial-expression analysis endpoint (image upload / camera capture).

Privacy: images are processed transiently and discarded; frames are never
written to disk or the database.
"""
import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, File, Form, Request, UploadFile

from app.api._common import map_service_errors
from app.config import settings
from app.services.vision_service import vision_analyzer
from app.utils.errors import success
from app.utils.rate_limit import limiter

logger = logging.getLogger("app.vision")

router = APIRouter(prefix="/api/vision", tags=["vision"])

ALLOWED_MIME_PREFIXES = ("image/",)


@router.post("/analyze")
@limiter.limit("30/minute")
async def analyze_image(
    request: Request,
    file: UploadFile = File(..., description="JPEG/PNG image of a single face"),
    face_index: Optional[int] = Form(default=None),
    explain: bool = Form(default=True),
):
    """Detect a face and estimate the visible facial expression."""
    content_type = (file.content_type or "").lower()
    if not any(content_type.startswith(p) for p in ALLOWED_MIME_PREFIXES):
        from app.utils.errors import ApiError

        raise ApiError(
            "INVALID_IMAGE_TYPE",
            f"Unsupported content type {content_type!r}. Upload a JPEG or PNG image.",
            status_code=415,
            suggestion="Re-export the file as JPEG or PNG.",
        )

    data = await file.read()
    if len(data) > settings.max_image_bytes:
        from app.utils.errors import ApiError

        raise ApiError(
            "IMAGE_TOO_LARGE",
            f"Image exceeds the {settings.max_image_bytes // (1024 * 1024)} MB limit.",
            status_code=413,
            suggestion="Resize or compress the image and try again.",
        )

    with map_service_errors("Facial-expression analysis failed."):
        result = vision_analyzer.analyze_image(
            data,
            face_index=face_index,
            explain=explain,
        )
    return success(result)
