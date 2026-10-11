"""Shared helpers for API routes: service-error -> ApiError mapping."""
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

from app.utils.errors import ApiError


@contextmanager
def map_service_errors(user_hint: str = "The analysis could not be completed.") -> Iterator[None]:
    """Translate service-layer exceptions into consistent API errors."""
    try:
        yield
    except ApiError:
        raise
    except Exception as exc:  # noqa: BLE001
        code = getattr(exc, "code", None) or getattr(exc, "reason", None)
        message = getattr(exc, "message", None) or str(exc)
        suggestion = getattr(exc, "suggestion", None)

        if code == "MODEL_UNAVAILABLE":
            raise ApiError(
                "MODEL_UNAVAILABLE", message, status_code=503,
                suggestion=suggestion or "Run the training pipeline (scripts/).",
            ) from exc
        if code in ("NO_FACE_DETECTED", "MULTIPLE_FACES_DETECTED", "INVALID_IMAGE",
                    "IMAGE_TOO_LARGE", "EMPTY_IMAGE", "FACE_INDEX_OUT_OF_RANGE"):
            details = getattr(exc, "details", {}) or {}
            raise ApiError(
                code, message, status_code=400,
                suggestion=suggestion or details.get("primary_face_policy"),
            ) from exc
        if code == "INVALID_INPUT":
            raise ApiError("INVALID_INPUT", message, status_code=400,
                           suggestion="Provide meaningful text to analyze.") from exc
        if code in ("DATASET_UNAVAILABLE", "DATASET_INVALID"):
            raise ApiError(code, message, status_code=503,
                           suggestion="Check network connectivity or the configured dataset revision.") from exc
        if code == "DATABASE_UNAVAILABLE":
            raise ApiError("DATABASE_UNAVAILABLE",
                           "The database is not reachable right now.", status_code=503,
                           suggestion="Try again shortly; if it persists, check MONGODB_URI.") from exc
        raise ApiError("INTERNAL_ERROR", f"{user_hint} Unexpected error.", status_code=500) from exc
