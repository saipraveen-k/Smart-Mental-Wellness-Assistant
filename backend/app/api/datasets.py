"""Dataset status / refresh endpoints (dynamic Hugging Face datasets)."""
import logging

from fastapi import APIRouter, Depends, Request

from app.api._common import map_service_errors
from app.database.mongo import database
from app.ml.dataset_manager import dataset_manager
from app.utils.errors import success
from app.utils.rate_limit import limiter
from app.utils.security import get_current_user

logger = logging.getLogger("app.datasets_api")

router = APIRouter(prefix="/api/datasets", tags=["datasets"])


@router.get("/status")
@limiter.limit("30/minute")
def datasets_status(request: Request, user: dict = Depends(get_current_user)):
    """Provenance of every dataset loaded so far (lazy: loads on demand)."""
    with map_service_errors("Dataset status unavailable."):
        status = dataset_manager.get_status()
    return success(status)


@router.get("/statistics/{kind}")
@limiter.limit("10/minute")
def dataset_statistics(kind: str, request: Request, user: dict = Depends(get_current_user)):
    """Live-computed statistics for 'text' or 'cv' (never hardcoded)."""
    if kind not in ("text", "cv"):
        from app.utils.errors import ApiError

        raise ApiError("INVALID_DATASET_KIND", "kind must be 'text' or 'cv'.", status_code=404)
    with map_service_errors("Dataset statistics unavailable."):
        stats = dataset_manager.get_dataset_statistics(kind)
    return success(stats)


@router.post("/refresh")
@limiter.limit("2/minute")
def refresh_datasets(request: Request, user: dict = Depends(get_current_user)):
    """Force a fresh download of both datasets (revalidation included)."""
    with map_service_errors("Dataset refresh failed."):
        text_prov = dataset_manager.load_text_dataset(refresh=True)
        cv_prov = dataset_manager.load_cv_dataset(refresh=True)
        status = dataset_manager.get_status()
    try:
        database.insert_dataset_run({"kind": "text", **status["datasets"]["text"]})
        database.insert_dataset_run({"kind": "cv", **status["datasets"]["cv"]})
    except Exception as exc:  # noqa: BLE001
        logger.warning("dataset run not recorded: %s", exc)
    return success({"refreshed": True, "datasets": status["datasets"]})


@router.post("/cache/clear")
@limiter.limit("2/minute")
def clear_cache(request: Request, user: dict = Depends(get_current_user)):
    """Clear local dataset caches (they will re-download on next use)."""
    with map_service_errors("Cache clear failed."):
        results = {
            "text": dataset_manager.clear_dataset_cache("text"),
            "cv": dataset_manager.clear_dataset_cache("cv"),
        }
    return success(results)
