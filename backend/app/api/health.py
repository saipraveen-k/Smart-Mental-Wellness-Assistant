"""Liveness + readiness endpoint."""
from datetime import datetime, timezone
from typing import Any, Dict

from fastapi import APIRouter

from app.config import settings
from app.database.mongo import database
from app.services.model_manager import model_manager
from app.utils.errors import success

router = APIRouter(tags=["system"])


@router.get("/health")
def health() -> Dict[str, Any]:
    """Health of the backend, database and every model."""
    db_ok = database.ping()
    statuses = model_manager.get_model_status()
    models = statuses["models"]
    critical_ok = db_ok
    payload = {
        "status": "ok" if critical_ok else "degraded",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "environment": settings.app_env,
        "database": {"connected": db_ok, "name": settings.mongodb_database},
        "models": {
            name: {"available": m.get("available", False)}
            for name, m in models.items()
        },
        "device": statuses["device"]["selected"],
    }
    return success(payload)
