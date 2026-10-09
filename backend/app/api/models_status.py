"""Model management status endpoint."""
from pathlib import Path

from fastapi import APIRouter, Depends, Request

from app.config import settings
from app.services.model_manager import model_manager
from app.utils.errors import success
from app.utils.rate_limit import limiter
from app.utils.security import get_current_user

router = APIRouter(prefix="/api/models", tags=["models"])


def _metrics_snapshot() -> dict:
    """Read real saved metrics from training artifacts (no hardcoded values)."""
    out = {}
    for name, path in (
        ("text_emotion", settings.text_model_dir / "metrics.json"),
        ("cv_emotion", Path(settings.cv_model_file).parent / "metrics.json"),
        ("crisis", settings.crisis_model_dir / "metrics.json"),
    ):
        p = Path(path)
        if p.exists():
            import json

            try:
                doc = json.loads(p.read_text(encoding="utf-8"))
                metrics = doc.get("test_metrics", {})
                out[name] = {
                    "generated_at": doc.get("generated_at"),
                    "accuracy": metrics.get("accuracy"),
                    "macro_f1": metrics.get("macro_f1"),
                    "weighted_f1": metrics.get("weighted_f1"),
                    "per_class": metrics.get("per_class"),
                }
            except Exception:  # noqa: BLE001
                out[name] = {"error": "METRICS_UNREADABLE"}
        else:
            out[name] = {"error": "NOT_TRAINED",
                         "message": "Model not trained yet. Run the training pipeline."}
    return out


@router.get("/status")
@limiter.limit("30/minute")
def models_status(request: Request, user: dict = Depends(get_current_user)):
    """Device, availability and real saved evaluation metrics per model."""
    status = model_manager.get_model_status()
    status["saved_metrics"] = _metrics_snapshot()
    return success(status)
