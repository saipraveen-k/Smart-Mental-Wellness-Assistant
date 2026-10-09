"""Mood forecasting endpoint."""
from fastapi import APIRouter, Depends, Query, Request

from app.config import settings
from app.database.mongo import database
from app.forecasting.forecaster import forecast
from app.utils.errors import success
from app.utils.rate_limit import limiter
from app.utils.security import get_current_user

router = APIRouter(prefix="/api/mood", tags=["forecast"])


@router.get("/forecast")
@limiter.limit("30/minute")
def mood_forecast(
    request: Request,
    horizon: int = Query(default=None, ge=1, le=30),
    user: dict = Depends(get_current_user),
):
    """Statistical mood forecast from this user's real check-in history.

    With fewer than FORECAST_MIN_OBSERVATIONS daily observations the response
    carries available=false and reason=INSUFFICIENT_DATA — no synthetic values.
    """
    records = database.mood_history(user["id"], limit=2000)
    result = forecast(records, horizon_days=horizon or settings.forecast_horizon_days)
    return success(result)
