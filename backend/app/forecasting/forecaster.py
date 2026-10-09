"""Mood forecasting — ARIMA / Holt-Winters on REAL stored mood history.

Valence mapping (documented): each check-in's fused emotion+confidence maps to
a valence in [-1, +1]:

    Joy      +1.0 * confidence
    Neutral   0.0
    Anger    -0.5 * confidence
    Sadness  -0.6 * confidence
    Anxiety  -0.7 * confidence

Daily series = mean valence per calendar day. Forecasting only runs when at
least FORECAST_MIN_OBSERVATIONS real daily observations exist; otherwise the
caller receives an explicit INSUFFICIENT_DATA response. No synthetic history
is ever invented.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from app.config import settings
from app.services.wellness_service import emotion_valence

logger = logging.getLogger("app.forecasting")

STRESS_THRESHOLD = -0.5  # forecast day below this valence is flagged


def build_daily_series(records: List[Dict]) -> List[Dict]:
    """Aggregate raw mood records into a daily valence series (real data)."""
    by_day: Dict[str, List[float]] = defaultdict(list)
    for rec in records:
        emotion = rec.get("fused_emotion")
        conf = float(rec.get("fused_confidence") or 0.0)
        ts = rec.get("timestamp")
        if isinstance(ts, str):
            try:
                ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except ValueError:
                continue
        if ts is None or emotion is None:
            continue
        day = ts.astimezone(timezone.utc).date().isoformat()
        by_day[day].append(emotion_valence(emotion, conf))
    return [
        {"date": day, "valence": round(sum(vals) / len(vals), 4), "checkins": len(vals)}
        for day, vals in sorted(by_day.items())
    ]


def analyze_trend(series: List[Dict]) -> Dict[str, Any]:
    """Real trend statistics: slope, window means, weekday patterns."""
    if len(series) < 2:
        return {"available": False, "reason": "INSUFFICIENT_DATA"}

    vals = [p["valence"] for p in series]
    n = len(vals)
    import numpy as np

    x = np.arange(n, dtype=np.float64)
    y = np.array(vals, dtype=np.float64)
    slope, intercept = np.polyfit(x, y, 1)
    recent7 = float(np.mean(vals[-7:])) if n >= 7 else float(np.mean(vals))
    prev7 = float(np.mean(vals[-14:-7])) if n >= 14 else float(np.mean(vals[:-7])) if n > 7 else None
    change = None if prev7 is None else round(recent7 - prev7, 4)
    declining = bool(slope < -0.02 and (change is None or change < 0))

    weekday_means: Dict[str, Optional[float]] = {}
    by_weekday: Dict[int, List[float]] = defaultdict(list)
    for p in series:
        d = datetime.fromisoformat(p["date"])
        by_weekday[d.weekday()].append(p["valence"])
    for i, name in enumerate(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]):
        weekday_means[name] = round(float(np.mean(by_weekday[i])), 4) if by_weekday.get(i) else None
    patterned = [k for k, v in weekday_means.items() if v is not None and v < -0.3]
    recurring = (
        {"weakest_weekdays": patterned,
         "note": "Weekdays whose mean valence is below -0.3 across your history."}
        if patterned and n >= 14
        else None
    )

    return {
        "available": True,
        "slope_per_day": round(float(slope), 5),
        "direction": "declining" if slope < -0.02 else ("improving" if slope > 0.02 else "flat"),
        "recent_7d_mean": round(recent7, 4),
        "previous_7d_mean": None if prev7 is None else round(prev7, 4),
        "change_vs_previous_7d": change,
        "declining": declining,
        "weekday_means": weekday_means,
        "recurring_patterns": recurring,
    }


def forecast(series: List[Dict], horizon_days: Optional[int] = None) -> Dict[str, Any]:
    """Forecast future daily valence with ARIMA / Holt-Winters (statsmodels).

    Returns an explicit INSUFFICIENT_DATA payload when fewer than
    FORECAST_MIN_OBSERVATIONS real daily observations exist. Never fabricates.
    """
    horizon = horizon_days or settings.forecast_horizon_days
    min_obs = settings.forecast_min_observations

    daily = build_daily_series(series)
    trend = analyze_trend(daily) if daily else {"available": False}

    base = {
        "daily_series": daily,
        "trend": trend,
        "min_observations_required": min_obs,
        "disclaimer": (
            "Forecasts are statistical projections from your own check-in history. "
            "They are not predictions of clinical outcomes."
        ),
    }

    if not settings.enable_forecasting:
        base.update({"available": False, "reason": "FORECASTING_DISABLED"})
        return base

    if len(daily) < min_obs:
        base.update(
            {
                "available": False,
                "reason": "INSUFFICIENT_DATA",
                "observations": len(daily),
                "message": (
                    "Not enough mood history yet. Continue checking in to unlock "
                    "personalized trend analysis."
                ),
            }
        )
        return base

    import numpy as np

    y = np.array([p["valence"] for p in daily], dtype=np.float64)
    method = None
    fitted = None
    predicted = None

    # Prefer ARIMA(1,1,0); fall back to Holt's linear trend; then to a naive
    # drift estimator. If everything fails we report FORECAST_UNAVAILABLE —
    # we do NOT invent numbers.
    try:
        from statsmodels.tsa.arima.model import ARIMA

        model = ARIMA(y, order=(1, 1, 0))
        res = model.fit()
        predicted = np.asarray(res.forecast(horizon))
        fitted = np.asarray(res.fittedvalues)
        method = "ARIMA(1,1,0)"
    except Exception as exc:  # noqa: BLE001
        logger.warning("ARIMA failed (%s); trying Holt", exc)
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing

            model = ExponentialSmoothing(y, trend="add", seasonal=None)
            res = model.fit()
            predicted = np.asarray(res.forecast(horizon))
            fitted = np.asarray(res.fittedvalues)
            method = "Holt linear trend"
        except Exception as exc2:  # noqa: BLE001
            logger.warning("Holt failed (%s); falling back to drift", exc2)
            try:
                slope = float((y[-1] - y[0]) / max(1, len(y) - 1))
                predicted = np.array([y[-1] + slope * (i + 1) for i in range(horizon)])
                fitted = None
                method = "linear drift (fallback)"
            except Exception as exc3:  # noqa: BLE001
                logger.error("all forecasting methods failed: %s", exc3)
                base.update({"available": False, "reason": "FORECAST_UNAVAILABLE",
                             "message": "Forecasting failed on your history; try again later."})
                return base

    last_date = datetime.fromisoformat(daily[-1]["date"])
    forecast_days = []
    for i, value in enumerate(predicted):
        day = (last_date + timedelta(days=i + 1)).isoformat()
        v = float(np.clip(value, -1.0, 1.0))
        forecast_days.append(
            {
                "date": day,
                "forecast_valence": round(v, 4),
                "potential_stress_period": v <= STRESS_THRESHOLD,
            }
        )

    return {
        **base,
        "available": True,
        "method": method,
        "observations": len(daily),
        "horizon_days": horizon,
        "forecast": forecast_days,
        "stress_periods": [d["date"] for d in forecast_days if d["potential_stress_period"]],
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
