"""FastAPI application — Smart Mental Wellness Assistant backend.

Provides AI-assisted wellness support. It is NOT a medical diagnostic system,
therapist, or emergency service, and never claims diagnosis, clinical
accuracy, guaranteed crisis detection, or suicide prevention.
"""
from __future__ import annotations

import logging
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded

from app.api import (
    auth,
    chat,
    crisis,
    datasets,
    emotion,
    forecast,
    health,
    interventions,
    models_status,
    mood,
    multimodal,
    privacy,
    stressor,
    vision,
)
from app.config import settings
from app.database.mongo import database
from app.safety.resources import get_crisis_resources
from app.services.model_manager import model_manager
from app.utils.errors import failure, register_exception_handlers
from app.utils.logging_config import install_middleware, setup_logging
from app.utils.rate_limit import limiter

logger = logging.getLogger("app.main")

DESCRIPTION = """
AI-assisted wellness platform combining text emotion recognition,
facial-expression analysis, multimodal fusion, crisis screening, stressor
detection, mood history, forecasting, and personalized CBT-inspired /
mindfulness interventions.

**Safety:** This application provides AI-assisted wellness support and is not
a medical diagnostic system, therapist, or emergency service. Facial-expression
analysis estimates visible expressions only — it cannot determine a person's
true internal emotional state. Crisis screening is conservative but cannot
guarantee detection.
"""

lifespan_cm = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging(settings.log_level)
    logger.info("starting %s (%s)", settings.app_name, settings.app_env)

    # DB indexes in a worker thread so startup is not blocked for long.
    def _init_db():
        try:
            if database.ping():
                database.ensure_indexes()
            else:
                logger.error("mongodb unreachable at %s — running degraded", settings.mongodb_uri)
        except Exception as exc:  # noqa: BLE001
            logger.error("database init failed: %s", exc)

    threading.Thread(target=_init_db, name="db-init", daemon=True).start()
    # Model warm-up in the background; /api/models/status reflects progress.
    threading.Thread(target=model_manager.warm_up, name="model-warmup", daemon=True).start()
    yield
    logger.info("shutting down")
    model_manager.unload_models()


app = FastAPI(
    title=settings.app_name,
    description=DESCRIPTION,
    version="1.0.0",
    lifespan=lifespan,
    contact={"name": "Smart Mental Wellness Assistant"},
    license_info={"name": "Academic prototype — see docs/SAFETY.md"},
)

app.state.limiter = limiter
register_exception_handlers(app)
install_middleware(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["authorization", "content-type", "x-request-id"],
)


@app.exception_handler(RateLimitExceeded)
async def _rate_limited(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content=failure(
            "RATE_LIMITED",
            "Too many requests. Please slow down.",
            f"Limit: {exc.detail}. Try again in a moment.",
        ),
    )


@app.get("/api/safety/resources")
def safety_resources(request: Request):
    """Crisis resources for the deployment country (real, curated numbers)."""
    from app.utils.errors import success

    return success(get_crisis_resources())


for router in (
    health.router, auth.router, emotion.router, vision.router, multimodal.router,
    crisis.router, stressor.router, mood.router, forecast.router,
    interventions.router, datasets.router, models_status.router, privacy.router,
    chat.router,
):
    app.include_router(router)


@app.get("/")
def root():
    from app.utils.errors import success

    return success(
        {
            "service": settings.app_name,
            "docs": "/docs",
            "health": "/health",
            "disclaimer": (
                "AI-assisted wellness support — not a medical diagnostic system, "
                "therapist, or emergency service."
            ),
        }
    )
