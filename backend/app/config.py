"""Application settings.

All configuration comes from environment variables (optionally loaded from a
`.env` file at the repository root). No secrets are hard-coded.
"""
from __future__ import annotations

import os
import secrets
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")


def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be a number, got {raw!r}") from exc


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer, got {raw!r}") from exc


def _resolve_path(raw: str, default: Path) -> Path:
    if raw and raw.strip():
        p = Path(raw.strip())
        return p if p.is_absolute() else (REPO_ROOT / p)
    return default


@dataclass
class Settings:
    # App
    app_env: str = "development"
    app_name: str = "Smart Mental Wellness Assistant"
    log_level: str = "INFO"

    # Database
    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_database: str = "mental_wellness"
    mongodb_timeout_ms: int = 5000

    # Datasets
    text_dataset_id: str = "SetFit/go_emotions"
    cv_dataset_id: str = "Jeneral/fer-2013"
    crisis_dataset_id: str = "av9ash/CSSR-S_labelled_suicidewatch_posts_reddit"
    crisis_control_dataset_id: str = "SetFit/go_emotions"
    hf_dataset_revision: str = "main"
    hf_cache_dir: str = ""

    # Models
    text_model_name: str = "distilbert-base-uncased"
    text_model_path: str = "ml/artifacts/text_emotion/model"
    cv_model_path: str = "ml/artifacts/cv_emotion/model.pt"
    crisis_model_path: str = "ml/artifacts/crisis/model"
    model_device: str = "auto"

    # Fusion
    text_modality_weight: float = 0.65
    vision_modality_weight: float = 0.35

    # Feature switches
    enable_camera: bool = True
    enable_forecasting: bool = True
    save_images_by_default: bool = False

    # Forecasting
    forecast_min_observations: int = 10
    forecast_horizon_days: int = 7

    # Security
    jwt_secret: str = ""
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 1440
    cors_origins: List[str] = field(default_factory=lambda: ["http://localhost:5173"])
    rate_limit_per_minute: int = 60
    auth_rate_limit_per_minute: int = 10
    max_image_bytes: int = 4 * 1024 * 1024
    max_text_length: int = 4000

    # Safety
    crisis_resources_country: str = "US"
    crisis_resources_file: Path = field(default_factory=lambda: REPO_ROOT / "config" / "crisis_resources.json")

    # Paths
    artifacts_dir: Path = field(default_factory=lambda: REPO_ROOT / "ml" / "artifacts")

    # Derived
    repo_root: Path = REPO_ROOT

    @property
    def text_model_dir(self) -> Path:
        p = Path(self.text_model_path)
        return p if p.is_absolute() else (REPO_ROOT / p)

    @property
    def cv_model_file(self) -> Path:
        p = Path(self.cv_model_path)
        return p if p.is_absolute() else (REPO_ROOT / p)

    @property
    def crisis_model_dir(self) -> Path:
        p = Path(self.crisis_model_path)
        return p if p.is_absolute() else (REPO_ROOT / p)

    @property
    def is_production(self) -> bool:
        return self.app_env.lower() == "production"

    @property
    def hf_cache(self) -> Optional[str]:
        return self.hf_cache_dir or None


def load_settings() -> Settings:
    settings = Settings(
        app_env=os.getenv("APP_ENV", "development"),
        app_name=os.getenv("APP_NAME", "Smart Mental Wellness Assistant"),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        mongodb_uri=os.getenv("MONGODB_URI", "mongodb://localhost:27017"),
        mongodb_database=os.getenv("MONGODB_DATABASE", "mental_wellness"),
        mongodb_timeout_ms=_get_int("MONGODB_TIMEOUT_MS", 5000),
        text_dataset_id=os.getenv("TEXT_DATASET_ID", "SetFit/go_emotions"),
        cv_dataset_id=os.getenv("CV_DATASET_ID", "Jeneral/fer-2013"),
        crisis_dataset_id=os.getenv("CRISIS_DATASET_ID", "av9ash/CSSR-S_labelled_suicidewatch_posts_reddit"),
        crisis_control_dataset_id=os.getenv("CRISIS_CONTROL_DATASET_ID", "SetFit/go_emotions"),
        hf_dataset_revision=os.getenv("HF_DATASET_REVISION", "main"),
        hf_cache_dir=os.getenv("HF_CACHE_DIR", ""),
        text_model_name=os.getenv("TEXT_MODEL_NAME", "distilbert-base-uncased"),
        text_model_path=os.getenv("TEXT_MODEL_PATH", "ml/artifacts/text_emotion/model"),
        cv_model_path=os.getenv("CV_MODEL_PATH", "ml/artifacts/cv_emotion/model.pt"),
        crisis_model_path=os.getenv("CRISIS_MODEL_PATH", "ml/artifacts/crisis/model"),
        model_device=os.getenv("MODEL_DEVICE", "auto"),
        text_modality_weight=_get_float("TEXT_MODALITY_WEIGHT", 0.65),
        vision_modality_weight=_get_float("VISION_MODALITY_WEIGHT", 0.35),
        enable_camera=_get_bool("ENABLE_CAMERA", True),
        enable_forecasting=_get_bool("ENABLE_FORECASTING", True),
        save_images_by_default=_get_bool("SAVE_IMAGES_BY_DEFAULT", False),
        forecast_min_observations=_get_int("FORECAST_MIN_OBSERVATIONS", 10),
        forecast_horizon_days=_get_int("FORECAST_HORIZON_DAYS", 7),
        jwt_secret=os.getenv("JWT_SECRET", "CHANGE_ME"),
        jwt_algorithm=os.getenv("JWT_ALGORITHM", "HS256"),
        jwt_expire_minutes=_get_int("JWT_EXPIRE_MINUTES", 1440),
        cors_origins=[
            o.strip()
            for o in os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",")
            if o.strip()
        ],
        rate_limit_per_minute=_get_int("RATE_LIMIT_PER_MINUTE", 60),
        auth_rate_limit_per_minute=_get_int("AUTH_RATE_LIMIT_PER_MINUTE", 10),
        max_image_bytes=_get_int("MAX_IMAGE_BYTES", 4 * 1024 * 1024),
        max_text_length=_get_int("MAX_TEXT_LENGTH", 4000),
        crisis_resources_country=os.getenv("CRISIS_RESOURCES_COUNTRY", "US"),
        crisis_resources_file=_resolve_path(
            os.getenv("CRISIS_RESOURCES_FILE", "config/crisis_resources.json"),
            REPO_ROOT / "config" / "crisis_resources.json",
        ),
        artifacts_dir=_resolve_path(os.getenv("ARTIFACTS_DIR", "ml/artifacts"), REPO_ROOT / "ml" / "artifacts"),
    )

    # --- Validation -----------------------------------------------------
    total = settings.text_modality_weight + settings.vision_modality_weight
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"TEXT_MODALITY_WEIGHT + VISION_MODALITY_WEIGHT must sum to 1.0 (got {total:.4f})."
        )

    if settings.jwt_secret == "CHANGE_ME" or not settings.jwt_secret:
        if settings.is_production:
            raise RuntimeError(
                "JWT_SECRET must be set to a strong random value in production. "
                "Generate one with: python -c \"import secrets; print(secrets.token_urlsafe(48))\""
            )
        # Development: ephemeral secret. Tokens will not survive restarts.
        settings.jwt_secret = secrets.token_urlsafe(48)
        print(
            "[config] WARNING: JWT_SECRET not set — using an ephemeral development secret. "
            "Set JWT_SECRET in .env before deploying.",
            file=sys.stderr,
        )

    return settings


settings = load_settings()
