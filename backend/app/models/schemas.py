"""Pydantic request/response schemas with validation."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, EmailStr, Field, field_validator

from app.config import settings


# ---------------------------------------------------------------------- #
# Auth
# ---------------------------------------------------------------------- #
class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128,
                          description="At least 8 characters.")
    name: str = Field(min_length=1, max_length=80)

    @field_validator("password")
    @classmethod
    def _password_strength(cls, v: str) -> str:
        if v.isdigit() or v.isalpha() and v.islower() and v.isdigit() == 0 and len(set(v)) < 4:
            raise ValueError("Password is too simple; mix letters and numbers.")
        return v


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=128)


class AuthResponse(BaseModel):
    token: str
    user: Dict[str, Any]


# ---------------------------------------------------------------------- #
# Analysis
# ---------------------------------------------------------------------- #
class TextEmotionRequest(BaseModel):
    text: str = Field(min_length=1, max_length=settings.max_text_length)
    explain: bool = True


class VisionAnalyzeRequest(BaseModel):
    """Optional JSON fields when the client posts multipart form data."""
    face_index: Optional[int] = Field(default=None, ge=0, le=20)
    explain: bool = True


class MultimodalRequest(BaseModel):
    text: Optional[str] = Field(default=None, max_length=settings.max_text_length)
    image_base64: Optional[str] = Field(default=None, description="Base64-encoded image (data URL or raw b64).")
    face_index: Optional[int] = Field(default=None, ge=0, le=20)
    explain: bool = True
    save_checkin: bool = False

    @field_validator("text", "image_base64")
    @classmethod
    def _strip_if_present(cls, v: Optional[str]) -> Optional[str]:
        return v.strip() if isinstance(v, str) else v

    @field_validator("image_base64")
    @classmethod
    def _image_required_if_no_text(cls, v: Optional[str], info) -> Optional[str]:
        return v


class CrisisRequest(BaseModel):
    text: str = Field(min_length=1, max_length=settings.max_text_length)


class StressorRequest(BaseModel):
    text: str = Field(min_length=1, max_length=settings.max_text_length)


# ---------------------------------------------------------------------- #
# Mood / check-in
# ---------------------------------------------------------------------- #
class MoodRecordOut(BaseModel):
    id: str
    timestamp: str
    fused_emotion: Optional[str] = None
    fused_confidence: Optional[float] = None
    wellness_state: Optional[str] = None
    stressor: Optional[str] = None


class SaveCheckinRequest(BaseModel):
    text_emotion: Optional[str] = None
    text_confidence: Optional[float] = Field(default=None, ge=0, le=1)
    vision_emotion: Optional[str] = None
    vision_confidence: Optional[float] = Field(default=None, ge=0, le=1)
    fused_emotion: str
    fused_confidence: float = Field(ge=0, le=1)
    stressor: Optional[str] = None
    severity: Optional[str] = Field(default=None, pattern="^(low|medium|high)$")
    wellness_state: Optional[str] = None
    crisis_level: Optional[str] = Field(default=None, pattern="^(LOW|MODERATE|HIGH)$")
    intervention_id: Optional[str] = None


class FeedbackRequest(BaseModel):
    intervention_id: str = Field(min_length=1, max_length=64)
    rating: str = Field(pattern="^(much_better|a_little_better|no_change|worse)$")
    notes: Optional[str] = None
    context: Optional[Dict[str, Any]] = None

    @field_validator("rating", mode="before")
    @classmethod
    def _normalize_rating(cls, v: str) -> str:
        if isinstance(v, str):
            return v.strip().lower().replace(" ", "_").replace("-", "_")
        return v


class InterventionListResponse(BaseModel):
    crisis_mode: bool
    headline: str
    interventions: List[Dict[str, Any]]


# ---------------------------------------------------------------------- #
# Privacy
# ---------------------------------------------------------------------- #
class ConsentRequest(BaseModel):
    save_mood_history: Optional[bool] = None
    camera_access: Optional[bool] = None
    store_images: Optional[bool] = None


# ---------------------------------------------------------------------- #
# Generic envelopes
# ---------------------------------------------------------------------- #
class ErrorDetail(BaseModel):
    code: str
    message: str
    suggestion: Optional[str] = None


class ApiResponse(BaseModel):
    success: bool
    data: Optional[Any] = None
    error: Optional[ErrorDetail] = None
