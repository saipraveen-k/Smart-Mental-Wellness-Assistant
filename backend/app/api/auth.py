"""Authentication endpoints (register / login)."""
import logging

from fastapi import APIRouter, Request

from app.config import settings
from app.database.mongo import database
from app.models.schemas import LoginRequest, RegisterRequest
from app.utils.errors import ApiError, success
from app.utils.rate_limit import limiter
from app.utils.security import create_access_token, hash_password, verify_password

logger = logging.getLogger("app.auth")

router = APIRouter(prefix="/api/auth", tags=["auth"])


def _public_user(doc) -> dict:
    return {
        "id": doc["id"],
        "email": doc["email"],
        "name": doc["name"],
        "consent": doc.get("consent", {}),
        "created_at": doc["created_at"].isoformat() if doc.get("created_at") else None,
    }


@router.post("/register")
@limiter.limit(f"{settings.auth_rate_limit_per_minute}/minute")
def register(payload: RegisterRequest, request: Request):
    """Create an account. Passwords are bcrypt-hashed; never stored in plain text."""
    existing = database.find_user_by_email(payload.email)
    if existing:
        raise ApiError("EMAIL_ALREADY_REGISTERED",
                       "An account with this email already exists.", status_code=409,
                       suggestion="Try logging in instead.")
    try:
        user = database.create_user(payload.email, hash_password(payload.password), payload.name)
    except ValueError as exc:
        if str(exc) == "EMAIL_ALREADY_REGISTERED":
            raise ApiError("EMAIL_ALREADY_REGISTERED",
                           "An account with this email already exists.", status_code=409) from exc
        raise
    token = create_access_token(user["id"], user["email"])
    logger.info("user registered user_id=%s", user["id"])
    return success({"token": token, "user": _public_user(user)})


@router.post("/guest")
@router.get("/guest")
def guest_session():
    """Create or retrieve a guest session instantly without credentials."""
    guest_email = "guest@smwa.local"
    try:
        if database.ping():
            guest_user = database.find_user_by_email(guest_email)
            if not guest_user:
                guest_user = database.create_user(guest_email, hash_password("guest123"), "Guest User")
            token = create_access_token(guest_user["id"], guest_user["email"])
            return success({"token": token, "user": _public_user(guest_user)})
    except Exception as exc:
        logger.warning("guest session fallback due to error: %s", exc)

    guest_dict = {
        "id": "guest_user",
        "email": guest_email,
        "name": "Guest User",
        "consent": {"save_mood_history": True, "camera_access": True, "store_images": False},
        "created_at": None,
    }
    token = create_access_token("guest_user", guest_email)
    return success({"token": token, "user": guest_dict})

