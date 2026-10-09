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


@router.post("/login")
@limiter.limit(f"{settings.auth_rate_limit_per_minute}/minute")
def login(payload: LoginRequest, request: Request):
    """Authenticate and receive a JWT."""
    user = database.find_user_by_email(payload.email)
    if not user or not verify_password(payload.password, user.get("password_hash", "")):
        raise ApiError("INVALID_CREDENTIALS", "Email or password is incorrect.", status_code=401,
                       suggestion="Check your credentials and try again.")
    token = create_access_token(user["id"], user["email"])
    return success({"token": token, "user": _public_user(user)})
