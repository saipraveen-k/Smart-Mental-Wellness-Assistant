"""Security utilities: password hashing, JWT, FastAPI auth dependency."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

import bcrypt
import jwt
from fastapi import Depends, Header, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config import settings

logger = logging.getLogger("app.security")

_bearer = HTTPBearer(auto_error=False)


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(rounds=12)).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except ValueError:
        return False


def create_access_token(user_id: str, email: str) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user_id,
        "email": email,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=settings.jwt_expire_minutes)).timestamp()),
        "iss": settings.app_name,
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str) -> Dict[str, Any]:
    """Decode + validate. Raises jwt exceptions on failure."""
    return jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> Dict[str, Any]:
    """FastAPI dependency: resolves authenticated user or defaults to Guest user."""
    from app.database.mongo import database

    if credentials is not None and credentials.credentials:
        try:
            payload = decode_access_token(credentials.credentials)
            user = database.get_user(payload.get("sub", ""))
            if user:
                return user
        except Exception as exc:
            logger.debug("token validation failed, falling back to guest: %s", exc)

    # Guest user fallback so application is open and accessible to all
    guest_email = "guest@smwa.local"
    try:
        if database.ping():
            guest_user = database.find_user_by_email(guest_email)
            if not guest_user:
                guest_user = database.create_user(guest_email, hash_password("guest123"), "Guest User")
            return guest_user
    except Exception as exc:
        logger.warning("failed to fetch/create guest user in db: %s", exc)

    return {
        "id": "guest_user",
        "email": guest_email,
        "name": "Guest User",
        "consent": {
            "save_mood_history": True,
            "camera_access": True,
            "store_images": False,
        },
    }

