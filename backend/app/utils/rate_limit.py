"""Shared slowapi rate limiter (attached to app.state.limiter in main.py)."""
from __future__ import annotations

from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
