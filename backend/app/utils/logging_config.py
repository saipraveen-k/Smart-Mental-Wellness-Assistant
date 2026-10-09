"""Structured (JSON) logging + observability middleware.

Tracks request_id, timestamp, endpoint, latency, status, model usage and
error codes. Raw user conversation text is NEVER logged — only lengths and
outcome codes.
"""
from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Optional

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", request_id_var.get()),
        }
        for key in ("endpoint", "latency_ms", "status", "model", "error_code", "user_id"):
            val = getattr(record, key, None)
            if val is not None:
                payload[key] = val
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def setup_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    root.setLevel(level.upper())
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root.handlers = [handler]
    # Keep noisy third-party loggers manageable
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("multipart").setLevel(logging.WARNING)


def install_middleware(app) -> None:
    """Request-id + latency + status logging middleware."""
    from app.database.mongo import database

    @app.middleware("http")
    async def _observability(request: Request, call_next):  # noqa: ANN001
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        request.state.request_id = request_id
        token = request_id_var.set(request_id)
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            latency = (time.perf_counter() - start) * 1000
            logging.getLogger("app.access").info(
                "request failed",
                extra={"endpoint": request.url.path, "latency_ms": round(latency, 2),
                       "status": 500, "error_code": "INTERNAL_ERROR"},
            )
            raise
        latency = (time.perf_counter() - start) * 1000
        response.headers["x-request-id"] = request_id
        logging.getLogger("app.access").info(
            "request",
            extra={
                "endpoint": f"{request.method} {request.url.path}",
                "latency_ms": round(latency, 2),
                "status": response.status_code,
            },
        )
        # Fire-and-forget audit trail (firewall: never log bodies).
        try:
            if request.url.path.startswith("/api/"):
                database.insert_audit_log(
                    {
                        "request_id": request_id,
                        "method": request.method,
                        "path": request.url.path,
                        "status": response.status_code,
                        "latency_ms": round(latency, 2),
                    }
                )
        except Exception:  # audit must never break the request
            pass
        finally:
            request_id_var.reset(token)
        return response


from fastapi import Request  # noqa: E402  (used in middleware annotations)
