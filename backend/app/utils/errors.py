"""API error handling + consistent response envelopes.

Success: {"success": true,  "data": {...}, "error": null}
Error:   {"success": false, "data": null,  "error": {"code", "message", "suggestion"}}
Stack traces are logged server-side and never exposed to users.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

logger = logging.getLogger("app.errors")


class ApiError(Exception):
    def __init__(
        self,
        code: str,
        message: str,
        status_code: int = 400,
        suggestion: Optional[str] = None,
    ):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.suggestion = suggestion


def success(data: Any = None) -> Dict[str, Any]:
    return {"success": True, "data": data, "error": None}


def failure(code: str, message: str, suggestion: Optional[str] = None) -> Dict[str, Any]:
    err: Dict[str, Any] = {"code": code, "message": message}
    if suggestion:
        err["suggestion"] = suggestion
    return {"success": False, "data": None, "error": err}


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _api_error(request: Request, exc: ApiError):
        request_id = getattr(request.state, "request_id", "-")
        logger.warning(
            "api_error request_id=%s path=%s code=%s message=%s",
            request_id, request.url.path, exc.code, exc.message,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=failure(exc.code, exc.message, exc.suggestion),
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError):
        request_id = getattr(request.state, "request_id", "-")
        first = exc.errors()[0] if exc.errors() else {}
        loc = ".".join(str(p) for p in first.get("loc", []))
        msg = f"Invalid request: {loc} {first.get('error', '')}" if loc else "Invalid request."
        logger.warning("validation_error request_id=%s path=%s detail=%s",
                       request_id, request.url.path, str(exc.errors())[:300])
        return JSONResponse(
            status_code=422,
            content=failure("VALIDATION_ERROR", msg[:300], "Check the request body and try again."),
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        request_id = getattr(request.state, "request_id", "-")
        logger.exception("unhandled_error request_id=%s path=%s", request_id, request.url.path)
        return JSONResponse(
            status_code=500,
            content=failure(
                "INTERNAL_ERROR",
                "Something went wrong on our side. The issue has been logged.",
                "Try again in a moment. If it keeps happening, check the server logs.",
            ),
        )
