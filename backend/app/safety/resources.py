"""Crisis resources loader — real services only, configurable per country."""
from __future__ import annotations

import json
import logging
from functools import lru_cache
from typing import Any, Dict

from app.config import settings

logger = logging.getLogger("app.safety.resources")

FALLBACK_MESSAGE = {
    "international": [
        {
            "name": "Find A Helpline",
            "description": "Verified crisis helplines in 130+ countries.",
            "contact": "https://findahelpline.com",
            "type": "directory",
            "source": "https://findahelpline.com",
        }
    ],
    "emergency": {
        "description": (
            "If you are in immediate physical danger, contact your local "
            "emergency number."
        ),
        "type": "emergency",
    },
}


@lru_cache(maxsize=1)
def _load_file() -> Dict[str, Any]:
    path = settings.crisis_resources_file
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.error("crisis resources file unreadable (%s): %s", path, exc)
        return FALLBACK_MESSAGE


def get_crisis_resources(country: str | None = None) -> Dict[str, Any]:
    """Resources for the deployment country, falling back to an int'l directory.

    Numbers come from config/crisis_resources.json — a curated file of real,
    publicly documented services. Nothing here is invented.
    """
    doc = _load_file()
    code = (country or settings.crisis_resources_country or "US").upper()
    entries = doc.get("countries", {}).get(code)
    return {
        "country": code if entries else None,
        "message": (
            "We're concerned that you may be going through a serious situation. "
            "You don't have to handle this alone."
        ),
        "resources": entries or [],
        "international": doc.get("international", FALLBACK_MESSAGE["international"]),
        "emergency": doc.get("emergency", FALLBACK_MESSAGE["emergency"]),
        "note": None if entries else (
            f"No curated resources for country {code!r} — use the international "
            "directory to find verified helplines near you."
        ),
    }
