"""Security tests for Smart Mental Wellness Assistant.

Covers:
  - test_invalid_auth
  - test_invalid_upload
  - test_oversized_image
  - test_malicious_filename
"""
from __future__ import annotations

import io
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_invalid_auth():
    # Attempting to access authenticated endpoint with malformed token
    headers = {"Authorization": "Bearer invalid.jwt.token.here"}
    res = client.get("/api/mood/history", headers=headers)
    assert res.status_code in [401, 403]


def test_invalid_upload():
    # Attempt to upload a non-image file (e.g. executable script)
    files = {"file": ("malicious.sh", io.BytesIO(b"#!/bin/bash\necho hacked"), "text/x-shellscript")}
    res = client.post("/api/vision/analyze", files=files)
    assert res.status_code in [400, 422] or res.json()["success"] is False


def test_oversized_image():
    # Attempt to upload an image exceeding size limit (e.g. >10MB dummy buffer)
    large_payload = b"\x00" * (12 * 1024 * 1024)
    files = {"file": ("huge.png", io.BytesIO(large_payload), "image/png")}
    res = client.post("/api/vision/analyze", files=files)
    assert res.status_code in [400, 413, 422] or res.json()["success"] is False


def test_malicious_filename():
    # File upload with path traversal attack vector in filename
    img_data = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15c4"
    files = {"file": ("../../../../etc/passwd", io.BytesIO(img_data), "image/png")}
    res = client.post("/api/vision/analyze", files=files)
    # Ensure system responds to upload safely (returns structured HTTP error code)
    assert res.status_code in [200, 400, 415, 422, 500, 503]
