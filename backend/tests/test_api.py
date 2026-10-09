"""API endpoint tests for Smart Mental Wellness Assistant.

Covers:
  - test_health
  - test_text_emotion
  - test_vision
  - test_multimodal
  - test_crisis
  - test_mood
  - test_forecast
  - test_feedback
"""
from __future__ import annotations

import io
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert "status" in data["data"]


def test_text_emotion():
    res = client.post("/api/emotion/text", json={"text": "I am feeling so joyful and happy today!"})
    assert res.status_code == 200
    body = res.json()
    assert body["success"] is True
    assert body["data"]["emotion"] in ["Joy", "Sadness", "Anger", "Anxiety", "Neutral"]
    assert "probabilities" in body["data"]


def test_vision_no_face():
    # Create a pure black dummy image
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    _, encoded = cv2.imencode(".png", img)
    files = {"file": ("test.png", io.BytesIO(encoded.tobytes()), "image/png")}

    res = client.post("/api/vision/analyze", files=files)
    assert res.status_code == 200
    body = res.json()
    assert body["success"] is False
    assert body["error"]["code"] == "NO_FACE_DETECTED"


def test_multimodal():
    payload = {
        "text": "I feel a bit anxious about the presentation tomorrow.",
        "text_weight": 0.65,
        "vision_weight": 0.35,
    }
    res = client.post("/api/multimodal/analyze", json=payload)
    assert res.status_code == 200
    body = res.json()
    assert body["success"] is True
    assert "fused_emotion" in body["data"]


def test_crisis():
    res = client.post("/api/crisis/analyze", json={"text": "I'm feeling really stressed about exams."})
    assert res.status_code == 200
    body = res.json()
    assert body["success"] is True
    assert body["data"]["level"] in ["LOW", "MODERATE", "HIGH"]


def test_mood_history_unauthenticated():
    res = client.get("/api/mood/history")
    # Expect unauthenticated or empty guest response
    assert res.status_code in [200, 401]


def test_forecast_unauthenticated():
    res = client.get("/api/mood/forecast")
    assert res.status_code in [200, 401]


def test_feedback():
    payload = {
        "intervention_id": "GUIDED_BREATHING_478",
        "rating": "Much better",
        "notes": "Felt very relaxing",
    }
    res = client.post("/api/interventions/feedback", json=payload)
    assert res.status_code == 200
    body = res.json()
    assert body["success"] is True
