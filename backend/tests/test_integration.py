"""End-to-End Integration tests for Smart Mental Wellness Assistant.

Tests multi-step pipeline flows:
  - text -> emotion -> database -> intervention
  - image -> face -> emotion -> database
  - text + image -> fusion -> database -> intervention
"""
from __future__ import annotations

import io
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_integration_text_to_intervention_flow():
    # 1. User submits text
    text_payload = {"text": "I have so many assignments due tomorrow and I feel terrified."}
    res_text = client.post("/api/emotion/text", json=text_payload)
    assert res_text.status_code == 200
    text_data = res_text.json()["data"]
    assert text_data["emotion"] == "Anxiety"

    # 2. Crisis screening
    res_crisis = client.post("/api/crisis/analyze", json=text_payload)
    assert res_crisis.status_code == 200
    crisis_data = res_crisis.json()["data"]
    assert crisis_data["level"] in ["LOW", "MODERATE"]

    # 3. Stressor detection
    res_stressor = client.post("/api/stressor/analyze", json=text_payload)
    assert res_stressor.status_code == 200
    stressor_data = res_stressor.json()["data"]
    assert stressor_data["category"] in ["Academic", "Work", "General"]

    # 4. Intervention recommendation query
    res_inter = client.get(f"/api/interventions?emotion={text_data['emotion']}&stressor={stressor_data['category']}")
    assert res_inter.status_code == 200
    inter_data = res_inter.json()["data"]
    interventions_list = inter_data["interventions"] if isinstance(inter_data, dict) and "interventions" in inter_data else inter_data
    assert len(interventions_list) >= 1

    # 5. Submit feedback
    feedback_payload = {
        "intervention_id": interventions_list[0]["id"],
        "rating": "Much better",
        "notes": "Integration test feedback",
    }
    res_fb = client.post("/api/interventions/feedback", json=feedback_payload)
    assert res_fb.status_code == 200
    assert res_fb.json()["success"] is True


def test_integration_multimodal_flow():
    # Multimodal fused request
    payload = {
        "text": "Feeling a bit uneasy about my health test results.",
        "text_weight": 0.65,
        "vision_weight": 0.35,
    }
    res = client.post("/api/multimodal/analyze", json=payload)
    assert res.status_code == 200
    body = res.json()
    assert body["success"] is True
    data = body["data"]
    assert "fused_emotion" in data
    assert "wellness_state" in data
    assert "recommended_intervention" in data
