"""Unit tests for Smart Mental Wellness Assistant components.

Covers:
  - test_text_preprocessing
  - test_emotion_mapping
  - test_dataset_loader
  - test_cv_preprocessing
  - test_face_detection
  - test_fusion
  - test_crisis_classifier
  - test_stressor_detection
  - test_forecasting
  - test_intervention_selection
"""
from __future__ import annotations

import numpy as np
import pytest

from app.cv.image_utils import crop_and_normalize, decode_image, detect_faces
from app.cv.model import FacialEmotionCNN
from app.forecasting.forecaster import build_daily_series, forecast
from app.ml.dataset_manager import dataset_manager
from app.ml.emotion_mapping import (
    FER2013_LABELS,
    GOEMOTIONS_LABELS,
    WELLNESS_LABELS,
    map_fer_to_wellness,
    map_goemotions_to_wellness,
)
from app.ml.text_preprocessing import clean_text
from app.safety.crisis_classifier import crisis_classifier
from app.safety.stressor import detect_stressors
from app.services.fusion_service import compute_fusion
from app.services.intervention_service import intervention_engine


def test_text_preprocessing():
    raw = "  I feel VERY anxious & overwhelmed...   "
    cleaned = clean_text(raw)
    assert cleaned == "I feel VERY anxious & overwhelmed..."
    assert clean_text("") == ""


def test_emotion_mapping():
    assert map_goemotions_to_wellness("joy") == "Joy"
    assert map_goemotions_to_wellness("sadness") == "Sadness"
    assert map_goemotions_to_wellness("anger") == "Anger"
    assert map_goemotions_to_wellness("fear") == "Anxiety"
    assert map_goemotions_to_wellness("neutral") == "Neutral"

    assert map_fer_to_wellness("happy") == "Joy"
    assert map_fer_to_wellness("sad") == "Sadness"
    assert map_fer_to_wellness("angry") == "Anger"
    assert map_fer_to_wellness("fear") == "Anxiety"
    assert map_fer_to_wellness("neutral") == "Neutral"


def test_dataset_loader():
    status = dataset_manager.get_status()
    assert "configured" in status
    assert "text" in status["configured"]
    assert "cv" in status["configured"]


def test_cv_preprocessing():
    import cv2

    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    # Draw a simple white square
    cv2.rectangle(dummy_img, (20, 20), (80, 80), (255, 255, 255), -1)
    _, encoded = cv2.imencode(".png", dummy_img)
    decoded = decode_image(encoded.tobytes())
    assert decoded is not None
    assert decoded.shape[0] == 100 and decoded.shape[1] == 100


def test_face_detection():
    blank_img = np.zeros((200, 200, 3), dtype=np.uint8)
    boxes = detect_faces(blank_img)
    assert len(boxes) == 0


def test_fusion():
    text_probs = {"Joy": 0.1, "Sadness": 0.1, "Anger": 0.1, "Anxiety": 0.6, "Neutral": 0.1}
    vision_probs = {"Joy": 0.1, "Sadness": 0.1, "Anger": 0.1, "Anxiety": 0.5, "Neutral": 0.2}

    res = compute_fusion(
        text_distribution=text_probs,
        vision_distribution=vision_probs,
        text_weight=0.65,
        vision_weight=0.35,
    )
    assert res["emotion"] == "Anxiety"
    assert 0.0 <= res["confidence"] <= 1.0
    assert abs(sum(res["distribution"].values()) - 1.0) < 1e-3


def test_crisis_classifier():
    if crisis_classifier.available():
        low_res = crisis_classifier.predict("I am having a nice day studying for exams.")
        assert low_res["level"] in ("LOW", "MODERATE", "HIGH")
        assert "probabilities" in low_res

        high_res = crisis_classifier.predict("I cannot go on anymore and want to end my life right now")
        assert high_res["level"] in ("MODERATE", "HIGH")


def test_stressor_detection():
    res = detect_stressors("I have three exams tomorrow and I feel so stressed about failing.")
    assert res["category"] in ("Academic", "Work", "General", "Other")
    assert "confidence" in res
    assert len(res["matched_cues"]) >= 1


def test_forecasting():
    # Test with insufficient series points
    series_short = [{"timestamp": "2026-10-01T10:00:00Z", "fused_emotion": "Joy", "fused_confidence": 0.8}]
    res_insufficient = forecast(series_short)
    assert res_insufficient.get("available") is False

    # Test with 14 distinct daily mood records
    emotions = ["Joy", "Anxiety", "Sadness", "Neutral", "Joy", "Anxiety", "Joy"]
    records_sufficient = [
        {
            "timestamp": f"2026-10-{i+1:02d}T10:00:00Z",
            "fused_emotion": emotions[i % len(emotions)],
            "fused_confidence": 0.8,
        }
        for i in range(14)
    ]
    res_sufficient = forecast(records_sufficient)
    assert res_sufficient.get("available") is True


def test_intervention_selection():
    rec = intervention_engine.recommend(
        emotion="Anxiety",
        stressor="Academic",
        severity="MODERATE",
    )
    assert "title" in rec
    assert "category" in rec
    assert "steps" in rec
