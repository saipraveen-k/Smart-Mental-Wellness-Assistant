"""Wellness indicator — a non-clinical summary of the current signal.

Explicitly NOT a medical score. Deterministic rules over real inputs:
  * fused emotion + confidence
  * crisis level (screening signal from the dedicated crisis model)
  * stressor intensity (rule-based)
  * recent trend (real mood history, when available)

Thresholds are documented here and surfaced in the API response so the user
can see why a state was assigned.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

WELLNESS_STATES = ["STABLE", "NEEDS_ATTENTION", "HIGH_CONCERN"]

# Base concern per wellness emotion (documented constants).
EMOTION_CONCERN = {
    "Joy": 0.0,
    "Neutral": 0.15,
    "Anger": 0.45,
    "Sadness": 0.50,
    "Anxiety": 0.55,
}
STRESSOR_BONUS = {"low": 0.05, "medium": 0.10, "high": 0.20}
CRISIS_BONUS = {"LOW": 0.0, "MODERATE": 0.30}
TREND_PENALTY = 0.10  # declining 7-day trend

THRESHOLD_HIGH = 0.60
THRESHOLD_ATTENTION = 0.35


def compute_wellness_state(
    emotion: str,
    confidence: float,
    crisis_level: str = "LOW",
    stressor_intensity: Optional[str] = None,
    trend_declining: Optional[bool] = None,
) -> Dict[str, Any]:
    """Deterministic, documented wellness indicator."""
    components: Dict[str, float] = {}
    explanations: List[str] = []

    base = EMOTION_CONCERN.get(emotion, 0.3)
    components["emotion"] = round(base * max(0.0, min(1.0, confidence)), 4)
    explanations.append(
        f"{emotion} at {confidence:.0%} confidence contributes {components['emotion']:.2f}."
    )

    if crisis_level == "MODERATE":
        components["crisis"] = CRISIS_BONUS["MODERATE"]
        explanations.append("Moderate crisis-screening signal adds 0.30.")
    else:
        components["crisis"] = 0.0

    if stressor_intensity in STRESSOR_BONUS:
        components["stressor"] = STRESSOR_BONUS[stressor_intensity]
        explanations.append(
            f"{stressor_intensity}-intensity detected stressor adds "
            f"{STRESSOR_BONUS[stressor_intensity]:.2f}."
        )
    else:
        components["stressor"] = 0.0

    if trend_declining:
        components["trend"] = TREND_PENALTY
        explanations.append("A declining 7-day mood trend adds 0.10.")
    else:
        components["trend"] = 0.0

    score = min(1.0, sum(components.values()))
    if crisis_level == "HIGH":
        state = "HIGH_CONCERN"
        explanations.insert(
            0, "High crisis-screening signal: wellness indicator set to HIGH_CONCERN."
        )
    elif score >= THRESHOLD_HIGH:
        state = "HIGH_CONCERN"
    elif score >= THRESHOLD_ATTENTION:
        state = "NEEDS_ATTENTION"
    else:
        state = "STABLE"

    return {
        "state": state,
        "score": round(score, 4),
        "components": components,
        "explanations": explanations,
        "thresholds": {"attention": THRESHOLD_ATTENTION, "high_concern": THRESHOLD_HIGH},
        "disclaimer": (
            "This is a non-clinical wellness indicator computed from your check-in "
            "signals. It is not a medical score, diagnosis, or measure of mental illness."
        ),
    }


# Valence mapping used for history charts and forecasting (documented).
EMOTION_VALENCE = {
    "Joy": 1.0,
    "Neutral": 0.0,
    "Anger": -0.5,
    "Sadness": -0.6,
    "Anxiety": -0.7,
}


def emotion_valence(emotion: str, confidence: float) -> float:
    return EMOTION_VALENCE.get(emotion, 0.0) * float(confidence)
