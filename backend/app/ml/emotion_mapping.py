"""Emotion taxonomy and mapping layers.

Two mapping layers exist:

1. TEXT:  GoEmotions fine-grained labels  -> wellness taxonomy (5 classes)
2. VISION: FER-2013 expression labels     -> wellness taxonomy (5 classes)

Wellness taxonomy (used everywhere in the application):

    0 = Joy
    1 = Sadness
    2 = Anger
    3 = Anxiety
    4 = Neutral / Other

These mappings are documented in docs/ML_PIPELINE.md and docs/CV_PIPELINE.md.
Nothing here is guessed at runtime; every source label maps to an explicit
target or is explicitly dropped/handled as ambiguous.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Wellness taxonomy
# ---------------------------------------------------------------------------
JOY = "Joy"
SADNESS = "Sadness"
ANGER = "Anger"
ANXIETY = "Anxiety"
NEUTRAL = "Neutral"

WELLNESS_LABELS: List[str] = [JOY, SADNESS, ANGER, ANXIETY, NEUTRAL]
WELLNESS_LABEL2ID: Dict[str, int] = {name: i for i, name in enumerate(WELLNESS_LABELS)}
WELLNESS_ID2LABEL: Dict[int, str] = {i: name for name, i in WELLNESS_LABEL2ID.items()}

# ---------------------------------------------------------------------------
# TEXT mapping: GoEmotions (28 labels) -> wellness taxonomy
# ---------------------------------------------------------------------------
# Source: SetFit/go_emotions (Demszky et al., 2022 — GoEmotions, Reddit comments).
GOEMOTIONS_LABELS: List[str] = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization", "relief",
    "remorse", "sadness", "surprise", "neutral",
]

# Documented mapping decisions (see docs/ML_PIPELINE.md §Emotion mapping):
#
# Joy      — positive-affection + achievement + appreciation labels.
# Sadness  — sorrow, loss and self-reproach labels (remorse/embarrassment are
#            loss-of-status/self-directed distress, grouped with Sadness).
# Anger    — hostility and disapproval labels.
# Anxiety  — fear + nervousness (the GoEmotions closest match to anxiety).
# Neutral  — neutral text, plus low-affect "other" states (confusion, curiosity,
#            disgust, realization, surprise) that carry no wellness signal on
#            their own: they are grouped into Neutral/Other rather than being
#            forced into a wellness emotion they do not represent.
#
# Ambiguity rule: a sample whose active labels resolve to MORE THAN ONE
# wellness class is dropped from training (explicitly documented, never
# silently relabelled).
EMOTION_MAPPING: Dict[str, str] = {
    # -> Joy
    "joy": JOY,
    "amusement": JOY,
    "excitement": JOY,
    "gratitude": JOY,
    "optimism": JOY,
    "love": JOY,
    "pride": JOY,
    "relief": JOY,
    "admiration": JOY,
    "approval": JOY,
    "caring": JOY,
    "desire": JOY,
    # -> Sadness
    "sadness": SADNESS,
    "grief": SADNESS,
    "disappointment": SADNESS,
    "remorse": SADNESS,
    "embarrassment": SADNESS,
    # -> Anger
    "anger": ANGER,
    "annoyance": ANGER,
    "disapproval": ANGER,
    # -> Anxiety
    "fear": ANXIETY,
    "nervousness": ANXIETY,
    # -> Neutral / Other
    "neutral": NEUTRAL,
    "confusion": NEUTRAL,
    "curiosity": NEUTRAL,
    "disgust": NEUTRAL,
    "realization": NEUTRAL,
    "surprise": NEUTRAL,
}

assert set(EMOTION_MAPPING) == set(GOEMOTIONS_LABELS), "EMOTION_MAPPING must cover every GoEmotions label"


def resolve_text_label(active_labels: Sequence[str]) -> Optional[str]:
    """Resolve the active GoEmotions labels of one sample to a wellness class.

    Returns None when the sample is ambiguous (>=2 distinct wellness classes
    active) or when no known label is active. Callers must drop such samples
    from supervised training and record the reason.
    """
    targets = set()
    for lbl in active_labels:
        if lbl not in EMOTION_MAPPING:
            raise KeyError(f"Unknown GoEmotions label: {lbl!r}")
        targets.add(EMOTION_MAPPING[lbl])
    if len(targets) == 1:
        return next(iter(targets))
    return None


# ---------------------------------------------------------------------------
# VISION mapping: FER-2013 expressions -> wellness taxonomy
# ---------------------------------------------------------------------------
FER2013_LABELS: List[str] = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# Deliberate decisions (see docs/CV_PIPELINE.md):
#   happy   -> Joy
#   sad     -> Sadness
#   angry   -> Anger
#   fear    -> Anxiety   (closest wellness analogue; documented, not a claim
#                         that a facial expression equals an internal state)
#   neutral -> Neutral
#   disgust -> Neutral/Other   (NOT mapped to Anger)
#   surprise-> Neutral/Other   (NOT mapped to Joy)
CV_EMOTION_MAPPING: Dict[str, str] = {
    "happy": JOY,
    "sad": SADNESS,
    "angry": ANGER,
    "fear": ANXIETY,
    "neutral": NEUTRAL,
    "disgust": NEUTRAL,
    "surprise": NEUTRAL,
}

# Canonical display names used by the vision UI ("Facial-expression estimate")
CV_DISPLAY_LABELS: Dict[str, str] = {
    "angry": "Angry",
    "disgust": "Disgust",
    "fear": "Fear",
    "happy": "Happy",
    "neutral": "Neutral",
    "sad": "Sad",
    "surprise": "Surprise",
}


def map_cv_distribution(probs: Dict[str, float]) -> Dict[str, float]:
    """Collapse a 7-class FER distribution into the 5-class wellness taxonomy.

    Probabilities of source labels mapped to the same target are summed, then
    the result is renormalised to sum to 1.
    """
    out: Dict[str, float] = {label: 0.0 for label in WELLNESS_LABELS}
    for expr, p in probs.items():
        target = CV_EMOTION_MAPPING[expr]
        out[target] += float(p)
    total = sum(out.values())
    if total <= 0:
        raise ValueError("Probability distribution sums to zero")
    return {k: v / total for k, v in out.items()}


def map_text_distribution(probs: Dict[str, float]) -> Dict[str, float]:
    """Map a distribution over GoEmotions labels into the wellness taxonomy."""
    out: Dict[str, float] = {label: 0.0 for label in WELLNESS_LABELS}
    for expr, p in probs.items():
        target = EMOTION_MAPPING[expr]
        out[target] += float(p)
    total = sum(out.values())
    if total <= 0:
        raise ValueError("Probability distribution sums to zero")
    return {k: v / total for k, v in out.items()}


def map_goemotions_to_wellness(label: str) -> str:
    """Map a single GoEmotions label to the wellness taxonomy."""
    return EMOTION_MAPPING.get(label.lower(), NEUTRAL)


def map_fer_to_wellness(expression: str) -> str:
    """Map a single FER-2013 expression label to the wellness taxonomy."""
    return CV_EMOTION_MAPPING.get(expression.lower(), NEUTRAL)


def top_label(distribution: Dict[str, float]) -> Tuple[str, float]:
    """Return (label, confidence) of the argmax of a distribution."""
    if not distribution:
        raise ValueError("Empty distribution")
    label = max(distribution, key=distribution.get)  # type: ignore[arg-type]
    return label, float(distribution[label])

