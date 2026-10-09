"""Text Preprocessing & Taxonomy Mapping Utilities."""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

WELLNESS_TAXONOMY = ["Joy", "Sadness", "Anger", "Anxiety", "Neutral"]

# GoEmotions 28 raw labels mapped to 5 core categories
RAW_GOEMOTIONS_MAPPING: Dict[str, str] = {
    "amusement": "Joy",
    "excitement": "Joy",
    "joy": "Joy",
    "love": "Joy",
    "desire": "Joy",
    "optimism": "Joy",
    "pride": "Joy",
    "relief": "Joy",
    "gratitude": "Joy",
    "admiration": "Joy",
    "caring": "Joy",
    "approval": "Joy",
    "sadness": "Sadness",
    "disappointment": "Sadness",
    "embarrassment": "Sadness",
    "grief": "Sadness",
    "remorse": "Sadness",
    "anger": "Anger",
    "annoyance": "Anger",
    "disapproval": "Anger",
    "disgust": "Anger",
    "fear": "Anxiety",
    "nervousness": "Anxiety",
    "neutral": "Neutral",
}

def preprocess_text(text: str) -> str:
    """Light clean of raw input text."""
    if not text or not isinstance(text, str):
        return ""
    text = re.sub(r"\s+", " ", text.strip())
    return text

class GoEmotionsTaxonomyMapper:
    """Maps GoEmotions labels into normalized wellness categories."""
    
    def __init__(self, raw_mapping: Optional[Dict[str, str]] = None):
        self.mapping = raw_mapping or RAW_GOEMOTIONS_MAPPING
        self.categories = WELLNESS_TAXONOMY

    def map_label(self, raw_emotion: str) -> str:
        return self.mapping.get(raw_emotion.lower(), "Neutral")

    def get_class_index(self, category: str) -> int:
        return self.categories.index(category) if category in self.categories else 4
