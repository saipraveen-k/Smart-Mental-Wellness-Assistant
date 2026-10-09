"""Text preprocessing + GoEmotions -> wellness-label preparation for training.

Shared by ``scripts/train_text.py`` (training) and the inference service.
"""
from __future__ import annotations

import html
import re
import unicodedata
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

from app.ml.emotion_mapping import (
    EMOTION_MAPPING,
    GOEMOTIONS_LABELS,
    WELLNESS_LABELS,
    resolve_text_label,
)

_CONTROL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
_WS_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Light cleaning that preserves the emotional signal of the text."""
    if text is None:
        return ""
    text = html.unescape(str(text))
    text = text.replace("\ufffd", "'")
    text = _CONTROL_RE.sub(" ", text)
    text = unicodedata.normalize("NFKC", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def is_meaningful(text: str) -> bool:
    """Reject empty/too-short texts and texts without alphanumeric content."""
    if len(text) < 2:
        return False
    return any(ch.isalnum() for ch in text)


def resolve_sample(row: Dict) -> Optional[str]:
    """Map one GoEmotions row to a wellness class, or None if unusable."""
    active = [lbl for lbl in GOEMOTIONS_LABELS if row.get(lbl)]
    if not active:
        return None
    return resolve_text_label(active)


def prepare_split(hf_split, split_name: str = "train") -> Tuple[List[Dict], Dict[str, int]]:
    """Convert a Hugging Face GoEmotions split into training samples.

    Returns ``(samples, drop_reasons)`` where each sample is
    ``{"text": str, "label": int, "label_name": str}`` and ``drop_reasons``
    counts every sample that was excluded and why (for reporting; nothing is
    silently discarded).
    """
    samples: List[Dict] = []
    drops: Counter = Counter()

    # First pass: detect texts whose mapped labels conflict across the dataset.
    text_labels: Dict[str, set] = {}
    for row in hf_split:
        text = clean_text(row.get("text", ""))
        if not is_meaningful(text):
            drops["not_meaningful"] += 1
            continue
        label = resolve_sample(row)
        if label is None:
            active = [lbl for lbl in GOEMOTIONS_LABELS if row.get(lbl)]
            if not active:
                drops["no_active_label"] += 1
            else:
                drops["ambiguous_multiple_classes"] += 1
            continue
        text_labels.setdefault(text, set()).add(label)

    conflicting = {t for t, labels in text_labels.items() if len(labels) > 1}

    seen = set()
    for row in hf_split:
        text = clean_text(row.get("text", ""))
        if not is_meaningful(text) or text in seen:
            continue
        label = resolve_sample(row)
        if label is None:
            continue
        if text in conflicting:
            drops["conflicting_labels_across_annotators"] += 1
            seen.add(text)
            continue
        seen.add(text)
        samples.append(
            {
                "text": text,
                "label": WELLNESS_LABELS.index(label),
                "label_name": label,
                "split": split_name,
            }
        )
    return samples, dict(drops)


def class_distribution(samples: Sequence[Dict]) -> Dict[str, int]:
    counter = Counter(s["label_name"] for s in samples)
    return {name: counter.get(name, 0) for name in WELLNESS_LABELS}


def compute_class_weights(samples: Sequence[Dict]) -> List[float]:
    """Inverse-frequency class weights (normalized to mean 1.0) for the loss."""
    dist = class_distribution(samples)
    total = sum(dist.values())
    n_classes = len(WELLNESS_LABELS)
    weights = []
    for name in WELLNESS_LABELS:
        count = max(dist[name], 1)
        weights.append((total / (n_classes * count)))
    return [round(w, 6) for w in weights]
