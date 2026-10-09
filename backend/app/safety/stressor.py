"""Stressor detection — documented rule-based detector with real explanations.

Approach (see docs/ML_PIPELINE.md §Stressor detection):
  * Each stressor category owns a lexicon of weighted cue phrases.
  * A text's category score is the sum of the weights of every distinct cue
    it matches, plus an intensity bonus from intensifier phrases.
  * The reported confidence is a monotone function of that raw score,
    confidence = raw / (raw + K). It reflects how much lexical evidence was
    actually matched — it is NOT a model probability and is labelled as such.
  * Every prediction carries its matched cues so the UI can show the evidence.

No ML model is claimed: this component is rule-based, by design — there is no
public dataset with labels for these ten stressor categories, and we refuse to
fabricate one.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

STRESSOR_CATEGORIES = [
    "Academic", "Career", "Relationships", "Family",
    "Financial", "Health", "Sleep", "Social", "Work", "Other",
]

# Raw-score normalizer: confidence = raw / (raw + SCORE_HALF)
SCORE_HALF = 2.0

# Intensifier phrases add evidence weight (matched additively).
INTENSIFIERS: List[Tuple[str, float]] = [
    (r"\b(can'?not|can'?t)\s+(stop|sleep|focus|eat|think)\b", 0.8),
    (r"\b(terrif|petrifi|panic|meltdown|break(ing)? down)\w*", 1.0),
    (r"\b(overwhelm|drowning|falling apart|losing it|at (my|the) limit)\w*", 1.0),
    (r"\b(deadline|due tomorrow|last minute|all[- ]nighter)\b", 0.7),
    (r"\b(every ?day|constantly|non[- ]?stop|always)\b", 0.5),
    (r"\b(very|really|extremely|incredibly|so|terrified|scared|afraid)\b", 0.3),
    (r"\bsuicid|self[- ]?harm|hurt myself\b", 1.5),
]

# Category cue lexicons. Longer phrases are more specific; weights reflect
# how diagnostic the cue is (documented, not learned).
STRESSOR_LEXICON: Dict[str, List[Tuple[str, float]]] = {
    "Academic": [
        (r"\b(assignments?|essays?|homework|coursework)\b", 1.0),
        (r"\b(exams?|finals?|midterms?|test(s)? tomorrow|paper due)\b", 1.2),
        (r"\b(school|college|university|classes?|lectures?|semester|gpa|grades?)\b", 0.9),
        (r"\b(stud(y|ies|ying)|revision|revisioning)\b", 0.7),
        (r"\b(professor|teacher|prof|lecturer|classmate)\w*s?\b", 0.5),
        (r"\b(fail(ing|ed)?|flunk\w*|dropout)\b", 0.8),
        (r"\b(scholarship|thesis|dissertation|lab report|group project)\b", 0.9),
    ],
    "Career": [
        (r"\b(job (interview|offer|search|application)|interview(ing)?|resume|cv)\b", 1.1),
        (r"\b(career|promotion|layoff|laid off|fired|unemployed|quit my job)\b", 1.2),
        (r"\b(job hunt|recruiter|networking|portfolio)\b", 0.8),
        (r"\b(major|degree|graduating|post-?grad)\b", 0.5),
    ],
    "Work": [
        (r"\b(work(ing|load|place|hours|day|s)?|office|boss|colleague|coworker)\w*s?\b", 0.9),
        (r"\b(overtime|shift|meeting(s)?|deadline|performance review)\b", 0.9),
        (r"\b(remote work|wfh|9 to 5|9-5|back[- ]to[- ]back meetings)\b", 0.7),
        (r"\b(burn(ed|out|out)|overworked|stressed at work)\b", 1.1),
    ],
    "Relationships": [
        (r"\b(girlfriend|boyfriend|partner|husband|wife|spouse|fianc[eé])\b", 1.0),
        (r"\b(broke up|breakup|break up|divorce|separation|cheat\w*|ex[- ]?(partner|girlfriend|boyfriend))\b", 1.3),
        (r"\b(fighting with|argu(e|ing|ment)s? with|relationship (issues?|problems?))\b", 1.1),
        (r"\b(date|dating|romance|romantic|heartbroken|lonely (because )?of)\b", 0.8),
        (r"\b(dumped|dumped me|left me|unrequited)\b", 1.2),
    ],
    "Family": [
        (r"\b(mom|dad|mother|father|parents?|family|siblings?|brother|sister)\w*s?\b", 0.9),
        (r"\b(family (issues?|problems?|fight|drama|conflict)|my (mom|dad) (is|are))\b", 1.2),
        (r"\b(grandma|grandpa|grandmother|grandfather|aunt|uncle|cousin)\w*s?\b", 0.6),
        (r"\b(household|housemates?|step-?(mom|dad|parent))\b", 0.6),
    ],
    "Financial": [
        (r"\b(money|rent|bills?|debt|loans?|tuition|savings|afford\w*|bankrupt\w*)\b", 1.1),
        (r"\b(financial|broke|paycheck|overdraft|credit card|expenses?)\b", 1.0),
        (r"\b(can'?t afford|no money|out of money|broke)\b", 1.3),
    ],
    "Health": [
        (r"\b(sick|ill(ness)?|disease|diagnos\w*|symptoms?|pain|surgery|hospital)\b", 1.0),
        (r"\b(health|doctor|medical|therapy|medicat\w*|prescription|panic attacks?)\b", 0.9),
        (r"\b(anxiety|depress\w*|burnout|therapist|counsell?or|mental health)\b", 0.9),
        (r"\b(injured|injury|chronic|fatig\w*|exhaust\w*)\b", 0.8),
    ],
    "Sleep": [
        (r"\b(sleep|insomnia|can'?t (sleep|sleeping)|sleeping (poorly|badly)|nightmares?)\b", 1.2),
        (r"\b(awake|all[- ]night|up until|3 ?am|4 ?am|tired|exhausted)\b", 0.9),
        (r"\b(sleep(ing)? (too much|all day)|oversleep\w*|sleep schedule)\b", 0.9),
    ],
    "Social": [
        (r"\b(friends?|friendship|social(ly)?|party|hang(out|ing)|group chat)\b", 0.8),
        (r"\b(bull(y|ied|ying)|excluded|left out|left on read|ostrac\w*)\b", 1.2),
        (r"\b(social anxiety|scared of people|don'?t (have|fit in with) friends)\b", 1.2),
        (r"\b(popular|clique|classmates?)\b", 0.4),
    ],
    "Other": [
        # Deliberately sparse: "Other" only wins when a generic distress cue
        # exists and no specific category matched.
        (r"\b(stress\w*|overwhelm\w*|worried|worry|anxious|panic)\b", 0.4),
    ],
}

_COMPILED: Dict[str, List[Tuple[re.Pattern, float]]] = {
    cat: [(re.compile(pat, re.IGNORECASE), w) for pat, w in cues]
    for cat, cues in STRESSOR_LEXICON.items()
}
_COMPILED_INTENSIFIERS = [(re.compile(pat, re.IGNORECASE), w) for pat, w in INTENSIFIERS]


def detect_stressor(text: str) -> Dict[str, Any]:
    """Detect the most likely stressor category from text.

    Returns ``{"category": str|None, "confidence": float, "intensity":
    "low"|"medium"|"high", "matched_cues": [...], "method": "rule_based"}``.
    Confidence is the normalized rule score, not an ML probability.
    """
    if not text or not text.strip():
        return {
            "category": None,
            "confidence": 0.0,
            "intensity": "low",
            "matched_cues": [],
            "method": "rule_based",
            "scores": {},
        }

    scores: Dict[str, float] = {cat: 0.0 for cat in STRESSOR_CATEGORIES}
    matched: Dict[str, List[str]] = {cat: [] for cat in STRESSOR_CATEGORIES}

    for cat, cues in _COMPILED.items():
        for pattern, weight in cues:
            m = pattern.search(text)
            if m:
                scores[cat] += weight
                matched[cat].append(m.group(0).lower())

    intensifier_bonus = 0.0
    intensifier_hits: List[str] = []
    for pattern, weight in _COMPILED_INTENSIFIERS:
        m = pattern.search(text)
        if m:
            intensifier_bonus += weight
            intensifier_hits.append(m.group(0).lower())

    # Spread the intensifier bonus proportionally over matched categories.
    matched_cats = [c for c in STRESSOR_CATEGORIES if scores[c] > 0]
    if matched_cats and intensifier_bonus > 0:
        total = sum(scores[c] for c in matched_cats)
        for c in matched_cats:
            scores[c] += intensifier_bonus * (scores[c] / total)

    # "Other" only competes when nothing else matched.
    if any(scores[c] > 0 for c in STRESSOR_CATEGORIES if c != "Other"):
        scores["Other"] = 0.0
        matched["Other"] = []

    best_cat = max(scores, key=lambda c: scores[c])
    best_score = scores[best_cat]
    if best_score <= 0:
        return {
            "category": None,
            "confidence": 0.0,
            "intensity": "low",
            "matched_cues": [],
            "method": "rule_based",
            "scores": {},
        }

    confidence = best_score / (best_score + SCORE_HALF)
    intensity = "low"
    if best_score >= 3.0 or intensifier_bonus >= 1.5:
        intensity = "high"
    elif best_score >= 1.5 or intensifier_bonus >= 0.7:
        intensity = "medium"

    cues = [{"cue": c, "category": best_cat} for c in matched[best_cat][:6]]
    cues += [{"cue": c, "category": "intensifier"} for c in intensifier_hits[:4]]

    return {
        "category": best_cat,
        "confidence": round(float(confidence), 4),
        "intensity": intensity,
        "matched_cues": cues,
        "method": "rule_based",
        "raw_score": round(best_score, 3),
        "note": (
            "Rule-based detection: confidence is the normalized strength of the "
            "lexical cues actually matched in your text."
        ),
    }


detect_stressors = detect_stressor

