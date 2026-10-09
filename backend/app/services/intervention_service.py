"""Personalized intervention engine (CBT-inspired / mindfulness-oriented).

Inputs:  emotion, stressor, wellness state, crisis level, 7-day trend,
         per-intervention feedback statistics (real counts from the database).
Outputs: ranked recommendations with a rationale.

Feedback learning: recommendations are re-ranked with an additive boost/penalty
derived from THIS user's stored feedback — an explicit, documented formula.
This is NOT reinforcement learning and is never described as such.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

# feedback rating -> numeric value
FEEDBACK_VALUES = {"much_better": 2.0, "a_little_better": 1.0, "no_change": 0.0, "worse": -2.0}
FEEDBACK_WEIGHT = 0.35  # max rank adjustment from mean feedback

INTERVENTIONS: List[Dict[str, Any]] = [
    {
        "id": "GUIDED_BREATHING",
        "name": "Guided breathing exercise",
        "kind": "mindfulness",
        "duration_minutes": 5,
        "description": "A short paced-breathing exercise to settle your body's stress response.",
        "target_emotions": ["Anxiety", "Anger"],
        "target_states": ["NEEDS_ATTENTION", "HIGH_CONCERN", "STABLE"],
        "stressor_fit": ["Academic", "Work", "Career", "Health", "Financial"],
        "base_priority": 0.8,
        "steps": [
            "Sit comfortably and set a timer for 5 minutes.",
            "Breathe in through your nose for 4 seconds.",
            "Hold gently for 2 seconds.",
            "Breathe out through your mouth for 6 seconds.",
            "Repeat, letting your shoulders drop with each exhale.",
            "If your mind wanders, return to the count — that is normal.",
        ],
        "disclaimer": "A wellness exercise, not medical treatment.",
    },
    {
        "id": "CBT_REFRAMING",
        "name": "CBT-inspired thought reframing",
        "kind": "cbt",
        "duration_minutes": 10,
        "description": "Write down the stressful thought, the evidence for and against it, and a more balanced version.",
        "target_emotions": ["Anxiety", "Sadness", "Anger"],
        "target_states": ["NEEDS_ATTENTION", "STABLE"],
        "stressor_fit": ["Academic", "Career", "Work", "Relationships", "Family", "Financial", "Social"],
        "base_priority": 0.75,
        "steps": [
            "Write the situation in one sentence.",
            "Write the automatic thought that hurts most.",
            "List evidence that supports it — and evidence that does not.",
            "Write a balanced alternative a kind friend might say.",
            "Re-rate how strongly you believe the original thought (0-100).",
        ],
        "disclaimer": "A self-help exercise inspired by CBT; not a substitute for therapy.",
    },
    {
        "id": "JOURNALING",
        "name": "Structured journaling",
        "kind": "reflective",
        "duration_minutes": 10,
        "description": "A few focused prompts to externalise what is weighing on you.",
        "target_emotions": ["Sadness", "Anxiety", "Anger", "Neutral"],
        "target_states": ["STABLE", "NEEDS_ATTENTION"],
        "stressor_fit": ["Academic", "Family", "Relationships", "Social", "Career", "Other"],
        "base_priority": 0.7,
        "steps": [
            "Write for 5 minutes without editing — spelling does not matter.",
            "Name the emotion you notice most strongly.",
            "Name the situation it connects to.",
            "Write one small thing you can do about that situation tomorrow.",
        ],
        "disclaimer": "A wellness exercise, not medical treatment.",
    },
    {
        "id": "MINDFULNESS",
        "name": "5-4-3-2-1 mindfulness grounding",
        "kind": "mindfulness",
        "duration_minutes": 6,
        "description": "Reconnect with the present through your five senses.",
        "target_emotions": ["Anxiety", "Sadness"],
        "target_states": ["STABLE", "NEEDS_ATTENTION", "HIGH_CONCERN"],
        "stressor_fit": ["Health", "Social", "Family", "Relationships", "Other"],
        "base_priority": 0.7,
        "steps": [
            "Name 5 things you can see.",
            "Name 4 things you can feel (feet on floor, fabric, temperature).",
            "Name 3 things you can hear.",
            "Name 2 things you can smell.",
            "Name 1 thing you can taste.",
        ],
        "disclaimer": "A wellness exercise, not medical treatment.",
    },
    {
        "id": "GROUNDING",
        "name": "Distress tolerance grounding",
        "kind": "grounding",
        "duration_minutes": 8,
        "description": "Short, concrete steps to get through a moment of intense distress safely.",
        "target_emotions": ["Anxiety", "Anger", "Sadness"],
        "target_states": ["HIGH_CONCERN"],
        "stressor_fit": ["*"],
        "base_priority": 0.8,
        "steps": [
            "Move away from anything you could hurt yourself with.",
            "Hold something cold (ice, cold water) and name what you feel.",
            "Text or call one person you trust and say you are having a hard moment.",
            "Breathe out longer than you breathe in for two minutes.",
            "If the feeling persists, contact a crisis line or local emergency service.",
        ],
        "disclaimer": "Not a substitute for professional or emergency support.",
    },
    {
        "id": "REFLECTION_PROMPT",
        "name": "Daily reflection prompt",
        "kind": "reflective",
        "duration_minutes": 5,
        "description": "One question a day to notice what shapes your mood.",
        "target_emotions": ["Neutral", "Joy"],
        "target_states": ["STABLE"],
        "stressor_fit": ["*"],
        "base_priority": 0.5,
        "steps": [
            "What moment today took the most energy from me?",
            "What gave me even a little energy back?",
            "What would I like to do more of tomorrow?",
        ],
        "disclaimer": "A wellness exercise, not medical treatment.",
    },
    {
        "id": "PROFESSIONAL_SUPPORT",
        "name": "Consider professional support",
        "kind": "support",
        "duration_minutes": 0,
        "description": (
            "Talking to a licensed professional is the most effective next step when "
            "distress keeps showing up. A counselor or doctor can help you figure out "
            "what support fits your situation."
        ),
        "target_emotions": ["Anxiety", "Sadness", "Anger"],
        "target_states": ["HIGH_CONCERN", "NEEDS_ATTENTION"],
        "stressor_fit": ["*"],
        "base_priority": 0.85,
        "steps": [
            "Look up your campus or workplace counseling service, or your local clinic.",
            "Write down when the distress started and how often it happens.",
            "Book an initial appointment — most first visits are about listening, not labels.",
        ],
        "disclaimer": "A suggestion to seek help, not a clinical recommendation.",
    },
]

_BY_ID = {i["id"]: i for i in INTERVENTIONS}


def compute_priority(base: float, feedback_stats: Optional[Dict[str, Any]]) -> float:
    """Feedback-adjusted priority: base + FEEDBACK_WEIGHT * mean_feedback.

    mean_feedback is the mean of stored numeric feedback for this
    intervention (range -2..+2), 0 when the user has not rated it yet.
    """
    if not feedback_stats:
        return base
    total = feedback_stats.get("total", 0)
    if not total:
        return base
    mean = feedback_stats.get("mean_score", 0.0)
    return max(0.05, min(1.5, base + FEEDBACK_WEIGHT * mean))


def recommend_interventions(
    emotion: str,
    wellness_state: str,
    crisis_level: str = "LOW",
    stressor: Optional[str] = None,
    stressor_intensity: Optional[str] = None,
    trend_declining: Optional[bool] = None,
    feedback_stats: Optional[Dict[str, Dict[str, Any]]] = None,
    top_n: int = 3,
) -> Dict[str, Any]:
    """Rank interventions for the current signal, adjusted by user feedback."""
    feedback_stats = feedback_stats or {}
    crisis = crisis_level == "HIGH"

    scored: List[Dict[str, Any]] = []
    for item in INTERVENTIONS:
        reasons: List[str] = []
        score = item["base_priority"]

        if crisis:
            # Crisis safety override: only crisis-appropriate interventions.
            if item["id"] not in ("GROUNDING", "PROFESSIONAL_SUPPORT"):
                continue
            reasons.append("Crisis safety override: only grounding and support resources are offered.")
        else:
            if emotion in item["target_emotions"]:
                score += 0.15
                reasons.append(f"Targets {emotion}.")
            if wellness_state in item["target_states"]:
                score += 0.05
            fit = item["stressor_fit"]
            if stressor and ("*" in fit or stressor in fit):
                score += 0.10
                reasons.append(f"Fit for detected {stressor} stressor.")
            if stressor_intensity == "high" and item["id"] in ("GUIDED_BREATHING", "GROUNDING"):
                score += 0.05
            if trend_declining and item["id"] in ("MINDFULNESS", "JOURNALING"):
                score += 0.05
                reasons.append("Helpful for a declining mood trend.")

        stats = feedback_stats.get(item["id"])
        adjusted = compute_priority(score, stats)
        if stats and stats.get("total", 0) > 0:
            delta = adjusted - score
            if abs(delta) > 0.01:
                reasons.append(
                    f"Adjusted by {delta:+.2f} from your feedback "
                    f"({stats['total']} rating(s), mean {stats.get('mean_score', 0):+.1f})."
                )

        entry = {k: v for k, v in item.items() if k != "target_states"}
        entry.pop("target_emotions", None)
        entry.pop("stressor_fit", None)
        entry["priority"] = round(adjusted, 4)
        entry["base_priority"] = round(item["base_priority"], 4)
        entry["rationale"] = reasons or ["Generally applicable to your current state."]
        scored.append(entry)

    scored.sort(key=lambda x: x["priority"], reverse=True)
    chosen = scored[: max(1, top_n)]

    if crisis:
        headline = (
            "Your check-in suggests you may be going through a serious situation. "
            "You don't have to handle this alone — please reach out to support below."
        )
    else:
        headline = "Here is what may help right now based on your check-in."

    return {
        "crisis_mode": crisis,
        "headline": headline,
        "interventions": chosen,
        "learning_note": (
            "Ranking adapts to your intervention feedback using an additive "
            "adjustment (documented in docs/ML_PIPELINE.md). This is not "
            "reinforcement learning."
        ),
        "disclaimer": (
            "These are wellness exercises, not medical treatment or therapy."
        ),
    }


class InterventionEngine:
    """Intervention Engine wrapper providing convenience recommendation methods."""
    
    def recommend(
        self,
        emotion: str,
        stressor: Optional[str] = None,
        severity: str = "MODERATE",
    ) -> Dict[str, Any]:
        result = recommend_interventions(
            emotion=emotion,
            stressor=stressor,
            wellness_state=severity,
            top_n=1,
        )
        items = result.get("interventions", [])
        if items:
            rec = items[0]
            rec["title"] = rec.get("name", "Guided Exercise")
            rec["category"] = rec.get("kind", "wellness")
            return rec
        return {
            "id": "GUIDED_BREATHING",
            "title": "Guided breathing exercise",
            "category": "mindfulness",
            "steps": ["Breathe in for 4s", "Hold 2s", "Breathe out for 6s"],
        }


def get_catalog() -> List[Dict[str, Any]]:
    return [
        {k: v for k, v in i.items() if k not in ("target_states",)}
        for i in INTERVENTIONS
    ]


intervention_engine = InterventionEngine()



