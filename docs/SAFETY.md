# Smart Mental Wellness Assistant — Safety Framework & Crisis Protocol

## 1. Safety Principles & Limitations

> [!CAUTION]
> **Non-Clinical & Non-Therapeutic System:**
> - The Smart Mental Wellness Assistant is **not** a licensed therapist, clinician, or emergency crisis hotline.
> - The system does **not** make clinical diagnoses (e.g. Clinical Depression, Generalized Anxiety Disorder).
> - The system does **not** claim to prevent self-harm or suicide with 100% certainty.

---

## 2. Crisis Subsystem Architecture

The crisis detection subsystem runs independently of the standard text emotion classifier to ensure maximum sensitivity and recall for severe distress.

```text
User Input Text
       ↓
Crisis Classifier Subsystem (C-SSRS Trained Model)
       ↓
Assign Risk Tier: LOW | MODERATE | HIGH
       ↓
Is Risk HIGH or MODERATE?
  ├── NO (LOW)  → Proceed with standard wellness recommendations & chat.
  └── YES       → Trigger Safety Override UI & Emergency Resource Banner.
```

### Risk Tier Definitions

1. **LOW:** Everyday expression of feelings (including mild sadness, stress, or excitement).
2. **MODERATE:** Expressions of deep helplessness, passive ideation, or intense emotional distress without explicit self-harm intent.
3. **HIGH:** Explicit mentions of self-harm intent, suicidal ideation, or concrete plans.

---

## 3. Safety Override & Crisis Interface

When a **HIGH** risk score is detected:

1. **Motivational Response Suppression:** Standard casual conversational responses and lighthearted recommendations are suppressed.
2. **Safety Overlay Display:** The UI immediately displays prominent emergency helpline numbers based on user region or default international hotlines:
   - **United States:** 988 (Suicide & Crisis Lifeline)
   - **United Kingdom:** 111 (NHS) / 116 123 (Samaritans)
   - **India:** 9152987821 (KIRAN) / 112 (Emergency)
   - **International:** [Befrienders Worldwide](https://www.befrienders.org/)
3. **De-escalation Language:** Warm, empathetic, non-judgmental prompts encouraging the user to reach out to a professional or trusted contact.
