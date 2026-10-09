# Smart Mental Wellness Assistant — ML Pipeline Documentation

## Overview

The Machine Learning Subsystem processes user-generated text to produce:
1. Normalized 5-class emotion probability distributions (Joy, Sadness, Anger, Anxiety, Neutral).
2. Word-level explainability attributions.
3. Multi-label stressor classifications (Academic, Career, Relationships, Family, Financial, Health, Sleep, Social, Work, Other).

---

## 1. Text Emotion Architecture (`DistilBERT`)

- **Base Architecture:** `distilbert-base-uncased` (6 transformer layers, 768 hidden dimensions, 12 attention heads).
- **Classification Head:** Linear layer projecting pooled representations $\mathbb{R}^{768} \to \mathbb{R}^5$.
- **Loss Function:** Class-Weighted Cross-Entropy Loss to handle class imbalance (e.g. Anxiety frequency vs. Joy/Neutral frequency).

```text
User Text Input
      ↓
Light Cleaning (whitespace normalization, lowercasing)
      ↓
DistilBERT Tokenizer (Max Length = 64)
      ↓
DistilBERT Encoder
      ↓
Classification Head
      ↓
Softmax Layer → Probabilities [P(Joy), P(Sadness), P(Anger), P(Anxiety), P(Neutral)]
```

---

## 2. Text Explainability Engine

To provide transparent predictions without fabricating explanations, the model uses token-level gradient attribution (or integrated gradients fallback):

$$\text{Attribution}(t) = \left| \frac{\partial P(c)}{\partial e_t} \odot e_t \right|$$

Where $e_t$ is the embedding vector for token $t$ and $P(c)$ is the predicted class probability.

- Returns top contributing tokens (e.g., `"anxious"`, `"overwhelmed"`, `"cannot sleep"`).

---

## 3. Stressor Detection Subsystem

Identifies contextual domain drivers for distress:

- **Categories:** Academic, Career, Relationships, Family, Financial, Health, Sleep, Social, Work, Other.
- **Methodology:** Hybrid rule-based pattern extractor + TF-IDF multi-label classifier.
- **Output:** Category name, confidence score, and matched context keywords.

---

## 4. Model Training & Evaluation Command

```bash
# Fine-tune DistilBERT text emotion model
python scripts/train_text.py --epochs 2 --batch-size 32

# Model evaluation
python scripts/evaluate.py
```

### Saved Artifacts (`ml/artifacts/text_emotion/`)
- `model/` — PyTorch model weights (`pytorch_model.bin` / `model.safetensors`)
- `tokenizer/` — DistilBERT tokenizer config and vocab files
- `labels.json` — Label-to-ID mappings and taxonomy definition
- `metrics.json` — Evaluation metrics (Accuracy, Macro-F1, Per-Class F1, Confusion Matrix)
