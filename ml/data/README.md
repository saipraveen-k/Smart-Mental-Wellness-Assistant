# ML Data Directory

This directory contains configuration, cache metadata, and schemas for datasets dynamically loaded from Hugging Face Hub:

1. **Text Dataset**: `SetFit/go_emotions` (GoEmotions 27 fine-grained emotions mapped into wellness taxonomy)
2. **CV Dataset**: `Jeneral/fer-2013` (7 facial expression classes: angry, disgust, fear, happy, neutral, sad, surprise)
3. **Crisis Dataset**: `av9ash/CSSR-S_labelled_suicidewatch_posts_reddit` (C-SSRS labeled suicidal ideation/behavior risk dataset)

> **Note**: Raw dataset files are dynamically downloaded and cached locally in `~/.cache/huggingface/datasets`. Raw data binaries are omitted from Git tracking via `.gitignore`.
