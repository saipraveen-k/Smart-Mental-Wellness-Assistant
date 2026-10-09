"""DistilBERT Text Emotion Model Fine-Tuning Module."""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

def run_text_training(epochs: int = 1, batch_size: int = 64, lr: float = 2e-5) -> None:
    from scripts.train_text import main as train_main
    sys.argv = ["train_text.py", "--epochs", str(epochs), "--batch-size", str(batch_size), "--lr", str(lr)]
    train_main()
