"""FacialEmotionCNN CV Model Training Module."""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

def run_cv_training(epochs: int = 15, batch_size: int = 128, lr: float = 1e-3) -> None:
    from scripts.train_cv import main as train_cv_main
    sys.argv = ["train_cv.py", "--epochs", str(epochs), "--batch-size", str(batch_size), "--lr", str(lr)]
    train_cv_main()
