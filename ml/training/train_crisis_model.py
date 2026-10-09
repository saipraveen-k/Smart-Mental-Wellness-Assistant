"""C-SSRS Crisis Classifier Training Module."""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

def run_crisis_training() -> None:
    from scripts.train_crisis import main as train_crisis_main
    train_crisis_main()
