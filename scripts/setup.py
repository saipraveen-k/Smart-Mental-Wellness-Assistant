#!/usr/bin/env python
"""Environment setup and initialization script for Smart Mental Wellness Assistant.

Usage:
    python scripts/setup.py
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

def setup_environment():
    print("[1/3] Ensuring required directories exist ...")
    dirs = [
        REPO_ROOT / "ml" / "artifacts" / "text_emotion",
        REPO_ROOT / "ml" / "artifacts" / "cv_emotion",
        REPO_ROOT / "ml" / "artifacts" / "crisis",
        REPO_ROOT / "docs",
        REPO_ROOT / "backend" / "tests",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        print(f"      [OK] {d.relative_to(REPO_ROOT)}")

    print("[2/3] Checking environment configuration (.env) ...")
    env_file = REPO_ROOT / ".env"
    env_example = REPO_ROOT / ".env.example"
    if not env_file.exists():
        if env_example.exists():
            shutil.copy(env_example, env_file)
            print("      [OK] Created .env from .env.example")
        else:
            print("      [!] .env.example not found")
    else:
        print("      [OK] .env file present.")

    print("[3/3] Verification of core dependencies ...")
    try:
        import torch
        import transformers
        import datasets
        import cv2
        import fastapi
        import pymongo
        print("      [OK] Core dependencies import successfully.")
        print(f"        PyTorch {torch.__version__} | CUDA Available: {torch.cuda.is_available()}")
    except ImportError as exc:
        print(f"      [X] Missing dependency: {exc}")
        return 1

    print("Setup completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(setup_environment())

