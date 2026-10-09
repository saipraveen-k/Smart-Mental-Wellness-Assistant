"""Facial-expression CNN architecture (FER-2013, 7 classes, 48x48 grayscale).

A compact VGG-style CNN. The architecture is described in
docs/CV_PIPELINE.md; Grad-CAM explanations use ``conv_features`` (the last
convolutional block).
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FacialEmotionCNN(nn.Module):
    """3 conv blocks (32/64/128) + spatial dropout + 2 FC heads -> 7 classes."""

    ARCH_NAME = "FacialEmotionCNN-v1"

    def __init__(self, num_classes: int = 7, dropout: float = 0.4):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_p = dropout

        self.conv_features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),   # 48x48
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                              # 24x24
            nn.Dropout2d(0.25),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                              # 12x12
            nn.Dropout2d(0.25),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                              # 6x6
            nn.Dropout2d(0.25),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)                # 128
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.conv_features(x)
        return self.classifier(self.gap(feats))

    def arch_config(self) -> Dict:
        return {"arch": self.ARCH_NAME, "num_classes": self.num_classes,
                "dropout": self.dropout_p}

    @classmethod
    def from_config(cls, cfg: Dict) -> "FacialEmotionCNN":
        return cls(num_classes=cfg.get("num_classes", 7), dropout=cfg.get("dropout", 0.4))


def build_class_weights(labels: np.ndarray, num_classes: int) -> List[float]:
    """Inverse-frequency class weights normalized to mean 1.0."""
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    total = counts.sum()
    weights = total / (num_classes * counts)
    return [float(w) for w in weights]
