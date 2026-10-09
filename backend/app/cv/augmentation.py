"""On-the-fly FER-2013 augmentation (numpy + OpenCV, no extra deps).

Applied only to training samples; validation/test are never augmented so that
reported metrics stay reproducible.
"""
from __future__ import annotations

import random

import cv2
import numpy as np


class FERAugmenter:
    """Light geometric augmentation suited to aligned 48x48 face crops."""

    def __init__(
        self,
        flip_prob: float = 0.5,
        shift_limit: int = 4,
        rotation_deg: float = 12.0,
        zoom_range: float = 0.1,
        brightness_delta: int = 20,
        seed: int = 42,
    ):
        self.flip_prob = flip_prob
        self.shift_limit = shift_limit
        self.rotation_deg = rotation_deg
        self.zoom_range = zoom_range
        self.brightness_delta = brightness_delta
        self.rng = random.Random(seed)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        out = img

        # Horizontal flip (faces are roughly symmetric; standard for FER)
        if self.rng.random() < self.flip_prob:
            out = cv2.flip(out, 1)

        # Rotation
        if self.rotation_deg > 0:
            angle = self.rng.uniform(-self.rotation_deg, self.rotation_deg)
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            out = cv2.warpAffine(out, M, (w, h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)

        # Translation
        if self.shift_limit > 0:
            dx = self.rng.randint(-self.shift_limit, self.shift_limit)
            dy = self.rng.randint(-self.shift_limit, self.shift_limit)
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            out = cv2.warpAffine(out, M, (w, h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)

        # Zoom
        if self.zoom_range > 0:
            scale = 1.0 + self.rng.uniform(-self.zoom_range, self.zoom_range)
            M = cv2.getRotationMatrix2D((w / 2, h / 2), 0.0, scale)
            out = cv2.warpAffine(out, M, (w, h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)

        # Brightness jitter
        if self.brightness_delta > 0:
            delta = self.rng.randint(-self.brightness_delta, self.brightness_delta)
            out = cv2.convertScaleAbs(out, alpha=1.0, beta=delta)

        if out.shape != (h, w):
            out = cv2.resize(out, (w, h), interpolation=cv2.INTER_AREA)
        return out
