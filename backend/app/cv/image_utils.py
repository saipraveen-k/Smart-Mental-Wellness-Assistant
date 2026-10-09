"""Image decoding and face-detection helpers used by the CV pipeline."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

_cascade: Optional[cv2.CascadeClassifier] = None


def _get_cascade() -> cv2.CascadeClassifier:
    global _cascade
    if _cascade is None:
        cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
        if cascade.empty():
            raise RuntimeError(f"Could not load face cascade from {FACE_CASCADE_PATH}")
        _cascade = cascade
    return _cascade


def decode_face_image(img_bytes: bytes) -> Optional[np.ndarray]:
    """Decode encoded image bytes to a grayscale 48x48 array (or None)."""
    if not img_bytes:
        return None
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    if img.shape != (48, 48):
        img = cv2.resize(img, (48, 48), interpolation=cv2.INTER_AREA)
    return img


def decode_image(data: bytes) -> Optional[np.ndarray]:
    """Decode arbitrary image bytes to an BGR ndarray (or None when invalid)."""
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def detect_faces(image_bgr: np.ndarray, min_size: Tuple[int, int] = (36, 36)) -> List[Dict[str, Any]]:
    """Detect faces; returns list of {x, y, w, h, area} sorted by area desc."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    cascade = _get_cascade()
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=min_size)
    boxes = [
        {"x": int(x), "y": int(y), "w": int(w), "h": int(h), "area": int(w) * int(h)}
        for (x, y, w, h) in faces
    ]
    boxes.sort(key=lambda b: b["area"], reverse=True)
    return boxes


def crop_and_normalize(image_bgr: np.ndarray, box: Dict[str, Any]) -> np.ndarray:
    """Crop a face with a small margin and resize to 48x48 grayscale."""
    h_img, w_img = image_bgr.shape[:2]
    margin = int(0.1 * max(box["w"], box["h"]))
    x0 = max(0, box["x"] - margin)
    y0 = max(0, box["y"] - margin)
    x1 = min(w_img, box["x"] + box["w"] + margin)
    y1 = min(h_img, box["y"] + box["h"] + margin)
    crop = image_bgr[y0:y1, x0:x1]
    if crop.size == 0:
        return np.zeros((48, 48), dtype=np.uint8)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (48, 48), interpolation=cv2.INTER_AREA)
    gray = cv2.equalizeHist(gray)
    return gray
