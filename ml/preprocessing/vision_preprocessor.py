"""Computer Vision Preprocessing & Face Detection Guard."""
from __future__ import annotations

import cv2
import numpy as np
from typing import Tuple, Optional

# OpenCV Haar Cascade Classifier path
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

FER_CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

CV_TAXONOMY_MAPPING = {
    "happy": "Joy",
    "sad": "Sadness",
    "angry": "Anger",
    "fear": "Anxiety",
    "neutral": "Neutral",
    "disgust": "Anger",
    "surprise": "Joy",
}

def crop_and_align_face(image: np.ndarray, target_size: Tuple[int, int] = (48, 48)) -> Tuple[np.ndarray, bool]:
    """Detects primary face in image, crops, converts to grayscale, and resizes to target_size."""
    if image is None or image.size == 0:
        return np.zeros(target_size, dtype=np.uint8), False

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )

    if len(faces) == 0:
        # Fallback to center-crop if face detection fails on clear cropped faces
        h, w = gray.shape[:2]
        crop = gray[int(h*0.1):int(h*0.9), int(w*0.1):int(w*0.9)]
        resized = cv2.resize(crop, target_size, interpolation=cv2.INTER_AREA)
        return resized, False

    # Select primary face (largest bounding box)
    largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
    x, y, w, h = largest_face
    face_crop = gray[y:y+h, x:x+w]
    resized = cv2.resize(face_crop, target_size, interpolation=cv2.INTER_AREA)
    return resized, True

class VisionPreprocessor:
    """Preprocesses images for FacialEmotionCNN."""
    
    def __init__(self, img_size: int = 48):
        self.img_size = img_size

    def preprocess(self, raw_bytes: bytes) -> Tuple[np.ndarray, bool]:
        nparr = np.frombuffer(raw_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return np.zeros((self.img_size, self.img_size), dtype=np.uint8), False
        return crop_and_align_face(img, (self.img_size, self.img_size))
