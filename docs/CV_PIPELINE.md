# Smart Mental Wellness Assistant — Computer Vision Pipeline Documentation

## Overview

The Computer Vision Subsystem analyzes facial expressions from webcam frames or uploaded images to estimate visible emotional signals.

> [!IMPORTANT]
> **Terminology & Boundary:** The UI and documentation strictly refer to this module as **Facial-Expression Analysis**, NOT "True Emotion Detection". Visible facial expressions are external communicative signals and do not guarantee internal emotional state.

---

## 1. Computer Vision Pipeline Architecture

```text
Input Image (Webcam / Upload)
             ↓
OpenCV Haar Cascade / DNN Face Detection
             ↓
    Face Detected?
      ├── NO  → Return {"success": false, "reason": "NO_FACE_DETECTED"}
      └── YES → Select Primary Face (Largest Bounding Box)
             ↓
Crop Face Bounding Box
             ↓
Grayscale Conversion & Resize (48 x 48)
             ↓
Min-Max Intensity Normalization [0, 1]
             ↓
FacialEmotionCNN (4-stage Conv-BN-ReLU-MaxPool)
             ↓
Softmax → 7 raw class probabilities
             ↓
Taxonomy Normalization → 5 wellness categories
```

---

## 2. FacialEmotionCNN Architecture

```text
Input: (1 x 48 x 48)

Block 1: Conv2d(1 -> 32, 3x3) -> BatchNorm2d -> ReLU -> Conv2d(32 -> 32, 3x3) -> ReLU -> MaxPool2d(2x2) -> Dropout(0.25)
Block 2: Conv2d(32 -> 64, 3x3) -> BatchNorm2d -> ReLU -> Conv2d(64 -> 64, 3x3) -> ReLU -> MaxPool2d(2x2) -> Dropout(0.25)
Block 3: Conv2d(64 -> 128, 3x3) -> BatchNorm2d -> ReLU -> Conv2d(128 -> 128, 3x3) -> ReLU -> MaxPool2d(2x2) -> Dropout(0.3)
Block 4: Conv2d(128 -> 256, 3x3) -> BatchNorm2d -> ReLU -> Conv2d(256 -> 256, 3x3) -> ReLU -> MaxPool2d(2x2) -> Dropout(0.3)

FC Layer 1: Linear(256 * 3 * 3 -> 512) -> BatchNorm1d -> ReLU -> Dropout(0.5)
FC Layer 2: Linear(512 -> 7) (Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise)
```

---

## 3. Face Detection & Multi-Face Policy

1. **No Face Detected:**
   - Immediately returns HTTP 200 with structured status:
     ```json
     {
       "success": false,
       "reason": "NO_FACE_DETECTED",
       "message": "No face detected in the image. Please center your face in clear lighting."
     }
     ```
2. **Multiple Faces Detected:**
   - Rather than choosing randomly, the pipeline automatically isolates the primary face (largest bounding box area $W \times H$) and reports:
     ```json
     {
       "faces_detected": 3,
       "selected_face": "primary_largest_bbox"
     }
     ```

---

## 4. Model Training Command

```bash
# Train FacialEmotionCNN on FER-2013 dataset
python scripts/train_cv.py --epochs 20 --batch-size 128
```

### Saved Artifacts (`ml/artifacts/cv_emotion/`)
- `model.pt` — PyTorch model weights checkpoint
- `labels.json` — 7-class FER labels & normalization mapping
- `metrics.json` — Accuracy, Macro-F1, per-class metrics, confusion matrix
