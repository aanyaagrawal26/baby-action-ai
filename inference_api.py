"""
Baby Action AI — Infant Pose Inference Service
FastAPI / uvicorn service that accepts image uploads, runs MediaPipe Pose,
and classifies the detected posture using the trained MLP checkpoint.

Ported from mediapipe 0.10.x (mp.solutions) to mediapipe 0.10.30+
(mp.tasks.vision.PoseLandmarker). The landmark attributes (x, y, z, visibility)
and all downstream MLP logic are unchanged.

Required files:
  models/infant_pose_mlp.pt         — trained MLP checkpoint
  models/pose_landmarker_lite.task  — MediaPipe pose landmarker bundle
      Download: curl -L "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task" -o models/pose_landmarker_lite.task
"""
from __future__ import annotations

import io
import json
import os
import threading
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from torch import nn

try:
    import mediapipe as mp
    _BaseOptions = mp.tasks.BaseOptions
    _PoseLandmarker = mp.tasks.vision.PoseLandmarker
    _PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    _RunningMode = mp.tasks.vision.RunningMode
    _MpImage = mp.Image
    _MpImageFormat = mp.ImageFormat
except ImportError:
    mp = None  # type: ignore[assignment]

ROOT = Path(__file__).resolve().parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "models" / "infant_pose_mlp.pt"))
POSE_TASK_PATH = Path(
    os.getenv("POSE_TASK_PATH", ROOT / "models" / "pose_landmarker_lite.task")
)
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.56"))
SMOOTHING_WINDOW = int(os.getenv("SMOOTHING_WINDOW", "5"))
LABELS = ["supine", "prone", "sitting", "standing", "all-fours"]


# ---------------------------------------------------------------------------
# Model architecture (must match the trained checkpoint)
# ---------------------------------------------------------------------------

class PoseMLP(nn.Module):
    def __init__(self, n_classes: int = 5) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(51, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.25),
            nn.Linear(128, 96), nn.LayerNorm(96), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(96, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Pose normalisation (unchanged from original)
# ---------------------------------------------------------------------------

def normalize_keypoints(points: np.ndarray) -> np.ndarray:
    """Hip-centred, scale-normalised 51-D feature vector."""
    points = np.asarray(points, dtype=np.float32)
    hip = points[0]
    if not np.isfinite(hip).all():
        hip = np.nanmean(points, axis=0)
    centered = points - hip
    shoulder = points[8]
    torso = float(np.linalg.norm(shoulder - hip)) if np.isfinite(shoulder).all() else 0.0
    spread = np.nanpercentile(points[:, 0], 90) - np.nanpercentile(points[:, 0], 10)
    height = np.nanpercentile(points[:, 1], 90) - np.nanpercentile(points[:, 1], 10)
    scale = max(torso, spread, height, 1e-3)
    visibility = np.nan_to_num(points[:, 2], nan=0.0, posinf=0.0, neginf=0.0)
    xy = np.nan_to_num(
        (centered[:, :2] / scale).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0
    )
    return np.concatenate([xy, visibility]).astype(np.float32)


# ---------------------------------------------------------------------------
# Temporal smoother (unchanged)
# ---------------------------------------------------------------------------

class TemporalSmoother:
    def __init__(self, size: int):
        self.values: deque[np.ndarray] = deque(maxlen=max(size, 1))

    def update(self, probs: np.ndarray) -> np.ndarray:
        self.values.append(probs)
        return np.mean(np.stack(list(self.values)), axis=0)


# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------

app = FastAPI(title="Baby Action AI Infant Pose Inference")
model: PoseMLP | None = None
pose_landmarker: "_PoseLandmarker | None" = None  # type: ignore[type-arg]
smoother = TemporalSmoother(SMOOTHING_WINDOW)
lock = threading.Lock()


# ---------------------------------------------------------------------------
# Runtime initialisation
# ---------------------------------------------------------------------------

def load_runtime() -> None:
    global model, pose_landmarker

    # --- MLP checkpoint ---
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"MLP checkpoint not found: {MODEL_PATH}. "
            "Run train_public.py or ensure models/infant_pose_mlp.pt is present."
        )
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    global LABELS
    LABELS = checkpoint.get("labels", LABELS)
    model = PoseMLP(len(LABELS))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    # --- MediaPipe PoseLandmarker (tasks API) ---
    if mp is None:
        raise RuntimeError("mediapipe is not installed")

    if not POSE_TASK_PATH.exists():
        raise RuntimeError(
            f"MediaPipe pose task bundle not found: {POSE_TASK_PATH}.\n"
            "Download it with:\n"
            "  curl -L \"https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task\" "
            "-o models/pose_landmarker_lite.task"
        )

    options = _PoseLandmarkerOptions(
        base_options=_BaseOptions(model_asset_path=str(POSE_TASK_PATH)),
        running_mode=_RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.45,
        min_pose_presence_confidence=0.45,
        min_tracking_confidence=0.45,
    )
    pose_landmarker = _PoseLandmarker.create_from_options(options)


@app.on_event("startup")
def startup() -> None:
    load_runtime()


# ---------------------------------------------------------------------------
# Landmark extraction
# ---------------------------------------------------------------------------

def extract_points(image_rgb: np.ndarray) -> np.ndarray | None:
    """
    Run MediaPipe PoseLandmarker on an RGB image and return the
    normalised 51-D feature vector, or None if no pose is detected.

    Landmark index mapping (Human3.6M-compatible, same as original):
      0  hip_mid      (mean of 23, 24)
      1  right_hip    (24)
      2  right_knee   (26)
      3  right_ankle  (28)
      4  left_hip     (23)
      5  left_knee    (25)
      6  left_ankle   (27)
      7  spine        (mid of hip_mid and shoulder_mid)
      8  thorax       (mean of 11, 12)
      9  neck         (2/3 thorax + 1/3 nose)
      10 nose         (0)
      11 left_shoulder  (11)
      12 left_elbow    (13)
      13 left_wrist    (15)
      14 right_shoulder (12)
      15 right_elbow   (14)
      16 right_wrist   (16)
    """
    if pose_landmarker is None:
        raise RuntimeError("PoseLandmarker is not initialised")

    mp_image = _MpImage(image_format=_MpImageFormat.SRGB, data=image_rgb)
    result = pose_landmarker.detect(mp_image)

    if not result.pose_landmarks:
        return None

    lm = result.pose_landmarks[0]  # first detected person

    def pt(i: int) -> list[float]:
        return [lm[i].x, lm[i].y, float(getattr(lm[i], "visibility", 1.0))]

    hip = np.mean([pt(23), pt(24)], axis=0)
    shoulder_mid = np.mean([pt(11), pt(12)], axis=0)
    spine = (hip + shoulder_mid) / 2.0
    thorax = shoulder_mid
    neck = (thorax * 2.0 + pt(0)) / 3.0

    points = np.array([
        hip,        # 0 hip_mid
        pt(24),     # 1 right_hip
        pt(26),     # 2 right_knee
        pt(28),     # 3 right_ankle
        pt(23),     # 4 left_hip
        pt(25),     # 5 left_knee
        pt(27),     # 6 left_ankle
        spine,      # 7 spine
        thorax,     # 8 thorax
        neck,       # 9 neck
        pt(0),      # 10 nose
        pt(11),     # 11 left_shoulder
        pt(13),     # 12 left_elbow
        pt(15),     # 13 left_wrist
        pt(12),     # 14 right_shoulder
        pt(14),     # 15 right_elbow
        pt(16),     # 16 right_wrist
    ], dtype=np.float32)

    return normalize_keypoints(points)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict_image(image_bgr: np.ndarray) -> dict:
    # Convert BGR (from opencv) to RGB for mediapipe
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    features = extract_points(image_rgb)
    if features is None:
        return {
            "action": "unknown",
            "confidence": 0.0,
            "framesAnalyzed": 0,
            "modelProvenance": "InfActPrimitive-public-pose-finetune",
            "reason": "No confident person pose detected",
        }

    with torch.no_grad():
        logits = model(torch.from_numpy(features).unsqueeze(0))  # type: ignore[arg-type]
        probs = torch.softmax(logits, dim=1).numpy()[0]

    with lock:
        stable = smoother.update(probs)

    idx = int(stable.argmax())
    confidence = float(stable[idx])
    labels: list[str] = LABELS
    action = labels[idx] if confidence >= CONFIDENCE_THRESHOLD else "unknown"

    return {
        "action": action,
        "confidence": confidence,
        "framesAnalyzed": 1,
        "modelProvenance": "InfActPrimitive-public-pose-finetune",
        "rawAction": labels[idx],
        "smoothed": True,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    return {
        "ok": model is not None and pose_landmarker is not None,
        "model": str(MODEL_PATH),
        "poseTask": str(POSE_TASK_PATH),
        "labels": LABELS,
        "smoothingWindow": SMOOTHING_WINDOW,
        "confidenceThreshold": CONFIDENCE_THRESHOLD,
    }


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)) -> dict:
    content = await file.read()
    if len(content) > 50 * 1024 * 1024:
        raise HTTPException(413, "File exceeds the 50 MB limit")

    if file.content_type and file.content_type.startswith("image/"):
        # PIL → numpy RGB → BGR for opencv convention used in predict_image
        image_rgb = np.array(Image.open(io.BytesIO(content)).convert("RGB"))
        image_bgr = image_rgb[:, :, ::-1].copy()
        return predict_image(image_bgr)

    raise HTTPException(
        415,
        "This service accepts image inputs only. "
        "Video inference is not yet supported by the FastAPI service.",
    )
