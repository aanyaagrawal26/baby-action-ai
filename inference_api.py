"""Local image-inference service for the Baby Action AI research preview.

A photo is classified independently. No pose probabilities are retained between
requests, so a new upload cannot inherit the previous photo's predicted posture.
"""
from __future__ import annotations

import io
import os
import threading
from pathlib import Path

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError
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
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
MAX_IMAGE_PIXELS = 32_000_000
MAX_IMAGE_SIDE = 2048
SUPPORTED_IMAGE_FORMATS = {"JPEG", "PNG", "WEBP"}
LABELS = ["supine", "prone", "sitting", "standing", "all-fours"]


class PoseMLP(nn.Module):
    def __init__(self, n_classes: int = 5) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(51, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(128, 96),
            nn.LayerNorm(96),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(96, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def normalize_keypoints(points: np.ndarray) -> np.ndarray:
    """Return the trained checkpoint's 51-D hip-centred feature vector."""
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


app = FastAPI(title="Baby Action AI Photo Inference")
model: PoseMLP | None = None
pose_landmarker: "_PoseLandmarker | None" = None  # type: ignore[type-arg]
landmarker_lock = threading.Lock()


def load_runtime() -> None:
    """Load the supplied research checkpoint and MediaPipe task bundle."""
    global model, pose_landmarker, LABELS

    if not MODEL_PATH.is_file():
        raise RuntimeError(f"MLP checkpoint not found: {MODEL_PATH}")

    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("state_dict"), dict):
        raise RuntimeError("The MLP checkpoint does not contain a compatible state_dict")
    labels = checkpoint.get("labels", LABELS)
    if not isinstance(labels, list) or not labels or not all(isinstance(item, str) for item in labels):
        raise RuntimeError("The MLP checkpoint has an invalid class-label list")

    candidate = PoseMLP(len(labels))
    candidate.load_state_dict(checkpoint["state_dict"])
    candidate.eval()

    if mp is None:
        raise RuntimeError("MediaPipe is not installed")
    if not POSE_TASK_PATH.is_file():
        raise RuntimeError(f"MediaPipe pose task bundle not found: {POSE_TASK_PATH}")

    options = _PoseLandmarkerOptions(
        base_options=_BaseOptions(model_asset_path=str(POSE_TASK_PATH)),
        running_mode=_RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.45,
        min_pose_presence_confidence=0.45,
        min_tracking_confidence=0.45,
    )
    landmarker = _PoseLandmarker.create_from_options(options)
    model = candidate
    LABELS = labels
    pose_landmarker = landmarker


@app.on_event("startup")
def startup() -> None:
    load_runtime()


def upright_posture_from_landmarks(landmarks: list) -> str | None:
    """Recognize only clear upright leg patterns; otherwise defer to the MLP."""
    def point(index: int) -> tuple[np.ndarray, float] | None:
        landmark = landmarks[index]
        xy = np.asarray([landmark.x, landmark.y], dtype=np.float32)
        visibility = float(np.nan_to_num(getattr(landmark, "visibility", 1.0), nan=0.0))
        if (
            not np.isfinite(xy).all()
            or visibility < 0.25
            or np.any(xy < -0.03)
            or np.any(xy > 1.03)
        ):
            return None
        return xy, visibility

    def joint(indices: tuple[int, ...]) -> tuple[np.ndarray, float] | None:
        visible = [row for index in indices if (row := point(index)) is not None]
        if not visible:
            return None
        xy = np.stack([row[0] for row in visible])
        weight = np.asarray([row[1] for row in visible], dtype=np.float32)
        center = np.average(xy, axis=0, weights=np.maximum(weight, 0.01))
        return center.astype(np.float32), float(weight.mean())

    head = point(0)
    shoulders = joint((11, 12))
    hips = joint((23, 24))
    knee_points = (point(25), point(26))
    ankle_points = (point(27), point(28))
    knees = joint((25, 26))
    ankles = joint((27, 28))
    if any(row is None for row in (head, shoulders, hips, knees, ankles, *knee_points, *ankle_points)):
        return None

    head_xy = head[0]
    shoulder_xy = shoulders[0]
    hip_xy = hips[0]
    knee_xy = knees[0]
    ankle_xy = ankles[0]
    knee_y_spread = abs(float(knee_points[0][0][1] - knee_points[1][0][1]))

    torso = hip_xy - shoulder_xy
    torso_length = float(np.linalg.norm(torso))
    torso_verticality = float(torso[1] / max(torso_length, 1e-6))
    head_above_hip = float(hip_xy[1] - head_xy[1])

    # Upright labels require a head-up torso and visible knees/ankles.
    # Tummy-time babies often lift the chest too, so torso angle alone is not
    # enough: the direction of both leg segments must also support the label.
    if torso[1] < 0.08 or head_above_hip < 0.08 or torso_verticality < 0.78:
        return None

    thigh = knee_xy - hip_xy
    shin = ankle_xy - knee_xy
    thigh_length = max(float(np.linalg.norm(thigh)), 1e-6)
    shin_length = max(float(np.linalg.norm(shin)), 1e-6)
    thigh_verticality = float(thigh[1] / thigh_length)
    shin_verticality = float(shin[1] / shin_length)

    # Require both knee-to-hip and ankle-to-knee to extend distinctly downward.
    if thigh[1] >= 0.04 and shin[1] >= 0.04 and thigh_verticality >= 0.72 and shin_verticality >= 0.66:
        return "standing"

    # Sitting commonly has knees near hip height and thighs reaching forward.
    # Reject knees that are lifted well above the hips (common in tummy-time).
    if (
        -0.02 <= float(thigh[1]) <= 0.06
        and abs(float(thigh[0])) >= max(0.04, abs(float(thigh[1])) * 1.2)
        and (float(shin[1]) >= 0.025 or abs(float(shin[0])) >= 0.045)
        and knee_y_spread <= 0.08
    ):
        return "sitting"

    return None


def body_pose_in_frame(landmarks: list) -> bool:
    """Require a head, torso, arm and leg evidence—not only guessed lower-body joints."""
    def visible(index: int) -> bool:
        landmark = landmarks[index]
        visibility = float(np.nan_to_num(getattr(landmark, "visibility", 1.0), nan=0.0))
        return bool(
            visibility >= 0.25
            and np.isfinite([landmark.x, landmark.y]).all()
            and -0.03 <= landmark.x <= 1.03
            and -0.03 <= landmark.y <= 1.03
        )

    return (
        visible(0)
        and sum(visible(index) for index in (11, 12)) >= 1
        and sum(visible(index) for index in (23, 24)) >= 1
        and sum(visible(index) for index in (13, 14, 15, 16)) >= 2
        and sum(visible(index) for index in (25, 26, 27, 28)) >= 2
    )


def prone_support_from_landmarks(landmarks: list) -> str | None:
    """Recognize a narrow chest-raised, arms-supported tummy-time pattern."""
    def point(index: int) -> np.ndarray | None:
        landmark = landmarks[index]
        visibility = float(np.nan_to_num(getattr(landmark, "visibility", 1.0), nan=0.0))
        xy = np.asarray([landmark.x, landmark.y], dtype=np.float32)
        if (
            visibility < 0.25
            or not np.isfinite(xy).all()
            or np.any(xy < -0.03)
            or np.any(xy > 1.03)
        ):
            return None
        return xy

    def joint(indices: tuple[int, ...]) -> np.ndarray | None:
        values = [value for index in indices if (value := point(index)) is not None]
        return np.mean(values, axis=0) if values else None

    head = point(0)
    shoulders = joint((11, 12))
    hips = joint((23, 24))
    knees = joint((25, 26))
    ankles = joint((27, 28))
    wrists = joint((15, 16))
    if any(value is None for value in (head, shoulders, hips, knees, ankles, wrists)):
        return None

    # A raised head, hands below the chest and both legs still above the hips
    # form a specific supported-prone pattern, unlike simple supine limb motion.
    if (
        head[1] < shoulders[1] - 0.02
        and wrists[1] > shoulders[1] + 0.10
        and knees[1] < hips[1] - 0.03
        and ankles[1] < hips[1] - 0.025
    ):
        return "prone"
    return None


def all_fours_support_from_landmarks(landmarks: list) -> str | None:
    """Match a clear head-up, arms-and-knees support pose without requiring visible feet."""
    def point(index: int) -> np.ndarray | None:
        landmark = landmarks[index]
        visibility = float(np.nan_to_num(getattr(landmark, "visibility", 1.0), nan=0.0))
        xy = np.asarray([landmark.x, landmark.y], dtype=np.float32)
        if (
            visibility < 0.25
            or not np.isfinite(xy).all()
            or np.any(xy < -0.03)
            or np.any(xy > 1.03)
        ):
            return None
        return xy

    def joint(indices: tuple[int, ...]) -> np.ndarray | None:
        values = [value for index in indices if (value := point(index)) is not None]
        return np.mean(values, axis=0) if values else None

    head = point(0)
    required_pairs = ((11, 12), (13, 14), (15, 16), (23, 24))
    if (
        head is None
        or float(np.nan_to_num(getattr(landmarks[0], "visibility", 1.0), nan=0.0)) < 0.30
        or any(point(index) is None for pair in required_pairs for index in pair)
    ):
        return None
    shoulders = joint((11, 12))
    elbows = joint((13, 14))
    wrists = joint((15, 16))
    hips = joint((23, 24))
    support_offsets = []
    for hip_index, knee_index in ((23, 25), (24, 26)):
        hip, knee = point(hip_index), point(knee_index)
        if hip is not None and knee is not None and abs(float(knee[0] - hip[0])) <= 0.26:
            support_offsets.append(float(knee[1] - hip[1]))
    hip_knee_support = any(offset >= 0.025 for offset in support_offsets)
    if any(value is None for value in (head, shoulders, elbows, wrists, hips)):
        return None

    # Two visibly extended legs are a stronger upright cue than a hands-and-
    # knees cue. A tucked/asymmetric knee pattern may still support all-fours.
    if len(support_offsets) >= 2 and min(support_offsets) > 0.12:
        return None

    # Look for a visible support chain: head above shoulders, hands below the
    # chest, and knees tucked under the hips. Ankles are deliberately optional.
    if (
        shoulders[1] - head[1] >= 0.08
        and hips[1] - shoulders[1] >= 0.16
        and wrists[1] - shoulders[1] >= 0.10
        and elbows[1] - shoulders[1] >= 0.04
        and hip_knee_support
        and abs(float(wrists[0] - shoulders[0])) >= 0.035
        and abs(float(wrists[0] - shoulders[0])) <= 0.24
    ):
        return "all-fours"
    return None


def recumbent_pose_is_ambiguous(landmarks: list) -> bool:
    """Flag lying-pose guesses where 2D landmarks do not distinguish back from tummy."""
    def point(index: int) -> np.ndarray | None:
        landmark = landmarks[index]
        visibility = float(np.nan_to_num(getattr(landmark, "visibility", 1.0), nan=0.0))
        xy = np.asarray([landmark.x, landmark.y], dtype=np.float32)
        return xy if visibility >= 0.25 and np.isfinite(xy).all() else None

    def joint(indices: tuple[int, ...]) -> np.ndarray | None:
        values = [value for index in indices if (value := point(index)) is not None]
        return np.mean(values, axis=0) if values else None

    head = point(0)
    shoulders = joint((11, 12))
    hips = joint((23, 24))
    if any(value is None for value in (head, shoulders, hips)):
        return False

    torso = hips - shoulders
    torso_length = max(float(np.linalg.norm(torso)), 1e-6)
    verticality = abs(float(torso[1])) / torso_length
    horizontality = abs(float(torso[0])) / torso_length
    head_above_shoulder = float(shoulders[1] - head[1])
    horizontal_lie = verticality <= 0.55 and horizontality >= 0.80
    unclear_upright_direction = verticality >= 0.80 and head_above_shoulder < 0.06
    return horizontal_lie or unclear_upright_direction


def extract_pose(image_rgb: np.ndarray) -> tuple[np.ndarray, str | None, bool, str | None] | None:
    """Detect one in-frame pose and return features plus a conservative shape match."""
    if pose_landmarker is None:
        raise RuntimeError("PoseLandmarker is not initialised")

    mp_image = _MpImage(image_format=_MpImageFormat.SRGB, data=image_rgb)
    with landmarker_lock:
        result = pose_landmarker.detect(mp_image)
    if not result.pose_landmarks:
        return None

    landmarks = result.pose_landmarks[0]
    body_in_frame = body_pose_in_frame(landmarks)
    all_fours_action = all_fours_support_from_landmarks(landmarks)
    if body_in_frame:
        geometry_action = (
            upright_posture_from_landmarks(landmarks)
            or prone_support_from_landmarks(landmarks)
            or all_fours_action
        )
    else:
        # A visible hands-and-knees support pose can still be classified when
        # the feet are cropped; unrelated partial detections remain unknown.
        geometry_action = all_fours_action
        if geometry_action is None:
            return None

    def pt(index: int) -> list[float]:
        point = landmarks[index]
        return [point.x, point.y, float(getattr(point, "visibility", 1.0))]

    hip = np.mean([pt(23), pt(24)], axis=0)
    shoulder_mid = np.mean([pt(11), pt(12)], axis=0)
    spine = (hip + shoulder_mid) / 2.0
    thorax = shoulder_mid
    neck = (thorax * 2.0 + pt(0)) / 3.0

    points = np.array(
        [
            hip,
            pt(24),
            pt(26),
            pt(28),
            pt(23),
            pt(25),
            pt(27),
            spine,
            thorax,
            neck,
            pt(0),
            pt(11),
            pt(13),
            pt(15),
            pt(12),
            pt(14),
            pt(16),
        ],
        dtype=np.float32,
    )
    ambiguity_signal = geometry_action is None and recumbent_pose_is_ambiguous(landmarks)
    return normalize_keypoints(points), geometry_action, ambiguity_signal, all_fours_action


def extract_points(image_rgb: np.ndarray) -> np.ndarray | None:
    """Compatibility helper returning only the checkpoint's 51-D features."""
    pose = extract_pose(image_rgb)
    return None if pose is None else pose[0]


def select_pose_geometry_action(
    model_action: str,
    pose_geometry_action: str | None,
    all_fours_action: str | None,
) -> str | None:
    """Prefer a clear pose-specific geometry match before using the all-fours fallback."""
    if pose_geometry_action == "all-fours" and model_action in {"standing", "sitting"}:
        return None
    if pose_geometry_action is not None:
        return pose_geometry_action
    if all_fours_action == "all-fours" and model_action in {"supine", "prone", "all-fours", "unknown"}:
        return all_fours_action
    return None


def predict_image(image_bgr: np.ndarray) -> dict:
    """Classify one image without carrying state across images or sessions."""
    if model is None:
        raise RuntimeError("The posture model is not initialised")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pose = extract_pose(image_rgb)
    if pose is None:
        return {
            "action": "unknown",
            "confidence": 0.0,
            "modelConfidence": 0.0,
            "decisionSource": "model",
            "framesAnalyzed": 1,
            "modelProvenance": "InfActPrimitive-public-pose-finetune",
            "reason": "No confident person pose detected",
            "classScores": [],
        }

    features, geometry_action, ambiguity_signal, all_fours_action = pose
    with torch.inference_mode():
        logits = model(torch.from_numpy(features).unsqueeze(0))
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]

    top_index = int(probabilities.argmax())
    score = float(probabilities[top_index])
    model_action = LABELS[top_index] if score >= CONFIDENCE_THRESHOLD else "unknown"
    geometry_action = select_pose_geometry_action(model_action, geometry_action, all_fours_action)
    geometry_override = geometry_action is not None and geometry_action != model_action
    pose_ambiguity = (
        geometry_action is None
        and model_action in {"supine", "prone"}
        and ambiguity_signal
    )
    action = (
        geometry_action
        if geometry_override
        else "unknown" if pose_ambiguity else model_action
    )
    class_scores = [
        {"action": LABELS[index], "score": float(probabilities[index])}
        for index in np.argsort(probabilities)[::-1]
    ]

    return {
        "action": action,
        "confidence": 0.0 if geometry_override or pose_ambiguity else score,
        "modelConfidence": score,
        "decisionSource": (
            "pose_geometry" if geometry_override
            else "pose_ambiguity" if pose_ambiguity
            else "model"
        ),
        "framesAnalyzed": 1,
        "modelProvenance": "InfActPrimitive-public-pose-finetune",
        "rawAction": LABELS[top_index],
        "reason": (
            f"A separate pose-shape check matched {geometry_action}; the learned classifier favored {LABELS[top_index]}"
            if geometry_override
            else "Ambiguous pose direction; label withheld" if pose_ambiguity
            else None if action != "unknown" else "Below the model confidence threshold"
        ),
        "classScores": class_scores,
    }


def decode_photo(content: bytes) -> np.ndarray:
    """Decode a supported photo, honor EXIF rotation, and cap inference dimensions."""
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded photo is empty")

    try:
        with Image.open(io.BytesIO(content)) as original:
            if original.format not in SUPPORTED_IMAGE_FORMATS:
                raise HTTPException(
                    status_code=415,
                    detail="Use a JPEG, PNG or WebP photo. This model does not accept videos.",
                )
            image = ImageOps.exif_transpose(original)
            width, height = image.size
            if width < 32 or height < 32:
                raise HTTPException(
                    status_code=400,
                    detail="This photo is too small to analyze. Choose a larger image.",
                )
            if width * height > MAX_IMAGE_PIXELS:
                raise HTTPException(
                    status_code=413,
                    detail="This photo has too many pixels. Choose a smaller image.",
                )
            image.load()
            image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE), Image.Resampling.LANCZOS)
            if image.mode in ("RGBA", "LA") or (
                image.mode == "P" and "transparency" in image.info
            ):
                rgba = image.convert("RGBA")
                background = Image.new("RGB", rgba.size, (255, 255, 255))
                background.paste(rgba, mask=rgba.getchannel("A"))
                image_rgb = np.asarray(background, dtype=np.uint8)
            else:
                image_rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    except HTTPException:
        raise
    except Image.DecompressionBombError as error:
        raise HTTPException(
            status_code=413, detail="This photo has too many pixels. Choose a smaller image."
        ) from error
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise HTTPException(
            status_code=400,
            detail="We couldn’t read this image. Try a valid JPEG, PNG or WebP photo.",
        ) from error

    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)


@app.get("/health")
def health() -> dict:
    return {
        "ok": model is not None and pose_landmarker is not None,
        "labels": LABELS,
        "confidenceThreshold": CONFIDENCE_THRESHOLD,
        "statelessPhotoInference": True,
    }


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)) -> dict:
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Photo exceeds the 50 MB upload limit")

    return predict_image(decode_photo(content))
