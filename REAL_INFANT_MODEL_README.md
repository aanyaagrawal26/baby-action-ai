# Baby Action AI — Public Infant Pose Model

This is the improved real-data model package for Baby Action AI. It trains on the official public **InfActPrimitive** infant pose release from Northeastern University / Augmented Cognition Lab.

## What changed

- Replaced synthetic-only training with real infant pose annotations.
- Corrected the feature layout to the release's **17 Human3.6M joints**.
- Added hip-centered and torso/extent-scaled normalization.
- Added per-joint visibility features.
- Added class-weighted cross-entropy for imbalanced labels.
- Added mild pose-jitter and horizontal-mirror augmentation.
- Added label smoothing, AdamW, gradient clipping, and validation checkpointing.
- Added confidence gating: unclear frames return `unknown`.
- Added temporal probability smoothing for live-camera stability.
- Added a FastAPI image endpoint using MediaPipe Pose.

## Dataset and license

The downloaded release is the public InfActPrimitive pose-only dataset:

- Official repository: https://github.com/ostadabbas/Video-Based-Infant-Action-Recognition
- Official data link: https://drive.google.com/file/d/1TiuTul5b5XtJgKZeOCnrAH8WKmxb6Rld/view?usp=sharing
- Dataset release: 792 annotated posture sequences, split into train/validation IDs.
- Labels: `supine`, `prone`, `sitting`, `standing`, `all-fours`.
- The dataset provides skeleton/pose data, not raw RGB baby images or videos.
- The official repository states the dataset is for non-commercial research, teaching, scientific publication, and personal experimentation, and is not intended for diagnosis or incorporation into a product. Review the official terms before deployment.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train

The downloaded pickle is expected at:

```text
data/infactprimitive/InfActPrimitive/2d/InfAct_plus.pkl
```

Run:

```bash
python train_public.py \
  --data data/infactprimitive/InfActPrimitive/2d/InfAct_plus.pkl \
  --out models \
  --epochs 50 \
  --max-frames-per-sequence 24
```

Outputs:

- `models/infant_pose_mlp.pt` — best validation checkpoint.
- `models/labels.json` — class order.
- `models/metrics.json` — accuracy, balanced accuracy, per-class report, and confusion matrix.
- `models/training_history.json` — epoch history.

## Training result from this run

The corrected run used 9,600 sampled train frames and 9,307 validation frames. The best checkpoint was selected by balanced accuracy:

- Frame accuracy: **64.21%**
- Balanced accuracy: **55.45%**
- Mean confidence: **77.87%**

This is a validation result on pose sequences from the public release, not a guarantee for arbitrary webcam images. The validation set is highly imbalanced toward supine/prone examples, and performance is weaker for sitting/standing. Collecting consented RGB examples from the target camera setup is the next accuracy step.

## Run the live-image API

```bash
uvicorn inference_api:app --host 127.0.0.1 --port 8000
```

Check health:

```bash
curl http://127.0.0.1:8000/health
```

Analyze an image:

```bash
curl -X POST http://127.0.0.1:8000/analyze \
  -F file=@/path/to/image.jpg
```

Environment settings:

```bash
MODEL_PATH=models/infant_pose_mlp.pt
CONFIDENCE_THRESHOLD=0.56
SMOOTHING_WINDOW=5
```

The API accepts images and uses MediaPipe Pose to extract landmarks. Video/live-camera clients should send sampled JPEG frames. The website's `/api/infer` proxy can forward these frames to this service.

## Important limitation

The public release contains pose data without raw RGB images. Therefore this model can be trained and evaluated on infant pose features, but live-image performance depends on MediaPipe successfully estimating an infant pose in the camera environment. For production-quality live images, fine-tune with consented RGB frames/videos from the actual camera and lighting setup, using person-held-out validation.
