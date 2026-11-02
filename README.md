# Baby Action AI

A responsive research prototype for still-photo posture review and short-clip posture-transition demonstrations. Choose one JPEG, PNG or WebP photo for a single pose estimate, or review a short MP4/WebM clip through sequential sampled frames and a transparent event rule.

> **Research prototype only.** This is not a medical, safety or developmental assessment. The supplied InfActPrimitive model package states that it is for non-commercial research, teaching, scientific publication and personal experimentation and is not intended for product deployment. Review the source terms before using or distributing this model as a product.

## What it can recognize

The supplied classifier has five pose labels: `standing`, `sitting`, `all-fours`, `prone` (on the tummy), and `supine` (on the back). The learned model returns `unknown` when MediaPipe does not find a usable pose or its highest score is below the 0.56 threshold, unless one of the clearly disclosed, narrow posture-shape safeguards applies. A still photo can show an all-fours posture; it cannot establish that a baby is moving or crawling. The model detects one person pose and does not verify that the person is a baby.

Scores are relative softmax outputs, not calibrated probabilities. The checkpoint's validation report lists 64.21% frame accuracy and 55.45% balanced accuracy over pose-sequence data, with weaker validation recall for sitting and standing. The source data contains pose landmarks rather than raw RGB baby photos, so these numbers are not a guarantee for a user's photo.

## Conservative posture safeguards

For a clearly visible upright pose, the service checks both knee-and-ankle chains and torso geometry for standing or sitting. A narrow chest-raised, arms-supported tummy-time pattern can also override the learned classifier. Clear upright and supported-tummy geometry are resolved before the all-fours fallback.

A head-up hands-and-knees posture can be recognized without the baby's feet or both knees in frame, but only when the landmarks supporting that shape are strong: the head must be at least 0.30 visibility; both shoulders, elbows, wrists and hips must be in-frame at 0.25 visibility or higher; and at least one same-side hip-to-knee support pair must also be at 0.25 or higher. A low-confidence singleton cannot create a synthetic center point. This all-fours correction can replace a learned back/tummy/all-fours/unknown label when the separate support shape is clear, but never a learned standing or sitting label. The UI names the shape-based correction and shows all five raw model scores separately.

These rules are cautious pose-shape estimates, not calibrated confidence or a validation guarantee. Other cropped, low-visibility, partly covered or unusually angled photos can still return `unknown` or an incorrect label. When 2D landmarks cannot distinguish back from tummy, the service withholds the guess rather than treating a high score as proof.

## Photo privacy and limits

The browser sends the selected photo through the app's same-origin API proxy to the local inference service. The service processes the photo in memory and does not save the image. The photo is not used by the optional explanation endpoint; that endpoint receives only the predicted label and score. Good light and a clear view of the baby's head, shoulders, hips and supporting limbs help. Full-body framing is useful but is not mandatory for every supported shape.

## Clip event review and alert demonstration

The event workbench accepts MP4/WebM clips between 2.5 and 20 seconds and at most 50 MB. The original video stays in the browser. It samples up to eight compressed still frames, sends them sequentially to the existing image endpoint, and discards temporary canvas data. It does not store the clip or sampled frames. A usable event candidate requires at least two consecutive **model-only** samples before and two after a supported posture change, with each raw model score at or above 0.70; unknown, ambiguous, weak, geometry-overridden and long-gap samples break the evidence chain. The score is uncalibrated and this guard is not validated event reliability.

The page may display a local **demo inbox alert preview** for a qualifying posture transition, with the clip-relative time and supporting sample counts. It never sends push notifications, SMS, email or parent-account messages. It does not run in the background or monitor continuously. A scripted “See example” walkthrough is labeled as simulated and is not a baby video or classifier prediction.

## Why use GenAI alongside computer vision?

MediaPipe and the existing neural network perform the visual labeling; a deterministic rule checks repeated labels and decides whether to show a candidate. GenAI is only an optional language layer after that decision: it can turn the fixed, allow-listed posture labels and time into a short caregiver-readable summary. The event-summary endpoint receives structured text only—not a photo, video or open-ended prompt. It cannot change a classification or trigger an alert, and a deterministic text fallback is used when GenAI is unavailable. The benefit is flexible wording, **not** better vision accuracy; if personalized wording is not needed, a template is simpler.

## Alert reliability boundary

This is a capstone research demo, not a dependable baby monitor. The current model reports 64.21% frame accuracy and 55.45% balanced accuracy on pose-sequence validation; these values do not measure video-event accuracy, false alerts per hour, alert latency or recipient-delivery reliability. A real notification service would require caregiver consent and verified recipient setup, a secure background provider/queue, retry and delivery receipts, rate limiting, audit/privacy controls, and independent event-level evaluation on representative consented videos. No safety, health, developmental, danger or emergency meaning is inferred here.

## Run locally

Requirements: Node.js 20+, pnpm 10.4.1 (pinned in `package.json`), and Python 3.10–3.14 with pip.

```bash
pnpm install
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Start the photo-analysis API in one terminal:

```bash
.venv/bin/uvicorn inference_api:app --host 127.0.0.1 --port 8000
```

In a second terminal, start the website:

```bash
pnpm dev
```

Open `http://localhost:3000`. The website proxies still frames to `http://127.0.0.1:8000` by default. Set `INFERENCE_API_URL` to change that internal target. The optional `/api/event-summary` route uses the configured built-in language model when available and automatically falls back to a fixed summary. The FastAPI task bundle and checkpoint are included under `models/` in this research copy. The browser interface caps photos at 15 MB and video clips at 50 MB and 20 seconds; the API image limit is 50 MB.

## Checks

```bash
pnpm run check
pnpm test
pnpm run build
.venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v
```

Python runtime dependencies are listed in `requirements.txt`. `GET /health` reports whether the model and pose task are loaded; `POST /analyze` accepts a multipart photo. The app provides `GET /api/infer/health` and `POST /api/infer` through the website origin.

See [`FACULTY_RESPONSE.md`](FACULTY_RESPONSE.md) for the faculty feedback response, design rationale, evidence threshold and prerequisites for real notifications.

## Attribution and source terms

The checkpoint was trained on the public InfActPrimitive pose-only release described in `REAL_INFANT_MODEL_README.md`, including its official source links and non-commercial research-use warning. The source model and associated pose-task license/terms should be reviewed before any distribution or public/product deployment. Do not include private project configuration, API keys, environment files or another user's Git history when packaging this project.
