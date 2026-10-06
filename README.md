# Baby Action AI — GestureVerse-AI

**Applied Generative AI | 5th Semester B.Tech | Review-2 Project**

A local-first infant posture recognition system that combines a trained MLP classifier with Generative AI to produce natural-language explanations of detected actions.

> **Research prototype only.** Not a medical device. Not validated for clinical, developmental, or safety decisions.

---

## Project Objective

> Recognise infant posture/action from visual input and transform the model's structured prediction into understandable natural-language information.

---

## Architecture Overview

```
Image / Video Input
       ↓
MediaPipe Pose Detection  (33 landmarks → 17 H3.6M joints)
       ↓
Pose Normalisation        (hip-centred, scale-normalised)
       ↓
51-D Feature Construction (34 normalised xy + 17 visibility scores)
       ↓
Trained MLP Classifier    (2-layer neural network, 5 posture classes)
       ↓
Confidence & Unknown Handling  (5-frame smoothing, threshold 0.56)
       ↓
Prompt Construction       (Strategy A / B / C — server-side)
       ↓
LLM / Generative AI       (Forge API — OpenAI-compatible)
       ↓
Natural-Language Explanation  (displayed with prediction + confidence)
```

**The LLM does not perform classification. The MLP does not generate language. These are two separate systems.**

---

## Features

- **Image upload inference** — upload a JPG/PNG and get a real MLP prediction
- **Live camera inference** — webcam frames sampled every 2.2 s, sent to MLP
- **Three prompt engineering strategies:**
  - Strategy A: Simple one-sentence description
  - Strategy B: Caregiver-friendly explanation (no medical claims)
  - Strategy C: Technical analysis distinguishing ML from GenAI output
- **Prompt Engineering Lab** — compare all three strategies side by side
- **Accurate pipeline visualization** — 10-stage numbered diagram
- **Model analysis section** — real validation metrics, per-class recall chart
- **Rule-based fallback** — if LLM key is missing/fails, a deterministic fallback is clearly labelled

---

## Technology Stack

| Layer | Technology |
|---|---|
| Frontend | React 19, TypeScript, Tailwind CSS v4, tRPC, Recharts |
| Backend | Node.js, Express, tRPC |
| ML Inference | Python 3.14, FastAPI, uvicorn |
| Pose Detection | MediaPipe Pose (Tasks API) |
| Classifier | PyTorch — 2-layer normalized MLP |
| Generative AI | Forge API (OpenAI-compatible LLM) |
| Package manager | pnpm |

---

## ML Model Details

| Property | Value |
|---|---|
| Architecture | 2-layer normalized MLP |
| Input | 51-D (34 normalised xy + 17 visibility) |
| Classes | supine, prone, sitting, standing, all-fours |
| Dataset | InfActPrimitive public pose-only release |
| Frame accuracy | 64.21% (offline validation, 9 307 frames) |
| Balanced accuracy | 55.45% |
| Mean confidence | 77.87% (**not** the same as accuracy) |
| Confidence threshold | 0.56 (below → "unknown") |

---

## Prerequisites

### Node.js / pnpm

- Node.js 18+
- pnpm — install with `npm install -g pnpm`

### Python

- Python 3.10–3.14 (tested on 3.14.2, macOS ARM64)
- pip

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/aanyaagrawal26/baby-action-ai.git
cd baby-action-ai
```

### 2. Install Node dependencies

```bash
pnpm install
```

### 3. Set up the Python virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

pip install --upgrade pip
pip install -r requirements.txt
```

> **Note:** `requirements.txt` uses `mediapipe>=0.10.30,<1.0`. mediapipe 1.x removed the `mp.solutions` API and is not compatible.

### 4. Download the MediaPipe pose model bundle

This file (~5.5 MB) is excluded from the repository. Download it once:

```bash
curl -L "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task" \
     -o models/pose_landmarker_lite.task
```

### 5. Configure environment variables

```bash
cp .env.example .env
# Edit .env and fill in BUILT_IN_FORGE_API_KEY
```

See [Environment Variables](#environment-variables) below.

---

## Running the Project

You need **two terminals**.

### Terminal 1 — Python inference service

```bash
cd baby-action-ai
source .venv/bin/activate
uvicorn inference_api:app --host 127.0.0.1 --port 8000
```

Verify it started:
```bash
curl http://127.0.0.1:8000/health
# Expected: {"ok":true,"model":"...infant_pose_mlp.pt",...}
```

### Terminal 2 — Node dev server

```bash
cd baby-action-ai
pnpm dev
```

Open **http://localhost:3000** in your browser.

---

## Environment Variables

Copy `.env.example` to `.env` and configure:

| Variable | Required | Description |
|---|---|---|
| `BUILT_IN_FORGE_API_KEY` | For live LLM | API key for the Forge LLM endpoint |
| `BUILT_IN_FORGE_API_URL` | No | Override Forge base URL (default: `https://forge.manus.im`) |
| `INFERENCE_API_URL` | No | Python service URL (default: `http://127.0.0.1:8000`) |
| `MODEL_PATH` | No | Path to MLP checkpoint (default: `models/infant_pose_mlp.pt`) |
| `POSE_TASK_PATH` | No | Path to MediaPipe task bundle (default: `models/pose_landmarker_lite.task`) |

**Without `BUILT_IN_FORGE_API_KEY`:** The `/api/explain` route returns a rule-based fallback clearly labelled "LLM generation unavailable". ML inference works normally.

**Never commit `.env` to version control.**

---

## API Routes

| Route | Method | Description |
|---|---|---|
| `/api/infer` | POST | Node proxy → Python `/analyze` → MLP inference |
| `/api/explain` | POST | Server-side prompt construction + LLM call (or fallback) |
| `http://127.0.0.1:8000/health` | GET | Python service health check |
| `http://127.0.0.1:8000/analyze` | POST | Direct MLP inference (multipart image) |

### POST /api/infer

```json
{
  "filename": "frame.jpg",
  "contentType": "image/jpeg",
  "data": "<base64-encoded image>"
}
```

Response:
```json
{
  "action": "prone",
  "confidence": 0.834,
  "framesAnalyzed": 1,
  "rawAction": "prone",
  "smoothed": true,
  "modelProvenance": "InfActPrimitive-public-pose-finetune"
}
```

### POST /api/explain

```json
{
  "action": "prone",
  "rawAction": "prone",
  "confidence": 0.834,
  "strategy": "B"
}
```

Response:
```json
{
  "explanation": "The posture detector identified the infant as lying face-down...",
  "status": "llm_success",
  "statusMessage": "Generated by LLM using Strategy B.",
  "strategyName": "Caregiver-Friendly Explanation",
  "displayPrompt": "[Strategy B — Caregiver-Friendly Explanation]...",
  "isUnknown": false
}
```

---

## Project Structure

```
baby-action-ai/
├── client/src/
│   ├── pages/Home.tsx          # Main UI — inference, GenAI, pipeline, model analysis
│   └── components/             # shadcn/ui components
├── server/
│   ├── _core/
│   │   ├── llm.ts              # invokeLLM() — Forge API with retry/backoff
│   │   └── env.ts              # Environment variable definitions
│   ├── inferenceProxy.ts       # POST /api/infer — Node → Python proxy
│   ├── generateExplanation.ts  # POST /api/explain — prompt + LLM route
│   ├── promptEngineering.ts    # Three prompt strategies (pure functions)
│   └── routers.ts              # tRPC router
├── shared/                     # Shared TypeScript types
├── models/
│   ├── infant_pose_mlp.pt      # Trained MLP checkpoint (82 KB)
│   ├── labels.json             # ["supine","prone","sitting","standing","all-fours"]
│   ├── metrics.json            # Offline validation metrics
│   └── training_history.json   # Per-epoch training log
│   # pose_landmarker_lite.task — NOT in repo, download separately (5.5 MB)
├── inference_api.py            # FastAPI inference service (MediaPipe + MLP)
├── train_public.py             # Offline training script
├── requirements.txt            # Python dependencies
├── package.json                # Node dependencies
├── .env.example                # Environment variable template
└── STARTUP_GUIDE.md            # Detailed demo setup guide
```

---

## Running Tests

```bash
# TypeScript type check
pnpm check

# Unit tests (17 tests)
pnpm test

# Production build
pnpm build
```

All tests verified passing: TypeScript clean, 17/17 unit tests pass, build succeeds.

---

## Responsible Use

- **Not a medical device.** Do not use for diagnosis, developmental assessment, or clinical decisions.
- **Not a safety monitoring system.** Do not use for unattended infant monitoring or emergency alerts.
- **Dataset terms.** The InfActPrimitive dataset is for non-commercial research only. Respect the original licence.
- **LLM output.** Generated explanations may be inaccurate. Always display the original model prediction and confidence alongside any generated text.
- **Camera.** Camera access is opt-in. Frames are sent for inference only; no video is stored.

---

## Known Limitations

- Trained on pose-only annotations — does not generalise across all cameras, lighting, or body types
- Class imbalance: sitting (31.6% recall) and standing (38.9% recall) are underperformed
- Mean confidence (77.87%) ≠ accuracy (64.21%) — a confident prediction can still be wrong
- Video upload not yet supported by the inference service (images only)
- Live-webcam accuracy may differ from offline validation (domain shift)

---

## Team

Applied Generative AI — 5th Semester B.Tech  
Review-2 | October 2026

See `REVIEW2_TEAM_WORK.md` for proposed member responsibilities.

---

## License

Research prototype. Training data subject to the InfActPrimitive dataset's non-commercial research terms.
