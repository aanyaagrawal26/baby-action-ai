# REVIEW2_AUDIT.md — GestureVerse-AI / Baby Action AI

## 1. Workspace Layout (Verified)

```
/Users/hp/Desktop/baby-action-ai/        ← workspace root
├── inference_api.py                      ← Python FastAPI inference service
├── train_public.py                       ← offline training script
├── requirements.txt                      ← Python deps
├── run_api.sh                            ← launch helper
├── models/
│   ├── infant_pose_mlp.pt               ← trained checkpoint (PRESENT)
│   ├── labels.json                       ← ["supine","prone","sitting","standing","all-fours"]
│   ├── metrics.json                      ← offline validation metrics (PRESENT)
│   └── training_history.json             ← per-epoch loss/accuracy (PRESENT)
├── client/src/pages/Home.tsx            ← single-page React frontend
├── server/
│   ├── _core/index.ts                   ← Express entry; registers all routes
│   ├── _core/llm.ts                     ← FULL invokeLLM() helper (forge API)
│   ├── _core/env.ts                     ← ENV with BUILT_IN_FORGE_API_KEY
│   ├── inferenceProxy.ts               ← POST /api/infer → Python :8000/analyze
│   └── routers.ts                       ← tRPC router (only auth + system today)
└── shared/types.ts                      ← type re-exports
```

All model files are **inside** the workspace root. No separate folder to add.

---

## 2. Answers to the Ten Audit Questions

### Q1 — How does the frontend upload an image or video?
`Home.tsx` uses a `<input type="file">` that calls `handleUpload()`.  
`analyzeUpload()` reads the file with `FileReader`, base64-encodes it, and POSTs JSON  
`{ filename, contentType, data }` to `/api/infer`.

### Q2 — How does live-camera inference work?
`startLiveCamera()` calls `navigator.mediaDevices.getUserMedia`, attaches the stream to a  
`<video>` element. A `setInterval` at 2 200 ms calls `inferLiveFrame()`, which captures a  
frame to a hidden `<canvas>`, encodes it as JPEG (quality 0.78), and POSTs to `/api/infer`.  
Results are shown in `liveInference` state.

### Q3 — Which endpoint does the frontend call?
`POST /api/infer` (same for upload and live camera)

### Q4 — How does the Node backend communicate with Python?
`server/inferenceProxy.ts` receives the base64 JSON body, reconstructs a `Buffer`,  
wraps it in a `FormData`, and forwards it to  
`${INFERENCE_API_URL || "http://127.0.0.1:8000"}/analyze` via native `fetch`.

### Q5 — Is the trained checkpoint actually loaded?
Yes. `inference_api.py` → `load_runtime()` calls `torch.load(MODEL_PATH)` and  
`model.load_state_dict(checkpoint["state_dict"])`. The path defaults to  
`models/infant_pose_mlp.pt` which is present in the workspace.

### Q6 — Exact model input / output format
**Input:** 51-float `numpy.float32` vector  
- 17 keypoints × 2 xy coordinates (hip-centered, scale-normalized) = 34 floats  
- 17 keypoint visibility scores = 17 floats  
Keypoints are Human3.6M-compatible; constructed from MediaPipe Pose landmark indices.

**Output from `/analyze`:**  
```json
{
  "action":          "prone",        // LABELS[idx] if conf ≥ 0.56, else "unknown"
  "confidence":      0.834,          // float, max smoothed probability
  "framesAnalyzed":  1,
  "modelProvenance": "InfActPrimitive-public-pose-finetune",
  "rawAction":       "prone",        // actual argmax regardless of threshold
  "smoothed":        true
}
```
No-pose detected: `{"action":"unknown","confidence":0.0,"framesAnalyzed":0,"reason":"..."}`

### Q7 — Does a usable LLM integration exist?
**Yes — fully implemented in `server/_core/llm.ts`:**  
- `invokeLLM(params)` — builds an OpenAI-compatible chat completion request  
- Targets `BUILT_IN_FORGE_API_URL` (env var) with `BUILT_IN_FORGE_API_KEY`  
- Has retry/back-off, error handling, response-format normalization  
- Is **not wired to any route yet** — no tRPC procedure calls it for inference  

### Q8 — Dependencies and environment variables
Already installed: React 19, tRPC, Express, zod, recharts, lucide-react, framer-motion  
Python: torch, mediapipe, fastapi, opencv, Pillow, scikit-learn (all in requirements.txt)  
Required env vars: `BUILT_IN_FORGE_API_KEY`, `BUILT_IN_FORGE_API_URL` (optional — forge.manus.im is default)

### Q9 — Feature implementation status

| Feature | Status |
|---|---|
| Image upload → `/api/infer` | ✅ Implemented |
| Live camera inference (2.2 s interval) | ✅ Implemented |
| Video upload handling | ⚠️ Proxy accepts video MIME but Python API returns HTTP 415 |
| Trained MLP checkpoint loading | ✅ Implemented (Python) |
| MediaPipe Pose extraction | ✅ Implemented (Python) |
| 51-D feature vector + normalization | ✅ Implemented (Python) |
| 5-frame probability smoothing | ✅ Implemented (Python) |
| Unknown-prediction threshold (0.56) | ✅ Implemented (Python) |
| Inference result display (action + confidence) | ✅ Implemented (React) |
| **LLM / Generative AI route** | ❌ Missing — no tRPC or REST endpoint |
| **Prompt engineering strategies** | ❌ Missing |
| **Generated Explanation UI section** | ❌ Missing |
| **Prompt Engineering Lab UI** | ❌ Missing |
| **Pipeline visualization (accurate)** | ❌ Existing architecture section describes "LSTM / 30 frames" — inaccurate |
| **Model analysis section** | ❌ Missing (only hardcoded demo numbers shown) |
| **Per-class recall chart** | ❌ Missing |
| Review-2 documentation | ❌ Missing |

### Q10 — Changes necessary for Review-2 rubric

**Rubric criterion 1 (ML/DL/NLP/Transformer/GenAI — 10 marks)**  
Need: working LLM call + clear explanation of MLP vs GenAI.

**Rubric criterion 2 (Generated content + objectives + pipeline — 5 marks)**  
Need: generated explanation section, accurate pipeline diagram, objective statement.

**Rubric criterion 3 (Prompt engineering use cases — 5 marks)**  
Need: three strategies, selectable in UI, affecting actual prompts sent to LLM.

---

## 3. Architecture Corrections Required

1. **LSTM label is wrong.** `Home.tsx` architecture section (`#architecture`) currently shows:  
   `"03 / Temporal model — An LSTM reads 30 frames as movement…"`  
   The actual model is a **two-layer MLP**. The 5-frame window is post-classification probability smoothing, not sequential LSTM input.  
   → Fix: update the step description; replace "LSTM / 30 frames" with "MLP / frame-level + 5-frame smoothing".

2. **"33 landmarks"** statistic in `#notes` is standard MediaPipe output; the model uses 17 Human3.6M-compatible joints derived from those 33. Clarify.

3. Confidence is not accuracy. Mean confidence (77.87 %) ≠ frame accuracy (64.21 %).

---

## 4. Verified Metrics (from `models/metrics.json`)

| Metric | Value |
|---|---:|
| Frame accuracy | 64.21% |
| Balanced accuracy | 55.45% |
| Mean confidence | 77.87% |
| Supine recall | 77.31% |
| Prone recall | 62.55% |
| Sitting recall | 31.57% |
| Standing recall | 38.94% |
| All-fours recall | 66.88% |
| Val samples | 9 307 |

These match the `MODEL_CARD.md` values exactly.

---

## 5. Implementation Plan

### Phase 3 — Server: LLM generation route

**New file:** `server/generateExplanation.ts`  
- Registers `POST /api/explain` on Express (same pattern as `inferenceProxy.ts`)  
- Accepts `{ action, confidence, rawAction, strategy, modelInfo }`  
- Imports `invokeLLM` from `server/_core/llm.ts`  
- Graceful fallback if API key missing

**New file:** `server/promptEngineering.ts`  
- Exports `buildPrompt(strategy, context)` for strategies A / B / C  
- Pure functions — no side effects, easy to explain in viva

### Phase 4 — Three prompt strategies

Strategy A: Simple one-sentence description  
Strategy B: Caregiver-friendly, no medical inference  
Strategy C: Technical analysis with model context, limitations  

### Phase 5 — UI additions

- Add `GenAI` section to `Home.tsx` (after the inference result)  
  - Strategy selector (A / B / C)  
  - Generated explanation panel  
  - Prompt display (no secrets)  
  - Generation status badge  
- Prompt Engineering Lab as expandable sub-section

### Phase 6 — Pipeline + model analysis

- Fix `#architecture` section text to match MLP reality  
- Add numbered pipeline visualization (stages 01–10)  
- Add model analysis section with real metrics from `metrics.json`  
- Add per-class recall bar chart using recharts (already installed)

### Phase 7 — Testing

- TypeScript check: `pnpm check`  
- Build: `pnpm build`  
- Unit tests: `pnpm test`  
- Python import check  
- Prompt template unit tests

---

## 6. Risks and Blockers

| Risk | Mitigation |
|---|---|
| `BUILT_IN_FORGE_API_KEY` not set in `.env` | Implement deterministic fallback + clear status label |
| Python service not running during demo | Show clear error + "Start FastAPI first" instructions |
| mediapipe pose fails on demo image | API returns `action: "unknown"` — UI handles this already |
| Video inference returns 415 | Note as known limitation; image inference is fully functional |

---

## 7. Review-2 Rubric Mapping

| Criterion | Marks | Implementation target |
|---|---:|---|
| Justify ML/DL/NLP/Transformer/GenAI | 10 | Working LLM route + `REVIEW2_IMPLEMENTATION.md` |
| Generated content + pipeline + objectives | 5 | Explanation section + accurate pipeline + objective copy |
| Prompt engineering use cases | 5 | Three strategies + Lab UI + prompt display |
