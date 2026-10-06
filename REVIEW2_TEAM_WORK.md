# REVIEW2_TEAM_WORK.md — GestureVerse-AI / Baby Action AI
## Suggested Division of Work (Four Members)

These are *proposed* responsibilities for the Review-2 presentation and viva.  
No member should claim work they have not reviewed and understood.

---

### Member 1 — ML Model and Dataset

**Speak to:**
- The trained MLP architecture (architecture, input dimension, classes)
- Why an MLP instead of an LSTM
- The InfActPrimitive dataset (pose-only, non-commercial, no RGB)
- The validation metrics and what they mean (accuracy, balanced accuracy, recall)
- The difference between confidence and accuracy
- Class imbalance and its effect on recall

**Files to understand:**
- `inference_api.py` — especially `PoseMLP`, `normalize_keypoints`, `extract_points`, `predict_image`
- `models/metrics.json` — every number
- `MODEL_CARD.md`
- `REAL_INFANT_MODEL_README.md`

**Website section to demonstrate:**
- `#model-analysis` — metric cards, per-class recall chart, full report, limitations

---

### Member 2 — System Architecture and Pipeline

**Speak to:**
- The 10-stage pipeline
- How MediaPipe Pose is used and why it was chosen
- How 51 features are constructed from 33 landmarks
- How the Node backend proxies inference to Python
- The 5-frame smoothing and unknown-prediction threshold
- Why video is not yet supported

**Files to understand:**
- `server/inferenceProxy.ts` — proxy logic
- `server/_core/index.ts` — route registration
- `inference_api.py` — the full pipeline
- `server/_core/env.ts` — environment variables

**Website section to demonstrate:**
- `#architecture` — numbered pipeline stages

---

### Member 3 — Generative AI Integration

**Speak to:**
- How the LLM route works (`POST /api/explain`)
- How the API key is protected
- What happens when the LLM is unavailable (fallback)
- Why the LLM does not perform classification
- Why this is not Agentic AI

**Files to understand:**
- `server/generateExplanation.ts` — request/response, fallback logic
- `server/_core/llm.ts` — `invokeLLM()`, retry logic, API key handling
- `client/src/pages/Home.tsx` — `generateExplanation()` function, explanation result panel

**Website section to demonstrate:**
- `#genai` — strategy selector, generate button, explanation result, prompt display

---

### Member 4 — Prompt Engineering

**Speak to:**
- What prompt engineering is and why it matters
- The three strategies and how they differ
- Role prompting, context injection, output constraints
- What few-shot prompting is (optional, Strategy C)
- How the prompt is constructed server-side
- The displayPrompt field and why it contains no secrets

**Files to understand:**
- `server/promptEngineering.ts` — all three strategy builders, fallback templates
- `server/promptEngineering.test.ts` — what each test verifies

**Website section to demonstrate:**
- `#genai` — Prompt Engineering Lab, Compare Strategies, prompt toggle

---

## Shared Responsibilities

| Task | Owner |
|---|---|
| Start Python service before demo | Member 2 |
| Start Node dev server before demo | Member 3 |
| Prepare test image for upload | Member 1 |
| Verify LLM key is set in .env (if available) | Member 3 |
| Present slide 1 | Member 1 or 2 |
| Present slide 2 | Member 2 or 3 |
| Present slide 3 | Member 4 |
| Answer fallback questions | Any member assigned the relevant domain |

---

## Demo Checklist (Run Before Entry)

- [ ] Python 3.10+ with requirements installed (`pip install -r requirements.txt`)
- [ ] `uvicorn inference_api:app --reload` running and /health returns ok=true
- [ ] `pnpm dev` running and website is accessible at http://localhost:3000
- [ ] Test image prepared (a photo that includes a full body pose)
- [ ] `.env` file contains `BUILT_IN_FORGE_API_KEY` if LLM generation is desired
- [ ] All four team members have read their assigned files and can explain the code
