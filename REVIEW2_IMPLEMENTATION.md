# REVIEW2_IMPLEMENTATION.md — GestureVerse-AI / Baby Action AI

## 1. Implemented Functionality (Verified)

The following are genuinely implemented and tested as of this document:

| Component | Status | Notes |
|---|---|---|
| Image upload → inference | ✅ | `POST /api/infer` → Python :8000/analyze |
| Live camera inference (2.2 s interval) | ✅ | getUserMedia → canvas → JPEG → /api/infer |
| Trained MLP checkpoint loading | ✅ | `torch.load` on `models/infant_pose_mlp.pt` |
| MediaPipe Pose landmark extraction | ✅ | 33 → 17 H3.6M joints selected |
| 51-D feature vector construction | ✅ | 34 xy + 17 visibility, hip-centred, scale-normalised |
| 5-frame probability smoothing | ✅ | `TemporalSmoother` deque in inference_api.py |
| Unknown-prediction threshold (0.56) | ✅ | Action → "unknown" if confidence < 0.56 |
| `POST /api/explain` LLM route | ✅ | New — server/generateExplanation.ts |
| Strategy A — Simple description | ✅ | Pure function, unit-tested |
| Strategy B — Caregiver explanation | ✅ | Pure function, unit-tested |
| Strategy C — Technical analysis | ✅ | Pure function, unit-tested |
| Rule-based fallback | ✅ | Triggered when API key absent or LLM fails |
| Prompt display in UI | ✅ | Toggle-able, no secrets |
| Generated Explanation section | ✅ | In #genai section of Home.tsx |
| Prompt Engineering Lab | ✅ | Compare Strategies panel |
| Pipeline visualization (corrected) | ✅ | 10 stages, accurate labels |
| Per-class recall bar chart | ✅ | Uses real metrics.json values via recharts |
| Model analysis section | ✅ | Real architecture + metrics + limitations |
| TypeScript type check | ✅ | `pnpm check` exits 0 |
| All 17 unit tests | ✅ | `pnpm test` — 17/17 pass |
| Production build | ✅ | `pnpm build` succeeds |

---

## 2. ML/DL Justification

### Why ML (supervised classification)?
The project uses supervised learning: labelled pose sequences from the InfActPrimitive dataset are used to train a classifier that maps 51-D feature vectors to posture class labels. This is a classical supervised learning pipeline.

### Why DL (deep learning)?
The classifier is a Multi-Layer Perceptron (MLP) — a feedforward neural network. It uses non-linear activation functions (GELU) and regularisation (LayerNorm, Dropout). Whether an MLP with two hidden layers qualifies as "deep learning" depends on the convention used, but it is unambiguously a neural network that learns non-linear feature transformations, which is the core property of deep learning. The architecture:

```
Linear(51 → 128) → LayerNorm(128) → GELU → Dropout(0.25)
→ Linear(128 → 96) → LayerNorm(96) → GELU → Dropout(0.20)
→ Linear(96 → 5)
```

### Why an MLP instead of an LSTM?
- The dataset provides pose annotations extracted from video frames, not raw temporal sequences.
- Each frame's pose is independent after feature engineering (hip-centred normalisation removes spatial drift).
- An MLP processes each frame independently; temporal context is added post-classification via 5-frame probability smoothing.
- An LSTM would require ordered sequences and substantially more training data.
- The MLP achieved 64.21% frame accuracy on the validation set — a reasonable baseline for a pose-only model with limited class balance.

### Why NOT an LSTM?
The existing documentation on the website (now corrected) incorrectly stated "An LSTM reads 30 frames". The actual trained model is an MLP. No LSTM was trained. This document corrects that claim.

---

## 3. NLP and LLM Justification

Natural Language Processing is used to convert the MLP's structured numeric output into human-readable text. The LLM generates contextual language by:

1. Receiving a structured prompt containing the action label, confidence, and model metadata.
2. Applying the prompt constraints (role, task, output format, ethical boundaries).
3. Generating a response that is grounded in the provided context rather than hallucinated.

This is a Request–Response interaction with a language model, not an agentic system.

---

## 4. Transformer Justification

The LLM used is accessed via the Forge API (configured through `BUILT_IN_FORGE_API_URL` / `BUILT_IN_FORGE_API_KEY`). The Forge API is an OpenAI-compatible interface. The underlying model architecture is not explicitly exposed by the API. Based on published information:

- OpenAI-compatible APIs typically serve Transformer-based models (GPT family).
- The Forge API appears to route to models in the OpenAI / similar family.
- We cannot independently verify the exact architecture at runtime.

**Accurate claim:** "The LLM is accessed through an OpenAI-compatible API. Such APIs typically serve Transformer-based language models. We cannot independently confirm the architecture at the API level."

Do not claim Transformer-based generation with more certainty than this.

---

## 5. Generative AI Implementation

The GenAI component is implemented in:

- `server/promptEngineering.ts` — three prompt strategy builders (pure functions)
- `server/generateExplanation.ts` — Express route, calls `invokeLLM()`, returns explanation
- `client/src/pages/Home.tsx` — GenAI section, prompt lab, generated explanation panel

The LLM is called server-side only. The API key is never sent to the browser. The client receives:
- `explanation` — the generated text
- `status` — whether it came from the LLM or a fallback
- `statusMessage` — human-readable explanation of the status
- `displayPrompt` — the constructed prompt (no secrets)

---

## 6. Why Agentic AI Is NOT Used

Agentic AI involves autonomous decision-making, multi-step planning, tool-calling loops, or self-directed task execution. This project uses:

- **Request–Response LLM**: one prompt in, one response out.
- **No tool-calling**: the LLM has no tools and cannot call any external services.
- **No planning loop**: the system does not execute multiple LLM calls to decompose a task.
- **No autonomous state**: the LLM response does not affect subsequent system behaviour.

The architecture is a deterministic pipeline: image → pose → MLP → prompt construction → LLM request → display.

---

## 7. Actual Pipeline

```
01 Image / Video Input
     ↓
02 Pose Detection — MediaPipe Pose
     (33 landmarks → 17 H3.6M joints selected)
     ↓
03 Pose Normalisation
     (hip-centred, scale-normalised)
     ↓
04 51-D Feature Construction
     (34 xy + 17 visibility)
     ↓
05 Trained MLP Classification
     (2-layer normalised MLP, softmax output)
     ↓
06 Confidence & Unknown Handling
     (5-frame smoothing, threshold 0.56)
     ↓
07 Prompt Construction
     (server/promptEngineering.ts, strategy A/B/C)
     ↓
08 LLM / Generative AI
     (server/generateExplanation.ts → invokeLLM → Forge API)
     ↓
09 Generated Natural-Language Explanation
     (returned with status, prompt, fallback flag)
     ↓
10 User Interface & Analysis
     (Home.tsx — prediction + explanation + model analysis)
```

---

## 8. Prompt Engineering Strategies

### Strategy A — Simple Action Description
- **Role**: Factual assistant
- **Output**: One neutral sentence ≤ 30 words
- **Technique**: Task-specific role prompting + output length constraint
- **Unknown handling**: Acknowledge uncertainty explicitly

### Strategy B — Caregiver-Friendly Explanation
- **Role**: Friendly assistant translating ML output for non-technical caregivers
- **Output**: 2–3 plain sentences
- **Technique**: Role prompting + explicit ethical constraints + context injection
- **Constraints**: No medical, developmental, safety claims. Acknowledge prototype limitations.

### Strategy C — Technical Analysis
- **Role**: ML research assistant
- **Output**: ≤ 120 words technical paragraph
- **Technique**: Role prompting + deep context injection (architecture, metrics, dataset) + output constraints
- **Unique feature**: Explicitly distinguishes MLP prediction from LLM-generated text within the response

All three strategies are implemented as pure TypeScript functions in `server/promptEngineering.ts` and tested in `server/promptEngineering.test.ts` (15 tests, all pass).

---

## 9. Verified Evaluation Metrics (from models/metrics.json)

| Metric | Value | Source |
|---|---:|---|
| Frame accuracy | 64.21% | Offline val, 9 307 frames |
| Balanced accuracy | 55.45% | Offline val |
| Mean confidence | 77.87% | Not accuracy — see note |
| Supine recall | 77.31% | |
| Prone recall | 62.55% | |
| Sitting recall | 31.57% | Under-represented class |
| Standing recall | 38.94% | Under-represented class |
| All-fours recall | 66.88% | |

**Important:** Mean confidence (77.87%) is the average of the model's top softmax probability across all validation predictions. It does not mean 77.87% of predictions were correct. High confidence can coexist with incorrect predictions (overconfidence).

These metrics are from offline validation on the InfActPrimitive public dataset. Live-webcam performance is expected to differ.

---

## 10. Dataset Provenance

- Dataset: InfActPrimitive — official public release
- Content: Infant pose annotations (not raw RGB images/videos)
- Terms: Non-commercial research/teaching/scientific-publication/personal-experimentation
- Not intended for diagnosis or product incorporation (per dataset licence)
- Training samples: 9 600 · Validation samples: 9 307
- The model does NOT learn from RGB appearance, clothing, or lighting

---

## 11. Limitations

1. Pose-only training — no generalisation claims for all camera positions or body types.
2. Class imbalance — sitting and standing are substantially under-represented.
3. Mean confidence ≠ accuracy.
4. Not a medical device.
5. Live-webcam performance may differ from offline validation (domain shift).
6. Video inference not yet supported by the inference service (returns HTTP 415).
7. The LLM may produce inaccurate or unexpected output despite prompt constraints.
8. LLM generation requires `BUILT_IN_FORGE_API_KEY`; without it, the rule-based fallback is used.

---

## 12. Expected Viva Questions and Answers

See REVIEW2_DEMO_SCRIPT.md Section 6 for full Q&A.

---

## 13. Review-2 Rubric Mapping

| Criterion | Marks | Implementation |
|---|---:|---|
| ML/DL justified | — | MLP classifier, section 2 above |
| NLP/Transformer justified | — | LLM explanation, section 3–4 above |
| GenAI implemented | — | /api/explain route + UI section |
| Pipeline diagram refined | — | 10-stage diagram in #architecture |
| Generated content shown | — | Explanation section with real inference |
| Project objective connected | — | Objective banner in hero section |
| Prompt engineering — 3 strategies | — | Strategies A/B/C, unit-tested |
| Prompt engineering lab | — | Prompt Engineering Lab with compare |
| Prompt display | — | Toggle-able prompt in UI |
