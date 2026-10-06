# REVIEW2_SLIDE_CONTENT.md — GestureVerse-AI / Baby Action AI
## Three Content Slides (5-minute presentation)

Do not add headers, titles, or PowerPoint formatting beyond what is listed here.

---

## SLIDE 1 — JUSTIFICATION OF AI TECHNOLOGIES

**Title:** GestureVerse-AI: From Pose to Language

**Body — two columns:**

LEFT COLUMN — Technology Stack:

| Technology | Role in this project |
|---|---|
| **ML (Supervised)** | Labelled pose sequences → trained action classifier |
| **DL (Neural Network)** | MLP learns non-linear pose–posture mapping |
| **NLP** | LLM converts structured prediction → natural language |
| **Transformer / LLM** | Language generation via OpenAI-compatible Forge API |
| **Generative AI** | Contextual explanation generated from inference result |

RIGHT COLUMN — Flow Diagram (simplified):

```
Pose features (51-D)
     ↓  [ML: supervised learning]
Trained MLP  
     ↓  [DL: 2-layer neural network]
Predicted action + confidence
     ↓  [Prompt engineering]
Structured prompt (Strategy A / B / C)
     ↓  [Generative AI / LLM]
Natural-language explanation
```

**Footer note (small text):** "The LLM does not perform classification. The MLP does not generate language. These are separate systems with clearly separated responsibilities."

---

## SLIDE 2 — GENERATED CONTENT, ANALYSIS, AND OBJECTIVES

**Title:** Objective → Pipeline → Result

**Section A — Project Objective:**
> Recognise infant posture/action from visual input and transform the model's structured prediction into understandable natural-language information.

**Section B — Real Implementation Pipeline:**
01 Image Input → 02 MediaPipe Pose → 03 Normalisation → 04 51-D Features → 05 MLP Classification → 06 Confidence Handling → 07 Prompt Construction → 08 LLM → 09 Explanation → 10 UI

**Section C — Illustrative Example (label clearly as illustrative):**

*Note: The following is an illustrative example based on the system's design. A real result will be shown live during the demo.*

| Field | Value |
|---|---|
| Input | Test image (infant) |
| Detected action | prone |
| Model confidence | 83% |
| Threshold met? | Yes (≥ 0.56) |
| Strategy used | B — Caregiver Explanation |
| Generated explanation | "The posture detector identified the infant as lying face-down. This result comes from a research prototype trained on pose data — it should not be used for medical or safety decisions." |
| Generation status | LLM generated (Strategy B) / Rule-based fallback |

**Section D — Validation Metrics (from models/metrics.json — verified):**

| Metric | Value |
|---|---:|
| Frame accuracy | 64.21% |
| Balanced accuracy | 55.45% |
| Mean confidence* | 77.87% |
| Val samples | 9 307 |

*Mean confidence ≠ accuracy

**Footer note:** "Offline validation on InfActPrimitive public pose dataset. Live performance may differ."

---

## SLIDE 3 — PROMPT ENGINEERING USE CASES

**Title:** Three Strategies, One Inference Result

**Table — Side-by-side comparison:**

| | Strategy A | Strategy B | Strategy C |
|---|---|---|---|
| **Name** | Simple Description | Caregiver Explanation | Technical Analysis |
| **Audience** | General | Non-technical caregiver | Student / researcher |
| **Technique** | Role prompting, length constraint | Role + ethical constraints + context injection | Role + deep context injection + output constraint |
| **Output style** | One neutral sentence ≤ 30 words | 2–3 plain sentences, warm tone | ≤ 120 word technical paragraph |
| **Prompt excerpt** | "Write one neutral, factual sentence describing this observation." | "Do not infer emotions, intentions, health status, or safety." | "Explicitly note that YOUR response is LLM-generated text, not an additional ML prediction." |
| **Key constraint** | No inference, no advice | No medical/safety claims | Distinguish ML from LLM output |
| **Unknown handling** | Acknowledge explicitly | Explain low confidence in plain language | State threshold, mention uncertainty |

**Why three strategies?**
Different audiences need different levels of abstraction. Prompt engineering is the practice of designing this framing rather than retraining the model. The same inference result produces meaningfully different outputs depending on which strategy is selected — demonstrating that prompt design is a real engineering decision.

**Implementation note:**
All three strategies are implemented as pure TypeScript functions in `server/promptEngineering.ts` and tested with 15 unit tests (all pass).

**Footer note:** "No LLM outputs have been fabricated for this slide. Real outputs will be demonstrated live."
