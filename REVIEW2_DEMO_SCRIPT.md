# REVIEW2_DEMO_SCRIPT.md — GestureVerse-AI / Baby Action AI
## Strict 5-Minute Demo Plan

Total: **5:00**

---

### Segment 1 — Introduction and Objective (0:00–0:35)

**Speaker:** Team member 1  
**Time:** ~35 seconds

"Our project, Baby Action AI / GestureVerse-AI, has one objective:  
> Recognise infant posture/action from visual input and transform the model's structured prediction into understandable natural-language information.

We do this with two separate AI systems working in sequence: an MLP classifier for action recognition, and a language model for generating explanations. Neither system does the other's job."

*(Point to the hero section of the website, which shows the ML → MLP → Structured context → LLM → Natural language flow)*

---

### Segment 2 — Pipeline Walkthrough (0:35–1:10)

**Speaker:** Team member 2  
**Time:** ~35 seconds

*(Scroll to #architecture section — the 10-stage numbered pipeline)*

"Walk through stages 01 to 10 rapidly:
- 01 Image input via upload or live camera
- 02–03 MediaPipe Pose extracts and normalises 17 keypoints
- 04 We build a 51-dimensional feature vector: 34 normalised xy + 17 visibility scores
- 05 Our trained 2-layer MLP classifies into 5 postures: supine, prone, sitting, standing, all-fours
- 06 5-frame probability smoothing; predictions below 0.56 confidence become 'unknown'
- 07–08 The result is packaged into a prompt server-side and sent to the LLM
- 09–10 The explanation is displayed with full transparency — prediction, confidence, prompt, and source"

---

### Segment 3 — Live Inference Demo (1:10–2:20)

**Speaker:** Team member 3  
**Time:** ~70 seconds

"Let me show a real inference. The Python FastAPI service is running locally."

*(Ensure `uvicorn inference_api:app --reload` is running at http://127.0.0.1:8000)*

**Option A — Image upload:**
1. Scroll to #upload section
2. Drop a test image into the upload zone
3. Click "Analyze upload"
4. Wait ~2–3 seconds
5. Show the inference result: action, confidence, raw action, model provenance
6. Say: "The MLP returned [action] with [confidence]% confidence. This is a real checkpoint prediction, not a mock."

**Option B — Live camera:**
1. Click "Start camera"
2. Wait for first inference
3. Show the live action label and confidence updating

*(If the Python service is not running, show the error message and explain what it means — "The inference service at http://127.0.0.1:8000 is not reachable. In the actual demo this service must be started first.")*

---

### Segment 4 — GenAI Explanation and Prompt Engineering (2:20–3:50)

**Speaker:** Team member 4  
**Time:** ~90 seconds

*(Scroll to #genai section)*

"This is the Generative AI part. Notice the three prompt strategies."

1. **Select Strategy B** (Caregiver-Friendly Explanation)
2. Click "Generate explanation"
3. Wait for response
4. Show the explanation panel — highlight:
   - The strategy name
   - The status badge (LLM generated / Rule-based fallback)
   - The generated explanation text
   - The action, confidence, and isUnknown indicators
5. Click "Show constructed prompt" — show the displayPrompt field
6. Say: "This prompt was built server-side in `promptEngineering.ts`. It includes the role, the model context, and explicit constraints like 'do not infer medical information'."

**Switch to Strategy C:**
1. Select Strategy C (Technical Analysis)
2. Click "Generate explanation"
3. Show how the response is now technical and explicitly distinguishes the MLP prediction from the LLM-generated text

**Open the Prompt Engineering Lab:**
1. Click "Prompt Engineering Lab"
2. Click "Compare all three strategies"
3. Show all three results side by side
4. Say: "Same inference result, three different prompt strategies, three different outputs. This is prompt engineering in action."

---

### Segment 5 — Model Analysis (3:50–4:25)

**Speaker:** Team member 1  
**Time:** ~35 seconds

*(Scroll to #model-analysis section)*

"Our model has real, honest metrics."

*(Point to the metric cards)*
- "Frame accuracy: 64.21% — not 95%, not fabricated."
- "Balanced accuracy: 55.45% — lower because of class imbalance."
- "Mean confidence: 77.87% — this is NOT accuracy. A confident prediction is not necessarily a correct one."

*(Point to the per-class recall bar chart)*
- "Sitting and standing are under-recalled because they're under-represented in our training data. This is a known limitation we document openly."

---

### Segment 6 — Closing and Q&A Prep (4:25–5:00)

**Speaker:** Team member 2  
**Time:** ~35 seconds

"To summarise:  
- ML: our MLP classifier predicts posture from 51-D pose features  
- Generative AI: three prompt strategies send the prediction to an LLM that generates natural-language explanations  
- We use prompt engineering — role prompting, context injection, output constraints  
- We are not using Agentic AI: one prompt in, one response out, no tool loops  
- All metrics are from the real validation set  
- The model is a research prototype, not a medical device"

---

## Section 6 — Viva Q&A Reference

**Q: What problem does the project solve?**  
A: Recognising infant posture from a camera and explaining the prediction in natural language — bridging ML inference with generative AI.

**Q: Why MediaPipe Pose?**  
A: It is an open-source, on-device pose estimation library that extracts body landmarks without storing raw video. It is pose-only, which aligns with our privacy-first research approach.

**Q: What are pose landmarks?**  
A: 2D coordinates (x, y) and a visibility score for each of 33 body keypoints (nose, shoulders, elbows, hips, knees, etc.). We use 17 of these, mapped to Human3.6M joint conventions.

**Q: Why 51 features?**  
A: 17 joints × 2 (xy coordinates, hip-centred and scale-normalised) = 34 floats, plus 17 visibility scores = 51 total.

**Q: Why an MLP instead of an LSTM?**  
A: Our dataset provides per-frame pose annotations rather than ordered sequences. The MLP treats each frame independently; temporal stability is added via 5-frame probability smoothing. An LSTM would require sequential input and substantially more data.

**Q: What is the difference between ML and Generative AI here?**  
A: The MLP is a discriminative model: it maps input features to a class label. The LLM is a generative model: it produces new text conditioned on a prompt. They do different jobs in the same pipeline.

**Q: What is the difference between confidence and accuracy?**  
A: Confidence is the model's softmax probability for its top prediction — how certain the model thinks it is. Accuracy is the fraction of predictions that are actually correct. A model can be 90% confident and still be wrong. Our mean confidence is 77.87% but our accuracy is only 64.21%.

**Q: What does balanced accuracy mean?**  
A: The average of per-class recall values, giving equal weight to each class regardless of how many samples it has. It penalises models that do well on large classes but ignore small ones.

**Q: Why is class-wise recall useful?**  
A: Frame accuracy alone can be misleading when classes are imbalanced. Class-wise recall shows that sitting (31.57%) and standing (38.94%) are poorly classified, even though overall accuracy looks reasonable.

**Q: What is prompt engineering?**  
A: The practice of designing input text (prompts) to guide a language model's output. This includes choosing the right framing, adding context, setting constraints, and selecting tone.

**Q: What is role prompting?**  
A: Assigning a role to the LLM in the system message, e.g. "You are a friendly assistant explaining a baby posture detector to a caregiver." This influences the register and constraints of the response.

**Q: What is context injection?**  
A: Including relevant data in the prompt — in our case, the action label, confidence, model architecture, validation metrics — so the LLM's response is grounded in facts rather than general knowledge.

**Q: What are output constraints?**  
A: Instructions in the prompt that restrict what the LLM may generate. For example: "Do not infer emotional state. Do not make medical claims. Keep the response under 30 words."

**Q: What is few-shot prompting?**  
A: Providing example input–output pairs in the prompt to show the model the desired format. We use it only in Strategy C where an example helps clarify the technical distinction between ML output and LLM output.

**Q: Where is the LLM actually called?**  
A: In `server/generateExplanation.ts`. It calls `invokeLLM()` from `server/_core/llm.ts`, which sends a POST request to the Forge API endpoint. The client never contacts the LLM directly.

**Q: How is the API key protected?**  
A: The key lives in a `.env` file (not committed to version control). It is read by `server/_core/env.ts` server-side only and never included in any HTTP response to the browser.

**Q: What happens when the LLM fails?**  
A: `server/generateExplanation.ts` catches the error and returns a deterministic fallback from `getFallbackExplanation()`. The UI displays the status label "LLM generation unavailable — showing a rule-based fallback." The ML inference result is unaffected.

**Q: Why is the application not Agentic AI?**  
A: Agentic AI involves autonomous planning, multi-step tool use, or self-directed decision-making loops. Our LLM receives one prompt and returns one response. There is no tool-calling, no planning loop, no autonomous behaviour.

**Q: What are the model's limitations?**  
A: (1) Pose-only training — no RGB generalisation. (2) Class imbalance. (3) Confidence ≠ accuracy. (4) Not a medical device. (5) Live-webcam domain shift. See the Model Analysis section.

**Q: Has model fine-tuning been performed?**  
A: No. The model was trained from scratch on the InfActPrimitive public pose dataset. No fine-tuning of any foundation model was performed. When the assignment says "refinement of the pipeline diagram," it means updating the diagram, not retraining the model.

**Q: Which features are actually implemented and tested?**  
A: See REVIEW2_IMPLEMENTATION.md Table 1. In brief: image inference, live camera, MLP checkpoint, prompt engineering (3 strategies, 15 unit tests), LLM route, fallback, pipeline diagram, model analysis with real metrics. All 17 unit tests pass; TypeScript type check passes; production build succeeds.
