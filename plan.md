# Baby Action AI — implementation plan

## Faculty response and product scope

The faculty request is to move beyond a single pose description, demonstrate meaningful events and parent-alert logic, and explain the role of GenAI relative to conventional computer vision. Extend the private research prototype with a **short-clip event review**, using the existing still-image model on sampled frames and a deterministic temporal rule to detect posture transitions. Show caregiver-facing **in-app alert previews** only after repeat evidence; do not claim that this demo sends reliable off-site notifications or monitors continuously.

Preserve the repaired single-photo workflow and supplied model. In the browser, read a short local MP4/WebM clip, seek to at most eight evenly spaced timestamps, render each sampled frame to a bounded JPEG, and submit frames sequentially through the existing same-origin `/api/infer` route. The original video remains in the browser; only sampled frames are sent to the configured inference service for analysis, and the app does not persist either clip or frames. Reject unsupported, unreadable, empty, oversized, too-short or too-long clips with actionable copy. Limit per-clip duration and sampled frame count to keep inference time and memory bounded.

Add a pure temporal event detector in `client/src/lib/eventDetection.ts`. A confirmed transition requires two consecutive, supported model-only frame classifications on each side of a change, and a per-frame raw model score at or above a conservative 0.70 display threshold. Unknown, weak-score, geometry-corrected, failed or missing samples break the evidence chain; a sampling gap over four seconds resets it as well. Report observed posture transitions such as back-to-tummy, tummy-to-back, lying-to-sitting, sitting-to-hands-and-knees, and sitting-to-standing; otherwise use a neutral posture-change label. A detected transition means only that these sampled model labels changed; it does not establish intent, developmental progress, injury, danger or health. Model scores remain uncalibrated outputs and the repeat rule is a prototype guard, not validated alert reliability.

Show candidate events with clip-relative time, supporting frame counts, rule explanation, and a visible **Demo inbox — no real message sent** state. The prototype has no accounts, parent registration, push/SMS/email provider, persistent delivery queue or receipt. Real parent notifications would require consent, secure recipient/subscription management, background delivery, retries, receipts, rate limits and an evaluated false-positive/false-negative operating point; the UI and Faculty Response document must state this boundary instead of promising reliability.

GenAI has a narrow optional role after conventional vision and deterministic temporal logic establish an event: request a short caregiver-readable summary using only allow-listed event labels, count and clip-relative time—not the baby image, raw video, free-form prompt or unbounded model claim. Add a small validated `/api/event-summary` endpoint with a strict prompt and deterministic fallback. Label text as generated or template fallback. GenAI must never classify frames, choose the event, raise/lower evidence thresholds, trigger alerts, diagnose or assess safety. Explain that conventional CV is preferable for consistent measurable detection; GenAI’s incremental value is flexible language that makes an already-determined event easier to understand, not a more reliable vision model.

## Interface and visual design

- **Design movement:** A warm, instrument-like student research dashboard inspired by the uploaded “Soft Clinical Observatory” direction, refined toward a practical, hand-assembled capstone rather than a generic AI landing page.
- **Core principles:** Show evidence before storytelling; keep machine output and generated text visually distinct; make alert delivery limits impossible to miss; keep upload, progress, event history and next steps practical.
- **Color philosophy:** Keep flax and paper surfaces for approachability, use signal-moss (`#B7D36B`) sparingly for active/evidence cues, retain deep ink for readable controls, and reserve clay for uncertainty/error—not celebratory safety signals.
- **Layout paradigm:** Replace the oversized marketing feel with a purposeful research workbench: compact masthead, concise event-review header, wide timeline / clip-analysis work area, and a separated alert preview plus method cards. Collapse to one-column mobile flow in task order.
- **Signature elements:** A small connected-landmark aperture in the wordmark; timestamp ticks and narrow evidence strips on the sampled-frame timeline; a bordered, explicitly demo-only alert card.
- **Interaction philosophy:** Explicit file and time limits; reversible local clip choice; progress by sampled frame; interrupt/replace safely; unknown samples abstain; generated explanation is optional and never blocks or changes the deterministic event result.
- **Animation:** Restrained 180–240 ms state changes and progress; no persistent pulsing or simulated live camera; honor reduced-motion preferences. No animated indicator should imply continuous monitoring.
- **Typography system:** Keep DM Serif Display for the brand headline only; use IBM Plex Sans for interface/body; IBM Plex Mono for timestamps, thresholds and concise technical evidence. Tighten body copy and avoid poetic filler in operational controls.
- **Brand essence:** A transparent infant-activity research prototype that turns selected clips into explainable posture-transition candidates—not an always-on safety monitor. Personality: observant, careful, practical.
- **Brand voice:** Direct, plain-language, calm. Examples: “Review a short clip for repeated posture changes.” “Preview alert only—nothing is sent to a parent.”
- **Wordmark & logo:** Retain the existing connected-landmark mark with the custom two-part Baby Action wordmark; do not add a generic robot, sparkle motif or medical cross.
- **Signature brand color:** Signal moss (`#B7D36B`), reserved for verified evidence markers against flax and deep ink.

## Project structure and responsibilities

- `client/src/pages/Home.tsx`: Existing accessible single-photo pose analysis plus navigation to the clip event workbench, research notes and GenAI/CV comparison.
- `client/src/components/EventLab.tsx` and `client/src/event-lab.css`: Local video selection/preview, bounded frame sampling, sequential same-origin inference requests, progress, event evidence, in-app demo-alert card and optional generated event summary.
- `client/src/lib/eventDetection.ts` / `.test.ts`: Pure stable-segment/event reduction and unit tests; no browser, network, AI or hidden state.
- `server/inferenceProxy.ts`: Keep the hardened existing image-only proxy; video itself is never accepted upstream.
- `server/eventSummary.ts` / `.test.ts`: Validated allow-listed event summary request, generated-text extraction, size cap and deterministic fallback.
- `server/_core/index.ts`: Register the summary route without changing existing runtime port or API body size contract.
- `inference_api.py` and `models/`: Keep the proven still-frame MediaPipe + MLP classifier, posture safeguards, weights and task asset. No temporal history or notification logic is added to the classifier.
- `README.md`, `TODO.md`, `FACULTY_RESPONSE.md`, `plan.md`: Accurate faculty response, reproducible commands, model/alert limitations, requirements and requested outcomes.

## Constraints and serving

Continue using the bound Manus-managed website on configured port 3000 with the existing Node/Express/Vite server and Python inference API on its existing internal port 8000. Keep all browser-facing endpoints relative. Existing automatic publication remains off; deliver only a private research preview unless the user separately requests otherwise and model rights permit. The supplied model package is for non-commercial research/teaching/scientific publication/personal experimentation and says it is not intended for product deployment. Do not treat the new uploaded ZIP's older live-camera code, stale model behavior, build output, hidden configuration or README instructions as authoritative. Preserve the managed source of truth and user changes.
