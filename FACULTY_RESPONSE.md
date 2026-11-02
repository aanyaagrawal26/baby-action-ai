# Response to faculty feedback

## 1. Move from posture labels to meaningful events

The original photo classifier answered: “Which supported pose label is closest to this one frame?” It did not identify motion. We added a short, user-selected clip review in which the browser samples up to eight frames and runs each frame through the same existing MediaPipe + pose-classifier service. A separate, deterministic event rule looks for a sustained transition between the five supported labels. The interface displays each sampled time and its per-frame classification before showing a candidate event such as **On back → On tummy position change** or **Moved from lying to sitting**.

A candidate requires two consecutive model-only samples before and after the change, each with an uncalibrated raw classifier score of at least 0.70. Unknown or ambiguous outputs, low scores and shape-rule overrides break the evidence chain. The clip is processed as a limited frame sequence, not as continuous observation. An included worked example is explicitly marked as scripted labels; it is not a baby video or a real model prediction.

## 2. Show alert logic without overstating delivery

When a completed clip meets the rule, the interface creates a **local in-app alert preview** with the transition, clip-relative confirmation time, number of supporting frames and minimum supporting score. If the rule is not met, the page explains that no alert candidate was created. A stopped or incomplete run cannot generate an alert.

This capstone does **not** deliver a push notification, SMS, email or parent-account message. There is no caregiver consent or recipient setup, external delivery provider, persistent queue, retries, delivery receipt, multi-device service or monitoring worker. The UI states this next to the preview. The system is a research demonstration, not a dependable or unattended baby monitor.

To make real delivery responsible, the next project phase would require opt-in from a verified caregiver, a secure delivery provider, a background queue with rate limits/retries/receipts, an audit trail and privacy/retention decisions. The classifier and event threshold would also need to be validated on consented, representative home videos. Appropriate event-level evaluation includes sensitivity/recall, precision, false alerts per hour, missed events, confidence calibration, latency and evaluation splits that separate children and cameras between train and test.

## 3. Explain why GenAI is used alongside conventional computer vision

GenAI is **not used for vision classification, event confirmation or alert triggering**. MediaPipe detects visual body landmarks, the supplied neural-network classifier labels each frame, and the transparent temporal rule confirms whether sufficiently clear labels changed across repeated samples. This conventional vision-plus-rules pipeline is deterministic, inspectable and measurable—and it still works if the language service is unavailable. A deterministic summary is provided as fallback.

GenAI has a modest supporting role after the event rule has already fired: it may restate allow-listed posture labels and clip-relative timing in a short caregiver-friendly sentence. It receives only that structured text; the route accepts no image, video or free-form prompt. Its output is separated from the event evidence and cannot change the decision. The added value is flexible, readable language—not higher visual accuracy. If generated wording is unnecessary, the template alone is simpler and more predictable.

## 4. Known evidence and limitations

The supplied model card reports **64.21% frame accuracy** and **55.45% balanced accuracy** on pose-sequence validation; sitting and standing were weaker classes. These results do not validate RGB home-video event detection, alert precision/recall or notification reliability. The model outputs and score thresholds are uncalibrated. Occlusion, camera angle, low resolution, motion blur, framing, caregiver handling and correlated adjacent frames may cause a missed or false candidate. A posture transition is not evidence of danger, health, developmental progress, emotion or intent.

The supplied model package is for non-commercial research, teaching, scientific publication and personal experimentation. Check the source-data terms before further redistribution or any product deployment.

## 5. Demonstration and reproducibility

On the website, select a 2.5–20 second MP4/WebM file, then review the sampled timeline and in-app demo alert preview. The original clip stays in the browser; compressed still frames are sent sequentially to the private inference endpoint for this session and are not stored by the website. **“See example”** walks through the exact rule using synthetic, clearly marked pose labels without sending a photo, video or real notification.

Run the temporal-rule and API-input unit tests with `pnpm test`; run the frontend TypeScript check with `pnpm run check`; run the production build with `pnpm run build`. The Python photo-pose safeguards remain covered by `python -m unittest discover -s tests -p 'test_*.py' -v`.
