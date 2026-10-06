/**
 * Prompt Engineering Module — GestureVerse-AI / Baby Action AI
 *
 * Defines three prompt strategies for transforming an MLP inference result
 * into a natural-language explanation via LLM.
 *
 * Strategy A — Simple Action Description
 * Strategy B — Caregiver-Friendly Explanation
 * Strategy C — Technical Analysis
 *
 * No agentic patterns, tool loops, or autonomous planning.
 * Each function is a pure transformation: context → prompt string.
 */

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type PromptStrategy = "A" | "B" | "C";

export interface InferenceContext {
  /** Thresholded action label, e.g. "prone" or "unknown" */
  action: string;
  /** Raw argmax label before threshold, e.g. "prone" */
  rawAction?: string;
  /** Smoothed max probability 0–1 */
  confidence: number;
  /** true when confidence < threshold (0.56) */
  isUnknown: boolean;
  /** Number of frames analysed */
  framesAnalyzed?: number;
}

export interface ModelInfo {
  architecture: string;
  inputDimension: number;
  classes: string[];
  frameAccuracy: number;
  balancedAccuracy: number;
  meanConfidence: number;
  confidenceThreshold: number;
  provenance: string;
}

export interface BuiltPrompt {
  strategy: PromptStrategy;
  strategyName: string;
  systemMessage: string;
  userMessage: string;
  /** Human-readable summary safe to display in UI (no secrets) */
  displayPrompt: string;
  maxTokens: number;
}

// ---------------------------------------------------------------------------
// Shared model context (used in strategies B and C)
// ---------------------------------------------------------------------------

const MODEL_CONTEXT: ModelInfo = {
  architecture: "2-layer normalized MLP (Multi-Layer Perceptron)",
  inputDimension: 51,
  classes: ["supine", "prone", "sitting", "standing", "all-fours"],
  frameAccuracy: 64.21,
  balancedAccuracy: 55.45,
  meanConfidence: 77.87,
  confidenceThreshold: 0.56,
  provenance:
    "Trained on InfActPrimitive public pose-only dataset; not trained on raw RGB images",
};

// ---------------------------------------------------------------------------
// Strategy A — Simple Action Description
// ---------------------------------------------------------------------------

/**
 * Produces one neutral, concise sentence describing the detected action.
 * Intended for a general audience. Does not infer intent, health, or safety.
 */
export function buildStrategyA(ctx: InferenceContext): BuiltPrompt {
  const actionLabel = ctx.isUnknown
    ? `an unrecognized or uncertain posture (raw prediction: ${ctx.rawAction ?? "unknown"}, confidence: ${(ctx.confidence * 100).toFixed(0)}%)`
    : `${ctx.action} (confidence: ${(ctx.confidence * 100).toFixed(0)}%)`;

  const systemMessage =
    "You are a factual assistant. Respond with exactly one short, neutral sentence that describes a detected body posture or action. " +
    "Do not infer emotions, intentions, health status, development, or safety. " +
    "Do not provide advice or make claims beyond the single detected label. " +
    "Keep the sentence under 30 words.";

  const userMessage =
    `An infant posture classification model detected: ${actionLabel}. ` +
    "Write one neutral, factual sentence describing this observation. " +
    "If the result is uncertain or unknown, acknowledge the uncertainty explicitly rather than guessing a confident action.";

  const displayPrompt =
    `[Strategy A — Simple Description]\n` +
    `Detected: ${actionLabel}\n` +
    `Task: Write one neutral sentence describing the detected posture.\n` +
    `Constraint: ≤ 30 words, no medical inference.`;

  return {
    strategy: "A",
    strategyName: "Simple Action Description",
    systemMessage,
    userMessage,
    displayPrompt,
    maxTokens: 80,
  };
}

// ---------------------------------------------------------------------------
// Strategy B — Caregiver-Friendly Explanation
// ---------------------------------------------------------------------------

/**
 * Produces a short, accessible explanation for a non-technical caregiver audience.
 * Explicitly prevents medical, developmental, and safety inferences.
 * Uses context injection to ground the LLM in the model's actual output.
 */
export function buildStrategyB(ctx: InferenceContext): BuiltPrompt {
  const confidencePct = (ctx.confidence * 100).toFixed(0);

  const systemMessage =
    "You are a friendly assistant explaining a baby posture detector to a caregiver. " +
    "Your role is to translate a machine learning prediction into plain, accessible language. " +
    "STRICT CONSTRAINTS — you must not:\n" +
    "- Infer the baby's emotional state, mood, or intention\n" +
    "- Make or imply any medical, developmental, or safety assessments\n" +
    "- Suggest the posture is safe, unsafe, normal, or abnormal\n" +
    "- Invent information not present in the model output\n" +
    "Keep the response to 2–3 sentences. Use warm, neutral, plain language.";

  let userMessage: string;
  if (ctx.isUnknown) {
    userMessage =
      `A posture detection model tried to identify an infant's position from a photo, ` +
      `but its confidence was too low (${confidencePct}%) to report a reliable result. ` +
      `The closest prediction was "${ctx.rawAction ?? "unclear"}". ` +
      "Please explain to a caregiver in 2–3 plain sentences what this means, " +
      "making it clear that the model could not determine the posture with enough certainty. " +
      "Do not suggest any action or make health-related comments.";
  } else {
    userMessage =
      `A posture detection model identified an infant as being in a "${ctx.action}" position ` +
      `with ${confidencePct}% model confidence. ` +
      "Please explain to a caregiver in 2–3 plain sentences what this posture generally looks like, " +
      "without suggesting what the baby is feeling or whether this is medically significant. " +
      "Acknowledge that the model is a research prototype and may make mistakes.";
  }

  const displayPrompt =
    `[Strategy B — Caregiver-Friendly Explanation]\n` +
    `Model output: action="${ctx.action}", confidence=${confidencePct}%, unknown=${ctx.isUnknown}\n` +
    `Role: Translate the prediction into plain language for a caregiver.\n` +
    `Constraints: No medical/safety claims. 2–3 sentences. Acknowledge uncertainty if present.`;

  return {
    strategy: "B",
    strategyName: "Caregiver-Friendly Explanation",
    systemMessage,
    userMessage,
    displayPrompt,
    maxTokens: 150,
  };
}

// ---------------------------------------------------------------------------
// Strategy C — Technical Analysis
// ---------------------------------------------------------------------------

/**
 * Produces a concise technical explanation for a student or research audience.
 * Distinguishes between the MLP classifier output and the LLM-generated text.
 * References actual model architecture, feature dimension, and limitations.
 * Uses role prompting + context injection + output constraints.
 */
export function buildStrategyC(ctx: InferenceContext): BuiltPrompt {
  const confidencePct = (ctx.confidence * 100).toFixed(0);
  const m = MODEL_CONTEXT;

  const systemMessage =
    "You are a machine learning research assistant explaining a pose-based action classifier. " +
    "Provide a technical description that clearly distinguishes between:\n" +
    "1. The MLP classifier's prediction (the ML component)\n" +
    "2. This generated text (the GenAI component)\n" +
    "CONSTRAINTS:\n" +
    "- Do not treat high confidence as equivalent to correctness\n" +
    "- Do not fabricate pose coordinates, class probabilities, or confusion matrix values\n" +
    "- Mention uncertainty when confidence is below 0.56 or the result is 'unknown'\n" +
    "- Do not claim this model performs clinical or safety assessment\n" +
    "Keep the response under 120 words.";

  const userMessage =
    `INFERENCE RESULT:\n` +
    `  Action (thresholded):  ${ctx.action}\n` +
    `  Raw argmax prediction: ${ctx.rawAction ?? ctx.action}\n` +
    `  Smoothed confidence:   ${confidencePct}% (threshold: ${(m.confidenceThreshold * 100).toFixed(0)}%)\n` +
    `  Is unknown:            ${ctx.isUnknown}\n\n` +
    `MODEL INFORMATION:\n` +
    `  Architecture: ${m.architecture}\n` +
    `  Input: ${m.inputDimension}-dimensional feature vector\n` +
    `         (17 hip-centered, scale-normalized 2D joints + 17 visibility scores)\n` +
    `  Classes: ${m.classes.join(", ")}\n` +
    `  Val frame accuracy: ${m.frameAccuracy}%\n` +
    `  Balanced accuracy:  ${m.balancedAccuracy}%\n` +
    `  Mean confidence:    ${m.meanConfidence}% (NOT the same as accuracy)\n` +
    `  Dataset: ${m.provenance}\n\n` +
    `Write a concise technical paragraph (≤120 words) explaining this inference result, ` +
    `the model's approach, and any relevant limitations. ` +
    `Explicitly note that YOUR response is LLM-generated text, not an additional ML prediction.`;

  const displayPrompt =
    `[Strategy C — Technical Analysis]\n` +
    `Model: ${m.architecture}, ${m.inputDimension}-D input, ${m.classes.length} classes\n` +
    `Result: action="${ctx.action}", confidence=${confidencePct}%, unknown=${ctx.isUnknown}\n` +
    `Val accuracy: ${m.frameAccuracy}%, balanced: ${m.balancedAccuracy}%\n` +
    `Task: Technical paragraph distinguishing ML prediction from LLM-generated text.\n` +
    `Constraints: ≤120 words, no fabricated metrics, mention uncertainty if present.`;

  return {
    strategy: "C",
    strategyName: "Technical Analysis",
    systemMessage,
    userMessage,
    displayPrompt,
    maxTokens: 200,
  };
}

// ---------------------------------------------------------------------------
// Main builder
// ---------------------------------------------------------------------------

/**
 * Builds a prompt for the given strategy and inference context.
 * Pure function — no side effects.
 */
export function buildPrompt(
  strategy: PromptStrategy,
  ctx: InferenceContext
): BuiltPrompt {
  switch (strategy) {
    case "A":
      return buildStrategyA(ctx);
    case "B":
      return buildStrategyB(ctx);
    case "C":
      return buildStrategyC(ctx);
    default: {
      const _exhaustive: never = strategy;
      throw new Error(`Unknown prompt strategy: ${String(_exhaustive)}`);
    }
  }
}

// ---------------------------------------------------------------------------
// Deterministic fallback (used when LLM is unavailable)
// ---------------------------------------------------------------------------

const FALLBACK_TEMPLATES: Record<string, string> = {
  supine:
    "The model detected a supine posture — the subject appears to be lying on their back.",
  prone:
    "The model detected a prone posture — the subject appears to be lying face-down.",
  sitting:
    "The model detected a sitting posture — the subject appears to be in an upright seated position.",
  standing:
    "The model detected a standing posture — the subject appears to be upright on their feet.",
  "all-fours":
    "The model detected an all-fours posture — the subject appears to be on hands and knees.",
  unknown:
    "The model could not determine the posture with sufficient confidence (threshold: 56%). A clearer image may improve detection.",
};

export function getFallbackExplanation(ctx: InferenceContext): string {
  const key = ctx.isUnknown ? "unknown" : ctx.action.toLowerCase();
  return (
    FALLBACK_TEMPLATES[key] ??
    `The model reported action "${ctx.action}" with ${(ctx.confidence * 100).toFixed(0)}% confidence.`
  );
}
