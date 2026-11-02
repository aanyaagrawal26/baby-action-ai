import type { Express, Request, Response } from "express";
import { invokeLLM } from "./_core/llm";

const POSTURES = ["supine", "prone", "sitting", "standing", "all-fours"] as const;
type Posture = (typeof POSTURES)[number];
const LABELS: Record<Posture, string> = {
  supine: "on their back",
  prone: "on their tummy",
  sitting: "sitting",
  standing: "standing",
  "all-fours": "on hands and knees",
};

export interface EventSummaryInput {
  from: Posture;
  to: Posture;
  confirmedAtSeconds: number;
  fromFrameCount: number;
  toFrameCount: number;
  minimumRawScore: number;
}

export function validateEventSummaryInput(input: unknown): EventSummaryInput | null {
  if (!input || typeof input !== "object" || Array.isArray(input)) return null;
  const body = input as Record<string, unknown>;
  if (!(POSTURES as readonly unknown[]).includes(body.from) || !(POSTURES as readonly unknown[]).includes(body.to)) return null;
  if (body.from === body.to) return null;
  if (typeof body.confirmedAtSeconds !== "number" || !Number.isFinite(body.confirmedAtSeconds) || body.confirmedAtSeconds < 0 || body.confirmedAtSeconds > 120) return null;
  if (!Number.isInteger(body.fromFrameCount) || Number(body.fromFrameCount) < 2 || Number(body.fromFrameCount) > 8) return null;
  if (!Number.isInteger(body.toFrameCount) || Number(body.toFrameCount) < 2 || Number(body.toFrameCount) > 8) return null;
  if (typeof body.minimumRawScore !== "number" || !Number.isFinite(body.minimumRawScore) || body.minimumRawScore < 0.7 || body.minimumRawScore > 1) return null;
  return {
    from: body.from as Posture,
    to: body.to as Posture,
    confirmedAtSeconds: body.confirmedAtSeconds,
    fromFrameCount: Number(body.fromFrameCount),
    toFrameCount: Number(body.toFrameCount),
    minimumRawScore: body.minimumRawScore,
  };
}

export function buildEventSummaryMessages(event: EventSummaryInput) {
  const seconds = Math.floor(event.confirmedAtSeconds);
  const time = `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`;
  return [
    {
      role: "system" as const,
      content:
        "Write a brief, plain-language summary of a posture transition that a separate computer-vision classifier and deterministic rule have already identified. You do not classify images or decide whether to send an alert. Describe only the provided posture-label change and clip-relative time. Never infer emotion, intent, development, injury, danger, safety, sleep state or medical meaning. Do not recommend an action. State that this is a research estimate, not a confirmed observation. Use at most two short sentences, under 55 words.",
    },
    {
      role: "user" as const,
      content: `The rule-based system reported that sampled labels changed from ${LABELS[event.from]} to ${LABELS[event.to]}, confirmed at ${time} in a short selected clip. The transition was supported by ${event.fromFrameCount} consecutive frame classifications before and ${event.toFrameCount} after. The minimum model score among those supporting samples was ${(event.minimumRawScore * 100).toFixed(0)}%, which is an uncalibrated model score, not accuracy. Write the required careful summary. Do not claim that a message was sent.`,
    },
  ];
}

export function fallbackEventSummary(event: EventSummaryInput): string {
  return `The sampled pose labels changed from ${LABELS[event.from]} to ${LABELS[event.to]} around ${Math.floor(event.confirmedAtSeconds)} seconds into the clip. This is a computer-vision research estimate, not a confirmed observation and not a safety assessment.`;
}

export function sanitizeGeneratedEventSummary(candidate: string, event: EventSummaryInput): string {
  const summary = candidate.replace(/[\u0000-\u001f\u007f]/g, " ").replace(/\s+/g, " ").trim();
  const words = summary.split(/\s+/).filter(Boolean);
  const textWithoutNegatedSafetyDisclaimer = summary.replace(/\bnot\s+(?:a\s+)?(?:safety|safe|danger|medical|health)\b/gi, "");
  const unsupportedClaims = /\b(safe|safety|danger|medical|health|diagnos\w*|development\w*|emotion\w*|intent\w*|injur\w*|emergenc\w*|sleep|breath\w*|chok\w*|suffocat\w*|risk|notification sent|message sent|alert sent)\b/i;
  if (!summary || summary.length > 500 || words.length > 55 || unsupportedClaims.test(textWithoutNegatedSafetyDisclaimer)) {
    return fallbackEventSummary(event);
  }
  return summary;
}

function textFromContent(content: unknown): string {
  if (typeof content === "string") return content.trim();
  if (!Array.isArray(content)) return "";
  return content
    .filter((part): part is { type: "text"; text: string } => Boolean(part) && typeof part === "object" && (part as { type?: unknown }).type === "text" && typeof (part as { text?: unknown }).text === "string")
    .map((part) => part.text)
    .join(" ")
    .trim();
}

export function registerEventSummaryRoute(app: Express): void {
  app.post("/api/event-summary", async (req: Request, res: Response) => {
    const event = validateEventSummaryInput(req.body);
    if (!event) return res.status(400).json({ detail: "A supported posture transition with repeated evidence is required" });

    const fallback = fallbackEventSummary(event);
    try {
      const result = await invokeLLM({ messages: buildEventSummaryMessages(event), model: "gpt-5-mini", max_completion_tokens: 2048 });
      const generatedSummary = textFromContent(result.choices?.[0]?.message?.content);
      const summary = sanitizeGeneratedEventSummary(generatedSummary, event);
      if (!summary) return res.json({ summary: fallback, source: "template" });
      return res.json({ summary, source: summary === fallback ? "template" : "genai" });
    } catch {
      return res.json({ summary: fallback, source: "template" });
    }
  });
}
