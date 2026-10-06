/**
 * LLM Explanation Route — GestureVerse-AI / Baby Action AI
 *
 * Registers POST /api/explain on the Express app.
 * Accepts the inference result + prompt strategy, constructs a prompt
 * server-side using promptEngineering.ts, calls invokeLLM(), and returns
 * the generated explanation.
 *
 * Falls back to a deterministic rule-based explanation if:
 *  - BUILT_IN_FORGE_API_KEY is not set
 *  - The LLM request fails
 *  - The LLM returns an empty response
 *
 * No API keys are returned to the client. No image data is stored.
 * No agentic patterns or tool loops.
 */

import type { Express, Request, Response } from "express";
import { invokeLLM } from "./_core/llm";
import {
  buildPrompt,
  getFallbackExplanation,
  type InferenceContext,
  type PromptStrategy,
} from "./promptEngineering";
import { ENV } from "./_core/env";

// ---------------------------------------------------------------------------
// Request / response types
// ---------------------------------------------------------------------------

interface ExplainRequestBody {
  action: string;
  rawAction?: string;
  confidence: number;
  framesAnalyzed?: number;
  strategy: PromptStrategy;
}

type GenerationStatus =
  | "llm_success"
  | "llm_fallback_no_key"
  | "llm_fallback_error"
  | "llm_fallback_empty";

interface ExplainResponseBody {
  explanation: string;
  status: GenerationStatus;
  statusMessage: string;
  strategy: PromptStrategy;
  strategyName: string;
  displayPrompt: string;
  action: string;
  confidence: number;
  isUnknown: boolean;
}

// ---------------------------------------------------------------------------
// Route registration
// ---------------------------------------------------------------------------

export function registerExplainRoute(app: Express): void {
  app.post("/api/explain", async (req: Request, res: Response) => {
    try {
      // --- Input validation ---
      const body = req.body as Partial<ExplainRequestBody>;

      if (typeof body.action !== "string" || body.action.trim() === "") {
        return res.status(400).json({ detail: "action is required" });
      }
      if (typeof body.confidence !== "number" || !isFinite(body.confidence)) {
        return res.status(400).json({ detail: "confidence must be a finite number" });
      }
      if (!["A", "B", "C"].includes(body.strategy ?? "")) {
        return res.status(400).json({ detail: 'strategy must be "A", "B", or "C"' });
      }

      const strategy = body.strategy as PromptStrategy;
      const confidence = Math.max(0, Math.min(1, body.confidence));
      const CONFIDENCE_THRESHOLD = 0.56;
      const isUnknown =
        body.action === "unknown" || confidence < CONFIDENCE_THRESHOLD;

      const ctx: InferenceContext = {
        action: body.action,
        rawAction: body.rawAction,
        confidence,
        isUnknown,
        framesAnalyzed: body.framesAnalyzed,
      };

      // --- Build prompt (pure function, always works) ---
      const built = buildPrompt(strategy, ctx);

      // --- Check API key before attempting LLM call ---
      if (!ENV.forgeApiKey) {
        const fallback = getFallbackExplanation(ctx);
        return res.status(200).json({
          explanation: fallback,
          status: "llm_fallback_no_key",
          statusMessage:
            "LLM generation unavailable — showing a rule-based fallback. Set BUILT_IN_FORGE_API_KEY to enable real generation.",
          strategy,
          strategyName: built.strategyName,
          displayPrompt: built.displayPrompt,
          action: ctx.action,
          confidence: ctx.confidence,
          isUnknown: ctx.isUnknown,
        } satisfies ExplainResponseBody);
      }

      // --- Attempt LLM call ---
      try {
        const result = await invokeLLM({
          messages: [
            { role: "system", content: built.systemMessage },
            { role: "user", content: built.userMessage },
          ],
          maxTokens: built.maxTokens,
        });

        const rawContent = result.choices?.[0]?.message?.content;
        const text =
          typeof rawContent === "string"
            ? rawContent.trim()
            : Array.isArray(rawContent)
              ? rawContent
                  .filter(p => p.type === "text")
                  .map(p => (p as { type: "text"; text: string }).text)
                  .join("")
                  .trim()
              : "";

        if (!text) {
          const fallback = getFallbackExplanation(ctx);
          return res.status(200).json({
            explanation: fallback,
            status: "llm_fallback_empty",
            statusMessage:
              "LLM generation unavailable — showing a rule-based fallback. The LLM returned an empty response.",
            strategy,
            strategyName: built.strategyName,
            displayPrompt: built.displayPrompt,
            action: ctx.action,
            confidence: ctx.confidence,
            isUnknown: ctx.isUnknown,
          } satisfies ExplainResponseBody);
        }

        return res.status(200).json({
          explanation: text,
          status: "llm_success",
          statusMessage: `Generated by LLM (${result.model || "forge"}) using Strategy ${strategy}.`,
          strategy,
          strategyName: built.strategyName,
          displayPrompt: built.displayPrompt,
          action: ctx.action,
          confidence: ctx.confidence,
          isUnknown: ctx.isUnknown,
        } satisfies ExplainResponseBody);
      } catch (llmError) {
        console.error("[Explain] LLM call failed", llmError);
        const fallback = getFallbackExplanation(ctx);
        return res.status(200).json({
          explanation: fallback,
          status: "llm_fallback_error",
          statusMessage:
            "LLM generation unavailable — showing a rule-based fallback. " +
            (llmError instanceof Error ? llmError.message : "Provider error."),
          strategy,
          strategyName: built.strategyName,
          displayPrompt: built.displayPrompt,
          action: ctx.action,
          confidence: ctx.confidence,
          isUnknown: ctx.isUnknown,
        } satisfies ExplainResponseBody);
      }
    } catch (error) {
      console.error("[Explain] Unexpected error", error);
      return res.status(500).json({ detail: "Explanation service error" });
    }
  });
}
