import { describe, it, expect } from "vitest";
import {
  buildPrompt,
  buildStrategyA,
  buildStrategyB,
  buildStrategyC,
  getFallbackExplanation,
  type InferenceContext,
} from "./promptEngineering";

const proneCtx: InferenceContext = {
  action: "prone",
  rawAction: "prone",
  confidence: 0.83,
  isUnknown: false,
  framesAnalyzed: 1,
};

const unknownCtx: InferenceContext = {
  action: "unknown",
  rawAction: "sitting",
  confidence: 0.41,
  isUnknown: true,
  framesAnalyzed: 1,
};

describe("buildStrategyA", () => {
  it("returns strategy A metadata", () => {
    const p = buildStrategyA(proneCtx);
    expect(p.strategy).toBe("A");
    expect(p.strategyName).toBe("Simple Action Description");
    expect(p.maxTokens).toBeLessThanOrEqual(100);
  });

  it("includes the action label in the user message", () => {
    const p = buildStrategyA(proneCtx);
    expect(p.userMessage).toContain("prone");
    expect(p.userMessage).toContain("83%");
  });

  it("acknowledges uncertainty for unknown context", () => {
    const p = buildStrategyA(unknownCtx);
    expect(p.userMessage.toLowerCase()).toContain("uncertain");
  });

  it("displayPrompt does not contain API keys or secrets", () => {
    const p = buildStrategyA(proneCtx);
    expect(p.displayPrompt).not.toMatch(/key|secret|token|bearer/i);
  });
});

describe("buildStrategyB", () => {
  it("returns strategy B metadata", () => {
    const p = buildStrategyB(proneCtx);
    expect(p.strategy).toBe("B");
    expect(p.strategyName).toBe("Caregiver-Friendly Explanation");
    expect(p.maxTokens).toBeGreaterThan(80);
  });

  it("system message forbids medical inference", () => {
    const p = buildStrategyB(proneCtx);
    expect(p.systemMessage.toLowerCase()).toContain("medical");
    expect(p.systemMessage.toLowerCase()).toContain("safety");
  });

  it("handles unknown result explicitly", () => {
    const p = buildStrategyB(unknownCtx);
    expect(p.userMessage.toLowerCase()).toContain("low");
  });
});

describe("buildStrategyC", () => {
  it("returns strategy C metadata", () => {
    const p = buildStrategyC(proneCtx);
    expect(p.strategy).toBe("C");
    expect(p.strategyName).toBe("Technical Analysis");
    expect(p.maxTokens).toBeGreaterThan(100);
  });

  it("includes actual model metrics in user message", () => {
    const p = buildStrategyC(proneCtx);
    expect(p.userMessage).toContain("64.21");   // frame accuracy
    expect(p.userMessage).toContain("55.45");   // balanced accuracy
    expect(p.userMessage).toContain("51");      // input dimension
  });

  it("distinguishes classifier from LLM in user message", () => {
    const p = buildStrategyC(proneCtx);
    expect(p.userMessage.toLowerCase()).toContain("llm");
  });
});

describe("buildPrompt dispatcher", () => {
  it("dispatches to A, B, C correctly", () => {
    expect(buildPrompt("A", proneCtx).strategy).toBe("A");
    expect(buildPrompt("B", proneCtx).strategy).toBe("B");
    expect(buildPrompt("C", proneCtx).strategy).toBe("C");
  });

  it("all three strategies produce non-empty system + user messages", () => {
    for (const s of ["A", "B", "C"] as const) {
      const p = buildPrompt(s, proneCtx);
      expect(p.systemMessage.length).toBeGreaterThan(20);
      expect(p.userMessage.length).toBeGreaterThan(20);
      expect(p.displayPrompt.length).toBeGreaterThan(10);
    }
  });
});

describe("getFallbackExplanation", () => {
  it("returns a non-empty string for known actions", () => {
    for (const action of ["supine", "prone", "sitting", "standing", "all-fours"]) {
      const ctx: InferenceContext = { action, confidence: 0.8, isUnknown: false };
      const result = getFallbackExplanation(ctx);
      expect(result.length).toBeGreaterThan(10);
      expect(result.toLowerCase()).toContain(action.split("-")[0]);
    }
  });

  it("returns uncertainty message for unknown", () => {
    const result = getFallbackExplanation(unknownCtx);
    expect(result.toLowerCase()).toMatch(/confidence|certain|unknown/);
  });

  it("fallback never contains the word 'AI-generated'", () => {
    for (const action of ["supine", "prone", "unknown"]) {
      const ctx: InferenceContext = { action, confidence: 0.7, isUnknown: action === "unknown" };
      expect(getFallbackExplanation(ctx)).not.toMatch(/AI.generated/i);
    }
  });
});
