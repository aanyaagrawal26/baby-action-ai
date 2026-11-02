import { describe, expect, it } from "vitest";
import { detectPostureEvents, describeEvent, formatClipTime, type FrameInference } from "./eventDetection";

function frame(timeSeconds: number, action: string, confidence = 0.82, decisionSource: FrameInference["decisionSource"] = "model"): FrameInference {
  return { timeSeconds, action, confidence, modelConfidence: confidence, decisionSource };
}

describe("detectPostureEvents", () => {
  it("confirms a back-to-tummy transition from two stable frames on each side", () => {
    const events = detectPostureEvents([
      frame(0, "supine"), frame(1, "supine"), frame(2, "prone", 0.74), frame(3, "prone", 0.78),
    ]);
    expect(events).toHaveLength(1);
    expect(events[0]).toMatchObject({
      kind: "possible-roll-transition", from: "supine", to: "prone",
      confirmedAtSeconds: 3, fromFrameCount: 2, toFrameCount: 2,
    });
    expect(describeEvent(events[0])).toContain("On back → On tummy");
  });

  it("sorts frame samples by their clip-relative timestamp", () => {
    const events = detectPostureEvents([
      frame(3, "sitting"), frame(2, "sitting"), frame(1, "supine"), frame(0, "supine"),
    ]);
    expect(events[0]?.from).toBe("supine");
    expect(events[0]?.to).toBe("sitting");
  });

  it("does not count duplicate timestamps or malformed timestamps as separate frames", () => {
    expect(detectPostureEvents([
      frame(0, "supine"), frame(0, "supine"), frame(1, "sitting"), frame(2, "sitting"),
      frame(Number.NaN, "standing"),
    ])).toEqual([]);
  });

  it("does not alert when either side of a transition appears only once", () => {
    expect(detectPostureEvents([frame(0, "supine"), frame(1, "prone"), frame(2, "prone")])).toEqual([]);
    expect(detectPostureEvents([frame(0, "supine"), frame(1, "supine"), frame(2, "prone")])).toEqual([]);
  });

  it("does not bridge unknown or low-score gaps", () => {
    const events = detectPostureEvents([
      frame(0, "supine"), frame(1, "supine"), frame(2, "unknown", 0.3),
      frame(3, "sitting"), frame(4, "sitting"),
    ]);
    expect(events).toEqual([]);
  });

  it("does not join stable labels across a long sample gap", () => {
    expect(detectPostureEvents([
      frame(0, "supine"), frame(1, "supine"),
      frame(8, "sitting"), frame(9, "sitting"),
    ])).toEqual([]);
  });

  it("uses the documented conservative score floor", () => {
    expect(detectPostureEvents([
      frame(0, "supine", 0.7), frame(1, "supine", 0.7),
      frame(2, "sitting", 0.699), frame(3, "sitting", 0.9),
    ])).toEqual([]);
  });

  it("does not treat geometry overrides as independent event-classifier evidence", () => {
    expect(detectPostureEvents([
      frame(0, "supine"), frame(1, "supine"),
      frame(2, "all-fours", 0.9, "pose_geometry"), frame(3, "all-fours", 0.9, "pose_geometry"),
    ])).toEqual([]);
  });

  it("reports sitting-to-hands-and-knees as a posture transition, not proof of crawling", () => {
    const events = detectPostureEvents([
      frame(0, "sitting"), frame(1, "sitting"), frame(2, "all-fours"), frame(3, "all-fours"),
    ]);
    expect(events[0]?.kind).toBe("sitting-to-hands-and-knees");
    expect(describeEvent(events[0]!)).toBe("Shifted from sitting to hands and knees");
  });
});

describe("formatClipTime", () => {
  it("formats relative time as minutes and seconds", () => {
    expect(formatClipTime(8.9)).toBe("0:08");
    expect(formatClipTime(67)).toBe("1:07");
  });
});
