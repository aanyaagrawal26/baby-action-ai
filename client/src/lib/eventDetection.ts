export const EVENT_MODEL_SCORE_FLOOR = 0.7;
export const MIN_STABLE_EVENT_FRAMES = 2;
export const MAX_EVENT_SAMPLE_GAP_SECONDS = 4;

export const POSTURE_LABELS = {
  supine: "On back",
  prone: "On tummy",
  sitting: "Sitting",
  standing: "Standing",
  "all-fours": "On hands and knees",
} as const;

export type Posture = keyof typeof POSTURE_LABELS;
export type EventDecisionSource = "model" | "pose_geometry" | "pose_ambiguity";

export interface FrameInference {
  timeSeconds: number;
  action: string;
  confidence: number;
  modelConfidence?: number;
  decisionSource?: EventDecisionSource;
}

export type EventKind =
  | "position-change"
  | "possible-roll-transition"
  | "lying-to-sitting"
  | "sitting-to-hands-and-knees"
  | "sitting-to-standing";

export interface PostureEvent {
  id: string;
  kind: EventKind;
  from: Posture;
  to: Posture;
  confirmedAtSeconds: number;
  fromFrameCount: number;
  toFrameCount: number;
  minimumRawScore: number;
}

interface StableRun {
  posture: Posture | null;
  frames: FrameInference[];
}

function isPosture(action: string): action is Posture {
  return Object.hasOwn(POSTURE_LABELS, action);
}

function usablePosture(frame: FrameInference): Posture | null {
  const score = frame.modelConfidence ?? frame.confidence;
  if (frame.decisionSource !== "model" || !isPosture(frame.action)) return null;
  if (!Number.isFinite(score) || score < EVENT_MODEL_SCORE_FLOOR) return null;
  return frame.action;
}

function eventKind(from: Posture, to: Posture): EventKind {
  if ((from === "supine" && to === "prone") || (from === "prone" && to === "supine")) {
    return "possible-roll-transition";
  }
  if ((from === "supine" || from === "prone") && to === "sitting") return "lying-to-sitting";
  if (from === "sitting" && to === "all-fours") return "sitting-to-hands-and-knees";
  if (from === "sitting" && to === "standing") return "sitting-to-standing";
  return "position-change";
}

/**
 * Turn sampled per-frame model labels into conservative posture-transition candidates.
 * Unknown, low-score, and geometry-corrected frames deliberately break the evidence chain.
 */
export function detectPostureEvents(frames: readonly FrameInference[]): PostureEvent[] {
  const ordered = [...frames]
    .filter((frame) => Number.isFinite(frame.timeSeconds) && frame.timeSeconds >= 0)
    .sort((a, b) => a.timeSeconds - b.timeSeconds)
    .filter((frame, index, sorted) => index === 0 || frame.timeSeconds - sorted[index - 1].timeSeconds > 0.001);
  const runs: StableRun[] = [];

  for (const frame of ordered) {
    const posture = usablePosture(frame);
    const previous = runs.at(-1);
    const previousFrame = previous?.frames.at(-1);
    if (previousFrame && frame.timeSeconds - previousFrame.timeSeconds > MAX_EVENT_SAMPLE_GAP_SECONDS) {
      runs.push({ posture: null, frames: [] });
    }
    const activeRun = runs.at(-1);
    if (activeRun?.posture === posture) {
      activeRun.frames.push(frame);
    } else {
      runs.push({ posture, frames: [frame] });
    }
  }

  const events: PostureEvent[] = [];
  for (let index = 1; index < runs.length; index += 1) {
    const before = runs[index - 1];
    const after = runs[index];
    if (
      !before.posture ||
      !after.posture ||
      before.posture === after.posture ||
      before.frames.length < MIN_STABLE_EVENT_FRAMES ||
      after.frames.length < MIN_STABLE_EVENT_FRAMES
    ) {
      continue;
    }

    const supportFrames = [...before.frames, ...after.frames];
    const supportScores = supportFrames.map((frame) => frame.modelConfidence ?? frame.confidence);
    events.push({
      id: `${before.posture}-${after.posture}-${events.length + 1}`,
      kind: eventKind(before.posture, after.posture),
      from: before.posture,
      to: after.posture,
      confirmedAtSeconds: after.frames[MIN_STABLE_EVENT_FRAMES - 1].timeSeconds,
      fromFrameCount: before.frames.length,
      toFrameCount: after.frames.length,
      minimumRawScore: Math.min(...supportScores),
    });
  }

  return events;
}

export function formatClipTime(seconds: number): string {
  const safeSeconds = Math.max(0, Math.floor(Number.isFinite(seconds) ? seconds : 0));
  const minutes = Math.floor(safeSeconds / 60);
  const remainder = safeSeconds % 60;
  return `${minutes}:${String(remainder).padStart(2, "0")}`;
}

export function describeEvent(event: Pick<PostureEvent, "kind" | "from" | "to">): string {
  if (event.kind === "possible-roll-transition") {
    return `${POSTURE_LABELS[event.from]} → ${POSTURE_LABELS[event.to]} position change`;
  }
  if (event.kind === "lying-to-sitting") return "Moved from lying to sitting";
  if (event.kind === "sitting-to-hands-and-knees") return "Shifted from sitting to hands and knees";
  if (event.kind === "sitting-to-standing") return "Shifted from sitting to standing";
  return `Posture changed: ${POSTURE_LABELS[event.from]} → ${POSTURE_LABELS[event.to]}`;
}
