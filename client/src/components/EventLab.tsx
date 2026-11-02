import { useEffect, useMemo, useRef, useState } from "react";
import {
  AlertCircle,
  Bell,
  Check,
  CircleHelp,
  LoaderCircle,
  Mail,
  Play,
  Send,
  Sparkles,
  Square,
  UserRound,
  Video,
} from "lucide-react";
import {
  detectPostureEvents,
  describeEvent,
  EVENT_MODEL_SCORE_FLOOR,
  formatClipTime,
  MIN_STABLE_EVENT_FRAMES,
  POSTURE_LABELS,
  type FrameInference,
  type PostureEvent,
} from "../lib/eventDetection";
import "../event-lab.css";

type WatchState = "idle" | "live" | "demo" | "error";
type SampleRow = FrameInference & { id: number; status: "clear" | "uncertain" | "shape-only"; reason?: string };
type EventSummary = { summary: string; source: "genai" | "template" };
type InferencePayload = { action: string; confidence: number; modelConfidence?: number; decisionSource?: FrameInference["decisionSource"]; reason?: string };

const MAX_LIVE_SAMPLES = 8;
const SAMPLE_EVERY_MS = 3500;
const WALKTHROUGH_FRAMES: SampleRow[] = [
  { id: 0, timeSeconds: 0, action: "supine", confidence: 0.84, modelConfidence: 0.84, decisionSource: "model", status: "clear" },
  { id: 1, timeSeconds: 2, action: "supine", confidence: 0.79, modelConfidence: 0.79, decisionSource: "model", status: "clear" },
  { id: 2, timeSeconds: 4, action: "supine", confidence: 0.82, modelConfidence: 0.82, decisionSource: "model", status: "clear" },
  { id: 3, timeSeconds: 6, action: "prone", confidence: 0.78, modelConfidence: 0.78, decisionSource: "model", status: "clear" },
  { id: 4, timeSeconds: 8, action: "prone", confidence: 0.81, modelConfidence: 0.81, decisionSource: "model", status: "clear" },
  { id: 5, timeSeconds: 10, action: "prone", confidence: 0.76, modelConfidence: 0.76, decisionSource: "model", status: "clear" },
];

function prettyScore(score: number) { return `${Math.round(Math.max(0, Math.min(1, score)) * 100)}%`; }
function modelEventFrames(samples: readonly SampleRow[]): FrameInference[] {
  return samples.map(({ timeSeconds, action, confidence, modelConfidence, decisionSource }) => ({ timeSeconds, action, confidence, modelConfidence, decisionSource }));
}
function readableTime(seconds: number) { return formatClipTime(seconds); }

export default function EventLab() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const intervalRef = useRef<number | null>(null);
  const watchStartedRef = useRef<number>(0);
  const sampleBusyRef = useRef(false);
  const sampleIdRef = useRef(0);
  const [watchState, setWatchState] = useState<WatchState>("idle");
  const [sampleRows, setSampleRows] = useState<SampleRow[]>([]);
  const [error, setError] = useState("");
  const [lastSignal, setLastSignal] = useState("Waiting for a camera signal");
  const [parentName, setParentName] = useState("Parent");
  const [parentContact, setParentContact] = useState("parent@example.com");
  const [parentChannel, setParentChannel] = useState<"email" | "sms">("email");
  const [sentAlertIds, setSentAlertIds] = useState<string[]>([]);
  const [activeEventId, setActiveEventId] = useState("");
  const [summary, setSummary] = useState<EventSummary | null>(null);
  const [summaryBusy, setSummaryBusy] = useState(false);

  const events = useMemo(() => detectPostureEvents(modelEventFrames(sampleRows)), [sampleRows]);
  const activeEvent = events.find((event) => event.id === activeEventId) ?? events[0];

  const stopLive = () => {
    if (intervalRef.current !== null) window.clearInterval(intervalRef.current);
    intervalRef.current = null;
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    if (watchState === "live") setWatchState("idle");
    setLastSignal("Watch mode paused");
  };

  useEffect(() => () => {
    if (intervalRef.current !== null) window.clearInterval(intervalRef.current);
    streamRef.current?.getTracks().forEach((track) => track.stop());
  }, []);

  const clearWatch = () => {
    stopLive();
    setSampleRows([]);
    setSentAlertIds([]);
    setActiveEventId("");
    setSummary(null);
    setError("");
    setLastSignal("Waiting for a camera signal");
  };

  const inferFrame = async (data: string, sampleTime: number) => {
    const response = await fetch("/api/infer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ filename: `live-watch-${Date.now()}.jpg`, contentType: "image/jpeg", data }),
    });
    const payload = await response.json().catch(() => ({})) as Partial<InferencePayload> & { detail?: string };
    if (!response.ok || typeof payload.action !== "string" || typeof payload.confidence !== "number") {
      throw new Error(payload.detail || "The live analyzer returned an incomplete signal.");
    }
    const source = payload.decisionSource;
    const status: SampleRow["status"] = source === "pose_geometry" ? "shape-only" : payload.action === "unknown" || source === "pose_ambiguity" ? "uncertain" : "clear";
    const row: SampleRow = {
      id: sampleIdRef.current++,
      timeSeconds: sampleTime,
      action: payload.action,
      confidence: payload.confidence,
      modelConfidence: payload.modelConfidence ?? payload.confidence,
      decisionSource: source,
      status,
      reason: payload.reason,
    };
    setSampleRows((current) => [...current, row].slice(-MAX_LIVE_SAMPLES));
    setLastSignal(status === "clear" ? `${POSTURE_LABELS[payload.action as keyof typeof POSTURE_LABELS] ?? "Posture observed"} · ${prettyScore(row.modelConfidence ?? row.confidence)}` : "Unclear frame · not counted toward an alert");
  };

  const sampleLiveFrame = async () => {
    const video = videoRef.current;
    if (!video || sampleBusyRef.current || video.readyState < 2 || watchState !== "live") return;
    sampleBusyRef.current = true;
    try {
      const scale = Math.min(1, 640 / Math.max(video.videoWidth, video.videoHeight));
      const canvas = document.createElement("canvas");
      canvas.width = Math.max(1, Math.round(video.videoWidth * scale));
      canvas.height = Math.max(1, Math.round(video.videoHeight * scale));
      const context = canvas.getContext("2d", { alpha: false });
      if (!context) throw new Error("Your browser could not prepare a live frame.");
      context.drawImage(video, 0, 0, canvas.width, canvas.height);
      const data = canvas.toDataURL("image/jpeg", 0.72).split(",", 2)[1] ?? "";
      canvas.width = 0;
      canvas.height = 0;
      if (!data) throw new Error("A live frame could not be prepared.");
      await inferFrame(data, (Date.now() - watchStartedRef.current) / 1000);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Live analysis failed. Try the exhibition demo.");
    } finally {
      sampleBusyRef.current = false;
    }
  };

  const startLive = async () => {
    clearWatch();
    setError("");
    try {
      if (!navigator.mediaDevices?.getUserMedia) throw new Error("Camera access is not available in this browser. Use the exhibition demo instead.");
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
      streamRef.current = stream;
      if (!videoRef.current) throw new Error("The camera preview could not be opened.");
      videoRef.current.srcObject = stream;
      await videoRef.current.play();
      watchStartedRef.current = Date.now();
      setWatchState("live");
      setLastSignal("Camera connected · first sample coming up");
      window.setTimeout(() => void sampleLiveFrame(), 500);
      intervalRef.current = window.setInterval(() => void sampleLiveFrame(), SAMPLE_EVERY_MS);
    } catch (caught) {
      stopLive();
      setWatchState("error");
      setError(caught instanceof Error ? caught.message : "Camera permission was not granted. Use the exhibition demo instead.");
    }
  };

  const showExhibitionDemo = () => {
    stopLive();
    setError("");
    setSampleRows(WALKTHROUGH_FRAMES);
    setSentAlertIds([]);
    setActiveEventId("supine-prone-1");
    setSummary(null);
    setWatchState("demo");
    setLastSignal("Demo signal · repeated change confirmed");
  };

  const generateSummary = async (event: PostureEvent) => {
    setActiveEventId(event.id);
    setSummaryBusy(true);
    setSummary(null);
    try {
      const response = await fetch("/api/event-summary", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ from: event.from, to: event.to, confirmedAtSeconds: event.confirmedAtSeconds, fromFrameCount: event.fromFrameCount, toFrameCount: event.toFrameCount, minimumRawScore: event.minimumRawScore }),
      });
      const payload = await response.json().catch(() => ({})) as Partial<EventSummary> & { detail?: string };
      if (!response.ok || typeof payload.summary !== "string") throw new Error(payload.detail || "The caregiver message could not be generated.");
      setSummary({ summary: payload.summary, source: payload.source === "genai" ? "genai" : "template" });
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "The caregiver message could not be generated.");
    } finally {
      setSummaryBusy(false);
    }
  };

  const sendDemoAlert = (eventId: string) => setSentAlertIds((current) => current.includes(eventId) ? current : [...current, eventId]);
  const cameraLabel = watchState === "live" ? "LIVE WATCH ON" : watchState === "demo" ? "EXHIBITION DEMO" : "READY TO WATCH";

  return (
    <section className="event-lab-section live-watch-section" id="event-lab" aria-labelledby="event-lab-title">
      <div className="content-width">
        <div className="event-lab-heading">
          <div>
            <div className="eyebrow section-eyebrow">02 <span className="eyebrow-line" /> Live watch</div>
            <h2 id="event-lab-title">See it. <em>Understand it.</em></h2>
            <p>Watch a live camera signal, sample clear moments, and show how a repeated posture change becomes a parent-alert candidate.</p>
          </div>
          <div className={`live-watch-stamp live-watch-stamp-${watchState}`}><span className="event-led" /> {cameraLabel}</div>
        </div>

        <div className="live-watch-grid">
          <section className="live-camera-card" aria-label="Live baby activity camera">
            <div className="live-card-title"><div className="live-card-icon"><Video size={18} aria-hidden="true" /></div><div><span className="event-mini-label">A / LIVE INPUT</span><h3>Camera watch</h3></div><span className="event-file-limit">ONE FRAME / 3.5 SEC</span></div>
            <div className="live-camera-view">
              <video ref={videoRef} muted playsInline aria-label="Live camera preview" />
              {watchState !== "live" && <div className="live-camera-placeholder"><Video size={27} aria-hidden="true" /><strong>{watchState === "demo" ? "Exhibition signal is ready" : "Camera is paused"}</strong><span>{watchState === "demo" ? "Use this scripted sequence to present the alert flow." : "Start Watch mode or use the instant exhibition demo."}</span></div>}
              {watchState === "live" && <span className="live-camera-badge"><span className="live-pulse" /> LIVE · LOCAL CAMERA</span>}
            </div>
            <div className="live-privacy"><Bell size={14} aria-hidden="true" /><span>Only compressed sample frames are analyzed. No video recording or continuous storage.</span></div>
            {error && <div className="event-inline-error" role="alert"><AlertCircle size={16} aria-hidden="true" /><span>{error}</span></div>}
            <div className="live-actions">
              {watchState === "live" ? <button className="live-primary-button live-stop-button" type="button" onClick={stopLive}><Square size={15} aria-hidden="true" /> Stop watch</button> : <button className="live-primary-button" type="button" onClick={() => void startLive}><Play size={15} aria-hidden="true" /> Start live watch</button>}
              <button className="live-secondary-button" type="button" onClick={showExhibitionDemo}><Sparkles size={15} aria-hidden="true" /> Instant exhibition demo</button>
            </div>
            <div className="live-signal-line"><span className="signal-dot" /> {lastSignal}</div>
          </section>

          <section className="live-alert-card" aria-labelledby="live-alert-title" aria-live="polite">
            <div className="live-card-title live-card-title-tight"><div className="live-card-icon live-alert-icon"><Bell size={18} aria-hidden="true" /></div><div><span className="event-mini-label">B / DECISION + DELIVERY</span><h3 id="live-alert-title">Parent alert center</h3></div></div>
            <div className="live-rule-banner"><Check size={14} aria-hidden="true" /><span><strong>Alert gate:</strong> {MIN_STABLE_EVENT_FRAMES} clear samples before + {MIN_STABLE_EVENT_FRAMES} after, each at least {prettyScore(EVENT_MODEL_SCORE_FLOOR)} raw score.</span></div>
            <div className="live-evidence-heading"><span>LIVE EVIDENCE</span><strong>{sampleRows.length ? `${sampleRows.length} SAMPLES` : "WAITING"}</strong></div>
            {sampleRows.length ? <div className="live-sample-list">{sampleRows.map((row) => <div className={`live-sample live-sample-${row.status}`} key={row.id}><time>{readableTime(row.timeSeconds)}</time><span className="live-sample-dot" /><span>{POSTURE_LABELS[row.action as keyof typeof POSTURE_LABELS] ?? "Unclear frame"}</span><strong>{row.status === "clear" ? prettyScore(row.modelConfidence ?? row.confidence) : "not counted"}</strong></div>)}</div> : <div className="live-empty"><CircleHelp size={22} aria-hidden="true" /><strong>No parent alert yet</strong><span>Start Watch mode or use the instant demo. A single still never triggers an alert.</span></div>}
            {events.length > 0 && <div className="live-event-stack">{events.map((event) => <article className="live-event-card" key={event.id}><div className="live-event-top"><span>CONFIRMED CHANGE</span><time>{readableTime(event.confirmedAtSeconds)}</time></div><h4>{describeEvent(event)}</h4><p>{event.fromFrameCount} samples before · {event.toFrameCount} after · lowest score {prettyScore(event.minimumRawScore)}</p><div className="live-event-actions"><button className="summary-button" type="button" onClick={() => void generateSummary(event)} disabled={summaryBusy}>{summaryBusy && activeEventId === event.id ? <LoaderCircle size={14} className="spin" aria-hidden="true" /> : <Sparkles size={14} aria-hidden="true" />}{summaryBusy && activeEventId === event.id ? "Generating" : "Generate GenAI message"}</button><button className="send-alert-button" type="button" onClick={() => sendDemoAlert(event.id)} disabled={sentAlertIds.includes(event.id)}>{sentAlertIds.includes(event.id) ? <Check size={14} aria-hidden="true" /> : <Send size={14} aria-hidden="true" />}{sentAlertIds.includes(event.id) ? "Demo sent" : "Send demo alert"}</button></div>{summary && activeEventId === event.id && <div className="event-summary-result"><div><Sparkles size={13} aria-hidden="true" /> {summary.source === "genai" ? "GENAI CAREGIVER MESSAGE" : "TEMPLATE FALLBACK"}</div><p>{summary.summary}</p><small>Generated after the vision rule; it cannot change the event decision.</small></div>}{sentAlertIds.includes(event.id) && <div className="parent-sent-state"><Check size={14} aria-hidden="true" /><span>Alert placed in {parentName || "the parent"}’s {parentChannel} inbox at {parentContact || "the saved contact"}.</span></div>}</article>)}</div>}
            <div className="parent-alert-center live-parent-routing"><div className="parent-alert-heading"><span><UserRound size={15} aria-hidden="true" /> ROUTE TO PARENT</span><span className="demo-only-tag">DEMO ROUTING</span></div><p>Set the caregiver recipient for your exhibition alert.</p><div className="parent-alert-fields"><label><span>Name</span><input value={parentName} onChange={(event) => setParentName(event.target.value)} placeholder="Parent" /></label><label><span>Email or phone</span><input value={parentContact} onChange={(event) => setParentContact(event.target.value)} placeholder="parent@example.com" /></label><label><span>Channel</span><select value={parentChannel} onChange={(event) => setParentChannel(event.target.value as "email" | "sms")}><option value="email">Email</option><option value="sms">SMS</option></select></label></div><div className="parent-alert-capability"><Mail size={14} aria-hidden="true" /><span>Demo inbox only. Production delivery needs consent, verified recipients, retries and receipts.</span></div></div>
          </section>
        </div>
        <div className="live-bottom-note"><Sparkles size={16} aria-hidden="true" /><div><strong>Why GenAI belongs here</strong><span>Computer vision observes. The evidence rule confirms. GenAI turns the confirmed event into a calm, readable caregiver message—without seeing the camera or deciding whether an alert should happen.</span></div><span className="rule-strip-tag">VISION → RULE → GENAI → PARENT</span></div>
      </div>
    </section>
  );
}
