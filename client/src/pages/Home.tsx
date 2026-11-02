import { useEffect, useRef, useState } from "react";
import {
  Activity,
  ArrowRight,
  Check,
  ChevronDown,
  CircleHelp,
  FileImage,
  Heart,
  ImagePlus,
  LoaderCircle,
  RefreshCw,
  ShieldCheck,
  Sparkles,
  X,
} from "lucide-react";
import type { DragEvent, KeyboardEvent } from "react";
import EventLab from "../components/EventLab";

type ServiceState = "checking" | "ready" | "offline";
type Photo = { file: File; mime: string; url: string };
type ClassScore = { action: string; score: number };
interface InferenceResult {
  action: string;
  confidence: number;
  modelConfidence?: number;
  decisionSource?: "model" | "pose_geometry" | "pose_ambiguity";
  framesAnalyzed: number;
  modelProvenance: string;
  rawAction?: string;
  reason?: string;
  classScores?: ClassScore[];
}
interface ExplanationResult {
  explanation: string;
  status: "llm_success" | "llm_fallback_no_key" | "llm_fallback_error" | "llm_fallback_empty";
  statusMessage: string;
  strategyName: string;
}

const MAX_FILE_BYTES = 20 * 1024 * 1024;
const MIME_BY_EXTENSION: Record<string, string> = {
  jpg: "image/jpeg",
  jpeg: "image/jpeg",
  png: "image/png",
  webp: "image/webp",
};
const POSE_LABELS: Record<string, string> = {
  supine: "On their back",
  prone: "On their tummy",
  sitting: "Sitting",
  standing: "Standing",
  "all-fours": "On all fours",
  unknown: "Not sure yet",
};
const POSE_GUIDANCE: Record<string, string> = {
  supine: "The closest posture match is lying on their back.",
  prone: "The closest posture match is lying on their tummy.",
  sitting: "The closest posture match is sitting.",
  standing: "The closest posture match is standing.",
  "all-fours": "The closest posture match is on all fours. One still photo cannot tell whether they are moving.",
};
const SUPPORTED_POSES = [
  { label: "Standing", icon: "01" },
  { label: "Sitting", icon: "02" },
  { label: "On all fours", icon: "03" },
  { label: "On tummy", icon: "04" },
  { label: "On back", icon: "05" },
];

function displayPose(action?: string) {
  if (!action) return "Unknown posture";
  return POSE_LABELS[action] ?? action.replaceAll("-", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function boundedScore(value: number) {
  return Math.round(Math.min(1, Math.max(0, Number.isFinite(value) ? value : 0)) * 100);
}

function formatFileSize(bytes: number) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function fileMime(file: File) {
  const type = file.type.toLowerCase();
  if (type.startsWith("image/")) return type;
  const extension = file.name.split(".").pop()?.toLowerCase() ?? "";
  if (!type || type === "application/octet-stream") return MIME_BY_EXTENSION[extension] ?? (extension ? `image/${extension}` : null);
  return null;
}

async function normalizePhoto(file: File): Promise<string> {
  const sourceUrl = URL.createObjectURL(file);
  try {
    const image = new Image();
    image.decoding = "async";
    image.src = sourceUrl;
    await image.decode();
    const longestSide = Math.max(image.naturalWidth, image.naturalHeight);
    if (!longestSide) throw new Error("empty image");
    const scale = Math.min(1, 2048 / longestSide);
    const canvas = document.createElement("canvas");
    canvas.width = Math.max(1, Math.round(image.naturalWidth * scale));
    canvas.height = Math.max(1, Math.round(image.naturalHeight * scale));
    const context = canvas.getContext("2d", { alpha: false });
    if (!context) throw new Error("canvas unavailable");
    context.fillStyle = "#ffffff";
    context.fillRect(0, 0, canvas.width, canvas.height);
    context.drawImage(image, 0, 0, canvas.width, canvas.height);
    const data = canvas.toDataURL("image/jpeg", 0.88).split(",", 2)[1] ?? "";
    canvas.width = 0;
    canvas.height = 0;
    if (!data) throw new Error("empty output");
    return data;
  } catch {
    throw new Error("We couldn’t open that image in this browser. Try a JPG, PNG or WebP export from your phone.");
  } finally {
    URL.revokeObjectURL(sourceUrl);
  }
}

function responseMessage(payload: unknown, fallback: string) {
  if (payload && typeof payload === "object" && "detail" in payload) {
    const detail = (payload as { detail?: unknown }).detail;
    if (typeof detail === "string" && detail.trim()) return detail;
  }
  if (payload && typeof payload === "object" && "error" in payload) {
    const error = (payload as { error?: unknown }).error;
    if (typeof error === "string" && error.trim()) return error;
  }
  return fallback;
}

export default function Home() {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const analysisSequence = useRef(0);
  const [photo, setPhoto] = useState<Photo | null>(null);
  const [dragging, setDragging] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [analysisStep, setAnalysisStep] = useState("");
  const [inference, setInference] = useState<InferenceResult | null>(null);
  const [analysisError, setAnalysisError] = useState("");
  const [serviceState, setServiceState] = useState<ServiceState>("checking");
  const [explanation, setExplanation] = useState<ExplanationResult | null>(null);
  const [explaining, setExplaining] = useState(false);
  const [explanationError, setExplanationError] = useState("");
  const [detailsOpen, setDetailsOpen] = useState(false);

  useEffect(() => {
    if (!photo) return;
    return () => URL.revokeObjectURL(photo.url);
  }, [photo]);

  useEffect(() => {
    const controller = new AbortController();
    fetch("/api/infer/health", { signal: controller.signal })
      .then(async (response) => {
        const payload = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(responseMessage(payload, "Analysis service unavailable"));
        setServiceState(payload.ready ? "ready" : "offline");
      })
      .catch(() => {
        if (!controller.signal.aborted) setServiceState("offline");
      });
    return () => controller.abort();
  }, []);

  const choosePhoto = (file?: File) => {
    if (!file || analyzing) return;
    const mime = fileMime(file);
    if (!mime) {
      setAnalysisError("Choose a common baby photo format such as JPG, PNG, WebP or a phone image. Videos belong in Event review.");
      return;
    }
    if (file.size === 0) {
      setAnalysisError("That file looks empty. Choose a different photo and try again.");
      return;
    }
    if (file.size > MAX_FILE_BYTES) {
      setAnalysisError("This photo is larger than 20 MB. Choose a smaller image and try again.");
      return;
    }
    analysisSequence.current += 1;
    setPhoto({ file, mime, url: URL.createObjectURL(file) });
    setInference(null);
    setAnalysisError("");
    setExplanation(null);
    setExplaining(false);
    setExplanationError("");
  };

  const handleDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setDragging(false);
    if (analyzing) return;
    choosePhoto(event.dataTransfer.files?.[0]);
  };

  const handleUploadKey = (event: KeyboardEvent<HTMLDivElement>) => {
    if (analyzing) return;
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      fileInputRef.current?.click();
    }
  };

  const removePhoto = () => {
    if (analyzing) return;
    analysisSequence.current += 1;
    setPhoto(null);
    setInference(null);
    setAnalysisError("");
    setExplanation(null);
    setExplaining(false);
    setExplanationError("");
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const analyzePhoto = async () => {
    if (!photo || analyzing) return;
    const requestedPhoto = photo;
    const requestId = ++analysisSequence.current;
    setAnalyzing(true);
    setInference(null);
    setAnalysisError("");
    setExplanation(null);
    setExplaining(false);
    setExplanationError("");
    setAnalysisStep("Preparing a clear copy of your photo…");
    try {
      const data = await normalizePhoto(requestedPhoto.file);
      if (requestId !== analysisSequence.current) return;
      setAnalysisStep("Looking for a clear body pose…");
      const response = await fetch("/api/infer", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ filename: `${requestedPhoto.file.name}.normalized.jpg`, contentType: "image/jpeg", data }),
      });
      const payload = await response.json().catch(() => ({}));
      if (requestId !== analysisSequence.current) return;
      if (!response.ok) {
        if (response.status >= 500) setServiceState("offline");
        throw new Error(responseMessage(payload, "We couldn’t analyze this photo. Please try again."));
      }
      if (!payload || typeof payload.action !== "string" || typeof payload.confidence !== "number") {
        throw new Error("The analysis service returned an incomplete result. Please try again.");
      }
      setInference(payload as InferenceResult);
      setServiceState("ready");
    } catch (error) {
      if (requestId !== analysisSequence.current) return;
      const message = error instanceof Error ? error.message : "We couldn’t analyze this photo. Please try again.";
      setAnalysisError(message.includes("Failed to fetch") ? "The photo analyzer isn’t connected. Please try again in a moment." : message);
      if (message.includes("Failed to fetch")) setServiceState("offline");
    } finally {
      if (requestId === analysisSequence.current) {
        setAnalyzing(false);
        setAnalysisStep("");
      }
    }
  };

  const explainResult = async () => {
    if (!inference || inference.action === "unknown" || explaining) return;
    const requestedInference = inference;
    const requestId = analysisSequence.current;
    setExplaining(true);
    setExplanationError("");
    try {
      const response = await fetch("/api/explain", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          action: requestedInference.action,
          rawAction: requestedInference.rawAction,
          confidence: requestedInference.confidence,
          framesAnalyzed: requestedInference.framesAnalyzed,
          strategy: "B",
        }),
      });
      const payload = await response.json().catch(() => ({}));
      if (requestId !== analysisSequence.current) return;
      if (!response.ok) throw new Error(responseMessage(payload, "The note isn’t available right now."));
      setExplanation(payload as ExplanationResult);
    } catch (error) {
      if (requestId === analysisSequence.current) {
        setExplanationError(error instanceof Error ? error.message : "The note isn’t available right now.");
      }
    } finally {
      if (requestId === analysisSequence.current) setExplaining(false);
    }
  };

  const geometryOverride = inference?.decisionSource === "pose_geometry";
  const poseAmbiguity = inference?.decisionSource === "pose_ambiguity";
  const score = inference ? boundedScore(inference.modelConfidence ?? inference.confidence) : 0;
  const hasClearMatch = Boolean(inference && inference.action !== "unknown");
  const resultScores = inference?.classScores?.slice(0, 5) ?? [];

  return (
    <div className="site-shell">
      <header className="site-header">
        <div className="header-inner">
          <a href="#top" className="brand" aria-label="Baby Action AI home">
            <span className="brand-mark" aria-hidden="true">
              <svg viewBox="0 0 40 40" fill="none">
                <path d="M11 11.5 20 19l9-7.5M20 19l-2.7 10" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
                <circle cx="11" cy="11" r="3" fill="currentColor" />
                <circle cx="29" cy="11" r="3" fill="currentColor" />
                <circle cx="20" cy="19" r="3.5" fill="currentColor" />
                <circle cx="17.3" cy="29" r="3" fill="currentColor" />
              </svg>
            </span>
            <span className="brand-wordmark"><strong>baby action</strong><small>AI</small></span>
          </a>
          <nav className="header-nav" aria-label="Main navigation">
            <a href="#event-lab">Live watch</a>
            <a href="#studio">Photo check</a>
            <a href="#method">How it works</a>
            <a href="#research">Research limits</a>
          </nav>
          <a className="header-cta" href="#event-lab">Start live watch <ArrowRight size={16} aria-hidden="true" /></a>
        </div>
      </header>

      <main id="top">
        <section className="hero content-width">
          <div className="hero-copy">
            <div className="eyebrow"><span className="eyebrow-dot" /> Baby activity · capstone research prototype</div>
            <h1>From a frame<br />to a <em>change.</em></h1>
            <p className="hero-description">Use any clear baby photo for a gentle posture read. For meaningful activity, use Live Watch: the vision model samples camera moments, then a transparent rule checks whether the same change repeats before showing a parent-alert preview.</p>
            <div className="hero-notes">
              <span><ShieldCheck size={17} aria-hidden="true" /> Video stays in your browser; sampled frames are not saved</span>
              <span><Heart size={16} aria-hidden="true" /> Demo alerts stay on this page</span>
            </div>
          </div>
          <aside className="hero-scope" aria-label="Posture labels recognized by the model">
              <div className="scope-top"><span><Activity size={15} aria-hidden="true" /> REVIEW PIPELINE</span><span>01—03</span></div>
              <div className="scope-main"><p>FRAME LABELS · REPEATED TRANSITION</p><strong>Vision reads.<br /><em>Rules confirm.</em></strong></div>
              <div className="scope-list"><span><i>01</i> Sampled frames</span><span><i>02</i> Stable pose change</span><span><i>03</i> In-app alert preview</span></div>
              <div className="scope-note"><span>GenAI can explain; it cannot detect or alert.</span><span>RESEARCH ONLY</span></div>
          </aside>
        </section>

        <section className="studio-section" id="studio">
          <div className="content-width">
            <div className="section-heading">
              <div>
                <div className="eyebrow section-eyebrow">01 <span className="eyebrow-line" /> Photo studio</div>
                <h2>Start with a <em>photo.</em></h2>
                <p>Portrait, landscape, close-up or full-body: choose the clearest baby photo you have. We resize a browser-readable image for the model and explain uncertainty instead of forcing a guess.</p>
              </div>
              <div className={`service-pill service-${serviceState}`} role="status" aria-live="polite">
                <span className="service-dot" />
                {serviceState === "ready" ? "Analyzer ready" : serviceState === "checking" ? "Checking analyzer" : "Analyzer offline"}
              </div>
            </div>

            <div className="workbench">
              <section className="photo-card" aria-labelledby="photo-heading">
                <div className="card-topline">
                  <div><span className="step-number">A</span><div><h3 id="photo-heading">Your photo</h3><p>Choose a clear image to begin</p></div></div>
                  <span className="input-tag"><FileImage size={14} aria-hidden="true" /> IMAGE</span>
                </div>

                {!photo ? (
                  <div
                    className={`drop-zone${dragging ? " is-dragging" : ""}`}
                    role="button"
                    tabIndex={0}
                    aria-label="Choose or drop a baby photo in any common image format, up to 20 megabytes"
                    onClick={() => fileInputRef.current?.click()}
                    onKeyDown={handleUploadKey}
                    onDragEnter={(event) => { event.preventDefault(); setDragging(true); }}
                    onDragOver={(event) => event.preventDefault()}
                    onDragLeave={(event) => { if (!event.currentTarget.contains(event.relatedTarget as Node | null)) setDragging(false); }}
                    onDrop={handleDrop}
                  >
                    <input
                      ref={fileInputRef}
                      className="visually-hidden"
                      type="file"
                      accept="image/*,.jpg,.jpeg,.png,.webp,.gif,.bmp,.heic,.heif"
                      aria-label="Choose a photo"
                      disabled={analyzing}
                      onChange={(event) => { choosePhoto(event.currentTarget.files?.[0]); event.currentTarget.value = ""; }}
                    />
                    <span className="upload-icon"><ImagePlus size={24} strokeWidth={1.7} aria-hidden="true" /></span>
                    <strong>Drop a photo here</strong>
                    <span className="drop-secondary">or <span>browse files</span></span>
                    <span className="file-types">JPG, PNG, WebP or phone image <i /> Up to 20 MB</span>
                  </div>
                ) : (
                  <div className="photo-preview-wrap">
                    <img src={photo.url} alt="Selected photo preview" className="photo-preview" />
                    <span className="preview-chip"><Check size={14} aria-hidden="true" /> Ready to analyze</span>
                    <button className="remove-photo" type="button" onClick={removePhoto} aria-label="Remove selected photo" disabled={analyzing}><X size={17} aria-hidden="true" /></button>
                    <div className="photo-file-meta"><span>{photo.file.name}</span><span>{formatFileSize(photo.file.size)}</span></div>
                    <button className="change-photo" type="button" onClick={() => fileInputRef.current?.click()} disabled={analyzing}>Choose a different photo</button>
                    <input
                      ref={fileInputRef}
                      className="visually-hidden"
                      type="file"
                      accept="image/*,.jpg,.jpeg,.png,.webp,.gif,.bmp,.heic,.heif"
                      aria-label="Choose a replacement photo"
                      disabled={analyzing}
                      onChange={(event) => { choosePhoto(event.currentTarget.files?.[0]); event.currentTarget.value = ""; }}
                    />
                  </div>
                )}

                {analysisError && <div className="inline-alert" role="alert"><CircleHelp size={18} aria-hidden="true" /><span>{analysisError}</span></div>}

                <div className="photo-card-bottom">
                  <div className="privacy-note"><ShieldCheck size={16} aria-hidden="true" /><span>Processed for this result; not stored</span></div>
                  <button className="analyze-button" type="button" onClick={analyzePhoto} disabled={!photo || analyzing}>
                    {analyzing ? <><LoaderCircle size={17} className="spin" aria-hidden="true" /> Reading photo</> : <>Analyze photo <ArrowRight size={17} aria-hidden="true" /></>}
                  </button>
                </div>
                {analyzing && <div className="progress-note" role="status" aria-live="polite"><span className="progress-dot" />{analysisStep}</div>}
                {serviceState === "offline" && <p className="offline-note">The page is ready, but live photo analysis needs its Python model service.</p>}
              </section>

              <section className="result-card" aria-labelledby="result-heading" aria-live="polite" aria-busy={analyzing}>
                <div className="card-topline result-topline">
                  <div><span className="step-number step-number-sage">B</span><div><h3 id="result-heading">The read</h3><p>Your closest posture match</p></div></div>
                  {inference && <span className={`read-status${hasClearMatch ? " read-status-clear" : " read-status-uncertain"}`}>{hasClearMatch ? (geometryOverride ? "BODY-SHAPE CHECK" : "POSE FOUND") : "UNCERTAIN"}</span>}
                </div>

                {analyzing ? (
                  <div className="result-empty result-loading"><LoaderCircle size={28} className="spin" aria-hidden="true" /><strong>Taking a closer look…</strong><span>{analysisStep || "Looking for a visible body pose"}</span></div>
                ) : inference ? (
                  <div className="result-content">
                    <div className="result-sparkle" aria-hidden="true">{hasClearMatch ? <Activity size={22} /> : <CircleHelp size={22} />}</div>
                    <div className="result-kicker">{hasClearMatch ? "Closest posture match" : "A clearer view may help"}</div>
                    <h4>{hasClearMatch ? displayPose(inference.action) : "Not sure yet"}</h4>
                    {geometryOverride && inference.rawAction && <p className="model-correction" role="status">A separate body-shape check suggests this posture. The learned model’s closest label was <strong>{displayPose(inference.rawAction)}</strong>, so treat this as a cautious estimate.</p>}
                    {!hasClearMatch && inference.rawAction && <p className="closest-note">{poseAmbiguity ? <>The body-shape check could not confirm the model’s closest guess, <strong>{displayPose(inference.rawAction)}</strong>.</> : <>The model’s closest guess was <strong>{displayPose(inference.rawAction)}</strong>, but it did not meet the confidence threshold.</>}</p>}
                    <p className="result-guidance">{hasClearMatch ? (POSE_GUIDANCE[inference.action] ?? "This is the closest posture match from the available labels.") : (inference.reason === "No confident person pose detected" ? "We couldn’t see enough clear posture landmarks. Try brighter light and keep the shoulders, hips, hands and at least one knee visible; feet don’t need to be in frame for every pose." : poseAmbiguity ? "The body is visible, but its direction is hard to tell from this still photo. Try a clearer view from another angle." : "The pose is ambiguous for this model. Try another angle or a clearer view of the shoulders, hips and supporting limbs.")}</p>

                    {hasClearMatch && !geometryOverride && <>
                      <div className="score-heading"><span>Model score</span><strong>{score}%</strong></div>
                      <div className="score-track" role="img" aria-label={`Model score ${score} percent`}><span style={{ width: `${score}%` }} /></div>
                      <p className="score-caption">A model estimate—not the probability that the answer is correct.</p>
                    </>}
                    {geometryOverride && <div className="geometry-read-note"><Activity size={15} aria-hidden="true" /><span>{inference.action === "all-fours" ? "Matched from the visible shoulders, hips, hands, elbows and a supporting knee; feet don’t need to be in frame for this pose." : "Matched from visible torso and leg alignment."} You can inspect the learned model’s raw scores below; they are not confidence in this shape-based estimate.</span></div>}

                    {resultScores.length > 0 && <details className="other-scores">
                      <summary>See the learned model’s scores <ChevronDown size={15} aria-hidden="true" /></summary>
                      <div className="score-list">{resultScores.map((item) => <div className="score-row" key={item.action}><span>{displayPose(item.action)}</span><span>{boundedScore(item.score)}%</span><i><b style={{ width: `${boundedScore(item.score)}%` }} /></i></div>)}</div>
                      <p>Relative model scores only; they are not calibrated odds.</p>
                    </details>}

                    <div className="result-footnote"><span>{geometryOverride ? "BODY-SHAPE CHECK" : "ONE PHOTO"}</span><i />{inference.framesAnalyzed || 1} MODEL PASS</div>

                    {hasClearMatch && !geometryOverride && <div className="explanation-block">
                      {!explanation ? <button className="explain-button" type="button" onClick={explainResult} disabled={explaining}>{explaining ? <LoaderCircle size={16} className="spin" aria-hidden="true" /> : <Sparkles size={16} aria-hidden="true" />}{explaining ? "Writing a short note…" : "Add a plain-language note"}</button> : <div className="explanation-result"><div><Sparkles size={15} aria-hidden="true" />{explanation.status === "llm_success" ? "A little context" : "Quick context"}</div><p>{explanation.explanation}</p><small>{explanation.status === "llm_success" ? "Language-model generated" : "Template explanation"}</small></div>}
                      {explanationError && <p className="explanation-error" role="alert">{explanationError}</p>}
                    </div>}
                  </div>
                ) : analysisError ? (
                  <div className="result-empty result-error"><span className="result-icon-error"><CircleHelp size={24} aria-hidden="true" /></span><strong>We couldn’t finish this read</strong><span>{analysisError}</span>{photo && <button type="button" className="retry-button" onClick={analyzePhoto}><RefreshCw size={15} aria-hidden="true" /> Try again</button>}</div>
                ) : (
                  <div className="result-empty"><span className="result-placeholder-icon"><Activity size={25} strokeWidth={1.5} aria-hidden="true" /></span><strong>A calm little read, when you’re ready</strong><span>Your result will appear here after you choose a photo and analyze it.</span></div>
                )}
                <div className="result-disclosure"><CircleHelp size={15} aria-hidden="true" /><span>This model recognizes five postures. It can’t verify a child’s age or confirm movement from a still image.</span></div>
              </section>
            </div>

            <div className="studio-caption"><span className="caption-lock"><ShieldCheck size={16} aria-hidden="true" /></span><span>Your photo is sent to the analysis service to make this result. This app does not save the image.</span><span className="caption-divider" /><span>One photo at a time</span></div>
          </div>
        </section>

        <EventLab />

        <section className="method-section" id="method" aria-labelledby="method-heading">
          <div className="content-width">
            <div className="method-heading">
              <div className="method-intro">
                <div className="eyebrow section-eyebrow">03 <span className="eyebrow-line" /> Why these methods</div>
                <h2 id="method-heading">Use the right tool<br />for <em>each step.</em></h2>
                <p>Computer vision handles the visual decision. A small rule checks whether it repeats. GenAI is optional—and only helps explain an event already found.</p>
              </div>
              <div className="method-side-note">GenAI does not improve this model’s measured vision accuracy. The posture classifier and event gate continue working if the language service is unavailable.</div>
            </div>
            <div className="method-cards">
              <article className="method-card">
                <div className="method-card-label"><span><Activity aria-hidden="true" /> 01 / VISION MODEL</span><span>THE OBSERVER</span></div>
                <h3>Label a single frame</h3>
                <p>MediaPipe estimates body landmarks; a small neural network selects one of five postures or abstains. Its visible score is a raw model output, not calibrated odds.</p>
                <span className="method-card-foot">REPEATABLE INPUT · MEASURABLE CLASSIFIER</span>
              </article>
              <article className="method-card">
                <div className="method-card-label"><span><Check aria-hidden="true" /> 02 / EVENT RULE</span><span>THE EVIDENCE GATE</span></div>
                <h3>Confirm a sustained change</h3>
                <p>Two consecutive clear samples on each side of a posture change create a demo event. Unclear, low-score and shape-only frames break the sequence.</p>
                <span className="method-card-foot">DETERMINISTIC · ALERT DOES NOT DEPEND ON GENAI</span>
              </article>
              <article className="method-card method-card-ai">
                <div className="method-card-label"><span><Sparkles aria-hidden="true" /> 03 / GENAI</span><span>OPTIONAL LANGUAGE LAYER</span></div>
                <h3>Explain the event in plain language</h3>
                <p>After the rule fires, GenAI may turn fixed posture labels and clip time into a short caregiver note. It receives no photo or video and cannot change the event.</p>
                <span className="method-card-foot">FLEXIBLE WORDING · NOT A VISION DETECTOR</span>
              </article>
            </div>
          </div>
        </section>

        <section className="posture-section" id="postures">
          <div className="content-width posture-layout">
            <div className="posture-intro">
              <div className="eyebrow section-eyebrow">04 <span className="eyebrow-line" /> Model labels</div>
              <h2>The poses behind<br />each <em>change.</em></h2>
              <p>The clip rule only compares these supported still-frame labels; it does not recognize open-ended actions.</p>
            </div>
            <div className="posture-list">{SUPPORTED_POSES.map((pose) => <div className="posture-item" key={pose.icon}><span className="posture-index">{pose.icon}</span><span>{pose.label}</span><ArrowRight size={15} aria-hidden="true" /></div>)}</div>
          </div>
        </section>

        <section className="research-section" id="research">
          <div className="content-width research-layout">
            <div>
              <div className="eyebrow section-eyebrow">05 <span className="eyebrow-line" /> Limits and research use</div>
              <h2>Curious, careful,<br /><em>and still research.</em></h2>
              <p className="research-copy">This project demonstrates a clip-analysis rule, not reliable unattended baby monitoring. It can miss or mislabel poses, especially when a baby is small in frame, partly covered or photographed at an unusual angle. The in-app alert preview does not send a message to a parent.</p>
              <details className="model-details" open={detailsOpen} onToggle={(event) => setDetailsOpen(event.currentTarget.open)}>
                <summary>How the model works <ChevronDown size={16} aria-hidden="true" /></summary>
                <div className="details-body">
                  <p>MediaPipe estimates body landmarks from each sampled frame. A small neural network compares those landmarks with five posture classes: standing, sitting, on all fours, on the tummy and on the back. A separate rule looks for consecutive model labels that change. It does not recognize open-ended activities or detect danger.</p>
                  <p>The supplied research validation reports 64.2% frame accuracy and 55.4% balanced accuracy on pose-sequence data; sitting and standing were weaker classes. Those metrics do not establish event-level accuracy on home videos, false-alert rate or notification reliability. The source dataset provides pose data, not representative raw home-video validation.</p>
                  <p>A real parent-notification product would need consent and verified recipients, a delivery service with retries/receipts, rate limiting, privacy controls and event-level evaluation on representative consented footage. None are part of this research demo.</p>
                  <p>The supplied model is intended for non-commercial research, teaching, scientific publication and personal experimentation. Check the source dataset terms before using or distributing it as a product.</p>
                </div>
              </details>
            </div>
            <aside className="research-stamp">
              <div className="stamp-icon"><Heart size={22} aria-hidden="true" /></div>
              <div className="stamp-label">A gentle reminder</div>
              <h3>Use your own<br />caregiver judgment.</h3>
              <p>This read describes a model’s guess from one image. It doesn’t assess health, development or safety.</p>
              <span className="stamp-rule" />
              <span className="stamp-meta">BABY ACTION AI <i /> RESEARCH PREVIEW</span>
            </aside>
          </div>
        </section>
      </main>

      <footer className="site-footer">
        <div className="content-width footer-inner"><a href="#top" className="brand footer-brand"><span className="brand-mark" aria-hidden="true"><svg viewBox="0 0 40 40" fill="none"><path d="M11 11.5 20 19l9-7.5M20 19l-2.7 10" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round"/><circle cx="11" cy="11" r="3" fill="currentColor"/><circle cx="29" cy="11" r="3" fill="currentColor"/><circle cx="20" cy="19" r="3.5" fill="currentColor"/><circle cx="17.3" cy="29" r="3" fill="currentColor"/></svg></span><span className="brand-wordmark"><strong>baby action</strong><small>AI</small></span></a><span>Capstone research prototype · not a monitoring product.</span><a href="#top" className="back-top">Back to top <ArrowRight size={14} aria-hidden="true" /></a></div>
      </footer>
    </div>
  );
}
