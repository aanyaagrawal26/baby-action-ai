/* Soft Clinical Observatory: warm editorial healthcare technology, asymmetric instrumentation, signal moss accents. */
import { useEffect, useRef, useState } from "react";
import { ArrowUpRight, Check, ChevronDown, CircleHelp, Copy, Github, Menu, Play, Radio, ShieldCheck, Sparkles, X, Cpu, Brain, FlaskConical, ChevronUp } from "lucide-react";
import { toast } from "sonner";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from "recharts";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type PromptStrategy = "A" | "B" | "C";

interface InferenceResult {
  action: string;
  confidence: number;
  framesAnalyzed: number;
  modelProvenance: string;
  rawAction?: string;
  smoothed?: boolean;
  reason?: string;
}

interface ExplanationResult {
  explanation: string;
  status: "llm_success" | "llm_fallback_no_key" | "llm_fallback_error" | "llm_fallback_empty";
  statusMessage: string;
  strategy: PromptStrategy;
  strategyName: string;
  displayPrompt: string;
  action: string;
  confidence: number;
  isUnknown: boolean;
}

// ---------------------------------------------------------------------------
// Static data
// ---------------------------------------------------------------------------

const actions = [
  { name: "Prone", score: 92, color: "bg-[#B7D36B]" },
  { name: "Supine", score: 5, color: "bg-[#D7D4C8]" },
  { name: "All-fours", score: 2, color: "bg-[#D7D4C8]" },
  { name: "Sitting", score: 1, color: "bg-[#D7D4C8]" },
  { name: "Standing", score: 1, color: "bg-[#D7D4C8]" },
];

const RECALL_DATA = [
  { class: "Supine", recall: 77.31, fill: "#B7D36B" },
  { class: "Prone", recall: 62.55, fill: "#879D4C" },
  { class: "All-fours", recall: 66.88, fill: "#9DB55A" },
  { class: "Sitting", recall: 31.57, fill: "#C8A85A" },
  { class: "Standing", recall: 38.94, fill: "#C8A85A" },
];

const STRATEGY_LABELS: Record<PromptStrategy, { name: string; description: string; badge: string }> = {
  A: { name: "Simple Description", description: "One neutral sentence describing the detected action.", badge: "bg-[#DDE5D0] text-[#3E5C28]" },
  B: { name: "Caregiver Explanation", description: "Plain language for a non-technical audience — no medical claims.", badge: "bg-[#E5DCCD] text-[#5C3E1E]" },
  C: { name: "Technical Analysis", description: "Research-level explanation distinguishing MLP prediction from LLM output.", badge: "bg-[#D8DCE5] text-[#1E2F5C]" },
};

function scrollToId(id: string) {
  document.getElementById(id)?.scrollIntoView({ behavior: "smooth" });
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function Home() {
  // Nav / demo state
  const [menuOpen, setMenuOpen] = useState(false);
  const [isLive, setIsLive] = useState(false);
  const [demoAction, setDemoAction] = useState("Prone");
  const [gestureIntent, setGestureIntent] = useState("Open palm");
  const [copied, setCopied] = useState(false);

  // Upload inference
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [analyzing, setAnalyzing] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [inference, setInference] = useState<InferenceResult | null>(null);
  const [inferenceError, setInferenceError] = useState("");

  // Live camera
  const [liveCamera, setLiveCamera] = useState(false);
  const [liveBusy, setLiveBusy] = useState(false);
  const [liveInference, setLiveInference] = useState<InferenceResult | null>(null);
  const [liveError, setLiveError] = useState("");
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);

  // GenAI / Prompt Lab
  const [strategy, setStrategy] = useState<PromptStrategy>("B");
  const [explanation, setExplanation] = useState<ExplanationResult | null>(null);
  const [explaining, setExplaining] = useState(false);
  const [explainError, setExplainError] = useState("");
  const [showPrompt, setShowPrompt] = useState(false);
  const [labOpen, setLabOpen] = useState(false);

  // Compare strategies
  const [comparing, setComparing] = useState(false);
  const [compareResults, setCompareResults] = useState<ExplanationResult[] | null>(null);

  // Model analysis toggle
  const [showMetrics, setShowMetrics] = useState(false);

  // ---------------------------------------------------------------------------
  // Handlers — upload / camera
  // ---------------------------------------------------------------------------

  const startDemo = () => {
    setIsLive(true);
    toast.success("Demo signal connected", { description: "Synthetic pose data is now flowing through the observatory." });
  };

  const handleUpload = (file?: File) => {
    if (!file) return;
    setUploadedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setAnalyzing(false);
    setUploadProgress(0);
    setInference(null);
    setInferenceError("");
    setExplanation(null);
    setExplainError("");
    setCompareResults(null);
    toast.success("Upload ready for analysis", { description: "The browser preview is prepared for the local inference workflow." });
  };

  const analyzeUpload = async () => {
    if (!uploadedFile) return;
    setAnalyzing(true);
    setInference(null);
    setInferenceError("");
    setExplanation(null);
    setExplainError("");
    setCompareResults(null);
    try {
      setUploadProgress(8);
      const data = await new Promise<string>((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => { setUploadProgress(35); resolve(String(reader.result).split(",")[1] || ""); };
        reader.onerror = () => reject(new Error("Could not read upload"));
        reader.readAsDataURL(uploadedFile);
      });
      const response = await fetch("/api/infer", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ filename: uploadedFile.name, contentType: uploadedFile.type, data }) });
      setUploadProgress(88);
      const payload = await response.json() as InferenceResult & { detail?: string };
      if (!response.ok) throw new Error(payload.detail || "Inference failed");
      setUploadProgress(100);
      setInference(payload);
      toast.success("Inference complete", { description: `${payload.action} · ${(payload.confidence * 100).toFixed(0)}% confidence` });
    } catch (error) {
      const message = error instanceof Error ? error.message : "Inference service unavailable";
      setInferenceError(message);
      toast.error("Could not analyze upload", { description: message });
    } finally {
      setAnalyzing(false);
    }
  };

  const inferLiveFrame = async () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || video.readyState < 2 || liveBusy) return;
    setLiveBusy(true);
    try {
      canvas.width = video.videoWidth || 640;
      canvas.height = video.videoHeight || 480;
      const context = canvas.getContext("2d");
      if (!context) throw new Error("Camera canvas unavailable");
      context.drawImage(video, 0, 0, canvas.width, canvas.height);
      const encoded = canvas.toDataURL("image/jpeg", 0.78).split(",")[1];
      const response = await fetch("/api/infer", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ filename: "live-camera-frame.jpg", contentType: "image/jpeg", data: encoded }) });
      const payload = await response.json() as InferenceResult & { detail?: string };
      if (!response.ok) throw new Error(payload.detail || "Live inference failed");
      setLiveInference(payload);
      setLiveError("");
    } catch (error) {
      setLiveError(error instanceof Error ? error.message : "Live inference unavailable");
    } finally {
      setLiveBusy(false);
    }
  };

  const stopLiveCamera = () => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setLiveCamera(false);
    setLiveBusy(false);
  };

  const startLiveCamera = async () => {
    setLiveError("");
    if (!navigator.mediaDevices?.getUserMedia) {
      setLiveError("This browser does not expose camera access.");
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } }, audio: false });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      setLiveCamera(true);
      toast.success("Live camera connected", { description: "Frames are sampled locally and sent for inference; no recording is stored." });
    } catch (error) {
      setLiveError(error instanceof Error ? error.message : "Camera permission was not granted");
    }
  };

  // ---------------------------------------------------------------------------
  // Handlers — GenAI explanation
  // ---------------------------------------------------------------------------

  const generateExplanation = async (inf: InferenceResult, strat: PromptStrategy) => {
    setExplaining(true);
    setExplainError("");
    setExplanation(null);
    try {
      const response = await fetch("/api/explain", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          action: inf.action,
          rawAction: inf.rawAction,
          confidence: inf.confidence,
          framesAnalyzed: inf.framesAnalyzed,
          strategy: strat,
        }),
      });
      const payload = await response.json() as ExplanationResult & { detail?: string };
      if (!response.ok) throw new Error(payload.detail || "Explanation service failed");
      setExplanation(payload);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Explanation service unavailable";
      setExplainError(message);
      toast.error("Could not generate explanation", { description: message });
    } finally {
      setExplaining(false);
    }
  };

  const compareStrategies = async (inf: InferenceResult) => {
    setComparing(true);
    setCompareResults(null);
    try {
      const results = await Promise.all(
        (["A", "B", "C"] as PromptStrategy[]).map(async (s) => {
          const r = await fetch("/api/explain", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ action: inf.action, rawAction: inf.rawAction, confidence: inf.confidence, framesAnalyzed: inf.framesAnalyzed, strategy: s }),
          });
          return r.json() as Promise<ExplanationResult>;
        })
      );
      setCompareResults(results);
      setLabOpen(true);
    } catch (error) {
      toast.error("Compare failed", { description: error instanceof Error ? error.message : "Unknown error" });
    } finally {
      setComparing(false);
    }
  };

  // ---------------------------------------------------------------------------
  // Effects
  // ---------------------------------------------------------------------------

  useEffect(() => {
    if (!liveCamera) return;
    const timer = window.setInterval(() => { void inferLiveFrame(); }, 2200);
    return () => window.clearInterval(timer);
  }, [liveCamera, liveBusy]);

  useEffect(() => () => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    if (previewUrl) URL.revokeObjectURL(previewUrl);
  }, [previewUrl]);

  const copyCommand = async () => {
    await navigator.clipboard?.writeText("streamlit run app.py");
    setCopied(true);
    toast.success("Command copied");
    window.setTimeout(() => setCopied(false), 1800);
  };

  // ---------------------------------------------------------------------------
  // Active inference for GenAI (upload preferred, live fallback)
  // ---------------------------------------------------------------------------
  const activeInference = inference ?? liveInference;

  // ---------------------------------------------------------------------------
  // Render helpers
  // ---------------------------------------------------------------------------

  const statusBadgeStyle = (status?: ExplanationResult["status"]) => {
    if (status === "llm_success") return "bg-[#DDE5D0] text-[#3E5C28] border-[#879D4C]/30";
    return "bg-[#F3E1D8] text-[#754A38] border-[#C98A67]/30";
  };

  const statusIcon = (status?: ExplanationResult["status"]) =>
    status === "llm_success" ? "✦" : "◈";

  // ---------------------------------------------------------------------------
  // JSX
  // ---------------------------------------------------------------------------

  return (
    <div className="min-h-screen overflow-x-hidden bg-[#F4F0E6] text-[#17211C]">

      {/* ── Header ── */}
      <header className="fixed left-0 right-0 top-0 z-50 border-b border-[#17211C]/10 bg-[#F4F0E6]/90 backdrop-blur-xl">
        <div className="mx-auto flex h-[76px] max-w-[1440px] items-center justify-between px-5 sm:px-8 lg:px-12">
          <button className="group flex items-center gap-3" onClick={() => scrollToId("top")} aria-label="Go to top">
            <span className="relative flex h-10 w-10 items-center justify-center overflow-hidden rounded-[13px] bg-[#17211C]">
              <span className="absolute h-2 w-2 rounded-full bg-[#B7D36B] left-[10px] top-[10px]" />
              <span className="absolute h-2 w-2 rounded-full bg-[#B7D36B] left-[19px] top-[17px]" />
              <span className="absolute h-2 w-2 rounded-full bg-[#B7D36B] left-[24px] top-[27px]" />
              <span className="absolute left-[13px] top-[14px] h-[1px] w-[11px] rotate-[32deg] bg-[#B7D36B]" />
              <span className="absolute left-[21px] top-[21px] h-[1px] w-[9px] rotate-[61deg] bg-[#B7D36B]" />
            </span>
            <span className="text-left leading-none"><strong className="font-display text-[17px] tracking-tight">baby action</strong><small className="ml-1 font-mono text-[10px] uppercase tracking-[0.18em] text-[#879D4C]">AI</small></span>
          </button>
          <nav className="hidden items-center gap-7 text-[11px] font-semibold uppercase tracking-[0.16em] text-[#5C675F] md:flex">
            <button onClick={() => scrollToId("observatory")} className="transition-colors hover:text-[#17211C]">Observatory</button>
            <button onClick={() => scrollToId("architecture")} className="transition-colors hover:text-[#17211C]">Pipeline</button>
            <button onClick={() => scrollToId("upload")} className="transition-colors hover:text-[#17211C]">Upload</button>
            <button onClick={() => scrollToId("genai")} className="transition-colors hover:text-[#17211C]">GenAI</button>
            <button onClick={() => scrollToId("model-analysis")} className="transition-colors hover:text-[#17211C]">Model</button>
            <button onClick={() => scrollToId("notes")} className="transition-colors hover:text-[#17211C]">Notes</button>
          </nav>
          <div className="flex items-center gap-3">
            <button onClick={() => scrollToId("run")} className="hidden rounded-full bg-[#17211C] px-5 py-3 text-[11px] font-semibold uppercase tracking-[0.14em] text-[#F4F0E6] transition-transform hover:-translate-y-0.5 active:scale-95 sm:block">Run the prototype <ArrowUpRight className="ml-2 inline h-3.5 w-3.5" /></button>
            <button onClick={() => setMenuOpen(!menuOpen)} className="rounded-full border border-[#17211C]/15 p-2 md:hidden" aria-label="Toggle menu">{menuOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}</button>
          </div>
        </div>
        {menuOpen && (
          <div className="border-t border-[#17211C]/10 bg-[#F4F0E6] px-5 py-5 md:hidden">
            <div className="flex flex-col gap-4 text-xs font-semibold uppercase tracking-[0.14em]">
              <button onClick={() => { scrollToId("observatory"); setMenuOpen(false); }}>Observatory</button>
              <button onClick={() => { scrollToId("architecture"); setMenuOpen(false); }}>Pipeline</button>
              <button onClick={() => { scrollToId("upload"); setMenuOpen(false); }}>Upload</button>
              <button onClick={() => { scrollToId("genai"); setMenuOpen(false); }}>GenAI</button>
              <button onClick={() => { scrollToId("model-analysis"); setMenuOpen(false); }}>Model</button>
              <button onClick={() => { scrollToId("notes"); setMenuOpen(false); }}>Notes</button>
            </div>
          </div>
        )}
      </header>

      <main id="top">

        {/* ── Hero ── */}
        <section className="relative mx-auto max-w-[1440px] px-5 pb-20 pt-36 sm:px-8 lg:px-12 lg:pb-28 lg:pt-48">
          <div className="absolute right-0 top-[110px] hidden font-mono text-[10px] uppercase tracking-[0.22em] text-[#879D4C] lg:block">01 / signal, not spectacle</div>
          <div className="grid items-end gap-12 lg:grid-cols-[minmax(0,0.85fr)_minmax(480px,1.15fr)] lg:gap-16">
            <div className="relative z-10 max-w-[610px]">
              <div className="mb-7 flex items-center gap-3 text-[10px] font-bold uppercase tracking-[0.22em] text-[#879D4C]"><span className="h-2 w-2 animate-pulse rounded-full bg-[#B7D36B] ring-4 ring-[#B7D36B]/20" /> Local-first activity recognition</div>
              <h1 className="font-display text-[clamp(3.8rem,8vw,8.4rem)] leading-[0.86] tracking-[-0.065em]">See the motion,<br /><em className="text-[#8A9F51]">not just</em> the frame.</h1>
              <p className="mt-8 max-w-[450px] text-base leading-7 text-[#59655C] sm:text-lg">Baby Action AI recognises infant posture from pose landmarks, then uses a language model to generate a natural-language explanation — a research prototype connecting ML inference with Generative AI.</p>

              {/* Objective statement */}
              <div className="mt-8 rounded-2xl border border-[#879D4C]/30 bg-[#DDE5D0]/60 px-5 py-4 text-sm leading-6 text-[#3E5C28]">
                <span className="font-semibold">Project objective: </span>Recognise infant posture/action from visual input and transform the model's structured prediction into understandable natural-language information.
              </div>

              {/* ML → GenAI flow */}
              <div className="mt-6 flex flex-wrap items-center gap-2 text-[11px] font-mono uppercase tracking-[0.13em] text-[#59655C]">
                <span className="rounded-md bg-[#17211C] px-2 py-1 text-[#B7D36B]">Pose features</span>
                <ArrowUpRight className="h-3.5 w-3.5 rotate-45 text-[#879D4C]" />
                <span className="rounded-md bg-[#17211C] px-2 py-1 text-[#B7D36B]">MLP prediction</span>
                <ArrowUpRight className="h-3.5 w-3.5 rotate-45 text-[#879D4C]" />
                <span className="rounded-md bg-[#17211C] px-2 py-1 text-[#B7D36B]">Structured context</span>
                <ArrowUpRight className="h-3.5 w-3.5 rotate-45 text-[#879D4C]" />
                <span className="rounded-md bg-[#17211C] px-2 py-1 text-[#B7D36B]">LLM</span>
                <ArrowUpRight className="h-3.5 w-3.5 rotate-45 text-[#879D4C]" />
                <span className="rounded-md bg-[#17211C] px-2 py-1 text-[#B7D36B]">Natural language</span>
              </div>

              <div className="mt-9 flex flex-wrap items-center gap-4">
                <button onClick={() => { scrollToId("observatory"); startDemo(); }} className="rounded-full bg-[#B7D36B] px-6 py-4 text-xs font-bold uppercase tracking-[0.13em] text-[#17211C] shadow-[0_10px_30px_rgba(139,159,81,.22)] transition-all hover:-translate-y-1 hover:bg-[#C6DF79] active:scale-95"><Play className="mr-2 inline h-3.5 w-3.5 fill-current" /> Open observatory</button>
                <button onClick={() => scrollToId("genai")} className="group px-2 py-4 text-xs font-bold uppercase tracking-[0.13em] text-[#59655C]">Try GenAI lab <Sparkles className="ml-1 inline h-3.5 w-3.5 transition-transform group-hover:-translate-y-0.5" /></button>
              </div>
              <div className="mt-14 flex items-center gap-4 border-t border-[#17211C]/10 pt-5 text-[10px] uppercase tracking-[0.16em] text-[#7A857C]"><ShieldCheck className="h-4 w-4 text-[#879D4C]" /> Research prototype — not for clinical decisions</div>
            </div>
            <div className="relative min-h-[390px] overflow-hidden rounded-[30px] bg-[#D8DDCF] shadow-[0_25px_70px_rgba(41,52,39,.13)] lg:min-h-[550px]">
              <img src="/manus-storage/baby-action-observatory_4a551411.png" alt="Abstract pose recognition observatory" className="absolute inset-0 h-full w-full object-cover" />
              <div className="absolute inset-0 bg-gradient-to-t from-[#17211C]/45 via-transparent to-[#F4F0E6]/10" />
              <div className="absolute left-5 top-5 flex items-center gap-2 rounded-full border border-white/35 bg-[#17211C]/55 px-3 py-2 font-mono text-[9px] uppercase tracking-[0.18em] text-[#F4F0E6] backdrop-blur-md"><Radio className="h-3 w-3 text-[#B7D36B]" /> visual input / ready</div>
              <div className="absolute bottom-5 left-5 right-5 flex items-end justify-between text-[#F4F0E6]">
                <div>
                  <div className="font-mono text-[9px] uppercase tracking-[0.18em] text-[#DDE9B7]">ML → GenAI pipeline</div>
                  <div className="mt-2 font-display text-3xl">A quiet read on movement.</div>
                </div>
                <div className="hidden text-right font-mono text-[10px] leading-5 text-white/70 sm:block">MLP classifier<br />51-D pose features</div>
              </div>
            </div>
          </div>
        </section>

        {/* ── Observatory ── */}
        <section id="observatory" className="border-y border-[#17211C]/10 bg-[#E9E9DE] py-20 sm:py-28">
          <div className="mx-auto max-w-[1440px] px-5 sm:px-8 lg:px-12">
            <div className="mb-10 flex flex-wrap items-end justify-between gap-5">
              <div>
                <div className="mb-4 font-mono text-[10px] uppercase tracking-[0.2em] text-[#879D4C]">02 / live observatory</div>
                <h2 className="font-display text-5xl leading-none tracking-[-0.05em] sm:text-6xl">A small window<br /><em>into the signal.</em></h2>
              </div>
              <div className="max-w-[320px] text-sm leading-6 text-[#677269]">A browser-side demonstration of the interface rhythm. The real prototype runs locally with Python, MediaPipe, and a trained MLP.</div>
            </div>
            <div className="grid gap-5 lg:grid-cols-[1.35fr_.65fr]">
              <div className="relative overflow-hidden rounded-[26px] bg-[#17211C] p-5 text-[#F4F0E6] sm:p-7">
                <div className="mb-7 flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className={`h-2.5 w-2.5 rounded-full ${isLive ? "animate-pulse bg-[#B7D36B]" : "bg-[#788177]"}`} />
                    <span className="font-mono text-[10px] uppercase tracking-[0.18em] text-[#C7CEC4]">{isLive ? "signal connected" : "standby / synthetic input"}</span>
                  </div>
                  <span className="font-mono text-[10px] text-[#778278]">MLP · 5-frame smoothing</span>
                </div>
                <div className="relative grid min-h-[275px] place-items-center overflow-hidden rounded-[17px] border border-[#F4F0E6]/10 bg-[#202D25]">
                  <div className="absolute inset-0 opacity-30" style={{ backgroundImage: "linear-gradient(rgba(183,211,107,.16) 1px, transparent 1px), linear-gradient(90deg, rgba(183,211,107,.16) 1px, transparent 1px)", backgroundSize: "48px 48px" }} />
                  <svg viewBox="0 0 420 280" className={`relative h-full w-full max-w-[530px] ${isLive ? "animate-[float_4s_ease-in-out_infinite]" : ""}`}>
                    <g fill="none" stroke="#B7D36B" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M205 60 L178 91 L166 139 L144 190 L129 238" />
                      <path d="M205 60 L233 92 L254 141 L271 190 L290 239" />
                      <path d="M178 91 L233 92" />
                      <path d="M178 91 L138 126 L103 153" />
                      <path d="M233 92 L274 126 L311 151" />
                      <path d="M166 139 L254 141" />
                      <path d="M166 139 L180 193 L190 242" />
                      <path d="M254 141 L235 194 L230 244" />
                    </g>
                    <g fill="#F4F0E6" stroke="#B7D36B" strokeWidth="3">
                      {[[205,60],[178,91],[233,92],[138,126],[274,126],[166,139],[254,141],[144,190],[271,190],[129,238],[290,239],[190,242],[230,244]].map(([cx,cy], i) => <circle key={i} cx={cx} cy={cy} r={i === 0 ? 7 : 4} />)}
                    </g>
                  </svg>
                  <div className="absolute bottom-3 left-3 font-mono text-[9px] uppercase tracking-[.16em] text-[#B7D36B]">17 keypoints · 51-D feature vector</div>
                  <div className="absolute right-3 top-3 font-mono text-[9px] text-white/45">hip-centred · scale-normalised</div>
                </div>
                <div className="mt-6 flex flex-wrap items-end justify-between gap-5">
                  <div>
                    <div className="font-mono text-[9px] uppercase tracking-[.18em] text-[#889489]">MLP frame read</div>
                    <div className="mt-1 font-display text-4xl text-[#B7D36B]">{demoAction}</div>
                  </div>
                  <button onClick={() => { setIsLive(!isLive); if (!isLive) toast.success("Synthetic signal connected"); }} className="rounded-full border border-[#F4F0E6]/20 px-4 py-3 text-[10px] font-bold uppercase tracking-[.14em] transition-colors hover:border-[#B7D36B] hover:text-[#B7D36B]">
                    {isLive ? "Pause signal" : "Connect signal"}
                  </button>
                </div>
              </div>
              <div className="rounded-[26px] bg-[#F4F0E6] p-5 sm:p-7">
                <div className="flex items-center justify-between">
                  <div className="font-mono text-[10px] uppercase tracking-[.18em] text-[#7A857C]">Model confidence</div>
                  <CircleHelp className="h-4 w-4 text-[#9AA39A]" />
                </div>
                <div className="mt-5 flex items-baseline gap-2">
                  <span className="font-display text-7xl tracking-[-.07em]">92</span>
                  <span className="font-mono text-sm text-[#879D4C]">%</span>
                </div>
                <div className="mt-2 text-sm text-[#677269]">Top signal: {demoAction.toLowerCase()}</div>
                <div className="mt-8 space-y-4">
                  {actions.map((action) => (
                    <button key={action.name} onClick={() => setDemoAction(action.name)} className="group block w-full text-left">
                      <div className="mb-1 flex justify-between font-mono text-[10px] uppercase tracking-[.13em] text-[#7A857C]">
                        <span className={demoAction === action.name ? "font-bold text-[#17211C]" : ""}>{action.name}</span>
                        <span>{action.score}%</span>
                      </div>
                      <div className="h-1.5 overflow-hidden rounded-full bg-[#DADBD0]">
                        <div className={`h-full rounded-full transition-all duration-500 ${demoAction === action.name ? "bg-[#B7D36B]" : action.color}`} style={{ width: `${demoAction === action.name ? Math.max(action.score, 8) : action.score}%` }} />
                      </div>
                    </button>
                  ))}
                </div>
                <div className="mt-9 border-t border-[#17211C]/10 pt-4 text-[11px] leading-5 text-[#7A857C]">These are synthetic demo values. Real inference runs via the local Python service.</div>
              </div>
            </div>
          </div>
        </section>

        {/* ── Contactless HCI ── */}
        <section id="interaction" className="border-y border-[#17211C]/10 bg-[#DDE5D0] py-20 sm:py-28">
          <div className="mx-auto max-w-[1440px] px-5 sm:px-8 lg:px-12">
            <div className="grid gap-10 lg:grid-cols-[.8fr_1.2fr] lg:items-end lg:gap-20">
              <div>
                <div className="mb-4 font-mono text-[10px] uppercase tracking-[.2em] text-[#6E8242]">03 / contactless HCI</div>
                <h2 className="font-display text-5xl leading-[.94] tracking-[-.05em] sm:text-6xl">Move beyond<br /><em>touch and clicks.</em></h2>
                <p className="mt-7 max-w-[450px] text-sm leading-6 text-[#59655C]">Traditional human–computer interaction can become inconvenient when hands are occupied. Baby Action AI explores a natural interface: recognise a gesture in real time and translate it into a meaningful action.</p>
                <div className="mt-6 rounded-2xl border border-[#17211C]/12 bg-[#F4F0E6]/70 p-4 text-xs leading-5 text-[#677269]"><strong className="text-[#17211C]">Two signals, one interface.</strong> Baby-action recognition describes what the child is doing. Caregiver gesture recognition can control the observatory without touching the screen. These are separate model intents.</div>
              </div>
              <div className="rounded-[26px] bg-[#17211C] p-5 text-[#F4F0E6] sm:p-7">
                <div className="flex items-center justify-between gap-4">
                  <div>
                    <div className="font-mono text-[9px] uppercase tracking-[.18em] text-[#B7D36B]">gesture intent / interaction contract</div>
                    <div className="mt-2 font-display text-3xl">{gestureIntent}</div>
                  </div>
                  <div className="grid h-12 w-12 place-items-center rounded-full border border-[#B7D36B]/40 text-2xl">{gestureIntent === "Open palm" ? "✋" : gestureIntent === "Thumbs up" ? "👍" : "↔"}</div>
                </div>
                <div className="mt-6 grid gap-3 sm:grid-cols-3">
                  {[{name:"Open palm", action:"Pause live read"},{name:"Thumbs up", action:"Pin current result"},{name:"Swipe right", action:"Next signal"}].map((item) => (
                    <button key={item.name} onClick={() => { setGestureIntent(item.name); toast.success(`${item.name} recognized`, { description: item.action }); }} className={`rounded-xl border p-3 text-left transition-colors ${gestureIntent === item.name ? "border-[#B7D36B] bg-[#26372B]" : "border-[#F4F0E6]/10 bg-[#203027] hover:border-[#B7D36B]/50"}`}>
                      <div className="font-mono text-[9px] uppercase tracking-[.12em] text-[#B7D36B]">{item.name}</div>
                      <div className="mt-2 text-xs leading-4 text-[#C7CEC4]">{item.action}</div>
                    </button>
                  ))}
                </div>
                <div className="mt-6 flex items-center gap-2 border-t border-[#F4F0E6]/10 pt-4 font-mono text-[9px] uppercase tracking-[.14em] text-[#89988A]"><span className="h-2 w-2 animate-pulse rounded-full bg-[#B7D36B]" /> contextual action mapping · demo controls</div>
              </div>
            </div>
          </div>
        </section>

        {/* ── Upload / Live Camera ── */}
        <section id="upload" className="bg-[#F4F0E6] py-20 sm:py-28">
          <div className="mx-auto max-w-[1440px] px-5 sm:px-8 lg:px-12">
            <div className="grid gap-10 lg:grid-cols-[.75fr_1.25fr] lg:items-start lg:gap-20">
              <div>
                <div className="mb-4 font-mono text-[10px] uppercase tracking-[.2em] text-[#879D4C]">upload / local handoff</div>
                <h2 className="font-display text-5xl leading-[.94] tracking-[-.05em] sm:text-6xl">Bring a frame<br /><em>into the read.</em></h2>
                <p className="mt-7 max-w-[390px] text-sm leading-6 text-[#677269]">Upload an image and get a real inference result from the trained MLP via the local Python service. After inference, generate a natural-language explanation using the GenAI section below.</p>
                <div className="mt-6 flex items-center gap-2 font-mono text-[9px] uppercase tracking-[.16em] text-[#7A857C]"><span className="h-2 w-2 rounded-full bg-[#B7D36B]" /> real trained checkpoint · InfActPrimitive</div>
              </div>
              <div className="rounded-[26px] border border-[#17211C]/15 bg-[#E9E9DE] p-5 sm:p-7">
                {/* Live camera card */}
                <div className="mb-5 rounded-[18px] border border-[#17211C]/12 bg-[#17211C] p-4 text-[#F4F0E6]">
                  <div className="flex items-center justify-between gap-4">
                    <div>
                      <div className="font-mono text-[9px] uppercase tracking-[.16em] text-[#B7D36B]">live camera / local sample</div>
                      <div className="mt-2 text-sm font-semibold">Watch the current movement signal.</div>
                    </div>
                    <button onClick={liveCamera ? stopLiveCamera : startLiveCamera} className="rounded-full bg-[#B7D36B] px-4 py-2.5 text-[10px] font-bold uppercase tracking-[.12em] text-[#17211C] transition-transform hover:-translate-y-0.5 active:scale-95">
                      {liveCamera ? "Stop camera" : "Start camera"}
                    </button>
                  </div>
                  <div className="mt-4 overflow-hidden rounded-xl border border-[#F4F0E6]/10 bg-[#203027]">
                    <video ref={videoRef} muted playsInline className={`aspect-video w-full object-cover ${liveCamera ? "opacity-100" : "opacity-35"}`} />
                    <canvas ref={canvasRef} className="hidden" />
                  </div>
                  {liveCamera && (
                    <div className="mt-4 flex items-end justify-between gap-4">
                      <div>
                        <div className="font-mono text-[9px] uppercase tracking-[.16em] text-[#89988A]">latest sampled action</div>
                        <div className="mt-1 font-display text-2xl text-[#B7D36B]">
                          {liveInference ? (liveInference.action === "unknown" ? "Needs a clearer read" : liveInference.action) : liveBusy ? "Reading frame…" : "Waiting for frame"}
                        </div>
                      </div>
                      <div className="text-right font-mono text-xs text-[#B7D36B]">{liveInference ? `${(liveInference.confidence * 100).toFixed(0)}%` : "—"}</div>
                    </div>
                  )}
                  {liveError && <div className="mt-3 text-xs leading-5 text-[#F0C5A9]">{liveError}</div>}
                  <div className="mt-3 text-[10px] leading-4 text-[#89988A]">Camera access is opt-in. Frames are sampled for inference and not recorded or stored.</div>
                </div>

                {/* File drop zone */}
                <label className="group flex min-h-[170px] cursor-pointer flex-col items-center justify-center rounded-[18px] border border-dashed border-[#17211C]/25 bg-[#F4F0E6] px-5 text-center transition-colors hover:border-[#879D4C] hover:bg-[#F8F6EF]">
                  <input type="file" accept="image/*" className="sr-only" onChange={(e) => handleUpload(e.target.files?.[0])} />
                  <div className="grid h-12 w-12 place-items-center rounded-full bg-[#B7D36B]/40 text-[#17211C]"><ArrowUpRight className="h-5 w-5 rotate-45" /></div>
                  <div className="mt-4 text-sm font-semibold">Drop an image here</div>
                  <div className="mt-1 text-xs text-[#7A857C]">JPG, PNG · images only (video not yet supported by inference service)</div>
                </label>

                {/* Uploaded file + analysis */}
                {uploadedFile && (
                  <div className="mt-5 grid gap-5 sm:grid-cols-[180px_1fr] sm:items-center">
                    <img src={previewUrl} alt="Uploaded action preview" className="h-32 w-full rounded-xl object-cover" />
                    <div>
                      <div className="font-mono text-[9px] uppercase tracking-[.16em] text-[#879D4C]">image input</div>
                      <div className="mt-2 truncate text-sm font-semibold">{uploadedFile.name}</div>
                      <div className="mt-1 text-xs text-[#7A857C]">{(uploadedFile.size / 1024 / 1024).toFixed(2)} MB · ready for inference</div>
                      <button onClick={analyzeUpload} disabled={analyzing} className="mt-4 rounded-full bg-[#17211C] px-4 py-3 text-[10px] font-bold uppercase tracking-[.13em] text-[#F4F0E6] transition-transform hover:-translate-y-0.5 active:scale-95 disabled:cursor-wait disabled:opacity-70">
                        {analyzing ? <><span className="mr-2 inline-block h-3 w-3 animate-spin rounded-full border-2 border-[#B7D36B] border-t-transparent align-[-2px]" />Reading signal…</> : "Analyze upload"}
                      </button>

                      {/* Progress bar */}
                      {analyzing && (
                        <div className="mt-4">
                          <div className="mb-2 flex justify-between font-mono text-[9px] uppercase tracking-[.14em] text-[#879D4C]"><span>inference status</span><span>{uploadProgress}%</span></div>
                          <div className="h-2 overflow-hidden rounded-full bg-[#D7D4C8]"><div className="h-full rounded-full bg-[#B7D36B] transition-all duration-500" style={{ width: `${uploadProgress}%` }} /></div>
                          <div className="mt-2 text-[10px] text-[#7A857C]">{uploadProgress < 35 ? "Preparing media…" : uploadProgress < 88 ? "Sending frame to pose model…" : "Reading the action signal…"}</div>
                        </div>
                      )}

                      {/* Inference result */}
                      {inference && (
                        <div className="mt-5 rounded-xl border border-[#879D4C]/30 bg-[#DDE5D0] p-4">
                          <div className="font-mono text-[9px] uppercase tracking-[.16em] text-[#6E8242]">MLP inference result</div>
                          <div className="mt-2 flex items-end justify-between">
                            <div className="font-display text-3xl">{inference.action === "unknown" ? "Needs a clearer read" : inference.action}</div>
                            <div className="font-mono text-2xl text-[#6E8242]">{(inference.confidence * 100).toFixed(0)}%</div>
                          </div>
                          <div className="mt-2 text-xs text-[#677269]">
                            {inference.framesAnalyzed} frame{inference.framesAnalyzed === 1 ? "" : "s"} analysed ·{" "}
                            {inference.action === "unknown" ? "below confidence threshold (0.56) — add a clearer image" : "above confidence threshold"}
                          </div>
                          {inference.rawAction && inference.rawAction !== inference.action && (
                            <div className="mt-1 text-[10px] text-[#7A857C]">Raw argmax: {inference.rawAction} (suppressed by threshold)</div>
                          )}
                          <div className="mt-3 border-t border-[#17211C]/10 pt-3 text-[10px] leading-4 text-[#677269]">{inference.modelProvenance}</div>
                          <button onClick={() => scrollToId("genai")} className="mt-3 text-[10px] font-semibold uppercase tracking-[.12em] text-[#6E8242] underline underline-offset-2">
                            → Generate explanation with GenAI ↓
                          </button>
                        </div>
                      )}

                      {/* Error */}
                      {inferenceError && (
                        <div className="mt-5 rounded-xl border border-[#C98A67]/35 bg-[#F3E1D8] p-4 text-xs leading-5 text-[#754A38]">
                          {inferenceError}
                          <div className="mt-2 font-mono text-[9px] uppercase tracking-[.14em]">Start the FastAPI service: uvicorn inference_api:app --reload</div>
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {!uploadedFile && (
                  <div className="mt-5 flex items-start gap-3 text-xs leading-5 text-[#7A857C]"><ShieldCheck className="mt-0.5 h-4 w-4 shrink-0 text-[#879D4C]" />This page calls the real backend inference route when the local Python service is running at http://127.0.0.1:8000.</div>
                )}
              </div>
            </div>
          </div>
        </section>

        {/* ── GenAI / Prompt Engineering Lab ── */}
        <section id="genai" className="border-y border-[#17211C]/10 bg-[#17211C] py-20 sm:py-28 text-[#F4F0E6]">
          <div className="mx-auto max-w-[1440px] px-5 sm:px-8 lg:px-12">
            <div className="mb-10 flex flex-wrap items-end justify-between gap-5">
              <div>
                <div className="mb-4 flex items-center gap-2 font-mono text-[10px] uppercase tracking-[.2em] text-[#B7D36B]"><Brain className="h-3.5 w-3.5" /> 05 / generative ai</div>
                <h2 className="font-display text-5xl leading-none tracking-[-0.05em] sm:text-6xl">From prediction<br /><em className="text-[#B7D36B]">to explanation.</em></h2>
              </div>
              <div className="max-w-[360px] text-sm leading-6 text-[#BBC4BB]">
                The MLP predicts the action. The LLM transforms that prediction into natural language. These are two separate systems — the LLM does not see the image or perform classification.
              </div>
            </div>

            {/* ML vs GenAI responsibility diagram */}
            <div className="mb-10 grid gap-4 sm:grid-cols-3">
              {[
                { icon: <Cpu className="h-5 w-5" />, label: "ML Model", role: "Predicts posture/action from 51-D pose features.", note: "MLP classifier", color: "border-[#B7D36B]/40 bg-[#1E2E23]" },
                { icon: <Sparkles className="h-5 w-5" />, label: "Generative AI", role: "Transforms the structured prediction into a contextual language response.", note: "LLM via Forge API", color: "border-[#B7D36B] bg-[#263B2C]" },
                { icon: <FlaskConical className="h-5 w-5" />, label: "User", role: "Receives an explanation with the original prediction and its uncertainty.", note: "Full transparency", color: "border-[#B7D36B]/40 bg-[#1E2E23]" },
              ].map((item) => (
                <div key={item.label} className={`rounded-[18px] border p-5 ${item.color}`}>
                  <div className="flex items-center gap-2 text-[#B7D36B]">{item.icon}<span className="font-mono text-[10px] uppercase tracking-[.16em]">{item.label}</span></div>
                  <p className="mt-3 text-sm leading-6 text-[#C7CEC4]">{item.role}</p>
                  <div className="mt-3 font-mono text-[9px] uppercase tracking-[.14em] text-[#89988A]">{item.note}</div>
                </div>
              ))}
            </div>

            {/* Strategy selector + generate */}
            <div className="rounded-[26px] border border-[#F4F0E6]/10 bg-[#203027] p-5 sm:p-7">
              <div className="mb-6 font-mono text-[10px] uppercase tracking-[.18em] text-[#89988A]">prompt strategy</div>
              <div className="grid gap-3 sm:grid-cols-3">
                {(["A", "B", "C"] as PromptStrategy[]).map((s) => (
                  <button key={s} onClick={() => setStrategy(s)} className={`rounded-[14px] border p-4 text-left transition-colors ${strategy === s ? "border-[#B7D36B] bg-[#1A3322]" : "border-[#F4F0E6]/10 bg-[#17211C] hover:border-[#B7D36B]/40"}`}>
                    <div className="flex items-center justify-between">
                      <span className="font-mono text-[10px] uppercase tracking-[.14em] text-[#B7D36B]">Strategy {s}</span>
                      {strategy === s && <Check className="h-3.5 w-3.5 text-[#B7D36B]" />}
                    </div>
                    <div className="mt-2 text-sm font-semibold text-[#E5EDE0]">{STRATEGY_LABELS[s].name}</div>
                    <div className="mt-1 text-xs leading-4 text-[#89988A]">{STRATEGY_LABELS[s].description}</div>
                  </button>
                ))}
              </div>

              {/* Generate button */}
              <div className="mt-6 flex flex-wrap items-center gap-4">
                <button
                  onClick={() => activeInference && generateExplanation(activeInference, strategy)}
                  disabled={!activeInference || explaining}
                  className="rounded-full bg-[#B7D36B] px-6 py-3.5 text-[10px] font-bold uppercase tracking-[.13em] text-[#17211C] transition-transform hover:-translate-y-0.5 active:scale-95 disabled:cursor-not-allowed disabled:opacity-50"
                >
                  {explaining ? <><span className="mr-2 inline-block h-3 w-3 animate-spin rounded-full border-2 border-[#17211C] border-t-transparent align-[-2px]" />Generating…</> : <><Sparkles className="mr-2 inline h-3.5 w-3.5" />Generate explanation</>}
                </button>
                {!activeInference && (
                  <span className="text-xs text-[#89988A]">Upload an image and run inference first to enable generation.</span>
                )}
              </div>

              {/* Explanation result */}
              {explanation && (
                <div className="mt-6 rounded-[18px] border border-[#F4F0E6]/15 bg-[#17211C] p-5">
                  <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
                    <div className="flex items-center gap-2">
                      <span className="font-mono text-[9px] uppercase tracking-[.16em] text-[#B7D36B]">Strategy {explanation.strategy} · {explanation.strategyName}</span>
                    </div>
                    <span className={`rounded-full border px-3 py-1 font-mono text-[9px] uppercase tracking-[.12em] ${statusBadgeStyle(explanation.status)}`}>
                      {statusIcon(explanation.status)} {explanation.status === "llm_success" ? "LLM generated" : "Rule-based fallback"}
                    </span>
                  </div>

                  {/* Inference context */}
                  <div className="mb-4 flex flex-wrap gap-3 font-mono text-[9px] uppercase tracking-[.12em] text-[#89988A]">
                    <span>Action: <span className="text-[#B7D36B]">{explanation.action}</span></span>
                    <span>Confidence: <span className="text-[#B7D36B]">{(explanation.confidence * 100).toFixed(0)}%</span></span>
                    <span>Unknown: <span className="text-[#B7D36B]">{explanation.isUnknown ? "yes" : "no"}</span></span>
                  </div>

                  {/* Generated text */}
                  <blockquote className="border-l-2 border-[#B7D36B] pl-4 text-[15px] leading-7 text-[#E5EDE0]">
                    {explanation.explanation}
                  </blockquote>

                  {/* Status message */}
                  <p className="mt-4 text-[11px] leading-5 text-[#89988A]">{explanation.statusMessage}</p>

                  {/* Show/hide prompt */}
                  <button onClick={() => setShowPrompt(!showPrompt)} className="mt-4 flex items-center gap-1.5 font-mono text-[9px] uppercase tracking-[.14em] text-[#89988A] hover:text-[#B7D36B]">
                    {showPrompt ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
                    {showPrompt ? "Hide prompt" : "Show constructed prompt"}
                  </button>

                  {showPrompt && (
                    <pre className="mt-3 whitespace-pre-wrap rounded-xl border border-[#F4F0E6]/10 bg-[#0F1A13] p-4 font-mono text-[10px] leading-5 text-[#B7D36B]">{explanation.displayPrompt}</pre>
                  )}
                </div>
              )}

              {/* Error */}
              {explainError && (
                <div className="mt-5 rounded-xl border border-[#C98A67]/35 bg-[#2A1C15] p-4 text-xs leading-5 text-[#F0C5A9]">
                  {explainError}
                  <div className="mt-2 font-mono text-[9px] uppercase tracking-[.14em] text-[#C98A67]">Check that the Node backend is running.</div>
                </div>
              )}
            </div>

            {/* Prompt Engineering Lab */}
            <div className="mt-6 rounded-[26px] border border-[#F4F0E6]/10 bg-[#203027]">
              <button onClick={() => setLabOpen(!labOpen)} className="flex w-full items-center justify-between p-5 sm:p-7">
                <div className="flex items-center gap-3">
                  <FlaskConical className="h-5 w-5 text-[#B7D36B]" />
                  <span className="font-mono text-[10px] uppercase tracking-[.18em] text-[#B7D36B]">Prompt Engineering Lab</span>
                </div>
                {labOpen ? <ChevronUp className="h-4 w-4 text-[#89988A]" /> : <ChevronDown className="h-4 w-4 text-[#89988A]" />}
              </button>

              {labOpen && (
                <div className="border-t border-[#F4F0E6]/10 px-5 pb-7 pt-6 sm:px-7">
                  <p className="mb-6 text-sm leading-6 text-[#BBC4BB]">
                    Compare all three strategies against the same inference result. Each strategy sends a different prompt to the LLM, demonstrating how prompt construction affects the generated output.
                  </p>

                  {/* Strategy comparison cards */}
                  <div className="mb-6 grid gap-4 sm:grid-cols-3">
                    {(["A", "B", "C"] as PromptStrategy[]).map((s) => (
                      <div key={s} className="rounded-[14px] border border-[#F4F0E6]/10 bg-[#17211C] p-4">
                        <div className="mb-2 font-mono text-[9px] uppercase tracking-[.14em] text-[#B7D36B]">Strategy {s}</div>
                        <div className="text-sm font-semibold text-[#E5EDE0]">{STRATEGY_LABELS[s].name}</div>
                        <div className="mt-2 text-xs leading-4 text-[#89988A]">{STRATEGY_LABELS[s].description}</div>

                        {compareResults?.[["A","B","C"].indexOf(s)] && (() => {
                          const r = compareResults[["A","B","C"].indexOf(s)];
                          return (
                            <div className="mt-3 border-t border-[#F4F0E6]/10 pt-3">
                              <p className="text-[11px] leading-5 text-[#C7CEC4]">{r.explanation}</p>
                              <span className={`mt-2 inline-block rounded-full border px-2 py-0.5 font-mono text-[8px] uppercase tracking-[.1em] ${statusBadgeStyle(r.status)}`}>
                                {r.status === "llm_success" ? "LLM" : "Fallback"}
                              </span>
                            </div>
                          );
                        })()}
                      </div>
                    ))}
                  </div>

                  <button
                    onClick={() => activeInference && compareStrategies(activeInference)}
                    disabled={!activeInference || comparing}
                    className="rounded-full border border-[#B7D36B] px-6 py-3 text-[10px] font-bold uppercase tracking-[.13em] text-[#B7D36B] transition-colors hover:bg-[#B7D36B] hover:text-[#17211C] disabled:cursor-not-allowed disabled:opacity-50"
                  >
                    {comparing ? <><span className="mr-2 inline-block h-3 w-3 animate-spin rounded-full border-2 border-[#B7D36B] border-t-transparent align-[-2px]" />Comparing…</> : "Compare all three strategies"}
                  </button>
                  {!activeInference && <span className="ml-4 text-xs text-[#89988A]">Run inference first to compare.</span>}
                </div>
              )}
            </div>
          </div>
        </section>

        {/* ── Pipeline Visualization ── */}
        <section id="architecture" className="mx-auto max-w-[1440px] px-5 py-20 sm:px-8 sm:py-28 lg:px-12">
          <div className="grid gap-12 lg:grid-cols-[.75fr_1.25fr] lg:gap-20">
            <div>
              <div className="mb-4 font-mono text-[10px] uppercase tracking-[.2em] text-[#879D4C]">06 / implementation pipeline</div>
              <h2 className="max-w-[450px] font-display text-5xl leading-[.94] tracking-[-.05em] sm:text-6xl">Recognition starts<br /><em>before language.</em></h2>
              <p className="mt-7 max-w-[380px] text-sm leading-6 text-[#677269]">The LLM is not asked to guess what happened. It receives a structured signal <em>after</em> pose estimation and MLP classification have done the careful work.</p>
            </div>
            <div className="relative">
              <div className="absolute left-[24px] top-7 bottom-7 w-px bg-[#17211C]/15 sm:left-[28px]" />
              {[
                { n:"01", title:"Image / Video Input", body:"Webcam frame or uploaded image enters the pipeline." },
                { n:"02", title:"Pose Detection — MediaPipe", body:"MediaPipe Pose extracts 33 body landmarks from the frame. 17 Human3.6M-compatible joints are selected." },
                { n:"03", title:"Pose Normalization", body:"Keypoints are hip-centred and scale-normalised so the vector is body-size independent." },
                { n:"04", title:"51-D Feature Construction", body:"34 xy coordinates + 17 visibility scores form the 51-dimensional input vector for the classifier." },
                { n:"05", title:"Trained MLP Classification", body:"A 2-layer normalized MLP (Linear→LayerNorm→GELU→Dropout, ×2) predicts one of five posture classes." },
                { n:"06", title:"Confidence & Unknown Handling", body:"5-frame probability smoothing applied. Predictions below 0.56 confidence are labelled 'unknown'." },
                { n:"07", title:"Prompt Construction", body:"The inference result, confidence, and model context are assembled into a strategy-specific prompt server-side." },
                { n:"08", title:"LLM / Generative AI", body:"The prompt is sent to the Forge LLM API. The model generates a natural-language response." },
                { n:"09", title:"Generated Explanation", body:"The explanation is returned to the UI with status, prompt, and generation source clearly labelled." },
                { n:"10", title:"User Interface & Analysis", body:"The user sees the ML prediction, confidence, generated explanation, and model analysis side by side." },
              ].map((item) => (
                <div key={item.n} className="relative flex gap-6 pb-8 last:pb-0 sm:gap-8">
                  <div className="relative z-10 grid h-14 w-14 shrink-0 place-items-center rounded-full border border-[#17211C]/15 bg-[#F4F0E6] font-mono text-[11px] text-[#879D4C]">{item.n}</div>
                  <div className="pt-2">
                    <h3 className="font-display text-2xl tracking-[-.03em]">{item.title}</h3>
                    <p className="mt-2 max-w-[420px] text-sm leading-6 text-[#677269]">{item.body}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* ── Model Analysis ── */}
        <section id="model-analysis" className="border-t border-[#17211C]/10 bg-[#E9E9DE] py-20 sm:py-28">
          <div className="mx-auto max-w-[1440px] px-5 sm:px-8 lg:px-12">
            <div className="mb-10 flex flex-wrap items-end justify-between gap-5">
              <div>
                <div className="mb-4 font-mono text-[10px] uppercase tracking-[.2em] text-[#879D4C]">07 / model analysis</div>
                <h2 className="font-display text-5xl leading-none tracking-[-0.05em] sm:text-6xl">What the model<br /><em>actually knows.</em></h2>
              </div>
              <button onClick={() => setShowMetrics(!showMetrics)} className="flex items-center gap-2 rounded-full border border-[#17211C]/20 px-4 py-2.5 font-mono text-[10px] uppercase tracking-[.14em] text-[#5C675F] hover:border-[#879D4C]">
                {showMetrics ? <ChevronUp className="h-3.5 w-3.5" /> : <ChevronDown className="h-3.5 w-3.5" />}
                {showMetrics ? "Hide" : "Show"} full metrics
              </button>
            </div>

            {/* Architecture / provenance cards */}
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
              {[
                { label: "Architecture", value: "2-layer MLP", note: "Linear→LayerNorm→GELU→Dropout ×2" },
                { label: "Input dimension", value: "51-D", note: "34 normalised xy + 17 visibility scores" },
                { label: "Classes", value: "5", note: "supine · prone · sitting · standing · all-fours" },
                { label: "Confidence threshold", value: "0.56", note: "Below this → prediction labelled unknown" },
              ].map((item) => (
                <div key={item.label} className="rounded-[18px] border border-[#17211C]/15 bg-[#F4F0E6] p-5">
                  <div className="font-mono text-[9px] uppercase tracking-[.16em] text-[#879D4C]">{item.label}</div>
                  <div className="mt-2 font-display text-3xl tracking-[-0.03em]">{item.value}</div>
                  <div className="mt-2 text-xs leading-4 text-[#677269]">{item.note}</div>
                </div>
              ))}
            </div>

            {/* Headline metrics */}
            <div className="mt-6 grid gap-4 sm:grid-cols-3">
              {[
                { label: "Frame accuracy", value: "64.21%", note: "Val set — 9 307 frames", warn: false },
                { label: "Balanced accuracy", value: "55.45%", note: "Accounts for class imbalance", warn: false },
                { label: "Mean confidence", value: "77.87%", note: "Not the same as accuracy — high confidence ≠ correct prediction", warn: true },
              ].map((item) => (
                <div key={item.label} className={`rounded-[18px] border p-5 ${item.warn ? "border-[#C8A85A]/40 bg-[#F5EDD8]" : "border-[#879D4C]/30 bg-[#DDE5D0]"}`}>
                  <div className="font-mono text-[9px] uppercase tracking-[.16em] text-[#6E8242]">{item.label}</div>
                  <div className="mt-2 font-display text-4xl tracking-[-0.04em]">{item.value}</div>
                  <div className={`mt-2 text-[11px] leading-4 ${item.warn ? "text-[#8A6020]" : "text-[#677269]"}`}>{item.note}</div>
                </div>
              ))}
            </div>

            {/* Per-class recall chart */}
            <div className="mt-6 rounded-[18px] border border-[#17211C]/15 bg-[#F4F0E6] p-5 sm:p-7">
              <div className="mb-4 font-mono text-[10px] uppercase tracking-[.16em] text-[#879D4C]">per-class recall (validation set)</div>
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={RECALL_DATA} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
                  <XAxis dataKey="class" tick={{ fontSize: 11, fill: "#677269" }} axisLine={false} tickLine={false} />
                  <YAxis domain={[0, 100]} tick={{ fontSize: 10, fill: "#677269" }} axisLine={false} tickLine={false} tickFormatter={(v) => `${v}%`} />
                  <Tooltip formatter={(value: number) => [`${value.toFixed(1)}%`, "Recall"]} contentStyle={{ background: "#F4F0E6", border: "1px solid #17211C20", borderRadius: 8, fontSize: 11 }} />
                  <Bar dataKey="recall" radius={[4, 4, 0, 0]}>
                    {RECALL_DATA.map((entry) => <Cell key={entry.class} fill={entry.fill} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
              <div className="mt-4 text-[11px] leading-5 text-[#7A857C]">
                Recall varies significantly by class. Sitting (31.57%) and Standing (38.94%) are under-represented in the training data. These metrics reflect offline validation — live-webcam accuracy may differ due to domain shift.
              </div>
            </div>

            {/* Expanded metrics */}
            {showMetrics && (
              <div className="mt-6 rounded-[18px] border border-[#17211C]/15 bg-[#F4F0E6] p-5 sm:p-7">
                <div className="mb-4 font-mono text-[10px] uppercase tracking-[.16em] text-[#879D4C]">full classification report (offline validation)</div>
                <div className="overflow-x-auto">
                  <table className="w-full text-xs">
                    <thead>
                      <tr className="border-b border-[#17211C]/10 text-left font-mono text-[9px] uppercase tracking-[.14em] text-[#879D4C]">
                        <th className="pb-3 pr-6">Class</th>
                        <th className="pb-3 pr-6">Precision</th>
                        <th className="pb-3 pr-6">Recall</th>
                        <th className="pb-3 pr-6">F1</th>
                        <th className="pb-3">Support</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-[#17211C]/8 text-[#677269]">
                      {[
                        { cls: "supine",    p: 59.91, r: 77.31, f: 67.51, s: 3288 },
                        { cls: "prone",     p: 78.54, r: 62.55, f: 69.64, s: 4296 },
                        { cls: "sitting",   p: 69.01, r: 31.57, f: 43.32, s: 776 },
                        { cls: "standing",  p: 32.68, r: 38.94, f: 35.53, s: 470 },
                        { cls: "all-fours", p: 43.82, r: 66.88, f: 52.95, s: 477 },
                      ].map((row) => (
                        <tr key={row.cls}>
                          <td className="py-2.5 pr-6 font-semibold text-[#17211C]">{row.cls}</td>
                          <td className="py-2.5 pr-6">{row.p.toFixed(1)}%</td>
                          <td className="py-2.5 pr-6">{row.r.toFixed(1)}%</td>
                          <td className="py-2.5 pr-6">{row.f.toFixed(1)}%</td>
                          <td className="py-2.5">{row.s.toLocaleString()}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <div className="mt-5 rounded-xl border border-[#17211C]/10 bg-[#E9E9DE] p-4 text-[11px] leading-5 text-[#677269]">
                  <strong className="text-[#17211C]">Dataset:</strong> InfActPrimitive public pose-only release. Training on pose annotations only — the model does not learn from raw RGB appearance, clothing, or lighting. Val set: 9 307 frames. Confusion matrix available in models/metrics.json.
                </div>
              </div>
            )}

            {/* Limitations */}
            <div className="mt-6 rounded-[18px] border border-[#C98A67]/30 bg-[#F5EDD8] p-5">
              <div className="mb-3 font-mono text-[9px] uppercase tracking-[.16em] text-[#8A6020]">known limitations</div>
              <ul className="space-y-2 text-xs leading-5 text-[#5C3E1E]">
                <li>• Trained on pose-only annotations — does not generalise to all camera positions, clothing, or body types without validation.</li>
                <li>• Class imbalance: sitting (776 samples) and standing (470) are under-represented versus supine (3 288) and prone (4 296).</li>
                <li>• Mean confidence (77.87%) is not accuracy — a confident prediction can still be wrong.</li>
                <li>• Not a medical device. Not validated for clinical, developmental, or safety assessment.</li>
                <li>• Live-webcam performance may differ from offline validation due to domain shift.</li>
              </ul>
            </div>
          </div>
        </section>

        {/* ── Field Notes ── */}
        <section id="notes" className="border-t border-[#17211C]/10 bg-[#DDE5D0] py-20 sm:py-24">
          <div className="mx-auto max-w-[1440px] px-5 sm:px-8 lg:px-12">
            <div className="grid items-end gap-10 lg:grid-cols-[1fr_.8fr] lg:gap-24">
              <div>
                <div className="mb-4 font-mono text-[10px] uppercase tracking-[.2em] text-[#6E8242]">08 / field notes</div>
                <h2 className="font-display text-5xl leading-[.95] tracking-[-.05em] sm:text-6xl">Built to be<br /><em>looked at closely.</em></h2>
              </div>
              <p className="max-w-[420px] text-sm leading-6 text-[#59655C]">Five action classes. Pose-only training. Honest evaluation. This is the difference between an interesting demo and a useful research instrument.</p>
            </div>
            <div className="mt-14 grid gap-4 sm:grid-cols-3">
              <div className="border-t-2 border-[#17211C] pt-4"><div className="font-mono text-3xl text-[#17211C]">17</div><div className="mt-2 text-xs uppercase tracking-[.15em] text-[#667361]">pose keypoints (H3.6M-compatible)</div></div>
              <div className="border-t-2 border-[#17211C] pt-4"><div className="font-mono text-3xl text-[#17211C]">51</div><div className="mt-2 text-xs uppercase tracking-[.15em] text-[#667361]">feature dimensions per frame</div></div>
              <div className="border-t-2 border-[#17211C] pt-4"><div className="font-mono text-3xl text-[#17211C]">05</div><div className="mt-2 text-xs uppercase tracking-[.15em] text-[#667361]">action classes</div></div>
            </div>
          </div>
        </section>

        {/* ── Run ── */}
        <section id="run" className="bg-[#17211C] px-5 py-20 text-[#F4F0E6] sm:px-8 sm:py-28 lg:px-12">
          <div className="mx-auto grid max-w-[1440px] items-center gap-12 lg:grid-cols-[1fr_.7fr]">
            <div>
              <div className="mb-5 flex items-center gap-2 font-mono text-[10px] uppercase tracking-[.2em] text-[#B7D36B]"><Sparkles className="h-3.5 w-3.5" /> Open source prototype</div>
              <h2 className="max-w-[680px] font-display text-5xl leading-[.92] tracking-[-.055em] sm:text-7xl">Run the signal<br /><em className="text-[#B7D36B]">on your own machine.</em></h2>
              <p className="mt-7 max-w-[480px] text-base leading-7 text-[#BBC4BB]">A FastAPI inference service, a trained MLP checkpoint, and a React frontend with LLM explanation. Start the Python service, then start the Node dev server.</p>
              <div className="mt-9 flex flex-wrap gap-4">
                <button onClick={copyCommand} className="rounded-full bg-[#B7D36B] px-6 py-4 text-xs font-bold uppercase tracking-[.13em] text-[#17211C] transition-transform hover:-translate-y-1 active:scale-95">
                  {copied ? <Check className="mr-2 inline h-3.5 w-3.5" /> : <Copy className="mr-2 inline h-3.5 w-3.5" />}
                  {copied ? "Copied" : "Copy launch command"}
                </button>
                <button onClick={() => toast("The local README contains the full setup flow.")} className="rounded-full border border-[#F4F0E6]/25 px-6 py-4 text-xs font-bold uppercase tracking-[.13em] text-[#F4F0E6] transition-colors hover:border-[#B7D36B] hover:text-[#B7D36B]">Read the notes <ArrowUpRight className="ml-2 inline h-3.5 w-3.5" /></button>
              </div>
            </div>
            <div className="border-l border-[#F4F0E6]/15 pl-7 sm:pl-10">
              <div className="font-mono text-[10px] uppercase tracking-[.18em] text-[#89988A]">minimal start</div>
              <div className="mt-5 rounded-2xl border border-[#F4F0E6]/10 bg-[#203027] p-5 font-mono text-sm leading-8 text-[#D8E3D0]">
                <div><span className="text-[#879D4C]"># Python inference service</span></div>
                <div><span className="text-[#879D4C]">$</span> pip install -r requirements.txt</div>
                <div><span className="text-[#879D4C]">$</span> uvicorn inference_api:app --reload</div>
                <div className="mt-2"><span className="text-[#879D4C]"># Node dev server (new terminal)</span></div>
                <div><span className="text-[#879D4C]">$</span> pnpm dev</div>
              </div>
              <div className="mt-5 text-xs leading-5 text-[#89988A]">Set <code className="rounded bg-[#263B2C] px-1">BUILT_IN_FORGE_API_KEY</code> in a <code className="rounded bg-[#263B2C] px-1">.env</code> file to enable LLM generation. Without it, a rule-based fallback is used.</div>
            </div>
          </div>
        </section>

      </main>

      <footer className="bg-[#17211C] px-5 pb-10 text-[#89988A] sm:px-8 lg:px-12">
        <div className="mx-auto flex max-w-[1440px] flex-col justify-between gap-6 border-t border-[#F4F0E6]/10 pt-6 text-[10px] uppercase tracking-[.15em] sm:flex-row">
          <span>Baby Action AI · research prototype · GestureVerse-AI</span>
          <span>Not a medical device · not a safety guarantee</span>
          <span className="flex items-center gap-2"><Github className="h-3.5 w-3.5" /> local / open / inspectable</span>
        </div>
      </footer>
    </div>
  );
}
