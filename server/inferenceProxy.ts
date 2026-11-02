import { basename } from "node:path";
import type { Express, Request, Response } from "express";

export const MAX_UPLOAD_BYTES = 50 * 1024 * 1024;
const MAX_BASE64_LENGTH = 4 * Math.ceil(MAX_UPLOAD_BYTES / 3);
const MIME_TYPES = new Set(["image/jpeg", "image/png", "image/webp"]);
const BASE64_PATTERN = /^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$/;

type SupportedMime = "image/jpeg" | "image/png" | "image/webp";
export interface InferenceUpload {
  filename: string;
  contentType: SupportedMime;
  bytes: Buffer;
}
export type UploadValidation =
  | { ok: true; upload: InferenceUpload }
  | { ok: false; statusCode: number; detail: string };

function hasMatchingSignature(bytes: Buffer, mime: SupportedMime) {
  if (mime === "image/jpeg") return bytes.length >= 3 && bytes[0] === 0xff && bytes[1] === 0xd8 && bytes[2] === 0xff;
  if (mime === "image/png") return bytes.subarray(0, 8).equals(Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]));
  return bytes.length >= 12 && bytes.toString("ascii", 0, 4) === "RIFF" && bytes.toString("ascii", 8, 12) === "WEBP";
}

export function validateInferenceUpload(body: unknown): UploadValidation {
  if (!body || typeof body !== "object") {
    return { ok: false, statusCode: 400, detail: "Choose a JPEG, PNG or WebP photo to analyze" };
  }
  const input = body as Record<string, unknown>;
  if (typeof input.filename !== "string" || typeof input.contentType !== "string" || typeof input.data !== "string") {
    return { ok: false, statusCode: 400, detail: "filename, contentType, and base64 photo data are required" };
  }
  if (!MIME_TYPES.has(input.contentType)) {
    return { ok: false, statusCode: 415, detail: "Use a JPEG, PNG or WebP photo. Videos and other formats are not supported" };
  }
  if (!input.data || input.data.length > MAX_BASE64_LENGTH) {
    return { ok: false, statusCode: input.data.length > MAX_BASE64_LENGTH ? 413 : 400, detail: input.data.length ? "Photo exceeds the 50 MB upload limit" : "The uploaded photo is empty" };
  }
  if (input.data.length % 4 !== 0 || !BASE64_PATTERN.test(input.data)) {
    return { ok: false, statusCode: 400, detail: "The photo data is not valid base64" };
  }

  const bytes = Buffer.from(input.data, "base64");
  if (!bytes.length) return { ok: false, statusCode: 400, detail: "The uploaded photo is empty" };
  if (bytes.length > MAX_UPLOAD_BYTES) return { ok: false, statusCode: 413, detail: "Photo exceeds the 50 MB upload limit" };
  const contentType = input.contentType as SupportedMime;
  if (!hasMatchingSignature(bytes, contentType)) {
    return { ok: false, statusCode: 415, detail: "The file contents don’t match its image type. Choose a valid JPEG, PNG or WebP photo" };
  }

  const safeName = basename(input.filename).replace(/[\u0000-\u001f\u007f]/g, "").slice(0, 180) || "photo";
  return { ok: true, upload: { filename: safeName, contentType, bytes } };
}

function inferenceUrl(path: string) {
  const base = (process.env.INFERENCE_API_URL || "http://127.0.0.1:8000").replace(/\/+$/, "");
  return `${base}${path}`;
}

export function registerInferenceProxy(app: Express) {
  app.get("/api/infer/health", async (_req: Request, res: Response) => {
    try {
      const upstream = await fetch(inferenceUrl("/health"), { signal: AbortSignal.timeout(5_000) });
      const payload = await upstream.json().catch(() => ({})) as Record<string, unknown>;
      const ready = upstream.ok && payload.ok === true;
      return res.status(ready ? 200 : 503).json({
        ready,
        labels: Array.isArray(payload.labels) ? payload.labels : [],
        confidenceThreshold: typeof payload.confidenceThreshold === "number" ? payload.confidenceThreshold : null,
      });
    } catch {
      return res.status(503).json({ ready: false, detail: "Photo analysis service is not connected" });
    }
  });

  app.post("/api/infer", async (req: Request, res: Response) => {
    const validation = validateInferenceUpload(req.body);
    if (!validation.ok) return res.status(validation.statusCode).json({ detail: validation.detail });

    try {
      const { filename, contentType, bytes } = validation.upload;
      const form = new FormData();
      form.append("file", new Blob([new Uint8Array(bytes)], { type: contentType }), filename);
      const upstream = await fetch(inferenceUrl("/analyze"), {
        method: "POST",
        body: form,
        signal: AbortSignal.timeout(90_000),
      });
      const payload = await upstream.json().catch(() => ({ detail: "Photo analysis service returned invalid JSON" }));
      return res.status(upstream.status).json(payload);
    } catch (error) {
      const timedOut = error instanceof Error && error.name === "TimeoutError";
      if (!timedOut && error instanceof Error && error.name !== "AbortError") {
        console.error("[Inference] proxy could not reach analysis service", error.message);
      }
      return res.status(503).json({
        detail: timedOut
          ? "Photo analysis took too long. Try a smaller or clearer image."
          : "Photo analysis is unavailable. Please check that the Python model service is running and try again.",
      });
    }
  });
}
