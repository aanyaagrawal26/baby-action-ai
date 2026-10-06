import type { Express, Request, Response } from "express";

const MAX_UPLOAD_BYTES = 50 * 1024 * 1024;

export function registerInferenceProxy(app: Express) {
  app.post("/api/infer", async (req: Request, res: Response) => {
    try {
      const { filename, contentType, data } = req.body ?? {};
      if (typeof filename !== "string" || typeof contentType !== "string" || typeof data !== "string") {
        return res.status(400).json({ detail: "filename, contentType, and base64 data are required" });
      }
      if (!(contentType.startsWith("image/") || contentType.startsWith("video/"))) {
        return res.status(415).json({ detail: "Upload an image or video file" });
      }
      const bytes = Buffer.from(data, "base64");
      if (bytes.byteLength > MAX_UPLOAD_BYTES) {
        return res.status(413).json({ detail: "File exceeds the 50 MB upload limit" });
      }
      const form = new FormData();
      form.append("file", new Blob([bytes], { type: contentType }), filename);
      const target = `${process.env.INFERENCE_API_URL || "http://127.0.0.1:8000"}/analyze`;
      const upstream = await fetch(target, { method: "POST", body: form });
      const payload = await upstream.json().catch(() => ({ detail: "Inference service returned invalid JSON" }));
      return res.status(upstream.status).json(payload);
    } catch (error) {
      console.error("[Inference] proxy failure", error);
      return res.status(503).json({ detail: "Inference service unavailable. Start the local FastAPI service and try again." });
    }
  });
}
