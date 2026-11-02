import { describe, expect, it } from "vitest";
import { registerInferenceProxy, validateInferenceUpload } from "./inferenceProxy";

const jpegBytes = Buffer.from([0xff, 0xd8, 0xff, 0x00, 0x01]);
const encode = (bytes: Buffer) => bytes.toString("base64");
const validUpload = {
  filename: "baby.jpg",
  contentType: "image/jpeg",
  data: encode(jpegBytes),
};

describe("inference upload validation", () => {
  it("accepts an image with a matching JPEG signature", () => {
    const result = validateInferenceUpload(validUpload);
    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.upload.filename).toBe("baby.jpg");
      expect(result.upload.contentType).toBe("image/jpeg");
      expect(result.upload.bytes).toEqual(jpegBytes);
    }
  });

  it("rejects video and unsupported MIME types", () => {
    const result = validateInferenceUpload({ ...validUpload, contentType: "video/mp4" });
    expect(result).toMatchObject({ ok: false, statusCode: 415 });
  });

  it("rejects malformed base64", () => {
    const result = validateInferenceUpload({ ...validUpload, data: "not base64!" });
    expect(result).toMatchObject({ ok: false, statusCode: 400 });
  });

  it("rejects a MIME type that does not match the file signature", () => {
    const result = validateInferenceUpload({ ...validUpload, contentType: "image/png" });
    expect(result).toMatchObject({ ok: false, statusCode: 415 });
  });

  it("strips any supplied directory from the forwarded filename", () => {
    const result = validateInferenceUpload({ ...validUpload, filename: "../../baby.jpg" });
    expect(result.ok).toBe(true);
    if (result.ok) expect(result.upload.filename).toBe("baby.jpg");
  });
});

describe("inference proxy routes", () => {
  it("registers health and image-analysis endpoints", () => {
    const registered: string[] = [];
    const app = {
      get: (path: string) => registered.push(`GET ${path}`),
      post: (path: string) => registered.push(`POST ${path}`),
    } as never;

    registerInferenceProxy(app);

    expect(registered).toEqual(["GET /api/infer/health", "POST /api/infer"]);
  });
});
