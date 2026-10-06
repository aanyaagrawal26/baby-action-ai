import { describe, expect, it } from "vitest";
import { registerInferenceProxy } from "./inferenceProxy";

describe("inference proxy", () => {
  it("registers the upload endpoint", () => {
    let registeredPath = "";
    const app = { post: (path: string) => { registeredPath = path; } } as never;
    registerInferenceProxy(app);
    expect(registeredPath).toBe("/api/infer");
  });
});
