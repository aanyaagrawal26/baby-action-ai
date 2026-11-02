from pathlib import Path
p = Path('client/src/pages/Home.tsx')
s = p.read_text()
s = s.replace('const MAX_FILE_BYTES = 15 * 1024 * 1024;', 'const MAX_FILE_BYTES = 20 * 1024 * 1024;')
s = s.replace('if (Object.values(MIME_BY_EXTENSION).includes(type)) return type;', 'if (type.startsWith("image/")) return type;')
s = s.replace('if (!type || type === "application/octet-stream") return MIME_BY_EXTENSION[extension] ?? null;', 'if (!type || type === "application/octet-stream") return MIME_BY_EXTENSION[extension] ?? (extension ? `image/${extension}` : null);')
needle = '''function responseMessage(payload: unknown, fallback: string) {'''
insert = '''async function normalizePhoto(file: File): Promise<string> {
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

'''
if needle not in s: raise SystemExit('insert anchor missing')
s = s.replace(needle, insert + needle, 1)
s = s.replace('Choose a JPEG, PNG or WebP photo. Other formats and videos are not supported yet.', 'Choose a common baby photo format such as JPG, PNG, WebP or a phone image. Videos belong in Event review.')
s = s.replace('This photo is larger than 15 MB. Choose a smaller image and try again.', 'This photo is larger than 20 MB. Choose a smaller image and try again.')
old = '''      setAnalysisStep("Preparing your photo…");
      try {
        const dataUrl = await new Promise<string>((resolve, reject) => {
          const reader = new FileReader();
          reader.onload = () => resolve(String(reader.result ?? ""));
          reader.onerror = () => reject(new Error("We couldn’t read this photo. Choose another file and try again."));
          reader.readAsDataURL(requestedPhoto.file);
        });
        const data = dataUrl.split(",", 2)[1] ?? "";
        if (!data) throw new Error("We couldn’t read this photo. Choose another file and try again.");'''
new = '''      setAnalysisStep("Preparing a clear copy of your photo…");
      try {
        const data = await normalizePhoto(requestedPhoto.file);'''
if old not in s: raise SystemExit('analysis block missing')
s = s.replace(old, new, 1)
s = s.replace('filename: requestedPhoto.file.name, contentType: requestedPhoto.mime, data', 'filename: `${requestedPhoto.file.name}.normalized.jpg`, contentType: "image/jpeg", data', 1)
s = s.replace('aria-label="Choose or drop a JPEG, PNG or WebP photo, up to 15 megabytes"', 'aria-label="Choose or drop a baby photo in any common image format, up to 20 megabytes"')
s = s.replace('accept=".jpg,.jpeg,.png,.webp,image/jpeg,image/png,image/webp"', 'accept="image/*,.jpg,.jpeg,.png,.webp,.gif,.bmp,.heic,.heif"')
s = s.replace('JPEG, PNG or WebP <i /> Up to 15 MB', 'JPG, PNG, WebP or phone image <i /> Up to 20 MB')
s = s.replace('Review a short clip for repeated posture changes. The vision model labels sampled frames; a clear rule checks the sequence before the demo shows an in-app alert preview.', 'Use any clear baby photo for a gentle posture read. For meaningful activity, review a short clip: the vision model labels sampled frames, then a transparent rule checks whether the same change repeats before showing an alert preview.')
s = s.replace('Clear shoulders, hips and supporting limbs help the model; full-body framing helps when it’s available.', 'Portrait, landscape, close-up or full-body: choose the clearest baby photo you have. We resize a browser-readable image for the model and explain uncertainty instead of forcing a guess.')
p.write_text(s)
