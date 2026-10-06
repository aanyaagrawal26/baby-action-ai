# Startup Guide — GestureVerse-AI Demo
## Two-terminal quick start for the presentation

---

## Prerequisites (do once before the demo)

### Python environment and dependencies

```bash
# From the project root: /Users/hp/Desktop/baby-action-ai

# 1. Create the virtual environment (already done — skip if .venv exists)
python3 -m venv .venv

# 2. Install Python dependencies
.venv/bin/pip install --upgrade pip
.venv/bin/pip install "mediapipe>=0.10.30,<1.0" torch "fastapi>=0.115" "uvicorn[standard]>=0.30" "python-multipart>=0.0.9" "scikit-learn>=1.5" "numpy>=2.0" Pillow

# 3. Download the MediaPipe pose model bundle (already done — skip if models/pose_landmarker_lite.task exists)
curl -L "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task" \
     -o models/pose_landmarker_lite.task
```

### Node dependencies (do once)

```bash
pnpm install
```

### LLM API key (optional but recommended for live GenAI demo)

Create a `.env` file in the project root:

```
BUILT_IN_FORGE_API_KEY=your_api_key_here
```

If this is missing, the `/api/explain` route returns a clearly-labelled rule-based fallback.
The ML inference continues to work regardless.

---

## Demo day startup (two terminals)

### Terminal 1 — Python inference service

```bash
cd /Users/hp/Desktop/baby-action-ai
source .venv/bin/activate
uvicorn inference_api:app --host 127.0.0.1 --port 8000

# Verify it started correctly:
# You should see:  INFO:  Application startup complete.
# Test:  curl http://127.0.0.1:8000/health
# Expected: {"ok":true, "model":"...infant_pose_mlp.pt", ...}
```

### Terminal 2 — Node dev server

```bash
cd /Users/hp/Desktop/baby-action-ai
pnpm dev

# Opens at http://localhost:3000 (or next available port — check the output)
```

Open http://localhost:3000 in Chrome or Safari.

---

## Verified working (as of October 6, 2026)

| Check | Command | Expected result |
|---|---|---|
| Python imports | `.venv/bin/python -c "import torch, mediapipe, cv2; print('OK')"` | OK |
| MLP checkpoint | `.venv/bin/python -c "import torch; c=torch.load('models/infant_pose_mlp.pt'); print(c['labels'])"` | `['supine', 'prone', ...]` |
| Pose model | `ls models/pose_landmarker_lite.task` | File exists, ~5.5 MB |
| FastAPI health | `curl http://127.0.0.1:8000/health` | `{"ok":true, ...}` |
| TypeScript | `pnpm check` | exit 0 |
| Unit tests | `pnpm test` | 17/17 pass |
| Production build | `pnpm build` | exit 0 |
| Node /api/infer | POST to `http://localhost:3000/api/infer` | proxied to Python |
| Node /api/explain (no key) | POST to `http://localhost:3000/api/explain` | `status: llm_fallback_no_key` |

---

## What each route does

| Route | Method | What it does |
|---|---|---|
| `/api/infer` | POST | Node proxy → Python :8000/analyze → MLP inference |
| `/api/explain` | POST | Node server-side → builds prompt → calls LLM (or fallback) |
| `/health` (Python) | GET | Checks model + pose task loaded |

---

## Presentation checklist

**Night before:**
- [ ] Run `pnpm build` — confirm it exits 0
- [ ] Run `pnpm test` — confirm 17/17 pass
- [ ] Start both services and test a real image upload end-to-end
- [ ] Prepare 2–3 test images (photos with visible full-body pose)
- [ ] Confirm `.env` has `BUILT_IN_FORGE_API_KEY` if you want live LLM generation

**Morning of presentation:**
- [ ] Open Terminal 1, run `uvicorn inference_api:app --host 127.0.0.1 --port 8000`
- [ ] Confirm `/health` returns `ok: true` before opening browser
- [ ] Open Terminal 2, run `pnpm dev`
- [ ] Open http://localhost:3000 in browser
- [ ] Upload a test image and confirm inference result appears
- [ ] Click "Generate explanation" and confirm explanation appears (LLM or fallback — both are fine)
- [ ] Open Prompt Engineering Lab and click "Compare all three strategies"
- [ ] Scroll to `#model-analysis` and verify the per-class recall chart renders

**During presentation:**
- [ ] Do NOT refresh the page mid-demo (camera permission resets)
- [ ] Have a fallback screenshot ready in case of network issues
- [ ] All four team members know which section they're presenting (see REVIEW2_TEAM_WORK.md)

---

## If something goes wrong

**FastAPI won't start:**
```bash
# Check port 8000 is free
lsof -i :8000
# Check the error in terminal output
# Most common: pose_landmarker_lite.task missing → re-run the curl download command
```

**"No pose detected" on all images:**
- Use a photo where a full body (or at least torso, hips, and limbs) is clearly visible
- Avoid cropped faces or partial body shots
- Avoid photos taken from directly above

**LLM not generating:**
- This is expected without `BUILT_IN_FORGE_API_KEY`
- The fallback explanation is clearly labelled — this is demonstrable and correct behaviour
- You can show both states: "Here's what it says without the API key — fallback. And here's the actual LLM output when the key is present."

**Node server port conflict:**
- The server automatically tries ports 3000–3019
- Check the terminal output for the actual port: `Server running on http://localhost:XXXX`
