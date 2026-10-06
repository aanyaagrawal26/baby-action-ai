# Backend inference upgrade checklist

- [x] Upgrade the static project to a backend-capable web stack.
- [x] Add the trained checkpoint/runtime contract and inference dependencies.
- [x] Implement an endpoint for image and video uploads.
- [x] Return action, confidence, input metadata, and model provenance.
- [x] Replace the upload demo timeout with a real request and typed result state.
- [x] Handle unsupported files, inference failures, and low-confidence results.
- [x] Validate the end-to-end flow with representative test inputs.
- [x] Save and deliver the connected application checkpoint.

- [x] Add FastAPI and Uvicorn to the local requirements manifest.
- [x] Add thresholded unknown/low-confidence handling to the API and UI.
- [x] Run a successful end-to-end upload through `/api/infer` with a real test image or video.

- [x] Add animated upload inference progress and status messaging.
- [x] Add explicit local camera start/stop controls and consent messaging.
- [x] Capture webcam frames without recording or storing the live feed.
- [x] Send sampled live frames to the backend and show current action/confidence.
- [x] Handle camera permission denial, unavailable camera, backend errors, and cleanup.
- [x] Verify desktop/mobile UI and save the live-feed checkpoint.

- [x] Verify the live-camera and upload-progress UI at a narrow mobile breakpoint.
- [x] Save and deliver a new checkpoint after the live-feed changes.

- [x] Add the natural contactless HCI problem statement to the website narrative.
- [x] Explain the distinction between baby-action recognition and caregiver gesture commands.
- [x] Add an observatory interaction concept that maps recognized gestures/context to meaningful actions.
- [x] Document the HCI integration in the local project README.
- [x] Verify HCI copy and interaction states on desktop and mobile.
- [x] Save and deliver the HCI-integrated checkpoint.
