# Model Card — Baby Action AI Infant Pose MLP

## Model

- Architecture: 2-layer normalized MLP classifier
- Input: 17 Human3.6M-compatible 2D joints, hip-centered and scale-normalized, plus 17 keypoint visibility scores (51 features)
- Labels: `supine`, `prone`, `sitting`, `standing`, `all-fours`
- Runtime pose extractor: MediaPipe Pose
- Live stability: 5-frame probability moving average
- Unknown threshold: 0.56 by default

## Training data

Official public InfActPrimitive pose-only release, downloaded from the official repository's linked Google Drive file. The release contains infant pose annotations rather than raw RGB images/videos. The data is split by sequence IDs into train and validation sets.

## Evaluation

This run sampled 24 frames per sequence:

- Train samples: 9,600
- Validation samples: 9,307
- Frame accuracy: 64.21%
- Balanced accuracy: 55.45%
- Mean confidence: 77.87%

Per-class recall:

| Class | Recall |
|---|---:|
| Supine | 77.31% |
| Prone | 62.55% |
| Sitting | 31.57% |
| Standing | 38.94% |
| All-fours | 66.88% |

The release is imbalanced toward supine and prone validation frames. Accuracy and confidence should not be interpreted as clinical performance or as general live-webcam accuracy.

## Intended use

Research prototyping of privacy-first infant posture/action recognition from a single detected pose. Use for interface experimentation and dataset/model research only.

## Not intended for

- medical diagnosis
- developmental assessment
- emergency alerts
- unattended baby monitoring
- safety-critical decisions
- commercial deployment under the public dataset's non-commercial terms

## Live-image caveat

Because the public release is pose-only, this model does not learn RGB appearance, camera-specific lighting, clothing, or infant-camera domain shift. Live-image performance is limited by MediaPipe Pose detection and by the mismatch between the public pose-estimation pipeline and the user's camera. The next accuracy improvement requires consented images/videos from the target camera and person-held-out validation.

## License and provenance

Review the official dataset terms before any use. The official repository describes the data as non-commercial research/teaching/scientific-publication/personal-experimentation material and states it is not intended for diagnosis or product incorporation.
