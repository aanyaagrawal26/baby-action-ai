from __future__ import annotations

import argparse
import json
import pickle
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

LABELS = ["supine", "prone", "sitting", "standing", "all-fours"]
SEED = 42


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def normalize_keypoints(points: np.ndarray, scores: np.ndarray | None = None) -> np.ndarray:
    """Normalize 17 Human3.6M points and append visibility features."""
    points = np.asarray(points, dtype=np.float32)
    valid = np.isfinite(points).all(axis=-1)
    if valid.sum() < 6:
        return np.zeros(51, dtype=np.float32)
    p = points.copy()
    p[~valid] = np.nan
    # Public data uses Human3.6M order: pelvis=0, thorax=8.
    hip = p[0]
    if not np.isfinite(hip).all():
        hip = np.nanmean(p, axis=0)
    p = p - hip
    # Scale by shoulder-to-hip distance, then use a robust fallback for side views.
    shoulder = p[8]
    torso = float(np.linalg.norm(shoulder - hip)) if np.isfinite(shoulder).all() else 0.0
    spread = np.nanpercentile(points[:, 0], 90) - np.nanpercentile(points[:, 0], 10)
    height = np.nanpercentile(points[:, 1], 90) - np.nanpercentile(points[:, 1], 10)
    scale = max(torso, spread, height, 1.0)
    p = np.nan_to_num(p / scale, nan=0.0, posinf=0.0, neginf=0.0)
    if scores is None:
        scores = np.ones(17, dtype=np.float32)
    scores = np.nan_to_num(np.asarray(scores, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return np.concatenate([p.reshape(-1), scores]).astype(np.float32)


def iter_samples(path: Path, max_frames_per_sequence: int) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as f:
        payload = pickle.load(f)
    anns = payload["annotations"]
    split_map = {k: set(v) for k, v in payload["split"].items()}
    samples: dict[str, list[np.ndarray]] = {"train": [], "val": []}
    labels: dict[str, list[int]] = {"train": [], "val": []}
    for ann in anns:
        split = "train" if ann["frame_dir"] in split_map["train"] else "val"
        kp = np.asarray(ann["keypoint"], dtype=np.float32)
        # Shape is [person, time, joints, xy]. Use the first/only tracked infant.
        kp = kp[0]
        score = np.asarray(ann.get("keypoint_score", np.ones((1, len(kp), 17), dtype=np.float32)), dtype=np.float32)[0]
        n = len(kp)
        if n == 0:
            continue
        take = np.linspace(0, n - 1, min(n, max_frames_per_sequence), dtype=int)
        for idx in take:
            samples[split].append(normalize_keypoints(kp[idx], score[idx]))
            labels[split].append(int(ann["label"]))
    x_train = np.stack(samples["train"])
    y_train = np.asarray(labels["train"], dtype=np.int64)
    x_val = np.stack(samples["val"])
    y_val = np.asarray(labels["val"], dtype=np.int64)
    return (x_train, y_train), (x_val, y_val)


class PoseMLP(nn.Module):
    def __init__(self, n_classes: int = 5) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(51, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.25),
            nn.Linear(128, 96), nn.LayerNorm(96), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(96, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def augment(x: torch.Tensor) -> torch.Tensor:
    x = x.clone()
    # Feature layout is x,y pairs. Small noise models live-pose jitter.
    noise = torch.randn_like(x) * 0.012
    x = x + noise
    # Horizontal mirror in normalized coordinates with modest probability.
    mask = torch.rand(x.shape[0], device=x.device) < 0.5
    x[mask, 0:34:2] *= -1
    return x


def evaluate(model: nn.Module, x: np.ndarray, y: np.ndarray, device: torch.device) -> dict:
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(x).to(device))
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    pred = probs.argmax(1)
    report = classification_report(y, pred, labels=list(range(len(LABELS))), target_names=LABELS, output_dict=True, zero_division=0)
    return {
        "accuracy": float((pred == y).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "report": report,
        "confusion_matrix": confusion_matrix(y, pred, labels=list(range(len(LABELS)))).tolist(),
        "mean_confidence": float(probs.max(1).mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/infactprimitive/InfActPrimitive/2d/InfAct_plus.pkl")
    ap.add_argument("--out", default="models")
    ap.add_argument("--epochs", type=int, default=45)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--max-frames-per-sequence", type=int, default=24)
    args = ap.parse_args()
    seed_everything()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (x_train, y_train), (x_val, y_val) = iter_samples(Path(args.data), args.max_frames_per_sequence)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PoseMLP().to(device)
    counts = np.bincount(y_train, minlength=len(LABELS)).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0); weights = weights / weights.mean()
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device), label_smoothing=0.04)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=5)
    ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    best = -1.0; best_state = None; history = []
    for epoch in range(1, args.epochs + 1):
        model.train(); losses=[]
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(augment(xb)), yb)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            losses.append(float(loss.item()))
        metrics = evaluate(model, x_val, y_val, device)
        scheduler.step(metrics["balanced_accuracy"])
        row = {"epoch": epoch, "loss": float(np.mean(losses)), "val_accuracy": metrics["accuracy"], "val_balanced_accuracy": metrics["balanced_accuracy"]}
        history.append(row)
        print(json.dumps(row), flush=True)
        if metrics["balanced_accuracy"] > best:
            best = metrics["balanced_accuracy"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save({"state_dict": best_state, "labels": LABELS, "feature_dim": 51, "normalization": "human3.6m_hip_centered_scale_xy_plus_visibility", "source": "InfActPrimitive public pose release"}, out / "infant_pose_mlp.pt")
            (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    metrics = json.loads((out / "metrics.json").read_text())
    (out / "training_history.json").write_text(json.dumps(history, indent=2))
    (out / "labels.json").write_text(json.dumps(LABELS, indent=2))
    print("FINAL", json.dumps({"device": str(device), "train_samples": len(y_train), "val_samples": len(y_val), "best_balanced_accuracy": best, "metrics": metrics}))


if __name__ == "__main__":
    main()
