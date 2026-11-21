import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import cv2
import numpy as np

import agri
import guard

ROOT = Path(__file__).resolve().parent
INPUTS = ROOT / "demo" / "inputs"
OUTPUTS = ROOT / "demo" / "outputs"
TRAINING = ROOT / "evidence" / "training"


def create_inputs(mvp):
    images = mvp / "datasets/aerox_rescue/images/val"
    labels = mvp / "datasets/aerox_rescue/labels/val"
    candidates = []
    for image in sorted(images.glob("*.jpg")):
        count = len((labels / (image.stem + ".txt")).read_text().splitlines())
        if 18 <= count <= 60:
            candidates.append(image)
        if len(candidates) == 15:
            break
    selected = candidates[::5]
    if len(selected) != 3:
        raise RuntimeError("Expected three source images from the local development-validation split.")
    size = (960, 540)
    writer = cv2.VideoWriter(str(INPUTS / "visdrone_test_clip.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), 10, size)
    if not writer.isOpened():
        raise RuntimeError("Could not open MP4 sample writer.")
    try:
        for image in selected:
            frame = cv2.imread(str(image))
            if frame is None:
                raise RuntimeError(f"Could not read {image.name}")
            h, w = frame.shape[:2]
            scale = min(size[0] / w, size[1] / h)
            frame = cv2.resize(frame, (round(w * scale), round(h * scale)))
            pad_x, pad_y = size[0] - frame.shape[1], size[1] - frame.shape[0]
            frame = cv2.copyMakeBorder(frame, pad_y // 2, pad_y - pad_y // 2,
                                       pad_x // 2, pad_x - pad_x // 2, cv2.BORDER_CONSTANT)
            for _ in range(20):
                writer.write(frame)
    finally:
        writer.release()
    farm = np.zeros((480, 720, 3), dtype=np.uint8)
    farm[:] = (88, 121, 153)
    for row, counts in enumerate(((11, 13, 14, 13), (12, 14, 14, 12),
                                  (10, 8, 6, 10), (10, 6, 8, 9))):
        for col, count in enumerate(counts):
            for stripe in range(count):
                y, x = row * 120, col * 180 + stripe * 12
                farm[y:y + 120, x:x + 9] = (33, 130, 47)
    if not cv2.imwrite(str(INPUTS / "vegetation_test_pattern.png"), farm):
        raise RuntimeError("Failed to write AGRI fixture.")
    (INPUTS / "PROVENANCE.md").write_text(
        "# Demo input provenance\n\n"
        "GUARD uses three real aerial stills from the team's development-validation subset of VisDrone2019-DET. "
        "Each still is held for two seconds to exercise the MP4 pipeline. "
        "This clip does not show a flight sequence or establish field performance. Source image names: "
        + ", ".join(p.name for p in selected)
        + ". Dataset credit: AISKYEYE, Tianjin University. [Upstream dataset](https://github.com/VisDrone/VisDrone-Dataset). "
          "Confirm permission for public use before posting samples or derivative clips.\n\n"
          "AGRI uses a programmatically generated RGB field pattern with exactly 53.125% green pixels by construction. "
          "It exercises the code path and cannot estimate field accuracy.\n", encoding="utf-8")


def training_evidence(mvp):
    runs = {
        "RESCUE": ("detect/runs/aerox/rescue_v1-4", "metrics/mAP50(B)", "models/rescue.pt"),
        "SOLAR": ("classify/runs/solar/solar_v1", "metrics/accuracy_top1", "models/solar_best.pt"),
        "FIRE": ("classify/runs/fire/fire_v1", "metrics/accuracy_top1", "models/fire_best.pt"),
    }
    metrics = {}
    for name, (run, metric, model_file) in runs.items():
        source = mvp / "runs" / run
        target = TRAINING / name
        target.mkdir(parents=True, exist_ok=True)
        for f in ["results.csv", "args.yaml", "results.png", "confusion_matrix.png"]:
            shutil.copyfile(source / f, target / f)
        checkpoint = source / "weights" / "best.pt"
        dst = ROOT / model_file
        dst.parent.mkdir(exist_ok=True)
        shutil.copyfile(checkpoint if name == "RESCUE" else mvp / model_file, dst)
        with (target / "results.csv").open(newline="") as handle:
            rows = [{k.strip(): v for k, v in row.items()} for row in csv.DictReader(handle)]
        peak = max(rows, key=lambda row: float(row[metric]))
        metrics[name] = dict(metric=metric, peak_logged_value=float(peak[metric]), epoch=int(peak["epoch"]),
                             final_logged_value=float(rows[-1][metric]), epochs_logged=len(rows),
                             matches_run_best=(hashlib.sha256(dst.read_bytes()).digest()
                                               == hashlib.sha256(checkpoint.read_bytes()).digest()),
                             checkpoint_sha256=hashlib.sha256(dst.read_bytes()).hexdigest())
    (TRAINING / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-from", type=Path, help="Optional path to the original MVP")
    args = parser.parse_args()
    for directory in (INPUTS, OUTPUTS, TRAINING):
        directory.mkdir(parents=True, exist_ok=True)
    if args.prepare_from:
        create_inputs(args.prepare_from)
        print(json.dumps(training_evidence(args.prepare_from), indent=2))
    clip = INPUTS / "visdrone_test_clip.mp4"
    results = {}
    for device in ("auto", "cpu"):
        actual = guard.process_video(clip, ROOT / "models/rescue.pt", 0.25, 5, device)
        assert len(actual["frames"]) == 12
        assert actual["frames"][-1]["seconds"] == 5.5
        assert len(actual["detections"]) == sum(row["people"] for row in actual["frames"])
        assert actual["detections"] and actual["evidence"]
        filename = f"guard_{device}_report.zip"
        (OUTPUTS / filename).write_bytes(guard.report(actual, clip.name))
        results[device] = dict(compute=actual["device"], frames=len(actual["frames"]),
                               peak=max(r["people"] for r in actual["frames"]),
                               detections=len(actual["detections"]), evidence_frames=len(actual["evidence"]),
                               elapsed_seconds=round(actual["elapsed"], 2),
                               fallback=actual.get("fallback"))
    fixture = INPUTS / "vegetation_test_pattern.png"
    result = agri.analyze(fixture.read_bytes(), 4)
    assert abs(result["coverage"] - 53.125) < 0.00001
    assert sum(r["total_pixels"] for r in result["regions"]) == 480 * 720
    (OUTPUTS / "agri_report.zip").write_bytes(agri.report(result, fixture.name))
    results["agri"] = {"coverage_percent": result["coverage"], "regions": len(result["regions"]),
                       "review_cells": result["inspect"]}
    (OUTPUTS / "run.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

