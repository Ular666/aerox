import csv
import hashlib
import io
import math
from pathlib import Path
import time
from zipfile import ZipFile

import cv2

MAX_FRAMES = 1800
MAX_EVIDENCE = 24


def csv_bytes(rows, columns):
    out = io.StringIO(newline="")
    writer = csv.DictWriter(out, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue().encode("utf-8")


def load_detector(path):
    from ultralytics import YOLO
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("RESCUE checkpoint is missing. See README.md.")
    model = YOLO(str(path))
    if model.task != "detect":
        raise ValueError("The RESCUE checkpoint must be an object detector.")
    person_ids = [int(i) for i, name in model.names.items()
                  if name.strip().lower() == "person"]
    if not person_ids:
        raise ValueError("The checkpoint does not expose a person class.")
    return model, person_ids


def process_video(path, weights, confidence, stride, device="auto", progress=None):
    import torch
    prefer_gpu = device == "auto" and torch.cuda.is_available()
    if prefer_gpu:
        try:
            return _process(path, weights, confidence, stride, 0, progress)
        except (RuntimeError, NotImplementedError) as error:
            # Restart the entire run to keep all results on a single device.
            result = _process(path, weights, confidence, stride, "cpu", progress)
            result["fallback"] = f"GPU failed ({type(error).__name__}); CPU retry completed."
            return result
    return _process(path, weights, confidence, stride, "cpu", progress)


def _process(path, weights, confidence, stride, device, progress):
    started = time.perf_counter()
    model, person_ids = load_detector(weights)
    frames, detections, evidence = [], [], {}
    reader = cv2.VideoCapture(str(path))
    try:
        fps = reader.get(cv2.CAP_PROP_FPS)
        if not reader.isOpened() or not math.isfinite(fps) or fps <= 0:
            raise ValueError("Could not open this MP4 or determine its frame rate.")
        total = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
        if total and math.ceil(total / stride) > MAX_FRAMES:
            raise ValueError(f"Choose a larger frame step or trim the clip to at most {MAX_FRAMES} analyzed frames.")
        frame_index = 0
        previous_ms = -1
        first_empty = None
        while reader.grab():
            index = frame_index
            frame_index += 1
            if index % stride:
                continue
            if len(frames) >= MAX_FRAMES:
                raise ValueError("Analyzed frame limit reached. Use a larger step or trim the clip.")
            ok, frame = reader.retrieve()
            if not ok:
                raise ValueError(f"Cannot decode frame {index}.")
            measured_ms = reader.get(cv2.CAP_PROP_POS_MSEC)
            if math.isfinite(measured_ms) and measured_ms > previous_ms:
                ms = measured_ms
            else:
                ms = max(1000 * index / fps, previous_ms)
            previous_ms = ms
            result = model.predict(frame, classes=person_ids, conf=confidence,
                                   imgsz=640, device=device, verbose=False)[0]
            boxes = result.boxes
            scores = boxes.conf.cpu().tolist()
            frames.append(dict(frame=index, seconds=round(ms / 1000, 3), people=len(scores)))
            for score, xyxy in zip(scores, boxes.xyxy.cpu().tolist()):
                detections.append(dict(frame=index, seconds=frames[-1]["seconds"],
                                       confidence=round(score, 5),
                                       x1=round(xyxy[0], 1), y1=round(xyxy[1], 1),
                                       x2=round(xyxy[2], 1), y2=round(xyxy[3], 1)))
            if scores and len(evidence) < MAX_EVIDENCE or first_empty is None:
                annotated = result.plot()
                if first_empty is None:
                    first_empty = (index, _jpeg(annotated))
                if scores and len(evidence) < MAX_EVIDENCE:
                    evidence[index] = _jpeg(annotated)
            if progress:
                progress(len(frames), math.ceil(total / stride) if total else None)
        if not frames:
            raise ValueError("This MP4 has no decodable frames.")
        if not evidence:
            evidence[first_empty[0]] = first_empty[1]
    finally:
        reader.release()
    with Path(weights).open("rb") as checkpoint:
        checkpoint_hash = hashlib.file_digest(checkpoint, "sha256").hexdigest()
    return dict(frames=frames, detections=detections, evidence=evidence, fps=fps,
                stride=stride, confidence=confidence, device="GPU" if device == 0 else "CPU",
                elapsed=time.perf_counter() - started, source=Path(path).name,
                checkpoint=checkpoint_hash)


def _jpeg(image):
    height, width = image.shape[:2]
    if width > 1280:
        image = cv2.resize(image, (1280, round(height * 1280 / width)), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 87])
    if not ok:
        raise ValueError("Could not encode an evidence frame.")
    return encoded.tobytes()


def report(result, source):
    out = io.BytesIO()
    overview = (
        f"AEROX GUARD - Recorded video review\nInput: {source}\n"
        f"RESCUE checkpoint SHA-256: {result['checkpoint']}\n"
        f"Compute: {result['device']} | Elapsed seconds: {result['elapsed']:.2f}\n"
        f"Confidence cutoff: {result['confidence']:.2f} | Frame step: {result['stride']}\n"
        f"Analyzed frames: {len(result['frames'])}\n"
        f"Peak detections per analyzed frame: {max(r['people'] for r in result['frames'])}\n"
        "Timestamps are video-relative. Counts can repeat the same person across frames.\n"
        "Experimental person detection. Review the footage and predictions manually.\n"
    )
    with ZipFile(out, "w") as archive:
        archive.writestr("summary.txt", overview)
        archive.writestr("frame_counts.csv", csv_bytes(result["frames"], ["frame", "seconds", "people"]))
        archive.writestr("detections.csv", csv_bytes(result["detections"],
                         ["frame", "seconds", "confidence", "x1", "y1", "x2", "y2"]))
        for index, data in result["evidence"].items():
            archive.writestr(f"evidence/frame_{index:06d}.jpg", data)
    return out.getvalue()

