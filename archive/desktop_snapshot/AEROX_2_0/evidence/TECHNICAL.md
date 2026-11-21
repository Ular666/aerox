# Technical evidence

## Architecture

```
Streamlit
  GUARD  MP4 → OpenCV sampled frames → RESCUE YOLO11n → counts, boxes, reports
  AGRI   RGB → ExG and color gates → mask → grid statistics, visualizations, report
  FLIGHT plan and checklist → in-memory route progress → virtual telemetry
```

The Windows environment was inspected before changes. The original app loads the three trained checkpoints for still-image detection/classification and uses an ExG baseline for AGRI. Sprint code is in a separate folder. The original trainer scripts and dataset tree were not edited.

## Run evidence

| Module | Observed result | Interpretation |
|---|---|---|
| GUARD | Original checkpoint, 12 sampled frames, 51 peak detection boxes, 348 boxes total on both GPU and CPU, cutoff 0.25 | Recorded-clip pipeline test on three held development-validation stills. A person may appear in several analyzed frames. This is neither a headcount nor a benchmark of speed or field accuracy. |
| AGRI | 53.125% cover on the constructed RGB input, 16 cells, C3/D2/C2 flagged | Exact fixture result. Field accuracy and agronomic interpretation are To Be Verified. |
| FLIGHT | Five patrol waypoints and seven survey waypoints. Virtual route progress depends on checklist completion. | Simulation. Actual telemetry is unavailable. |

The GPU and CPU outputs include annotated JPEGs, frame counts, detection-level confidences/coordinates, timestamps and checkpoint SHA-256. The AGRI archive contains the mask, overlay, color grid, table and summary. Exact elapsed times from the run are in `demo/outputs/run.json`. These times include initialization and are not comparable benchmarks.

## Original model validation

| Module | Logged validation peak | Source |
|---|---|---|
| RESCUE | mAP50 38.458% at epoch 30 (38.5% rounded) | training/RESCUE/results.csv |
| SOLAR | Top-1 98.000% at epoch 15 | training/SOLAR/results.csv |
| FIRE | Top-1 98.889% at epoch 28 (98.9% rounded) | training/FIRE/results.csv |
| AGRI | To Be Verified | Deterministic RGB color baseline |

The three bundled checkpoints match corresponding run best.pt files by SHA-256. Values above come from saved training histories. No independent field evaluation was performed. Checkpoint selection may use a fitness measure beyond a single column. Use the original charts and matrices with their run labels.

- RESCUE fine-tuned pretrained YOLO11n, 30 epochs, 640 px input, batch 8. The local converter merges VisDrone2019-DET-train pedestrian and people labels and makes an 80/20 internal split (5,176 train / 1,295 validation). An official split, site separation, collection/approval terms and independent field performance remain To Be Verified.
- SOLAR and FIRE fine-tuned pretrained YOLO11n classification models, 30 epochs, 224 px input, batch 32. SOLAR directories label Bird-drop, Clean, Dusty, Electrical-damage, Physical-Damage and Snow-Covered. FIRE uses fire, Smoke and non fire. Original sources, licenses, duplicate screening, dataset versions and independent test performance are To Be Verified.
- AGRI applies `2G - R - B` to channels scaled to [0, 1], threshold 0.08 plus green dominance and brightness gates. Grid cells compare pixel fractions, without georeferencing or crop diagnosis.

## Operating boundaries

GUARD is experimental. Missed and false detections are possible under occlusion, altitude changes, motion blur and low light. Detection scores represent model outputs. Review the source and individual frames before operational use. Use authorized media and restrict access to evidence exports. Uploaded MP4 processing uses a temporary local file that is cleaned up after processing. Downloads may contain imagery of people and need a retention/access policy before shared deployment. Privacy/legal grounds for datasets and real deployment, release licensing and formal AI risk classification are To Be Verified.

PLANNED · Independent validation, authorized operating trials, verified F405 model and firmware, bench telemetry, then controlled flight integration. Crop-specific yield prediction requires measured yield, growing stage and held-out validation. Authorized missing-person assistance and explicit distress-signal recognition require further labeled data and human-supervised evaluation.

## Evidence still needed for public submission

Capture genuine app screenshots and an unedited live screen recording on the development PC. Replace the VisDrone excerpt with team-authorized media for a public video unless use of the excerpt is cleared. Acquire a real farmland RGB photograph and log location/date permission and imaging conditions if available. Record app/runtime versions and the exact checkpoint used.



Update — 20 November: The team captured genuine browser screenshots of all three modules. See BROWSER_DEMO.md for the observations, scope and remaining recording work.
