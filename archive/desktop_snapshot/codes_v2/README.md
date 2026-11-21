# AEROX AI CORE 2.0

One Platform. Multiple Intelligent Missions.

IT FEST 2026 StartUp Battle qualification prototype. GUARD runs experimental person detection on recorded MP4, AGRI compares visible green pixel coverage across an RGB image, and FLIGHT steps through virtual waypoints. The original still-image MVP remains at `D:\Ular\Documents\AEROX-AI`.

## Run on the development PC

Extract this folder, open a PowerShell window in it, and run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\start.ps1
```

Open http://127.0.0.1:8501. The launcher reuses the existing project's Python environment if there is no local virtual environment. It uses the bundled RESCUE checkpoint. To point to another trusted local RESCUE checkpoint, set `$env:AEROX_RESCUE_WEIGHTS` before starting.

For a clean Python 3.11 installation, make a local virtual environment and install the requirements. Install a compatible PyTorch build for the machine first if GPU acceleration is needed. The verified development PC used PyTorch 2.10.0+cu128 and torchvision 0.25.0+cu128, RTX 5060 Laptop GPU, and the versions pinned in requirements.txt.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
powershell -NoProfile -ExecutionPolicy Bypass -File .\start.ps1
```

## Reproducible demo

1. **GUARD** · Enable the bundled input. Keep confidence at 0.25, analyze every fifth frame, and select Auto. Click **Analyze video**. Review the frame picker, timestamp, confidence table and report. The 6 s input uses three VisDrone aerial stills assembled into a test MP4. Use **Upload recorded aerial MP4** for authorized original footage.
2. **AGRI** · Enable the bundled RGB pattern, select a 4×4 grid and click **Analyze field**. Review source, overlay, mask and regional coverage. The test pattern has exactly 53.125% green pixels. Select an actual field photograph for a field-data demo when available.
3. **FLIGHT** · Choose Scheduled patrol or Agricultural survey, set local start and interval, review the checklist, then move the waypoint control. Battery, GPS, link and controller values are simulated. Waypoints use local coordinates.

`demo/outputs/` contains reproducible analysis archives. `evidence/training/` contains original run logs, confusion matrices, curves and checkpoint hashes. `evidence/TECHNICAL.md` explains how to interpret the results. See `pitch/VIDEO.md` for the recording guide and `sanzhar/` for the English/Russian handoff.

To regenerate the bundled demo results and run the tests using the original environment:

```powershell
$py = 'D:\Ular\Documents\AEROX-AI\.venv\Scripts\python.exe'
& $py .\run_demo.py
& $py .\check_sprint.py
```

## Scope

- GUARD handles up to 1,800 analyzed frames per run. Raise the sampling step or trim long clips. Evidence images are selected samples. Frame counts can repeat people across time.
- AGRI uses a color rule with no verified field accuracy. Lands with different crop colors, soil or lighting can produce misleading comparisons. Yield Prediction: Not Yet Validated.
- FLIGHT is a simulator. Real telemetry, flight commands, controller firmware and hardware integration are pending.
- Legacy SOLAR and FIRE weights appear in the evidence package for checkpoint traceability. Their interfaces remain in the original MVP.
- The bundle is for team review. Check source media rights and model/software redistribution terms before sharing it publicly.

