import hashlib
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import streamlit as st

import agri
import flight
import guard

ROOT = Path(__file__).resolve().parent
SAMPLES = ROOT / "demo" / "inputs"
WEIGHTS = Path(os.environ.get("AEROX_RESCUE_WEIGHTS", ROOT / "models" / "rescue.pt"))
GUARD_SAMPLE = "visdrone_test_clip.mp4"
AGRI_SAMPLE = "vegetation_test_pattern.png"

st.set_page_config(page_title="AEROX AI CORE 2.0", layout="wide")
st.title("AEROX AI CORE 2.0")
st.caption("One Platform. Multiple Intelligent Missions.")


def selected_input(key, label, formats, sample_name, provenance):
    sample = SAMPLES / sample_name
    use_sample = sample.is_file() and st.checkbox("Use bundled demo input", key=f"{key}-sample")
    if use_sample:
        st.caption(provenance)
        return sample.name, sample.read_bytes()
    uploaded = st.file_uploader(label, type=formats, key=key)
    if uploaded is None:
        return None, None
    return uploaded.name, uploaded.getvalue()


def guard_page():
    st.header("GUARD")
    st.caption("Recorded video review with the experimental RESCUE person detector.")
    st.info("Use authorized footage. Detections are for human review. Counts are per analyzed frame.")
    filename, data = selected_input("video", "Upload recorded aerial MP4", ["mp4"], GUARD_SAMPLE,
                                    "Demo input • Real VisDrone aerial stills assembled into an MP4. No team flight.")
    confidence = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
    stride = st.number_input("Analyze every Nth frame", 1, 60, 5)
    device = st.selectbox("Compute device", ["auto", "cpu"], format_func={"auto": "Auto (GPU if available)", "cpu": "CPU"}.get)
    if not WEIGHTS.is_file():
        st.error("RESCUE checkpoint missing. Configure it as shown in README.md.")
    if data is None:
        st.session_state.pop("guard-result", None)
        return
    signature = (filename, hashlib.sha256(data).digest(), confidence, stride, device)
    if st.button("Analyze video", type="primary", disabled=not WEIGHTS.is_file()):
        st.session_state.pop("guard-result", None)
        bar = st.progress(0, text="Loading the RESCUE checkpoint…")
        def show_progress(done, total):
            bar.progress(min(done / total, 1.0) if total else 0.0, text=f"Analyzed {done} frames")
        try:
            with TemporaryDirectory() as directory:
                path = Path(directory) / "input.mp4"
                path.write_bytes(data)
                result = guard.process_video(path, WEIGHTS, confidence, stride, device, show_progress)
            st.session_state["guard-result"] = signature, result
        except Exception as error:
            st.error(f"Video analysis failed: {error}")
        finally:
            bar.empty()
    saved = st.session_state.get("guard-result")
    if saved is None or saved[0] != signature:
        return
    result = saved[1]
    counts = pd.DataFrame(result["frames"])
    peak = int(counts.people.max())
    st.caption(f"{filename} • {result['device']} • {result['elapsed']:.2f} s • frame step {stride}")
    if result.get("fallback"):
        st.warning(result["fallback"])
    left, right = st.columns(2)
    left.metric("Analyzed frames", len(counts))
    right.metric("Peak detections per frame", peak)
    frames = {row["frame"]: row for row in result["frames"]}
    picks = list(result["evidence"])
    reviews = result.setdefault("reviews", {})
    scope = f"{signature[1].hex()}-{confidence}-{stride}-{device}-{result['elapsed']}"
    picker = "evidence-" + scope
    st.subheader("Human review")
    st.caption("Suggested frames have the highest detection counts at least two seconds apart. "
               "Review decisions apply to this analysis session. Download the report to keep them.")
    for column, index in zip(st.columns(3), result["candidates"]):
        column.button(f"Frame {index} | {frames[index]['seconds']:.3f} s | {frames[index]['people']} detections",
                      on_click=st.session_state.__setitem__, args=(picker, index), key=f"jump-{scope}-{index}")
    if picker not in st.session_state:
        st.session_state[picker] = max(picks, key=lambda i: frames[i]["people"])
    frame = st.selectbox("Evidence frame", picks, key=picker,
                         format_func=lambda i: f"{frames[i]['seconds']:.3f} s / frame {i}")
    st.image(result["evidence"][frame], caption=f"{frames[frame]['people']} people detected in this frame")
    status, note = reviews.get(frame, (guard.REVIEW_STATES[0], ""))
    status_key = f"status-{scope}-{frame}"
    note_key = f"note-{scope}-{frame}"
    if status_key not in st.session_state:
        st.session_state[status_key] = status
    if note_key not in st.session_state:
        st.session_state[note_key] = note
    status = st.radio("Review status", guard.REVIEW_STATES, horizontal=True, key=status_key)
    note = st.text_input("Operator note", key=note_key, max_chars=500)
    reviews[frame] = (status, note)
    st.dataframe(pd.DataFrame([d for d in result["detections"] if d["frame"] == frame],
                             columns=["frame", "seconds", "confidence", "x1", "y1", "x2", "y2"]), hide_index=True)
    st.line_chart(counts, x="seconds", y="people")
    st.download_button("Download GUARD mission report", guard.report(result, filename, reviews), "aerox_guard.zip",
                       "application/zip", on_click="ignore")


def agri_page():
    st.header("AGRI")
    st.caption("Excess Green analysis for a field-only RGB aerial photograph.")
    filename, data = selected_input("image", "Upload farmland RGB image", ["jpg", "jpeg", "png"], AGRI_SAMPLE,
                                    "Demo input • Synthetic RGB test pattern with known green area.")
    grid = st.slider("Grid rows and columns", 2, 8, 4)
    if data is None:
        st.session_state.pop("agri-result", None)
        return
    signature = (filename, hashlib.sha256(data).digest(), grid)
    if st.button("Analyze field", type="primary"):
        st.session_state.pop("agri-result", None)
        try:
            st.session_state["agri-result"] = signature, agri.analyze(data, grid)
        except (OSError, ValueError) as error:
            st.error(f"Image analysis failed: {error}")
    saved = st.session_state.get("agri-result")
    if saved is None or saved[0] != signature:
        return
    result = saved[1]
    st.metric("Green vegetation pixel coverage", f"{result['coverage']:.2f}%")
    st.info(agri.YIELD_STATUS)
    cols = st.columns(4)
    for col, image, title in zip(cols, ("image", "overlay", "mask", "grid_map"),
                                ("Source", "Vegetation overlay", "Binary mask", "Coverage by region")):
        col.image(result[image], caption=title)
    if result["inspect"]:
        st.write("Review these lower-cover cells against the source: " + ", ".join(result["inspect"]))
    else:
        st.write("No cells met the lower-cover review rule.")
    st.caption("The rule flags up to three cells at least 5 percentage points below overall coverage. "
               "Shadows, green non-crop objects and soil appearance can change the result. "
               "The map shows RGB cover, with no diagnosis of crop condition.")
    st.dataframe(pd.DataFrame(result["regions"]), hide_index=True)
    cells = st.multiselect("Cells for the next field inspection",
                           [r["region"] for r in result["regions"]], default=result["inspect"],
                           key=f"cells-{signature[1].hex()}-{grid}")
    if st.button("Save inspection list to FLIGHT", disabled=not cells):
        st.session_state["field-inspection"] = dict(source=filename, grid=grid,
            cells=[r for r in result["regions"] if r["region"] in cells])
        st.success("Inspection list saved. Open FLIGHT and select Agricultural survey.")
    st.download_button("Download AGRI analysis report", agri.report(result, filename), "aerox_agri.zip",
                       "application/zip", on_click="ignore")


page = st.sidebar.radio("Module", ["GUARD", "AGRI", "FLIGHT"])
{"GUARD": guard_page, "AGRI": agri_page, "FLIGHT": flight.render}[page]()

