
import streamlit as st
from ultralytics import YOLO
from PIL import Image
from pathlib import Path
from agri_ai import analyze_agriculture
import torch
import io

st.set_page_config(
    page_title="AEROX AI CORE",
    page_icon="✈️",
    layout="wide"
)

st.markdown("""
<style>
.block-container {
    max-width: 1250px;
    padding-top: 2rem;
}
h1, h2, h3 {
    letter-spacing: -0.5px;
}
div[data-testid="stMetric"] {
    background: #161d2b;
    padding: 16px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

st.title("AEROX AI CORE")
st.subheader("Intelligence for Every Mission.")
st.caption("One AI Platform. Multiple Drone Applications.")

st.divider()

MODELS = [
    {
        "id": "rescue",
        "title": "01 — RESCUE AI",
        "description": "Aerial Person Detection",
        "path": "runs/detect/runs/aerox/rescue_v1-4/weights/best.pt",
        "purpose": "Detect people in aerial imagery.",
        "task": "detect"
    },
    {
        "id": "solar",
        "title": "02 — SOLAR AI",
        "description": "Solar Panel Condition Classification",
        "path": "models/solar_best.pt",
        "purpose": "Classify solar panel images into six condition categories.",
        "task": "classify"
    },
{
    "id": "fire",
    "title": "03 — FIRE AI",
    "description": "Forest Fire & Smoke Classification",
    "path": "models/fire_best.pt",
    "purpose": "Classify images as fire, smoke, or non-fire.",
    "task": "classify"
},
    {
    "id": "agri",
    "title": "04 — AGRI AI",
    "description": "Agricultural Vegetation Analysis",
    "path": "models/agri_best.pt",
    "purpose": "Estimate visible green vegetation coverage.",
    "task": "prototype"
}
]


@st.cache_resource
def load_model(path):
    return YOLO(path)


def render_module(module):
    module_id = module["id"]
    task = module["task"]

    st.header(module["title"])
    st.write(module["description"])
    st.caption(module["purpose"])

    if module_id == "agri":
        st.info("Vegetation Analysis Prototype — No trained crop model")

        uploaded = st.file_uploader(
            "Upload agricultural aerial image",
            type=["jpg", "jpeg", "png"],
            key="agri_upload"
        )

        field_area = st.number_input(
            "Field Area (hectares)",
            min_value=0.0,
            value=1.0,
            step=0.1,
            help="Enter the actual field area, not an AI estimate."
        )

        if uploaded is not None:
            image = Image.open(uploaded).convert("RGB")

            if st.button("Analyze Field", type="primary"):
                result = analyze_agriculture(
                    image,
                    field_area_ha=field_area
                )

                left, right = st.columns(2)

                with left:
                    st.subheader("Original Field")
                    st.image(image, use_container_width=True)

                with right:
                    st.subheader("Vegetation Map")
                    st.image(
                        result["image"],
                        use_container_width=True
                    )

                st.metric(
                    "Green Vegetation Coverage",
                    f'{result["coverage_percent"]:.1f}%'
                )

                st.metric(
                    "Estimated Vegetation Area",
                    f'{result["vegetation_area_ha"]:.2f} ha'
                )

                st.caption(
                    "Experimental color-based analysis. "
                    "Requires a field-only image and "
                    "independent validation. "
                    "Not a crop yield prediction."
                )

        return


    model_path = Path(module["path"])

    if not model_path.exists():
        st.info(
            "Module in development. "
            "A trained model will be integrated here."
        )
        return

    st.success("AI Model Available")

    uploaded_file = st.file_uploader(
        "Upload image",
        type=["jpg", "jpeg", "png"],
        key=f"upload_{module_id}"
    )

    confidence = None

    if task == "detect":
        confidence = st.slider(
            "Confidence Threshold",
            min_value=0.05,
            max_value=0.95,
            value=0.25,
            step=0.05,
            key=f"confidence_{module_id}"
        )

    if uploaded_file is None:
        st.session_state.pop(f"result_{module_id}", None)
        return

    image_bytes = uploaded_file.getvalue()

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        st.error("Unable to read this image.")
        return

    left, right = st.columns(2)

    with left:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    image_key = (image_bytes, confidence)

    saved = st.session_state.get(f"result_{module_id}")

    if saved is not None and saved["key"] != image_key:
        st.session_state.pop(f"result_{module_id}", None)
        saved = None

    if st.button(
        "Run AI Analysis",
        key=f"run_{module_id}",
        type="primary"
    ):
        try:
            model = load_model(str(model_path))
            device = 0 if torch.cuda.is_available() else "cpu"

            with st.spinner("AI is analyzing the image..."):
                kwargs = {
                    "source": image,
                    "device": device,
                    "verbose": False
                }

                if task == "detect":
                    kwargs["conf"] = confidence

                results = model.predict(**kwargs)
                result = results[0]

            if task == "classify":
                if result.probs is None:
                    st.error("Classification output is unavailable.")
                    return

                probabilities = result.probs.data.cpu().tolist()

                saved = {
                    "key": image_key,
                    "task": "classify",
                    "probabilities": probabilities,
                    "names": dict(model.names)
                }

            else:
                if result.boxes is None:
                    st.error("Detection output is unavailable.")
                    return

                annotated = Image.fromarray(
                    result.plot()[:, :, ::-1]
                )

                detections = []

                for box in result.boxes:
                    class_id = int(box.cls[0])
                    score = float(box.conf[0])

                    detections.append({
                        "name": model.names[class_id],
                        "confidence": score
                    })

                buffer = io.BytesIO()
                annotated.save(buffer, format="PNG")

                saved = {
                    "key": image_key,
                    "task": "detect",
                    "image": buffer.getvalue(),
                    "detections": detections
                }

            st.session_state[f"result_{module_id}"] = saved

        except Exception as error:
            st.error(f"AI analysis failed: {error}")
            return

    saved = st.session_state.get(f"result_{module_id}")

    if saved is None:
        return

    with right:
        st.subheader("AI Analysis Results")

        if saved["task"] == "classify":
            probabilities = saved["probabilities"]
            names = saved["names"]

            ranked = sorted(
                enumerate(probabilities),
                key=lambda item: item[1],
                reverse=True
            )

            top_id, top_score = ranked[0]

            st.metric(
                "Predicted Condition",
                names[top_id]
            )

            st.metric(
                "Model Confidence",
                f"{top_score:.1%}"
            )

            st.subheader("Class Probabilities")

            for class_id, score in ranked:
                st.write(
                    f"**{names[class_id]}** — {score:.1%}"
                )
                st.progress(
                    min(max(float(score), 0.0), 1.0)
                )

            st.caption(
                "Experimental image classification. "
                "Predictions are not verified incpection findings."
            )

        else:
            st.image(
                saved["image"],
                use_container_width=True
            )

            detections = saved["detections"]

            st.metric(
                "Detected Objects",
                len(detections)
            )

            for index, detection in enumerate(detections):
                st.write(
                    f"{index + 1}. "
                    f"{detection['name']}: "
                    f"{detection['confidence']:.1%}"
                )

            st.download_button(
                "Download Detection Result",
                data=saved["image"],
                file_name=f"aerox_{module_id}.png",
                mime="image/png",
                key=f"download_{module_id}"
            )


available = sum(
    Path(module["path"]).exists()
    for module in MODELS
)

c1, c2, c3 = st.columns(3)

c1.metric("AI Modules", len(MODELS))
c2.metric("Models Available", available)
c3.metric(
    "Compute",
    "NVIDIA GPU" if torch.cuda.is_available() else "CPU"
)

st.divider()

for module in MODELS:
    render_module(module)
    st.divider()

st.caption(
    "AEROX AI CORE v0.3 | "
    "Experimental AI platform. "
    "Modules require independent validation "
    "before operational deployment."
)
