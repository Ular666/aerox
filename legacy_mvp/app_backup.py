
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import torch
import io

st.set_page_config(
    page_title="AEROX RESCUE AI",
    page_icon="✈️",
    layout="wide"
)

st.title("AEROX AI CORE")
st.caption("One AI Platform. Multiple Drone Applications.")
st.subheader("RESCUE AI — Aerial Person Detection")

MODEL_PATH = (
    "runs/detect/runs/aerox/"
    "rescue_v1-4/weights/best.pt"
)

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

st.sidebar.header("Detection Settings")

confidence = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.05,
    max_value=0.95,
    value=0.25,
    step=0.05
)

st.sidebar.write(
    "GPU:",
    torch.cuda.get_device_name(0)
    if torch.cuda.is_available()
    else "CPU"
)

uploaded_file = st.file_uploader(
    "Upload an aerial image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    if st.button("Run AI Detection", type="primary"):
        device = 0 if torch.cuda.is_available() else "cpu"

        with st.spinner("Analyzing image..."):
            results = model.predict(
                source=image,
                device=device,
                conf=confidence,
                verbose=False
            )

        result = results[0]

        annotated = Image.fromarray(
            result.plot()[:, :, ::-1]
        )

        with col2:
            st.subheader("AI Detection Results")
            st.image(
                annotated,
                use_container_width=True
            )

            count = len(result.boxes)

            st.metric(
                "Detected Persons",
                count
            )

            for i, box in enumerate(result.boxes):
                confidence_score = float(box.conf[0])

                st.write(
                    f"Person {i + 1}: "
                    f"{confidence_score:.1%}"
                )

            buffer = io.BytesIO()
            annotated.save(buffer, format="PNG")

            st.download_button(
                "Download Result",
                data=buffer.getvalue(),
                file_name="aerox_rescue_detection.png",
                mime="image/png"
            )

st.divider()

st.caption(
    "AEROX RESCUE AI v1 | "
    "Experimental aerial person detection. "
    "Not validated for operational search and rescue."
)
