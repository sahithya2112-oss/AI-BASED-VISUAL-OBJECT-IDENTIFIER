import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.set_page_config(
    page_title="AI Object Identifier",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 AI Based Visual Object Identifier")

st.write(
    "Upload an image or capture an image using your webcam. "
    "The YOLO AI model will identify objects automatically."
)

st.divider()


@st.cache_resource
def load_model():
    return YOLO("yolo11n.pt")


try:
    model = load_model()
    st.success("✅ YOLO model loaded successfully")
except Exception as e:
    st.error("❌ Could not load the YOLO model.")
    st.error(str(e))
    st.stop()


option = st.radio(
    "Choose input method:",
    ["🖼️ Upload Image", "📷 Use Camera"],
    horizontal=True
)

image = None

if option == "🖼️ Upload Image":

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "webp"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

elif option == "📷 Use Camera":

    camera_image = st.camera_input(
        "Take a picture using your webcam"
    )

    if camera_image is not None:
        image = Image.open(camera_image).convert("RGB")


if image is not None:

    st.divider()

    image_array = np.array(image)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    with st.spinner("🔎 Detecting objects..."):

        results = model(
            image_array,
            conf=0.25,
            verbose=False
        )

    result = results[0]

    detected_image = result.plot()

    with col2:
        st.subheader("Detected Objects")
        st.image(
            detected_image,
            channels="BGR",
            use_container_width=True
        )

    st.divider()

    st.subheader("📊 Detection Results")

    boxes = result.boxes

    if boxes is None or len(boxes) == 0:

        st.warning("⚠️ No objects detected.")

    else:

        object_count = len(boxes)

        st.metric(
            "Total Objects Detected",
            object_count
        )

        detections = []

        for box in boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = model.names[class_id]

            detections.append({
                "Object": class_name,
                "Confidence": f"{confidence * 100:.2f}%"
            })

        st.dataframe(
            detections,
            use_container_width=True,
            hide_index=True
        )

        st.subheader("📦 Object Summary")

        object_counts = {}

        for detection in detections:

            object_name = detection["Object"]

            if object_name not in object_counts:
                object_counts[object_name] = 0

            object_counts[object_name] += 1

        for object_name, count in object_counts.items():

            st.write(
                f"**{object_name}** : {count}"
            )


st.divider()

st.caption(
    "AI Based Visual Object Identifier | "
    "Python + Streamlit + YOLO + OpenCV"
)
