import streamlit as st
import cv2
import numpy as np
from collections import Counter
from ultralytics import YOLO


# -----------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------

st.set_page_config(
    page_title="AI Object Detection",
    page_icon="🤖",
    layout="wide"
)


# -----------------------------------------
# LOAD YOLO MODEL
# -----------------------------------------

@st.cache_resource
def load_model():

    return YOLO("yolo11n.pt")


model = load_model()


# -----------------------------------------
# TITLE
# -----------------------------------------

st.title("🤖 AI Object Detection System")

st.write(
    "Real-time object detection using YOLO and OpenCV"
)


# -----------------------------------------
# SIDEBAR
# -----------------------------------------

st.sidebar.header("⚙️ Detection Settings")

confidence = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.5,
    step=0.05
)


# -----------------------------------------
# INPUT MODE
# -----------------------------------------

mode = st.radio(
    "Select Input Mode",
    ["📷 Webcam", "🖼️ Image", "🎥 Video"],
    horizontal=True
)


# -----------------------------------------
# IMAGE MODE
# -----------------------------------------

if mode == "🖼️ Image":

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        file_bytes = np.asarray(
            bytearray(uploaded_file.read()),
            dtype=np.uint8
        )

        image = cv2.imdecode(
            file_bytes,
            cv2.IMREAD_COLOR
        )

        results = model(
            image,
            conf=confidence,
            verbose=False
        )

        annotated_image = results[0].plot()

        # Convert BGR → RGB
        annotated_image = cv2.cvtColor(
            annotated_image,
            cv2.COLOR_BGR2RGB
        )

        st.image(
            annotated_image,
            caption="Detection Result",
            use_container_width=True
        )

        # ---------------------------------
        # STATISTICS
        # ---------------------------------

        detected_objects = []
        confidence_values = []

        for box in results[0].boxes:

            class_id = int(box.cls[0])

            object_name = model.names[class_id]

            detected_objects.append(object_name)

            confidence_values.append(
                float(box.conf[0])
            )

        object_counts = Counter(
            detected_objects
        )

        total_objects = len(
            detected_objects
        )

        if confidence_values:

            avg_confidence = (
                sum(confidence_values)
                / len(confidence_values)
            )

        else:

            avg_confidence = 0

        # ---------------------------------
        # DISPLAY METRICS
        # ---------------------------------

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Objects Detected",
            total_objects
        )

        col2.metric(
            "Average Confidence",
            f"{avg_confidence * 100:.1f}%"
        )

        col3.metric(
            "Confidence Threshold",
            f"{confidence * 100:.0f}%"
        )

        # ---------------------------------
        # OBJECT COUNTS
        # ---------------------------------

        if object_counts:

            st.subheader("📊 Detected Objects")

            for name, count in object_counts.items():

                st.write(
                    f"**{name}** : {count}"
                )

        else:

            st.info(
                "No objects detected."
            )


# -----------------------------------------
# WEBCAM MODE
# -----------------------------------------

elif mode == "📷 Webcam":

    st.info(
        "Webcam mode will be upgraded in the next step."
    )


# -----------------------------------------
# VIDEO MODE
# -----------------------------------------

elif mode == "🎥 Video":

    st.info(
        "Video mode will be upgraded in the next step."
    )
