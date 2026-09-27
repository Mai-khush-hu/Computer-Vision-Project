
import os
from collections import Counter

import cv2
import numpy as np
import streamlit as st

from detector import ObjectDetector


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Object Detection",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# LOAD YOLO MODEL
# ============================================================

@st.cache_resource
def load_detector():
    """Load and cache the YOLO object detector."""
    return ObjectDetector("yolo11n.pt")


detector = load_detector()


# ============================================================
# PAGE TITLE
# ============================================================

st.title("🤖 AI Object Detection System")

st.write(
    "Object detection using YOLO11 and OpenCV"
)


# ============================================================
# SIDEBAR SETTINGS
# ============================================================

st.sidebar.header("⚙️ Detection Settings")

confidence = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.5,
    step=0.05
)


# ============================================================
# INPUT MODE
# ============================================================

mode = st.radio(
    "Select Input Mode",
    ["📷 Webcam", "🖼️ Image", "🎥 Video"],
    horizontal=True
)


# ============================================================
# IMAGE MODE
# ============================================================

if mode == "🖼️ Image":

    st.subheader("🖼️ Image Object Detection")

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        # Read uploaded image
        file_bytes = np.asarray(
            bytearray(uploaded_file.read()),
            dtype=np.uint8
        )

        image = cv2.imdecode(
            file_bytes,
            cv2.IMREAD_COLOR
        )

        if image is None:

            st.error("Could not decode the uploaded image.")

        else:

            # Run detection
            results = detector.detect(
                image,
                confidence
            )

            # Draw detections
            annotated_image = detector.draw_detections(
                results
            )

            # Convert BGR to RGB for Streamlit
            annotated_image = cv2.cvtColor(
                annotated_image,
                cv2.COLOR_BGR2RGB
            )

            # Display image
            st.image(
                annotated_image,
                caption="Detection Result",
                use_container_width=True
            )

            # Get statistics
            statistics = detector.get_statistics(
                results
            )

            total_objects = statistics.get(
                "total_objects",
                0
            )

            object_counts = statistics.get(
                "object_counts",
                {}
            )

            avg_confidence = statistics.get(
                "average_confidence",
                0
            )

            # Display metrics
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

            # Display detected objects
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


# ============================================================
# WEBCAM MODE
# ============================================================

elif mode == "📷 Webcam":

    st.subheader("📷 Webcam Object Detection")

    st.info(
        "Take a picture using your webcam to detect objects."
    )

    camera_image = st.camera_input(
        "Take a picture"
    )

    if camera_image is not None:

        # Read camera image
        file_bytes = np.asarray(
            bytearray(camera_image.read()),
            dtype=np.uint8
        )

        image = cv2.imdecode(
            file_bytes,
            cv2.IMREAD_COLOR
        )

        if image is None:

            st.error(
                "Could not read the webcam image."
            )

        else:

            # Run detection
            results = detector.detect(
                image,
                confidence
            )

            # Draw detections
            annotated_image = detector.draw_detections(
                results
            )

            # Convert BGR to RGB
            annotated_image = cv2.cvtColor(
                annotated_image,
                cv2.COLOR_BGR2RGB
            )

            # Display result
            st.image(
                annotated_image,
                caption="Webcam Detection Result",
                use_container_width=True
            )

            # Get statistics
            statistics = detector.get_statistics(
                results
            )

            total_objects = statistics.get(
                "total_objects",
                0
            )

            object_counts = statistics.get(
                "object_counts",
                {}
            )

            avg_confidence = statistics.get(
                "average_confidence",
                0
            )

            # Display metrics
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
                "Threshold",
                f"{confidence * 100:.0f}%"
            )

            # Display object counts
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


# ============================================================
# VIDEO MODE
# ============================================================

elif mode == "🎥 Video":

    st.subheader("🎥 Video Object Detection")

    uploaded_video = st.file_uploader(
        "Upload a video",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_video is not None:

        # Create folders if they don't exist
        os.makedirs(
            "outputs",
            exist_ok=True
        )

        # Save uploaded video
        input_path = os.path.join(
            "outputs",
            "input_video.mp4"
        )

        output_path = os.path.join(
            "outputs",
            "detected_video.mp4"
        )

        with open(
            input_path,
            "wb"
        ) as file:

            file.write(
                uploaded_video.getbuffer()
            )

        st.success(
            "Video uploaded successfully!"
        )

        # Open video
        cap = cv2.VideoCapture(
            input_path
        )

        if not cap.isOpened():

            st.error(
                "Could not open the uploaded video."
            )

        else:

            # Get video properties
            width = int(
                cap.get(
                    cv2.CAP_PROP_FRAME_WIDTH
                )
            )

            height = int(
                cap.get(
                    cv2.CAP_PROP_FRAME_HEIGHT
                )
            )

            fps = cap.get(
                cv2.CAP_PROP_FPS
            )

            total_frames = int(
                cap.get(
                    cv2.CAP_PROP_FRAME_COUNT
                )
            )

            # Prevent invalid FPS
            if fps <= 0:
                fps = 30.0

            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(
                *"mp4v"
            )

            out = cv2.VideoWriter(
                output_path,
                fourcc,
                fps,
                (width, height)
            )

            if not out.isOpened():

                cap.release()

                st.error(
                    "Could not create the output video."
                )

            else:

                # Progress UI
                progress_bar = st.progress(0)

                status_text = st.empty()

                frame_count = 0

                # Process video frame by frame
                while True:

                    ret, frame = cap.read()

                    if not ret:
                        break

                    # Run YOLO detection
                    results = detector.detect(
                        frame,
                        confidence
                    )

                    # Draw detections
                    annotated_frame = (
                        detector.draw_detections(
                            results
                        )
                    )

                    # Write processed frame
                    out.write(
                        annotated_frame
                    )

                    frame_count += 1

                    # Update progress
                    if total_frames > 0:

                        progress = min(
                            frame_count / total_frames,
                            1.0
                        )

                        progress_bar.progress(
                            progress
                        )

                        status_text.text(
                            f"Processing frame "
                            f"{frame_count}/{total_frames}"
                        )

                # Release resources
                cap.release()
                out.release()

                progress_bar.progress(1.0)

                status_text.success(
                    "Video processing completed!"
                )

                # Display processed video
                st.subheader(
                    "🎯 Detection Result"
                )

                if os.path.exists(output_path):

                    st.video(
                        output_path
                    )

                    # Download button
                    with open(
                        output_path,
                        "rb"
                    ) as file:

                        st.download_button(
                            label="⬇️ Download Processed Video",
                            data=file.read(),
                            file_name="detected_video.mp4",
                            mime="video/mp4"
                        )

                else:

                    st.error(
                        "Output video was not created."
                    )
