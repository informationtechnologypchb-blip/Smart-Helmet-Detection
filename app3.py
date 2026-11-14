
import streamlit as st
import os
import cv2
import av
import tempfile
import numpy as np

from PIL import Image
from datetime import datetime
from ultralytics import YOLO
from streamlit_webrtc import (
    webrtc_streamer,
    WebRtcMode,
    VideoProcessorBase
)


# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Smart Helmet Detection AI",
    page_icon="🪖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==================================================
# 2. PROJECT PATHS
# ==================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "models", "best.pt")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==================================================
# 3. PROFESSIONAL CSS
# ==================================================

st.markdown("""
<style>

.stApp {
    background: #0b1220;
    color: #f1f5f9;
}

[data-testid="stSidebar"] {
    background: #111c30;
    border-right: 1px solid #263b57;
}

.block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.hero {
    background: linear-gradient(120deg, #172c4a, #123b4b);
    padding: 30px;
    border-radius: 18px;
    border: 1px solid #284966;
    margin-bottom: 25px;
}

.hero h1 {
    color: white;
    font-size: 34px;
    font-weight: 750;
}

.hero p {
    color: #b9c9da;
    font-size: 16px;
}

.metric-card {
    background: #142238;
    padding: 22px;
    border-radius: 14px;
    border: 1px solid #263d5c;
}

.metric-label {
    color: #9eb2ca;
    font-size: 14px;
}

.metric-value {
    color: white;
    font-size: 30px;
    font-weight: 700;
}

.section-card {
    background: #142238;
    border: 1px solid #263d5c;
    border-radius: 14px;
    padding: 20px;
}

.stButton button {
    border-radius: 9px;
    font-weight: 600;
    min-height: 42px;
}

</style>
""", unsafe_allow_html=True)


# ==================================================
# 4. LOAD TRAINED MODEL
# ==================================================

@st.cache_resource
def load_model(path):
    return YOLO(path)


if not os.path.exists(MODEL_PATH):

    st.error("Trained model not found!")

    st.code("models/best.pt")

    st.stop()


try:

    model = load_model(MODEL_PATH)

except Exception as error:

    st.error(f"Model loading failed: {error}")
    st.stop()


# ==================================================
# 5. SESSION STATISTICS
# ==================================================

if "total_detections" not in st.session_state:
    st.session_state.total_detections = 0

if "with_helmet" not in st.session_state:
    st.session_state.with_helmet = 0

if "without_helmet" not in st.session_state:
    st.session_state.without_helmet = 0


# ==================================================
# 6. HELPER FUNCTIONS
# ==================================================

def get_counts(result):

    with_count = 0
    without_count = 0

    for box in result.boxes:

        class_id = int(box.cls[0])

        if class_id == 0:
            with_count += 1

        elif class_id == 1:
            without_count += 1

    return with_count, without_count


def update_statistics(with_count, without_count):

    st.session_state.total_detections += (
        with_count + without_count
    )

    st.session_state.with_helmet += with_count

    st.session_state.without_helmet += without_count


def save_uploaded_image(uploaded_file):

    """
    Save image using a YOLO-supported extension.
    JFIF images are converted to JPG.
    """

    extension = os.path.splitext(
        uploaded_file.name
    )[1].lower()

    if extension == ".jfif":

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        temp = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".jpg"
        )

        image.save(
            temp.name,
            format="JPEG",
            quality=95
        )

        temp.close()

        return temp.name

    # Preserve original image file for other formats
    temp = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=extension
    )

    temp.write(uploaded_file.getvalue())
    temp.close()

    return temp.name


def save_uploaded_video(uploaded_file):

    extension = os.path.splitext(
        uploaded_file.name
    )[1].lower()

    if extension not in [".mp4", ".avi", ".mov", ".mkv"]:
        extension = ".mp4"

    temp = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=extension
    )

    temp.write(uploaded_file.getvalue())
    temp.close()

    return temp.name


def display_details(result):

    if len(result.boxes) == 0:

        st.warning("No objects detected.")
        return

    for index, box in enumerate(
        result.boxes,
        start=1
    ):

        class_id = int(box.cls[0])

        confidence = float(box.conf[0]) * 100

        class_name = model.names[class_id]

        st.write(
            f"**{index}. {class_name}** "
            f"| Confidence: {confidence:.2f}%"
        )


# ==================================================
# 7. LIVE CAMERA PROCESSOR
# ==================================================

class HelmetVideoProcessor(VideoProcessorBase):

    def __init__(self, confidence):

        self.confidence = confidence

    def recv(self, frame):

        frame_image = frame.to_ndarray(
            format="bgr24"
        )

        results = model.predict(
            source=frame_image,
            conf=self.confidence,
            imgsz=640,
            verbose=False
        )

        annotated = results[0].plot()

        return av.VideoFrame.from_ndarray(
            annotated,
            format="bgr24"
        )


# ==================================================
# 8. SIDEBAR NAVIGATION
# ==================================================

with st.sidebar:

    st.markdown("## 🪖 Helmet AI")

    st.caption("Intelligent Safety Monitoring")

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Dashboard",
            "Image Detection",
            "Video Detection",
            "Live Camera",
            "Settings"
        ],
        key="navigation"
    )

    st.divider()

    st.markdown("### SYSTEM STATUS")

    st.success("Model Available")

    st.success("YOLOv8 Loaded")

    st.caption("Computer Vision System")


# ==================================================
# 9. MAIN HEADER
# ==================================================

st.markdown("""
<div class="hero">

    <h1>Smart Helmet Detection AI</h1>

    <p>
        AI-powered motorcycle rider safety monitoring
        using YOLOv8 computer vision.
    </p>

</div>
""", unsafe_allow_html=True)


# ==================================================
# 10. DASHBOARD
# ==================================================

if page == "Dashboard":

    st.subheader("System Overview")

    st.write(
        "Monitor detection activity and access "
        "the available detection modules."
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Detections</div>
            <div class="metric-value">
                {st.session_state.total_detections}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">With Helmet</div>
            <div class="metric-value"
                 style="color:#22c55e">
                {st.session_state.with_helmet}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Without Helmet</div>
            <div class="metric-value"
                 style="color:#ef4444">
                {st.session_state.without_helmet}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    st.subheader("Detection Modules")

    c1, c2, c3 = st.columns(3)

    with c1:

        st.markdown("""
        <div class="section-card">
            <h3>📷 Image Detection</h3>
            <p>Upload pictures and detect helmets.</p>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Open Image Detection",
                     use_container_width=True):

            st.session_state.navigation = "Image Detection"
            st.rerun()

    with c2:

        st.markdown("""
        <div class="section-card">
            <h3>🎞️ Video Detection</h3>
            <p>Analyze uploaded videos frame by frame.</p>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Open Video Detection",
                     use_container_width=True):

            st.session_state.navigation = "Video Detection"
            st.rerun()

    with c3:

        st.markdown("""
        <div class="section-card">
            <h3>🎥 Live Camera</h3>
            <p>Real-time helmet detection through webcam.</p>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Open Live Camera",
                     use_container_width=True):

            st.session_state.navigation = "Live Camera"
            st.rerun()


# ==================================================
# 11. IMAGE DETECTION
# ==================================================

elif page == "Image Detection":

    st.subheader("📷 Image Helmet Detection")

    st.write(
        "Upload an image to identify riders "
        "with and without helmets."
    )

    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png", "jfif"],
        key="image_upload"
    )

    confidence = st.slider(
        "Detection Confidence",
        min_value=0.10,
        max_value=0.90,
        value=0.25,
        step=0.05,
        key="image_confidence"
    )

    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

        if st.button(
            "Start Image Detection",
            type="primary",
            use_container_width=True
        ):

            image_path = None

            try:

                with st.spinner("AI analyzing image..."):

                    image_path = save_uploaded_image(
                        uploaded_file
                    )

                    # Original file path is passed to YOLO.
                    # JFIF has already been converted to JPG.

                    results = model.predict(
                        source=image_path,
                        conf=confidence,
                        imgsz=640,
                        save=False,
                        verbose=False
                    )

                    result = results[0]

                    annotated = result.plot()

                    annotated = cv2.cvtColor(
                        annotated,
                        cv2.COLOR_BGR2RGB
                    )

                    with_count, without_count = get_counts(
                        result
                    )

                    output_path = os.path.join(
                        OUTPUT_DIR,
                        "image_" +
                        datetime.now().strftime(
                            "%Y%m%d_%H%M%S"
                        ) +
                        ".jpg"
                    )

                    Image.fromarray(
                        annotated
                    ).save(output_path)

                update_statistics(
                    with_count,
                    without_count
                )

                st.success("Image detection completed!")

                st.subheader("Detection Result")

                st.image(
                    annotated,
                    caption="AI Detection Output",
                    use_container_width=True
                )

                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "Total Detected",
                    with_count + without_count
                )

                c2.metric(
                    "With Helmet",
                    with_count
                )

                c3.metric(
                    "Without Helmet",
                    without_count
                )

                st.subheader("Detection Details")

                display_details(result)

                with open(output_path, "rb") as file:

                    st.download_button(
                        "⬇️ Download Detection Image",
                        data=file.read(),
                        file_name=os.path.basename(
                            output_path
                        ),
                        mime="image/jpeg"
                    )

            except Exception as error:

                st.error(
                    f"Image detection failed: {error}"
                )

            finally:

                if image_path and os.path.exists(image_path):

                    os.remove(image_path)


# ==================================================
# 12. VIDEO DETECTION
# ==================================================

elif page == "Video Detection":

    st.subheader("🎞️ Video Helmet Detection")

    st.write(
        "Upload a video to detect helmet usage "
        "frame by frame."
    )

    uploaded_video = st.file_uploader(
        "Upload Video",
        type=["mp4", "avi", "mov", "mkv"],
        key="video_upload"
    )

    confidence = st.slider(
        "Video Confidence",
        min_value=0.10,
        max_value=0.90,
        value=0.25,
        step=0.05,
        key="video_confidence"
    )

    if uploaded_video is not None:

        st.video(uploaded_video)

        if st.button(
            "Start Video Detection",
            type="primary",
            use_container_width=True
        ):

            input_path = None
            output_path = None
            cap = None
            writer = None

            try:

                input_path = save_uploaded_video(
                    uploaded_video
                )

                output_path = os.path.join(
                    OUTPUT_DIR,
                    "video_" +
                    datetime.now().strftime(
                        "%Y%m%d_%H%M%S"
                    ) +
                    ".mp4"
                )

                cap = cv2.VideoCapture(input_path)

                if not cap.isOpened():

                    raise ValueError(
                        "Unable to open uploaded video."
                    )

                fps = cap.get(cv2.CAP_PROP_FPS)

                if not fps or fps <= 0:
                    fps = 25.0

                total_frames = int(
                    cap.get(cv2.CAP_PROP_FRAME_COUNT)
                )

                width = int(
                    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                )

                height = int(
                    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                )

                if width <= 0 or height <= 0:

                    raise ValueError(
                        "Invalid video dimensions."
                    )

                writer = cv2.VideoWriter(
                    output_path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (width, height)
                )

                if not writer.isOpened():

                    raise ValueError(
                        "Unable to create output video."
                    )

                progress = st.progress(0)

                status = st.empty()

                frame_number = 0
                with_count = 0
                without_count = 0

                with st.spinner("Processing video..."):

                    while True:

                        success, frame = cap.read()

                        if not success:
                            break

                        results = model.predict(
                            source=frame,
                            conf=confidence,
                            imgsz=640,
                            verbose=False
                        )

                        result = results[0]

                        annotated = result.plot()

                        writer.write(annotated)

                        frame_with, frame_without = (
                            get_counts(result)
                        )

                        with_count += frame_with

                        without_count += frame_without

                        frame_number += 1

                        if total_frames > 0:

                            progress.progress(
                                min(
                                    frame_number / total_frames,
                                    1.0
                                )
                            )

                        if frame_number % 10 == 0:

                            status.write(
                                f"Processed {frame_number} "
                                f"of {total_frames} frames"
                            )

                cap.release()
                cap = None

                writer.release()
                writer = None

                if frame_number == 0:

                    raise ValueError(
                        "No readable frames found."
                    )

                update_statistics(
                    with_count,
                    without_count
                )

                progress.progress(1.0)

                status.success(
                    "Video processing completed!"
                )

                st.success(
                    "Video detection completed successfully!"
                )

                with open(output_path, "rb") as file:

                    video_bytes = file.read()

                st.subheader("Processed Video")

                st.video(video_bytes)

                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "Frames Processed",
                    frame_number
                )

                c2.metric(
                    "With Helmet",
                    with_count
                )

                c3.metric(
                    "Without Helmet",
                    without_count
                )

                st.download_button(
                    "⬇️ Download Processed Video",
                    data=video_bytes,
                    file_name=os.path.basename(
                        output_path
                    ),
                    mime="video/mp4"
                )

                st.caption(
                    "Counts are frame-level detections. "
                    "The same rider can be counted repeatedly."
                )

            except Exception as error:

                st.error(
                    f"Video detection failed: {error}"
                )

            finally:

                if cap is not None:
                    cap.release()

                if writer is not None:
                    writer.release()

                if input_path and os.path.exists(input_path):
                    os.remove(input_path)


# ==================================================
# 13. LIVE CAMERA
# ==================================================

elif page == "Live Camera":

    st.subheader("🎥 Live Camera Detection")

    st.write(
        "Start your webcam to detect helmet usage "
        "in real time."
    )

    confidence = st.slider(
        "Live Camera Confidence",
        min_value=0.10,
        max_value=0.90,
        value=0.25,
        step=0.05,
        key="live_confidence"
    )

    st.info(
        "Press START and allow camera permission "
        "in your browser."
    )

    webrtc_streamer(
        key="helmet-live-camera",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=lambda: HelmetVideoProcessor(
            confidence
        ),
        media_stream_constraints={
            "video": True,
            "audio": False
        },
        rtc_configuration={
            "iceServers": [
                {
                    "urls": [
                        "stun:stun.l.google.com:19302"
                    ]
                }
            ]
        },
        async_processing=True
    )


# ==================================================
# 14. SETTINGS
# ==================================================

elif page == "Settings":

    st.subheader("⚙️ System Settings")

    st.markdown("### Model Information")

    st.write("Model: YOLOv8n")

    st.write("Framework: Ultralytics")

    st.write("Input Size: 640")

    st.write("Class 0: With Helmet")

    st.write("Class 1: Without Helmet")

    st.write("Model Path: models/best.pt")

    st.divider()

    st.markdown("### Dashboard Statistics")

    st.write(
        "Total detections:",
        st.session_state.total_detections
    )

    st.write(
        "With Helmet:",
        st.session_state.with_helmet
    )

    st.write(
        "Without Helmet:",
        st.session_state.without_helmet
    )

    if st.button("Reset Dashboard Statistics"):

        st.session_state.total_detections = 0

        st.session_state.with_helmet = 0

        st.session_state.without_helmet = 0

        st.success("Statistics reset successfully!")

        st.rerun()


# ==================================================
# 15. FOOTER
# ==================================================

st.divider()

st.caption(
    "Smart Helmet Detection AI | "
    "YOLOv8 Computer Vision | "
    "Python & Streamlit"
)
