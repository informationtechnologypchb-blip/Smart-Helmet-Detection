from pathlib import Path

app_code = r'''import os
import cv2
import av
import tempfile
import streamlit as st

from PIL import Image
from datetime import datetime
from ultralytics import YOLO
from streamlit_webrtc import (
    webrtc_streamer,
    WebRtcMode,
    VideoProcessorBase,
)

# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Smart Helmet Detection AI",
    page_icon="🪖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# 2. PATHS
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best.pt")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================================
# 3. PROFESSIONAL LIGHT THEME
# =========================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');

    :root {
        --navy: #14243A;
        --navy-2: #1D3553;
        --teal: #0F8B8D;
        --teal-dark: #087477;
        --page: #F4F7FB;
        --card: #FFFFFF;
        --text: #18283D;
        --muted: #718096;
        --border: #E4EAF1;
        --green: #16845B;
        --red: #D64545;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background: var(--page);
        color: var(--text);
    }

    [data-testid="stHeader"] {
        background: rgba(244, 247, 251, 0.92);
    }

    [data-testid="stSidebar"] {
        background: var(--navy);
        border-right: 1px solid #263B55;
    }

    [data-testid="stSidebar"] * {
        color: #EAF1F8;
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #AFC1D5;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        padding: 7px 10px;
        border-radius: 8px;
        font-weight: 600;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
        background: rgba(255,255,255,0.08);
    }

    [data-testid="stSidebar"] hr {
        border-color: #30445E;
    }

    .block-container {
        max-width: 1450px;
        padding-top: 1.7rem;
        padding-bottom: 3rem;
    }

    .brand-block {
        padding: 12px 4px 20px 4px;
    }

    .brand-icon {
        display: inline-flex;
        width: 43px;
        height: 43px;
        align-items: center;
        justify-content: center;
        background: #0F8B8D;
        border-radius: 12px;
        font-size: 23px;
        margin-bottom: 10px;
    }

    .brand-title {
        color: #FFFFFF;
        font-size: 20px;
        line-height: 1.2;
        font-weight: 800;
        letter-spacing: -0.4px;
    }

    .brand-subtitle {
        color: #AFC1D5;
        font-size: 11px;
        margin-top: 5px;
        letter-spacing: 1.3px;
        text-transform: uppercase;
    }

    .topline {
        color: var(--teal);
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 1.8px;
        text-transform: uppercase;
        margin-bottom: 7px;
    }

    .page-title {
        color: var(--navy);
        font-size: 31px;
        font-weight: 800;
        letter-spacing: -1px;
        line-height: 1.2;
        margin: 0 0 7px 0;
    }

    .page-subtitle {
        color: var(--muted);
        font-size: 14px;
        margin: 0 0 22px 0;
    }

    .hero {
        background: linear-gradient(115deg, #14243A 0%, #1D4260 100%);
        border-radius: 18px;
        padding: 28px 30px;
        margin: 4px 0 24px 0;
        border: 1px solid #294761;
        box-shadow: 0 10px 28px rgba(20, 36, 58, 0.10);
    }

    .hero-kicker {
        color: #69D0C8;
        text-transform: uppercase;
        letter-spacing: 1.8px;
        font-size: 11px;
        font-weight: 800;
        margin-bottom: 9px;
    }

    .hero h1 {
        color: #FFFFFF;
        font-size: 32px;
        font-weight: 800;
        letter-spacing: -0.8px;
        margin: 0 0 8px 0;
    }

    .hero p {
        color: #C5D3E1;
        font-size: 14px;
        margin: 0;
        max-width: 720px;
        line-height: 1.6;
    }

    .hero-pill {
        display: inline-block;
        margin-top: 17px;
        padding: 6px 11px;
        color: #C9F5ED;
        background: rgba(15,139,141,0.20);
        border: 1px solid rgba(105,208,200,0.35);
        border-radius: 30px;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.5px;
    }

    .metric-card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 19px 20px;
        min-height: 122px;
        box-shadow: 0 4px 14px rgba(24, 40, 61, 0.035);
    }

    .metric-top {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 10px;
        color: var(--muted);
        font-size: 12px;
        font-weight: 700;
    }

    .metric-icon {
        display: inline-flex;
        width: 34px;
        height: 34px;
        justify-content: center;
        align-items: center;
        border-radius: 10px;
        background: #EAF5F5;
        color: var(--teal-dark);
        font-size: 17px;
    }

    .metric-value {
        color: var(--navy);
        font-size: 29px;
        font-weight: 800;
        line-height: 1.2;
        margin-top: 11px;
    }

    .metric-note {
        color: #91A0B2;
        font-size: 11px;
        margin-top: 3px;
    }

    .module-card {
        background: #FFFFFF;
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 20px;
        min-height: 145px;
        box-shadow: 0 4px 14px rgba(24, 40, 61, 0.035);
    }

    .module-icon {
        display: inline-flex;
        width: 40px;
        height: 40px;
        align-items: center;
        justify-content: center;
        border-radius: 11px;
        background: #EAF5F5;
        color: var(--teal-dark);
        font-size: 20px;
        margin-bottom: 12px;
    }

    .module-card h3 {
        color: var(--navy);
        font-size: 16px;
        font-weight: 800;
        margin: 0 0 6px 0;
    }

    .module-card p {
        color: var(--muted);
        font-size: 12px;
        line-height: 1.5;
        margin: 0;
    }

    .info-panel {
        background: #FFFFFF;
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 19px 21px;
        box-shadow: 0 4px 14px rgba(24, 40, 61, 0.03);
    }

    .info-panel h3 {
        color: var(--navy);
        font-size: 16px;
        font-weight: 800;
        margin: 0 0 12px 0;
    }

    .status-line {
        display: flex;
        align-items: center;
        gap: 9px;
        color: #D9E6F2;
        font-size: 12px;
        margin: 10px 0;
    }

    .status-dot {
        width: 8px;
        height: 8px;
        display: inline-block;
        background: #36C58A;
        border-radius: 50%;
        box-shadow: 0 0 0 3px rgba(54,197,138,0.12);
    }

    .stButton > button, .stDownloadButton > button {
        border-radius: 9px;
        min-height: 42px;
        font-weight: 700;
        border: 1px solid #D7E1EA;
        transition: all 0.18s ease;
    }

    .stButton > button[kind="primary"] {
        background: var(--teal);
        border-color: var(--teal);
        color: white;
    }

    .stButton > button[kind="primary"]:hover {
        background: var(--teal-dark);
        border-color: var(--teal-dark);
        color: white;
    }

    .stDownloadButton > button {
        background: #FFFFFF;
        color: var(--navy);
    }

    .stDownloadButton > button:hover {
        border-color: var(--teal);
        color: var(--teal-dark);
    }

    [data-testid="stFileUploader"] {
        background: #FFFFFF;
        border: 1px dashed #B9C9D8;
        border-radius: 12px;
        padding: 8px;
    }

    [data-testid="stSlider"] [data-baseweb="slider"] {
        padding-top: 5px;
    }

    [data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid var(--border);
        padding: 14px 16px;
        border-radius: 12px;
    }

    [data-testid="stMetricLabel"] {
        color: var(--muted);
        font-weight: 700;
    }

    [data-testid="stMetricValue"] {
        color: var(--navy);
        font-weight: 800;
    }

    h2, h3 {
        color: var(--navy);
        letter-spacing: -0.3px;
    }

    .footer {
        color: #91A0B2;
        font-size: 11px;
        text-align: center;
        padding-top: 12px;
    }

    @media (max-width: 700px) {
        .page-title { font-size: 25px; }
        .hero { padding: 22px; }
        .hero h1 { font-size: 25px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# 4. MODEL LOADING
# =========================================================
@st.cache_resource
def load_model(path):
    return YOLO(path)


if not os.path.exists(MODEL_PATH):
    st.error("Trained model not found.")
    st.info("Please place your trained file at: models/best.pt")
    st.stop()

try:
    model = load_model(MODEL_PATH)
except Exception as error:
    st.error(f"Model loading failed: {error}")
    st.stop()

# =========================================================
# 5. SESSION STATE AND NAVIGATION
# =========================================================
for key, default in {
    "total_detections": 0,
    "with_helmet": 0,
    "without_helmet": 0,
    "navigation": "Dashboard",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


def go_to(page_name):
    # Callback runs before Streamlit recreates the radio widget.
    st.session_state.navigation = page_name


def update_statistics(with_count, without_count):
    st.session_state.total_detections += with_count + without_count
    st.session_state.with_helmet += with_count
    st.session_state.without_helmet += without_count


# =========================================================
# 6. HELPER FUNCTIONS
# =========================================================
def get_counts(result):
    with_count = 0
    without_count = 0

    if result.boxes is None:
        return with_count, without_count

    for box in result.boxes:
        class_id = int(box.cls[0])
        if class_id == 0:
            with_count += 1
        elif class_id == 1:
            without_count += 1

    return with_count, without_count


def save_uploaded_image(uploaded_file):
    extension = os.path.splitext(uploaded_file.name)[1].lower()

    if extension == ".jfif":
        image = Image.open(uploaded_file).convert("RGB")
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        image.save(temp.name, format="JPEG", quality=95)
        temp.close()
        return temp.name

    if extension not in [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"]:
        extension = ".jpg"

    temp = tempfile.NamedTemporaryFile(delete=False, suffix=extension)
    temp.write(uploaded_file.getvalue())
    temp.close()
    return temp.name


def save_uploaded_video(uploaded_file):
    extension = os.path.splitext(uploaded_file.name)[1].lower()
    if extension not in [".mp4", ".avi", ".mov", ".mkv"]:
        extension = ".mp4"

    temp = tempfile.NamedTemporaryFile(delete=False, suffix=extension)
    temp.write(uploaded_file.getvalue())
    temp.close()
    return temp.name


def page_heading(kicker, title, subtitle):
    st.markdown(
        f"""
        <div class="topline">{kicker}</div>
        <div class="page-title">{title}</div>
        <div class="page-subtitle">{subtitle}</div>
        """,
        unsafe_allow_html=True,
    )


def show_detection_details(result):
    if result.boxes is None or len(result.boxes) == 0:
        st.warning("No helmet-related objects were detected in this image.")
        return

    for index, box in enumerate(result.boxes, start=1):
        class_id = int(box.cls[0])
        confidence = float(box.conf[0]) * 100
        class_name = model.names[class_id]
        st.write(f"**{index}. {class_name}**  ·  Confidence: {confidence:.1f}%")


# =========================================================
# 7. LIVE CAMERA PROCESSOR
# =========================================================
class HelmetVideoProcessor(VideoProcessorBase):
    def __init__(self, confidence):
        self.confidence = confidence

    def recv(self, frame):
        frame_image = frame.to_ndarray(format="bgr24")
        results = model.predict(
            source=frame_image,
            conf=self.confidence,
            imgsz=640,
            verbose=False,
        )
        annotated = results[0].plot()
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")


# =========================================================
# 8. SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown(
        """
        <div class="brand-block">
            <div class="brand-icon">🪖</div>
            <div class="brand-title">Smart Helmet<br>Detection AI</div>
            <div class="brand-subtitle">Intelligent safety monitoring</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()
    st.markdown("#### WORKSPACE")

    page = st.radio(
        "Navigation",
        [
            "Dashboard",
            "Image Detection",
            "Video Detection",
            "Live Camera",
            "Settings",
        ],
        key="navigation",
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("#### SYSTEM STATUS")
    st.markdown(
        """
        <div class="status-line"><span class="status-dot"></span> AI model loaded</div>
        <div class="status-line"><span class="status-dot"></span> Detection engine ready</div>
        <div class="status-line"><span class="status-dot"></span> Output directory available</div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()
    st.caption("YOLOv8 · Computer Vision")
    st.caption("Smart Safety Monitoring")

# =========================================================
# 9. DASHBOARD
# =========================================================
if page == "Dashboard":
    st.markdown(
        """
        <div class="hero">
            <div class="hero-kicker">AI-powered safety platform</div>
            <h1>Smart Helmet Detection</h1>
            <p>
                Monitor helmet compliance with intelligent computer vision.
                Analyze images, process recorded videos, or run real-time
                detection through your camera.
            </p>
            <span class="hero-pill">● YOLOv8 MODEL READY</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page_heading(
        "OVERVIEW",
        "Dashboard",
        "A quick overview of detection activity in this session.",
    )

    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-top"><span>Total detections</span><span class="metric-icon">◎</span></div>
                <div class="metric-value">{st.session_state.total_detections:,}</div>
                <div class="metric-note">Objects detected in this session</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-top"><span>With helmet</span><span class="metric-icon">✓</span></div>
                <div class="metric-value" style="color:#16845B">{st.session_state.with_helmet:,}</div>
                <div class="metric-note">Helmet detected</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-top"><span>Without helmet</span><span class="metric-icon" style="background:#FFF0F0;color:#D64545">!</span></div>
                <div class="metric-value" style="color:#D64545">{st.session_state.without_helmet:,}</div>
                <div class="metric-note">No helmet detected</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Detection modules")
    st.caption("Choose a detection mode to get started.")

    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        st.markdown(
            """
            <div class="module-card">
                <div class="module-icon">▧</div>
                <h3>Image Detection</h3>
                <p>Upload a photo and identify riders with or without helmets.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.button(
            "Open Image Detection",
            key="go_image",
            use_container_width=True,
            on_click=go_to,
            args=("Image Detection",),
        )
    with c2:
        st.markdown(
            """
            <div class="module-card">
                <div class="module-icon">▶</div>
                <h3>Video Detection</h3>
                <p>Analyze a recorded video and download the annotated result.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.button(
            "Open Video Detection",
            key="go_video",
            use_container_width=True,
            on_click=go_to,
            args=("Video Detection",),
        )
    with c3:
        st.markdown(
            """
            <div class="module-card">
                <div class="module-icon">◉</div>
                <h3>Live Camera</h3>
                <p>Run real-time helmet detection using your webcam.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.button(
            "Open Live Camera",
            key="go_live",
            use_container_width=True,
            on_click=go_to,
            args=("Live Camera",),
        )

# =========================================================
# 10. IMAGE DETECTION
# =========================================================
elif page == "Image Detection":
    page_heading(
        "DETECTION MODULE",
        "Image Detection",
        "Upload a rider image to detect helmet compliance.",
    )

    left, right = st.columns([1.05, 0.95], gap="large")

    with left:
        st.markdown("### Upload image")
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=["jpg", "jpeg", "png", "jfif", "bmp", "webp", "tif", "tiff"],
            key="image_upload",
        )

        confidence = st.slider(
            "Detection confidence",
            min_value=0.10,
            max_value=0.90,
            value=0.25,
            step=0.05,
            key="image_confidence",
            help="Increase confidence to filter out weaker predictions.",
        )

        if uploaded_file is not None:
            preview = Image.open(uploaded_file).convert("RGB")
            st.image(preview, caption="Selected image", use_container_width=True)

            run_image = st.button(
                "Run image detection",
                type="primary",
                use_container_width=True,
                key="run_image_detection",
            )

            if run_image:
                image_path = None
                try:
                    with st.spinner("Analyzing image with YOLOv8..."):
                        image_path = save_uploaded_image(uploaded_file)
                        results = model.predict(
                            source=image_path,
                            conf=confidence,
                            imgsz=640,
                            save=False,
                            verbose=False,
                        )
                        result = results[0]
                        annotated = cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB)
                        with_count, without_count = get_counts(result)

                        output_path = os.path.join(
                            OUTPUT_DIR,
                            "image_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".jpg",
                        )
                        Image.fromarray(annotated).save(output_path, quality=95)

                    update_statistics(with_count, without_count)
                    st.session_state["last_image_result"] = {
                        "image": annotated,
                        "with": with_count,
                        "without": without_count,
                        "output": output_path,
                    }
                    st.success("Image detection completed.")
                except Exception as error:
                    st.error(f"Image detection failed: {error}")
                finally:
                    if image_path and os.path.exists(image_path):
                        os.remove(image_path)

    with right:
        st.markdown("### Detection output")
        last_result = st.session_state.get("last_image_result")
        if last_result:
            st.image(
                last_result["image"],
                caption="YOLOv8 annotated output",
                use_container_width=True,
            )
            a, b, c = st.columns(3)
            a.metric("Total", last_result["with"] + last_result["without"])
            b.metric("With helmet", last_result["with"])
            c.metric("Without helmet", last_result["without"])

            st.markdown("#### Detection summary")
            st.markdown(
                f"""
                <div class="info-panel">
                    <p style="margin:0 0 8px;color:#16845B;font-weight:700">✓ With helmet: {last_result['with']}</p>
                    <p style="margin:0;color:#D64545;font-weight:700">! Without helmet: {last_result['without']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            with open(last_result["output"], "rb") as output_file:
                st.download_button(
                    "Download annotated image",
                    data=output_file.read(),
                    file_name=os.path.basename(last_result["output"]),
                    mime="image/jpeg",
                    use_container_width=True,
                )
        else:
            st.markdown(
                """
                <div class="info-panel">
                    <h3>Ready for analysis</h3>
                    <p style="color:#718096;font-size:13px;line-height:1.6">
                        Select an image, adjust the confidence level, and run detection.
                        The annotated result and detection summary will appear here.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

# =========================================================
# 11. VIDEO DETECTION
# =========================================================
elif page == "Video Detection":
    page_heading(
        "DETECTION MODULE",
        "Video Detection",
        "Process a recorded video and generate an annotated output.",
    )

    uploaded_video = st.file_uploader(
        "Upload video",
        type=["mp4", "avi", "mov", "mkv"],
        key="video_upload",
    )

    confidence = st.slider(
        "Detection confidence",
        min_value=0.10,
        max_value=0.90,
        value=0.25,
        step=0.05,
        key="video_confidence",
    )

    if uploaded_video is not None:
        st.video(uploaded_video)

        if st.button(
            "Start video detection",
            type="primary",
            use_container_width=True,
            key="run_video_detection",
        ):
            input_path = None
            output_path = None
            cap = None
            writer = None

            try:
                input_path = save_uploaded_video(uploaded_video)
                output_path = os.path.join(
                    OUTPUT_DIR,
                    "video_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".mp4",
                )

                cap = cv2.VideoCapture(input_path)
                if not cap.isOpened():
                    raise ValueError("Unable to open uploaded video.")

                fps = cap.get(cv2.CAP_PROP_FPS)
                if not fps or fps <= 0:
                    fps = 25.0

                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                if width <= 0 or height <= 0:
                    raise ValueError("Invalid video dimensions.")

                writer = cv2.VideoWriter(
                    output_path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (width, height),
                )
                if not writer.isOpened():
                    raise ValueError("Unable to create output video.")

                progress = st.progress(0)
                status = st.empty()
                frame_number = 0
                with_count = 0
                without_count = 0

                with st.spinner("Processing video frames..."):
                    while True:
                        success, frame = cap.read()
                        if not success:
                            break

                        result = model.predict(
                            source=frame,
                            conf=confidence,
                            imgsz=640,
                            verbose=False,
                        )[0]

                        writer.write(result.plot())
                        frame_with, frame_without = get_counts(result)
                        with_count += frame_with
                        without_count += frame_without
                        frame_number += 1

                        if total_frames > 0:
                            progress.progress(min(frame_number / total_frames, 1.0))

                        if frame_number % 10 == 0:
                            status.write(f"Processed {frame_number:,} / {total_frames:,} frames")

                cap.release()
                cap = None
                writer.release()
                writer = None

                if frame_number == 0:
                    raise ValueError("No readable frames found in this video.")

                update_statistics(with_count, without_count)
                progress.progress(1.0)
                status.success("Video processing completed.")

                with open(output_path, "rb") as output_file:
                    video_bytes = output_file.read()

                st.session_state["last_video_result"] = {
                    "bytes": video_bytes,
                    "name": os.path.basename(output_path),
                    "frames": frame_number,
                    "with": with_count,
                    "without": without_count,
                }
            except Exception as error:
                st.error(f"Video detection failed: {error}")
            finally:
                if cap is not None:
                    cap.release()
                if writer is not None:
                    writer.release()
                if input_path and os.path.exists(input_path):
                    os.remove(input_path)

    last_video = st.session_state.get("last_video_result")
    if last_video:
        st.markdown("### Processed video")
        st.video(last_video["bytes"])
        v1, v2, v3 = st.columns(3)
        v1.metric("Frames processed", f"{last_video['frames']:,}")
        v2.metric("With helmet", f"{last_video['with']:,}")
        v3.metric("Without helmet", f"{last_video['without']:,}")
        st.download_button(
            "Download processed video",
            data=last_video["bytes"],
            file_name=last_video["name"],
            mime="video/mp4",
            use_container_width=True,
        )
        st.caption(
            "Counts are frame-level detections. The same rider may be counted in multiple frames."
        )

# =========================================================
# 12. LIVE CAMERA
# =========================================================
elif page == "Live Camera":
    page_heading(
        "REAL-TIME MODULE",
        "Live Camera",
        "Use your webcam to detect helmet compliance in real time.",
    )

    confidence = st.slider(
        "Detection confidence",
        min_value=0.10,
        max_value=0.90,
        value=0.25,
        step=0.05,
        key="live_confidence",
    )

    st.markdown(
        """
        <div class="info-panel">
            <h3>Camera instructions</h3>
            <p style="color:#718096;font-size:13px;line-height:1.6;margin:0">
                Select START below and allow camera access when your browser asks.
                The camera preview will display YOLOv8 detection boxes in real time.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    webrtc_streamer(
        key="helmet-live-camera",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=lambda: HelmetVideoProcessor(confidence),
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
        async_processing=True,
    )

# =========================================================
# 13. SETTINGS
# =========================================================
elif page == "Settings":
    page_heading(
        "CONFIGURATION",
        "System Settings",
        "Review model information and manage session statistics.",
    )

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown(
            """
            <div class="info-panel">
                <h3>Model information</h3>
                <p style="color:#718096;font-size:13px;line-height:2;margin:0">
                    <b>Model:</b> YOLOv8n<br>
                    <b>Framework:</b> Ultralytics<br>
                    <b>Input size:</b> 640 × 640<br>
                    <b>Class 0:</b> With Helmet<br>
                    <b>Class 1:</b> Without Helmet<br>
                    <b>Weights:</b> models/best.pt
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown("### Session statistics")
        st.metric("Total detections", st.session_state.total_detections)
        st.metric("With helmet", st.session_state.with_helmet)
        st.metric("Without helmet", st.session_state.without_helmet)

        if st.button(
            "Reset session statistics",
            type="primary",
            use_container_width=True,
        ):
            st.session_state.total_detections = 0
            st.session_state.with_helmet = 0
            st.session_state.without_helmet = 0
            st.success("Session statistics have been reset.")
            st.rerun()

# =========================================================
# 14. FOOTER
# =========================================================
st.markdown(
    """
    <div class="footer">
        SMART HELMET DETECTION AI &nbsp;·&nbsp; YOLOv8 Computer Vision &nbsp;·&nbsp; Python + Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)
'''

out = Path("/mnt/data/app.py")
out.write_text(app_code, encoding="utf-8")
print(f"Created {out} ({len(app_code.splitlines())} lines)")
