import base64
import io
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="Mbarira AI - Water Meter Reading", page_icon="\U0001F4A7", layout="wide")

APP_DIR = Path(__file__).parent
STAGE1_PATH = APP_DIR / "models" / "meter_detector.pt"
STAGE2_PATH = APP_DIR / "models" / "digit_detector.pt"

STAGE2_CLASSES = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "u"]
CROP_IMGSZ = 416
CROP_PAD = 0.12

COLORS = {
    "primary": "#00543b",
    "primary_container": "#0b6e4f",
    "on_primary_container": "#98edc6",
    "secondary": "#006781",
    "secondary_container": "#8fdfff",
    "on_secondary_container": "#00647d",
    "tertiary_container": "#974946",
    "background": "#f8f9ff",
    "surface": "#f8f9ff",
    "surface_container_lowest": "#ffffff",
    "surface_container_low": "#eff4ff",
    "surface_container": "#e5eeff",
    "surface_bright": "#f8f9ff",
    "on_surface": "#0b1c30",
    "on_surface_variant": "#3f4943",
    "outline": "#6f7a73",
    "outline_variant": "#bec9c1",
    "accent": "#0E7490",
    "error": "#ba1a1a",
}

BASE_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@500;700&family=Inter:wght@400;500;600&family=Sora:wght@700&display=swap');

:root {{
    color-scheme: light !important;
}}
html, body,
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stHeader"],
[data-testid="stToolbar"],
[class*="css"] {{
    font-family: 'Inter', sans-serif !important;
    background-color: {COLORS["background"]} !important;
    color: {COLORS["on_surface"]} !important;
}}
[data-testid="stHeader"] {{
    background-color: transparent !important;
}}

.block-container {{
    padding-top: 1.5rem;
    max-width: 1280px;
}}

#MainMenu, footer {{visibility: hidden;}}

.mb-navbar {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 4px 20px 4px;
    border-bottom: 1px solid {COLORS["outline_variant"]}55;
    margin-bottom: 24px;
}}
.mb-navbar .brand {{
    font-family: 'Sora', sans-serif;
    font-weight: 700;
    font-size: 22px;
    color: {COLORS["on_surface"]} !important;
}}
.mb-navbar .links a {{
    color: {COLORS["on_surface_variant"]} !important;
    text-decoration: none;
    margin-left: 24px;
    font-size: 15px;
}}
.mb-navbar .links a.active {{
    color: {COLORS["primary"]} !important;
    font-weight: 700;
    border-bottom: 2px solid {COLORS["primary"]};
    padding-bottom: 4px;
}}

.mb-header h1 {{
    font-family: 'Sora', sans-serif;
    font-size: 40px;
    line-height: 1.15;
    margin-bottom: 4px;
    color: {COLORS["on_surface"]} !important;
}}
.mb-header p {{
    font-size: 17px;
    color: {COLORS["on_surface_variant"]} !important;
    max-width: 640px;
}}

[data-testid="stVerticalBlockBorderWrapper"] {{
    background: {COLORS["surface_container_lowest"]} !important;
    border: 1px solid {COLORS["outline_variant"]} !important;
    border-radius: 12px !important;
    box-shadow: 0 1px 2px rgba(15,23,42,0.03);
}}
[data-testid="stVerticalBlockBorderWrapper"] * {{
    color: {COLORS["on_surface"]};
}}
.mb-card-title {{
    font-family: 'Sora', sans-serif;
    font-size: 20px;
    font-weight: 700;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    color: {COLORS["on_surface"]} !important;
}}

.mb-badge {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.05em;
    background: {COLORS["secondary_container"]}33;
    color: {COLORS["on_secondary_container"]};
    padding: 4px 12px;
    border-radius: 999px;
    display: inline-flex;
    align-items: center;
    gap: 6px;
}}
.mb-badge .dot {{
    width: 8px; height: 8px; border-radius: 999px;
    background: {COLORS["primary"]};
    animation: mb-pulse 1.4s infinite;
}}
@keyframes mb-pulse {{
    0%, 100% {{ opacity: 1; }}
    50% {{ opacity: 0.3; }}
}}

[data-testid="stFileUploaderDropzone"] {{
    background: transparent !important;
    border: 2px dashed {COLORS["outline_variant"]} !important;
    border-radius: 10px !important;
    padding: 12px !important;
}}
[data-testid="stFileUploaderDropzone"]:hover {{
    border-color: {COLORS["accent"]} !important;
}}

.mb-preview-frame {{
    position: relative;
    width: 100%;
    min-height: 320px;
    background: {COLORS["surface_container_low"]};
    border: 1px solid {COLORS["outline_variant"]};
    border-radius: 10px;
    overflow: hidden;
    display: flex;
    align-items: center;
    justify-content: center;
}}
.mb-preview-frame img {{
    width: 100%;
    height: auto;
    display: block;
}}
.mb-preview-empty {{
    color: {COLORS["on_surface_variant"]};
    font-size: 14px;
    text-align: center;
    padding: 32px;
}}

.mb-scanline {{
    position: absolute;
    top: 0; left: 0; width: 100%; height: 2px;
    background: linear-gradient(90deg, transparent, {COLORS["accent"]}, transparent);
    box-shadow: 0 0 10px {COLORS["accent"]}, 0 0 20px {COLORS["accent"]};
    animation: mb-scan 2.2s infinite linear;
    opacity: 0.85;
}}
@keyframes mb-scan {{
    0% {{ top: 0%; opacity: 0; }}
    10% {{ opacity: 1; }}
    90% {{ opacity: 1; }}
    100% {{ top: 100%; opacity: 0; }}
}}

.mb-label-caps {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: {COLORS["on_surface_variant"]};
    margin-bottom: 6px;
}}

.mb-reading-box {{
    background: {COLORS["surface_container"]};
    border: 1px solid {COLORS["outline_variant"]}88;
    border-radius: 10px;
    padding: 16px;
}}
.mb-reading-box .value {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 30px;
    font-weight: 700;
    letter-spacing: 0.12em;
    color: {COLORS["on_surface"]};
}}
.mb-reading-box.warn {{
    border-color: {COLORS["error"]}55;
}}
.mb-reading-box.warn .value {{
    color: {COLORS["error"]};
}}

.mb-metric-card {{
    border: 1px solid {COLORS["outline_variant"]}55;
    border-radius: 10px;
    padding: 12px;
    background: {COLORS["surface_bright"]};
}}
.mb-confbars {{
    display: flex;
    align-items: flex-end;
    gap: 3px;
    height: 32px;
    margin-bottom: 6px;
}}
.mb-confbars .bar {{
    flex: 1;
    background: {COLORS["primary_container"]};
    border-radius: 2px 2px 0 0;
}}
.mb-conf-footer {{
    display: flex;
    justify-content: space-between;
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: {COLORS["on_surface_variant"]};
}}

.mb-timeline {{
    list-style: none;
    padding-left: 20px;
    margin: 0;
    border-left: 1px solid {COLORS["outline_variant"]};
}}
.mb-timeline li {{
    position: relative;
    padding-bottom: 16px;
}}
.mb-timeline li:last-child {{
    padding-bottom: 0;
    color: {COLORS["primary_container"]};
}}
.mb-timeline li::before {{
    content: "";
    position: absolute;
    left: -25px;
    top: 4px;
    width: 8px;
    height: 8px;
    border-radius: 999px;
    background: {COLORS["primary_container"]};
}}
.mb-timeline .step-title {{
    font-size: 13px;
    font-weight: 600;
    color: {COLORS["on_surface"]};
}}
.mb-timeline .step-meta {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: {COLORS["on_surface_variant"]};
}}

.mb-footer {{
    margin-top: 48px;
    padding-top: 20px;
    border-top: 1px solid {COLORS["outline_variant"]};
    font-size: 12px;
    color: {COLORS["on_surface_variant"]};
    text-align: center;
}}

div.stButton > button {{
    border-radius: 8px;
    font-weight: 500;
}}
div.stButton > button[kind="primary"] {{
    background-color: {COLORS["primary_container"]};
    border-color: {COLORS["primary_container"]};
}}
</style>
"""

st.markdown(BASE_CSS, unsafe_allow_html=True)


@st.cache_resource
def load_models():
    if not STAGE1_PATH.exists() or not STAGE2_PATH.exists():
        st.error(
            f"Model weights missing. Expected files at:\n\n"
            f"- `{STAGE1_PATH}`\n- `{STAGE2_PATH}`\n\n"
            "Make sure models/meter_detector.pt and models/digit_detector.pt are committed "
            "to this repo and not excluded by .gitignore."
        )
        st.stop()
    return YOLO(str(STAGE1_PATH)), YOLO(str(STAGE2_PATH))


def find_window_box(result):
    if result.obb is None or len(result.obb) == 0:
        return None
    names = result.names
    xyxyxyxy = result.obb.xyxyxyxy.cpu().numpy()
    cls_ids = result.obb.cls.cpu().numpy().astype(int)
    best, best_area = None, -1
    for pts, cid in zip(xyxyxyxy, cls_ids):
        if names[cid] != "window":
            continue
        x0, y0 = pts[:, 0].min(), pts[:, 1].min()
        x1, y1 = pts[:, 0].max(), pts[:, 1].max()
        area = (x1 - x0) * (y1 - y0)
        if area > best_area:
            best_area, best = area, (x0, y0, x1, y1)
    return best


def pad_clip(box, w, h, pad_frac):
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    px, py = bw * pad_frac, bh * pad_frac
    return (max(0, int(x0 - px)), max(0, int(y0 - py)),
            min(w, int(x1 + px)), min(h, int(y1 + py)))


def letterbox(img, target):
    h, w = img.shape[:2]
    scale = min(target / w, target / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(img, (nw, nh))
    canvas = np.zeros((target, target, 3), dtype=img.dtype)
    px, py = (target - nw) // 2, (target - nh) // 2
    canvas[py:py + nh, px:px + nw] = resized
    return canvas


def pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def read_meter(image, stage1_model, stage2_model):
    timings = {}
    img = np.array(image.convert("RGB"))[:, :, ::-1].copy()
    h, w = img.shape[:2]

    t0 = time.time()
    r1 = stage1_model.predict(img, imgsz=640, conf=0.25, verbose=False)[0]
    timings["detect_meter"] = time.time() - t0
    annotated_full = Image.fromarray(r1.plot()[:, :, ::-1])

    window_box = find_window_box(r1)
    if window_box is None:
        return {
            "annotated_full": annotated_full,
            "annotated_crop": None,
            "status": "no_window",
            "message": "No digit window detected -- try a clearer, closer photo.",
            "table": None,
            "reading": None,
            "timings": timings,
        }

    x0, y0, x1, y1 = pad_clip(window_box, w, h, CROP_PAD)
    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        return {
            "annotated_full": annotated_full,
            "annotated_crop": None,
            "status": "crop_failed",
            "message": "Crop failed -- try a different photo.",
            "table": None,
            "reading": None,
            "timings": timings,
        }

    canvas = letterbox(crop, CROP_IMGSZ)
    t1 = time.time()
    r2 = stage2_model.predict(canvas, imgsz=CROP_IMGSZ, conf=0.25, verbose=False)[0]
    timings["recognize_digits"] = time.time() - t1
    annotated_crop = Image.fromarray(r2.plot()[:, :, ::-1])

    if r2.obb is None or len(r2.obb) == 0:
        return {
            "annotated_full": annotated_full,
            "annotated_crop": annotated_crop,
            "status": "no_digits",
            "message": "Window found, but no digits detected.",
            "table": None,
            "reading": None,
            "timings": timings,
        }

    t2 = time.time()
    xyxyxyxy = r2.obb.xyxyxyxy.cpu().numpy()
    cls_ids = r2.obb.cls.cpu().numpy().astype(int)
    confs = r2.obb.conf.cpu().numpy()

    digits = []
    for pts, cid, conf in zip(xyxyxyxy, cls_ids, confs):
        cx = pts[:, 0].mean()
        label = STAGE2_CLASSES[cid] if cid < len(STAGE2_CLASSES) else "?"
        digits.append((cx, label, float(conf)))
    digits.sort(key=lambda d: d[0])

    reading = "".join(d[1] if d[1] != "u" else "?" for d in digits)
    low_conf = any(d[2] < 0.5 for d in digits)
    has_unclear = any(d[1] == "u" for d in digits)
    timings["validate"] = time.time() - t2

    if has_unclear:
        status = "warn"
        message = "Contains an unclear digit (marked `?`) -- recommend manual check."
    elif low_conf:
        status = "warn"
        message = "Low confidence on one or more digits -- recommend manual check."
    else:
        status = "ok"
        message = "All digits detected with good confidence."

    table = pd.DataFrame(
        [{"Position": i + 1, "Digit": d[1], "Confidence": f"{d[2]:.1%}"} for i, d in enumerate(digits)]
    )

    return {
        "annotated_full": annotated_full,
        "annotated_crop": annotated_crop,
        "status": status,
        "message": message,
        "table": table,
        "reading": reading,
        "confidences": [d[2] for d in digits],
        "timings": timings,
    }


st.markdown(
    """
    <div class="mb-navbar">
        <span class="brand">Mbarira AI</span>
        <div class="links">
            <a href="#" class="active">Demo</a>
            <a href="#">Technology</a>
            <a href="#">About</a>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="mb-header">
        <h1>Meter Reading Analysis</h1>
        <p>Upload an image of a utility water meter to instantly extract readings using our
        two-stage YOLOv8-OBB detection pipeline.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.write("")

with st.spinner("Loading models..."):
    stage1_model, stage2_model = load_models()

if "result" not in st.session_state:
    st.session_state.result = None
if "uploaded_name" not in st.session_state:
    st.session_state.uploaded_name = None

col_upload, col_preview, col_results = st.columns([4, 5, 3], gap="large")

with col_upload:
    with st.container(border=True):
        st.markdown('<div class="mb-card-title">Upload Meter Image</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Drag & drop image here",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )
        st.caption("Supports JPG, PNG (Max 5MB)")
        reset = st.button("Analyze another", use_container_width=True)
        if reset:
            st.session_state.result = None
            st.session_state.uploaded_name = None
            st.rerun()

if uploaded is not None and uploaded.name != st.session_state.uploaded_name:
    image = Image.open(uploaded)
    with st.spinner("Analyzing meter..."):
        st.session_state.result = read_meter(image, stage1_model, stage2_model)
    st.session_state.uploaded_name = uploaded.name

result = st.session_state.result

with col_preview:
    with st.container(border=True):
        badge = (
            '<span class="mb-badge"><span class="dot"></span>PROCESSING</span>'
            if result is not None
            else '<span class="mb-badge" style="opacity:0.5;">IDLE</span>'
        )
        st.markdown(f'<div class="mb-card-title">Image Preview {badge}</div>', unsafe_allow_html=True)

        if result is None:
            st.markdown(
                '<div class="mb-preview-frame"><p class="mb-preview-empty">'
                'Upload a photo to get started. For best results, photograph the digit window '
                'straight-on, in good lighting, filling as much of the frame as possible.</p></div>',
                unsafe_allow_html=True,
            )
        else:
            b64_full = pil_to_b64(result["annotated_full"])
            st.markdown(
                f'<div class="mb-preview-frame"><img src="data:image/png;base64,{b64_full}"/>'
                f'<div class="mb-scanline"></div></div>',
                unsafe_allow_html=True,
            )
            if result["annotated_crop"] is not None:
                st.markdown('<p class="mb-label-caps" style="margin-top:14px;">Cropped digit window</p>', unsafe_allow_html=True)
                b64_crop = pil_to_b64(result["annotated_crop"])
                st.markdown(
                    f'<div class="mb-preview-frame" style="min-height:160px;">'
                    f'<img src="data:image/png;base64,{b64_crop}"/></div>',
                    unsafe_allow_html=True,
                )

with col_results:
    with st.container(border=True):
        st.markdown('<div class="mb-card-title">AI Results</div>', unsafe_allow_html=True)

        if result is None:
            st.markdown(
                '<p style="color:#3f4943;font-size:14px;">Results will appear here once an image is analyzed.</p>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown('<p class="mb-label-caps">Extracted Reading</p>', unsafe_allow_html=True)
            if result["reading"] is None:
                st.markdown(
                    f'<div class="mb-reading-box warn"><span class="value" style="font-size:16px;">'
                    f'{result["message"]}</span></div>',
                    unsafe_allow_html=True,
                )
            else:
                box_class = "mb-reading-box warn" if result["status"] == "warn" else "mb-reading-box"
                st.markdown(
                    f'<div class="{box_class}"><span class="value">{result["reading"]}</span></div>'
                    f'<p style="font-size:12px;color:#3f4943;margin-top:6px;">{result["message"]}</p>',
                    unsafe_allow_html=True,
                )

                confs = result.get("confidences", [])
                if confs:
                    bars = "".join(
                        f'<div class="bar" style="height:{max(8, c * 100)}%;" title="Digit {i+1}: {c:.1%}"></div>'
                        for i, c in enumerate(confs)
                    )
                    avg_conf = sum(confs) / len(confs)
                    st.markdown(
                        f"""
                        <div class="mb-metric-card" style="margin-top:16px;">
                            <p class="mb-label-caps" style="margin-bottom:8px;">Confidence Distribution</p>
                            <div class="mb-confbars">{bars}</div>
                            <div class="mb-conf-footer"><span>Avg: {avg_conf:.1%}</span></div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                total_time = sum(result["timings"].values())
                st.markdown(
                    f"""
                    <div class="mb-metric-card" style="margin-top:12px;">
                        <p class="mb-label-caps" style="margin-bottom:4px;">Total Time</p>
                        <p style="font-family:'JetBrains Mono',monospace;font-size:18px;">{total_time:.2f}s</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                step_labels = {
                    "detect_meter": ("Detect Meter", "YOLOv8-OBB"),
                    "recognize_digits": ("Recognize Digits", "YOLOv8-OBB"),
                    "validate": ("Validation", "Confidence check"),
                }
                items = "".join(
                    f'<li><p class="step-title">{step_labels.get(k, (k, ""))[0]}</p>'
                    f'<p class="step-meta">{step_labels.get(k, (k, ""))[1]} &middot; {v:.2f}s</p></li>'
                    for k, v in result["timings"].items()
                )
                st.markdown(
                    f"""
                    <p class="mb-label-caps" style="margin-top:16px;">Processing Workflow</p>
                    <ul class="mb-timeline">{items}</ul>
                    """,
                    unsafe_allow_html=True,
                )

                if result["table"] is not None:
                    with st.expander("Per-digit breakdown"):
                        st.dataframe(result["table"], use_container_width=True, hide_index=True)

                    csv = result["table"].to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download results (CSV)",
                        data=csv,
                        file_name="meter_reading.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

with st.expander("About this model"):
    st.markdown(
        "- **Stage 1** detects the meter body, the digit display window, and digits in the full photo.\n"
        "- **Stage 2** crops to the digit window and re-detects on the upscaled crop for tighter, "
        "more reliable digit reading.\n"
        "- `u` marks a digit the model found but couldn't classify confidently -- flagged for manual "
        "review rather than guessed.\n"
        "- Model weights are bundled directly in this repo's `models/` folder -- no external "
        "hosting or downloads required."
    )

st.markdown(
    '<div class="mb-footer">Mbarira AI Research Lab &middot; Precision computer vision for '
    'Rwanda water infrastructure.</div>',
    unsafe_allow_html=True,
)
