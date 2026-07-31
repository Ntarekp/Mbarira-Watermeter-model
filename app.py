import base64
import hashlib
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

STAGE2_CONF = 0.35
STAGE2_IOU = 0.45

ROW_GAP_FRAC = 0.9   # gap (x median digit height) that starts a new row
SPAN_OVERLAP_THRESH = 0.4

# Optional: purely a UI heads-up, never trims/pads the actual reading.
EXPECTED_DIGITS = None  # e.g. 8

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
[data-testid="stSidebar"] {{
    background-color: {COLORS["surface_container_low"]} !important;
}}

.block-container {{
    padding-top: 1.5rem;
    max-width: 1280px;
}}

#MainMenu, footer {{visibility: hidden;}}

.mb-navbar {{
    display: flex;
    align-items: center;
    gap: 20px;
    padding: 12px 4px 4px 4px;
    margin-bottom: 4px;
}}
.mb-navbar .brand {{
    font-family: 'Sora', sans-serif;
    font-weight: 700;
    font-size: 22px;
    color: {COLORS["on_surface"]} !important;
}}
.mb-navbar .current {{
    font-size: 15px;
    font-weight: 700;
    color: {COLORS["primary"]} !important;
    border-bottom: 2px solid {COLORS["primary"]};
    padding-bottom: 4px;
}}

.mb-navlinks {{
    border-bottom: 1px solid {COLORS["outline_variant"]}55;
    margin-bottom: 24px;
    padding-bottom: 4px;
}}
[data-testid="stPageLink"] {{
    width: auto !important;
}}
[data-testid="stPageLink"] a {{
    text-decoration: none !important;
    background: transparent !important;
    border: none !important;
    padding: 4px 2px !important;
    font-size: 15px !important;
    color: {COLORS["on_surface_variant"]} !important;
}}
[data-testid="stPageLink"] a:hover {{
    color: {COLORS["primary"]} !important;
}}
[data-testid="stPageLink"] p {{
    font-size: 15px !important;
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

div[data-testid="stRadio"] > div {{
    display: flex;
    gap: 6px;
    background: {COLORS["surface_container_low"]};
    padding: 4px;
    border-radius: 8px;
}}
div[data-testid="stRadio"] label {{
    flex: 1;
    justify-content: center;
    background: transparent;
    border-radius: 6px;
    padding: 6px 10px !important;
    margin: 0 !important;
    cursor: pointer;
}}
div[data-testid="stRadio"] label:has(input:checked) {{
    background: {COLORS["surface_container_lowest"]};
    box-shadow: 0 1px 2px rgba(15,23,42,0.08);
}}
div[data-testid="stRadio"] input {{
    display: none;
}}
</style>
"""

st.markdown(BASE_CSS, unsafe_allow_html=True)


def render_navbar(active="Demo"):
    """Real, clickable navigation. The only page that is NOT a link is the
    one you're currently on (a self-link is a no-op and was also the one
    combination that crashed Streamlit's page registry). Every other page
    link is a genuine st.page_link -- no silent fallback, no downgrade."""
    st.markdown(
        '<div class="mb-navbar"><span class="brand">Mbarira AI</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="mb-navlinks">', unsafe_allow_html=True)
    nav_cols = st.columns([1, 1, 8])
    with nav_cols[0]:
        if active == "Demo":
            st.markdown('<span class="current">Demo</span>', unsafe_allow_html=True)
        else:
            st.page_link("app.py", label="Demo", icon="🏠")
    with nav_cols[1]:
        if active == "Documentation":
            st.markdown('<span class="current">Documentation</span>', unsafe_allow_html=True)
        else:
            st.page_link("pages/1_📄_Documentation.py", label="Documentation", icon="📄")
    st.markdown('</div>', unsafe_allow_html=True)


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


def _box_geometry(pts):
    cx = pts[:, 0].mean()
    cy = pts[:, 1].mean()
    h = pts[:, 1].max() - pts[:, 1].min()
    span = (pts[:, 0].min(), pts[:, 0].max())
    return cx, cy, h, span


def _spans_overlap(a, b, overlap_frac=SPAN_OVERLAP_THRESH):
    ax0, ax1 = a
    bx0, bx1 = b
    inter = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    union = max(ax1, bx1) - min(ax0, bx0)
    return union > 0 and (inter / union) > overlap_frac


def _fit_row_line(centers):
    """Fit a line through the digit-box centers with SVD (total least
    squares) instead of assuming the row is horizontal, so a tilted photo
    doesn't get its real digits mis-filtered as off-row or duplicates."""
    pts = np.array(centers, dtype=float)
    origin = pts.mean(axis=0)
    centered = pts - origin
    _, _, vt = np.linalg.svd(centered)
    direction = vt[0] / np.linalg.norm(vt[0])
    perp = np.array([-direction[1], direction[0]])
    return origin, direction, perp


def _cluster_rows(perp_d, median_h, gap_frac=ROW_GAP_FRAC):
    """Group candidate boxes into rows by clustering their (signed)
    perpendicular offset from the fitted reference line. A gap bigger than
    gap_frac * median_h between consecutive offsets starts a new row. This
    is what lets a meter with a genuine second row -- e.g. a red
    sub-counter dial next to the main black digit wheels -- be recognized
    as a second row of real digits, instead of everything off the single
    fitted line being discarded as noise."""
    order = np.argsort(perp_d)
    sorted_d = perp_d[order]
    groups = [[int(order[0])]]
    for i in range(1, len(order)):
        if sorted_d[i] - sorted_d[i - 1] > gap_frac * median_h:
            groups.append([int(order[i])])
        else:
            groups[-1].append(int(order[i]))
    return groups


def clean_digit_detections(xyxyxyxy, cls_ids, confs):
    """Returns (kept, discarded). `kept` becomes the reading, ordered as:
    the main row (the cluster with the most detections) left-to-right,
    followed by any secondary row(s) -- e.g. a red sub-counter dial --
    also left-to-right. A box is discarded ONLY if it's a true duplicate:
    another box on the same physical slot within the same row. A box is
    never discarded just for sitting in a different row than the majority
    -- multi-row meters are real, and treating a second row as noise
    silently drops genuine digits."""
    if len(cls_ids) == 0:
        return [], []

    candidates = []
    for pts, cid, conf in zip(xyxyxyxy, cls_ids, confs):
        cx, cy, h, span = _box_geometry(pts)
        label = STAGE2_CLASSES[cid] if cid < len(STAGE2_CLASSES) else "?"
        candidates.append({
            "poly": pts, "cx": cx, "cy": cy, "h": h, "span": span,
            "label": label, "conf": float(conf), "reason": None,
        })

    if len(candidates) == 1:
        return candidates, []

    origin, direction, perp = _fit_row_line([(c["cx"], c["cy"]) for c in candidates])

    heights = np.array([c["h"] for c in candidates])
    median_h = float(np.median(heights)) or 1.0

    perp_d = np.array([
        (c["cx"] - origin[0]) * perp[0] + (c["cy"] - origin[1]) * perp[1]
        for c in candidates
    ])

    row_groups = _cluster_rows(perp_d, median_h)
    # Main row = whichever cluster has the most digits; ties broken by
    # closeness to the fitted reference line.
    row_groups.sort(key=lambda g: (-len(g), min(abs(perp_d[i]) for i in g)))

    for c in candidates:
        proj = (c["poly"][:, 0] - origin[0]) * direction[0] + (c["poly"][:, 1] - origin[1]) * direction[1]
        c["t_span"] = (float(proj.min()), float(proj.max()))
        c["t"] = float((proj.min() + proj.max()) / 2)

    kept, discarded = [], []
    for group in row_groups:
        row_candidates = sorted((candidates[i] for i in group), key=lambda c: c["conf"], reverse=True)
        row_kept = []
        for c in row_candidates:
            overlap_with = next((k for k in row_kept if _spans_overlap(c["t_span"], k["t_span"])), None)
            if overlap_with is None:
                row_kept.append(c)
            else:
                c["reason"] = f'duplicate of "{overlap_with["label"]}" on same slot'
                discarded.append(c)
        row_kept.sort(key=lambda c: c["t"])
        kept.extend(row_kept)

    return kept, discarded


def draw_digit_boxes(canvas_bgr, kept, discarded, show_discarded=False):
    """Draws ONLY the boxes that made it into the final reading (green),
    so the box count always equals the reading length. If show_discarded
    is on, filtered-out boxes are also drawn in gray with an 'x' prefix
    and the reason, for auditing -- never shown by default."""
    img = canvas_bgr.copy()

    if show_discarded:
        for c in discarded:
            pts = c["poly"].astype(int)
            cv2.polylines(img, [pts], isClosed=True, color=(150, 150, 150), thickness=1)
            x = int(c["span"][0])
            y = max(12, int(pts[:, 1].min()) - 4)
            cv2.putText(img, f'x {c["label"]} ({c["reason"]})', (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)

    for c in kept:
        pts = c["poly"].astype(int)
        cv2.polylines(img, [pts], isClosed=True, color=(60, 200, 130), thickness=2)
        x = int(c["span"][0])
        y = max(12, int(pts[:, 1].min()) - 4)
        cv2.putText(img, f'{c["label"]} {c["conf"]:.2f}', (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 200, 130), 1, cv2.LINE_AA)

    return img


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
            "annotated_full": annotated_full, "crop_canvas": None,
            "kept_digits": [], "discarded_digits": [],
            "status": "no_window",
            "message": "No digit window detected -- try a clearer, closer photo.",
            "table": None, "reading": None, "timings": timings,
        }

    x0, y0, x1, y1 = pad_clip(window_box, w, h, CROP_PAD)
    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        return {
            "annotated_full": annotated_full, "crop_canvas": None,
            "kept_digits": [], "discarded_digits": [],
            "status": "crop_failed",
            "message": "Crop failed -- try a different photo.",
            "table": None, "reading": None, "timings": timings,
        }

    canvas = letterbox(crop, CROP_IMGSZ)
    t1 = time.time()
    r2 = stage2_model.predict(canvas, imgsz=CROP_IMGSZ, conf=STAGE2_CONF, iou=STAGE2_IOU, verbose=False)[0]
    timings["recognize_digits"] = time.time() - t1

    if r2.obb is None or len(r2.obb) == 0:
        return {
            "annotated_full": annotated_full, "crop_canvas": canvas,
            "kept_digits": [], "discarded_digits": [],
            "status": "no_digits",
            "message": "Window found, but no digits detected.",
            "table": None, "reading": None, "timings": timings,
        }

    t2 = time.time()
    xyxyxyxy = r2.obb.xyxyxyxy.cpu().numpy()
    cls_ids = r2.obb.cls.cpu().numpy().astype(int)
    confs = r2.obb.conf.cpu().numpy()

    kept, discarded = clean_digit_detections(xyxyxyxy, cls_ids, confs)

    reading = "".join(c["label"] if c["label"] != "u" else "?" for c in kept)
    low_conf = any(c["conf"] < 0.5 for c in kept)
    has_unclear = any(c["label"] == "u" for c in kept)
    count_mismatch = EXPECTED_DIGITS is not None and len(kept) != EXPECTED_DIGITS
    timings["validate"] = time.time() - t2

    if has_unclear:
        status, message = "warn", "Contains a digit the model flagged as genuinely unclear (marked `?`) -- recommend manual check."
    elif low_conf:
        status, message = "warn", "Low confidence on one or more digits -- recommend manual check."
    elif count_mismatch:
        status, message = "warn", f"Detected {len(kept)} digits, expected {EXPECTED_DIGITS} -- check the photo/crop."
    else:
        status, message = "ok", "All digits detected with good confidence."

    table = pd.DataFrame(
        [{"Position": i + 1, "Digit": c["label"], "Confidence": f'{c["conf"]:.1%}'} for i, c in enumerate(kept)]
    )

    return {
        "annotated_full": annotated_full,
        "crop_canvas": canvas,
        "kept_digits": kept,
        "discarded_digits": discarded,
        "status": status,
        "message": message,
        "table": table,
        "reading": reading,
        "confidences": [c["conf"] for c in kept],
        "timings": timings,
    }


render_navbar("Demo")

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
st.page_link("pages/1_📄_Documentation.py", label="📄  View Documentation", icon=None)
st.write("")

with st.spinner("Loading models..."):
    stage1_model, stage2_model = load_models()

if "result" not in st.session_state:
    st.session_state.result = None
if "uploaded_key" not in st.session_state:
    st.session_state.uploaded_key = None

col_upload, col_preview, col_results = st.columns([4, 5, 3], gap="large")

with col_upload:
    with st.container(border=True):
        st.markdown('<div class="mb-card-title">Upload Meter Image</div>', unsafe_allow_html=True)

        input_mode = st.radio(
            "Source", ["Upload photo", "Use camera"],
            horizontal=True, label_visibility="collapsed", key="input_mode",
        )

        picked_file = None
        if input_mode == "Upload photo":
            picked_file = st.file_uploader(
                "Drag & drop image here", type=["jpg", "jpeg", "png"], label_visibility="collapsed",
            )
            st.caption("Supports JPG, PNG (Max 5MB)")
        else:
            picked_file = st.camera_input("Take a photo of the meter", label_visibility="collapsed")
            st.caption("On phones this opens your camera directly -- hold the meter's digit window steady and well-lit.")

        show_discarded = st.checkbox("Show filtered-out detections (debug)", value=False)

        reset = st.button("Analyze another", use_container_width=True)
        if reset:
            st.session_state.result = None
            st.session_state.uploaded_key = None
            st.rerun()

if picked_file is not None:
    file_bytes = picked_file.getvalue()
    file_key = hashlib.md5(file_bytes).hexdigest()
    if file_key != st.session_state.uploaded_key:
        image = Image.open(picked_file)
        with st.spinner("Analyzing meter..."):
            st.session_state.result = read_meter(image, stage1_model, stage2_model)
        st.session_state.uploaded_key = file_key

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
            if result["crop_canvas"] is not None:
                st.markdown('<p class="mb-label-caps" style="margin-top:14px;">Cropped digit window</p>', unsafe_allow_html=True)
                annotated_crop = draw_digit_boxes(
                    result["crop_canvas"], result["kept_digits"], result["discarded_digits"],
                    show_discarded=show_discarded,
                )
                crop_pil = Image.fromarray(annotated_crop[:, :, ::-1])
                b64_crop = pil_to_b64(crop_pil)
                st.markdown(
                    f'<div class="mb-preview-frame" style="min-height:160px;">'
                    f'<img src="data:image/png;base64,{b64_crop}"/></div>',
                    unsafe_allow_html=True,
                )
                if show_discarded and result["discarded_digits"]:
                    st.caption(
                        "Gray boxes = detections filtered out before the reading was built, with "
                        "the reason each was dropped -- shown for debugging only, never counted."
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
                        "Download results (CSV)", data=csv, file_name="meter_reading.csv",
                        mime="text/csv", use_container_width=True,
                    )

st.markdown(
    '<div class="mb-footer">Mbarira AI Research Lab &middot; Precision computer vision for '
    'Rwanda water infrastructure.</div>',
    unsafe_allow_html=True,
)
