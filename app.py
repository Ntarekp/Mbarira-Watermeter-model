import base64
import urllib.parse
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

APP_DIR = Path(__file__).parent
STAGE1_PATH = APP_DIR / "models" / "meter_detector.pt"
STAGE2_PATH = APP_DIR / "models" / "digit_detector.pt"

STAGE2_CLASSES = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "u"]
CROP_IMGSZ = 416

STAGE2_CONF = 0.35
STAGE2_IOU = 0.45

ROW_GAP_FRAC = 0.9
SPAN_OVERLAP_THRESH = 0.4

CROP_PAD_X_DIGITS = 0.8
CROP_PAD_Y_FRAC = 0.15

EXPECTED_DIGITS = None  # e.g. 8 -- purely a UI heads-up, never trims/pads the reading

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

:root {{ color-scheme: light !important; }}
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
[data-testid="stHeader"] {{ background-color: transparent !important; }}
[data-testid="stSidebar"] {{ background-color: {COLORS["surface_container_low"]} !important; }}

/* Extra top + right clearance so our own nav never renders underneath
   Streamlit Community Cloud's own floating toolbar (Share / Star / Edit /
   GitHub) -- that toolbar is platform chrome outside our app's DOM, so it
   can't be hidden from here; the fix is just not putting our content in
   the same screen region. */
.block-container {{
    padding-top: 3.5rem !important;
    padding-right: 2rem !important;
    max-width: 1280px;
}}
#MainMenu, footer {{visibility: hidden;}}

/* -- Top navbar: brand left, real page_link nav right, divider below -- */
.mb-topnav {{
    display: flex;
    align-items: center;
    padding-top: 4px;
    margin-top: 4px;
}}
.mb-brand {{
    font-family: 'Sora', sans-serif;
    font-weight: 700;
    font-size: 22px;
    letter-spacing: 0.02em;
    text-transform: uppercase;
    color: {COLORS["on_surface"]} !important;
    white-space: nowrap;
}}
.mb-navlink-active {{
    font-size: 15px;
    font-weight: 700;
    color: {COLORS["primary"]} !important;
    border-bottom: 2px solid {COLORS["primary"]};
    padding-bottom: 4px;
    display: inline-block;
    white-space: nowrap;
}}
[data-testid="stPageLink"] {{ width: auto !important; }}
[data-testid="stPageLink"] a {{
    text-decoration: none !important;
    background: transparent !important;
    border: none !important;
    padding: 4px 2px !important;
}}
[data-testid="stPageLink"] p {{
    font-size: 15px !important;
    color: {COLORS["on_surface_variant"]} !important;
    white-space: nowrap;
}}
[data-testid="stPageLink"]:hover p {{ color: {COLORS["primary"]} !important; }}
.mb-navdivider {{
    border: none;
    border-top: 1px solid {COLORS["outline_variant"]}88;
    margin: 8px 0 24px 0;
}}

.mb-header h1 {{
    font-family: 'Sora', sans-serif; font-size: 40px; line-height: 1.15;
    margin-bottom: 4px; color: {COLORS["on_surface"]} !important;
}}
.mb-header p {{
    font-size: 17px; color: {COLORS["on_surface_variant"]} !important; max-width: 640px;
}}

[data-testid="stVerticalBlockBorderWrapper"] {{
    background: {COLORS["surface_container_lowest"]} !important;
    border: 1px solid {COLORS["outline_variant"]} !important;
    border-radius: 12px !important;
    box-shadow: 0 1px 2px rgba(15,23,42,0.03);
}}
[data-testid="stVerticalBlockBorderWrapper"] * {{ color: {COLORS["on_surface"]}; }}
.mb-card-title {{
    font-family: 'Sora', sans-serif; font-size: 20px; font-weight: 700;
    margin-bottom: 16px; display: flex; align-items: center;
    justify-content: space-between; color: {COLORS["on_surface"]} !important;
}}

.mb-badge {{
    font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 700;
    letter-spacing: 0.05em; background: {COLORS["secondary_container"]}33;
    color: {COLORS["on_secondary_container"]}; padding: 4px 12px; border-radius: 999px;
    display: inline-flex; align-items: center; gap: 6px;
}}
.mb-badge .dot {{
    width: 8px; height: 8px; border-radius: 999px; background: {COLORS["primary"]};
    animation: mb-pulse 1.4s infinite;
}}
@keyframes mb-pulse {{ 0%, 100% {{ opacity: 1; }} 50% {{ opacity: 0.3; }} }}

[data-testid="stFileUploaderDropzone"] {{
    background: transparent !important;
    border: 2px dashed {COLORS["outline_variant"]} !important;
    border-radius: 10px !important; padding: 12px !important;
}}
[data-testid="stFileUploaderDropzone"]:hover {{ border-color: {COLORS["accent"]} !important; }}

.mb-preview-frame {{
    position: relative; width: 100%; min-height: 320px;
    background: {COLORS["surface_container_low"]};
    border: 1px solid {COLORS["outline_variant"]}; border-radius: 10px;
    overflow: hidden; display: flex; align-items: center; justify-content: center;
}}
.mb-preview-frame img {{ width: 100%; height: auto; display: block; }}
.mb-preview-empty {{
    color: {COLORS["on_surface_variant"]}; font-size: 14px;
    text-align: center; padding: 32px;
}}

.mb-scanline {{
    position: absolute; top: 0; left: 0; width: 100%; height: 2px;
    background: linear-gradient(90deg, transparent, {COLORS["accent"]}, transparent);
    box-shadow: 0 0 10px {COLORS["accent"]}, 0 0 20px {COLORS["accent"]};
    animation: mb-scan 2.2s infinite linear; opacity: 0.85;
}}
@keyframes mb-scan {{
    0% {{ top: 0%; opacity: 0; }} 10% {{ opacity: 1; }}
    90% {{ opacity: 1; }} 100% {{ top: 100%; opacity: 0; }}
}}

.mb-label-caps {{
    font-family: 'JetBrains Mono', monospace; font-size: 11px;
    letter-spacing: 0.05em; text-transform: uppercase;
    color: {COLORS["on_surface_variant"]}; margin-bottom: 6px;
}}

.mb-reading-box {{
    background: {COLORS["surface_container"]};
    border: 1px solid {COLORS["outline_variant"]}88; border-radius: 10px; padding: 16px;
}}
.mb-reading-box .value {{
    font-family: 'JetBrains Mono', monospace; font-size: 30px; font-weight: 700;
    letter-spacing: 0.12em; color: {COLORS["on_surface"]};
}}
.mb-reading-box.warn {{ border-color: {COLORS["error"]}55; }}
.mb-reading-box.warn .value {{ color: {COLORS["error"]}; }}

.mb-metric-card {{
    border: 1px solid {COLORS["outline_variant"]}55; border-radius: 10px;
    padding: 12px; background: {COLORS["surface_bright"]};
}}
.mb-confbars {{ display: flex; align-items: flex-end; gap: 3px; height: 32px; margin-bottom: 6px; }}
.mb-confbars .bar {{ flex: 1; background: {COLORS["primary_container"]}; border-radius: 2px 2px 0 0; }}
.mb-conf-footer {{
    display: flex; justify-content: space-between;
    font-family: 'JetBrains Mono', monospace; font-size: 10px;
    color: {COLORS["on_surface_variant"]};
}}

.mb-timeline {{ list-style: none; padding-left: 20px; margin: 0; border-left: 1px solid {COLORS["outline_variant"]}; }}
.mb-timeline li {{ position: relative; padding-bottom: 16px; }}
.mb-timeline li:last-child {{ padding-bottom: 0; color: {COLORS["primary_container"]}; }}
.mb-timeline li::before {{
    content: ""; position: absolute; left: -25px; top: 4px;
    width: 8px; height: 8px; border-radius: 999px; background: {COLORS["primary_container"]};
}}
.mb-timeline .step-title {{ font-size: 13px; font-weight: 600; color: {COLORS["on_surface"]}; }}
.mb-timeline .step-meta {{
    font-family: 'JetBrains Mono', monospace; font-size: 10px;
    color: {COLORS["on_surface_variant"]};
}}

.mb-footer {{
    margin-top: 48px; padding-top: 20px;
    border-top: 1px solid {COLORS["outline_variant"]};
    font-size: 12px; color: {COLORS["on_surface_variant"]}; text-align: center;
}}

div.stButton > button {{ border-radius: 8px; font-weight: 500; }}
div.stButton > button[kind="primary"] {{
    background-color: {COLORS["primary_container"]}; border-color: {COLORS["primary_container"]};
}}

div[data-testid="stRadio"] > div {{
    display: flex; gap: 6px; background: {COLORS["surface_container_low"]};
    padding: 4px; border-radius: 8px;
}}
div[data-testid="stRadio"] label {{
    flex: 1; justify-content: center; background: transparent; border-radius: 6px;
    padding: 6px 10px !important; margin: 0 !important; cursor: pointer;
}}
div[data-testid="stRadio"] label:has(input:checked) {{
    background: {COLORS["surface_container_lowest"]}; box-shadow: 0 1px 2px rgba(15,23,42,0.08);
}}
div[data-testid="stRadio"] input {{ display: none; }}

.mb-hero {{ text-align: center; max-width: 720px; margin: 0 auto 40px auto; padding-top: 8px; }}
.mb-hero h1 {{
    font-family: 'Sora', sans-serif; font-size: 44px; line-height: 1.15;
    color: {COLORS["on_surface"]}; margin-bottom: 12px;
}}
.mb-hero p {{ font-size: 18px; color: {COLORS["on_surface_variant"]}; }}
.mb-stat-value {{
    font-family: 'JetBrains Mono', monospace; font-size: 30px; font-weight: 700;
    color: {COLORS["primary_container"]};
}}
.mb-stat-label {{
    font-family: 'JetBrains Mono', monospace; font-size: 11px;
    letter-spacing: 0.05em; text-transform: uppercase;
    color: {COLORS["on_surface_variant"]}; margin-top: 4px;
}}
.mb-pipeline-num {{
    font-family: 'JetBrains Mono', monospace; font-size: 11px;
    letter-spacing: 0.05em; color: {COLORS["outline"]}; margin-bottom: 4px;
}}
.mb-caveat {{
    background: {COLORS["surface_container_low"]};
    border: 1px solid {COLORS["outline_variant"]}55;
    border-radius: 8px; padding: 10px 14px; font-size: 13px;
    color: {COLORS["on_surface_variant"]}; margin-top: 8px;
}}
</style>
"""


def inject_css():
    st.markdown(BASE_CSS, unsafe_allow_html=True)


def render_topnav(active):
    """Brand on the left, real working page_link navigation clustered to
    the right, divider line below. Coexists with Streamlit's own sidebar
    nav -- this is a second, additional way to navigate, not a replacement."""
    nav_items = [
        ("Home", home_page, ":material/home:"),
        ("Demo", demo_page, ":material/science:"),
        ("Technology", tech_page, ":material/memory:"),
        ("Documentation", docs_page, ":material/description:"),
    ]
    cols = st.columns([2, 2, 1, 1, 1, 1])
    with cols[0]:
        st.markdown('<span class="mb-brand">Mbarira AI</span>', unsafe_allow_html=True)
    # cols[1] is an empty spacer, pushing the links toward the right edge
    for col, (label, page, icon) in zip(cols[2:], nav_items):
        with col:
            if active == label:
                st.markdown(f'<span class="mb-navlink-active">{label}</span>', unsafe_allow_html=True)
            else:
                st.page_link(page, label=label, icon=icon)
    st.markdown('<hr class="mb-navdivider">', unsafe_allow_html=True)


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


def find_window_obb(result):
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
            best_area, best = area, pts
    return best


def deskew_window(img, poly, pad_x_digits=CROP_PAD_X_DIGITS, pad_y_frac=CROP_PAD_Y_FRAC):
    rect = cv2.minAreaRect(poly.astype(np.float32))
    (cx, cy), (w, h), angle = rect

    if w < h:
        w, h = h, w
        angle += 90.0

    h_img, w_img = img.shape[:2]
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rotated = cv2.warpAffine(
        img, M, (w_img, h_img), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )

    pad_x = h * pad_x_digits
    pad_y = h * pad_y_frac
    pw, ph = w + 2 * pad_x, h + 2 * pad_y

    x0 = int(max(0, cx - pw / 2))
    y0 = int(max(0, cy - ph / 2))
    x1 = int(min(w_img, cx + pw / 2))
    y1 = int(min(h_img, cy + ph / 2))
    return rotated[y0:y1, x0:x1]


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


def _cluster_rows(y_vals, median_h, gap_frac=ROW_GAP_FRAC):
    order = np.argsort(y_vals)
    sorted_y = y_vals[order]
    groups = [[int(order[0])]]
    for i in range(1, len(order)):
        if sorted_y[i] - sorted_y[i - 1] > gap_frac * median_h:
            groups.append([int(order[i])])
        else:
            groups[-1].append(int(order[i]))
    return groups


def clean_digit_detections(xyxyxyxy, cls_ids, confs):
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

    heights = np.array([c["h"] for c in candidates])
    median_h = float(np.median(heights)) or 1.0
    cy_vals = np.array([c["cy"] for c in candidates])

    row_groups = _cluster_rows(cy_vals, median_h)
    median_cy = float(np.median(cy_vals))
    row_groups.sort(key=lambda g: (-len(g), min(abs(cy_vals[i] - median_cy) for i in g)))

    kept, discarded = [], []
    for group in row_groups:
        row_candidates = sorted((candidates[i] for i in group), key=lambda c: c["conf"], reverse=True)
        row_kept = []
        for c in row_candidates:
            overlap_with = next((k for k in row_kept if _spans_overlap(c["span"], k["span"])), None)
            if overlap_with is None:
                row_kept.append(c)
            else:
                c["reason"] = f'duplicate of "{overlap_with["label"]}" on same slot'
                discarded.append(c)
        row_kept.sort(key=lambda c: c["cx"])
        kept.extend(row_kept)

    return kept, discarded


def draw_digit_boxes(canvas_bgr, kept, discarded, show_discarded=False):
    img = canvas_bgr.copy()

    if show_discarded:
        for c in discarded:
            pts = c["poly"].astype(int)
            cv2.polylines(img, [pts], isClosed=True, color=(150, 150, 150), thickness=1)
            x = int(c["span"][0])
            y = max(12, int(pts[:, 1].min()) - 4)
            cv2.putText(img, f'x {c["label"]}', (x, y),
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

    t0 = time.time()
    r1 = stage1_model.predict(img, imgsz=640, conf=0.25, verbose=False)[0]
    timings["detect_meter"] = time.time() - t0
    annotated_full = Image.fromarray(r1.plot()[:, :, ::-1])

    window_poly = find_window_obb(r1)
    if window_poly is None:
        return {
            "annotated_full": annotated_full, "crop_canvas": None,
            "kept_digits": [], "discarded_digits": [],
            "status": "no_window",
            "message": "No digit window detected -- try a clearer, closer photo.",
            "table": None, "reading": None, "timings": timings,
        }

    crop = deskew_window(img, window_poly)
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


def render_feedback_section():
    issue_title = "Feedback / Bug report"
    issue_body = "Describe what you saw and, if relevant, attach the photo you tested with.\n\n---\nApp: Mbarira AI water meter reader"
    issue_url = "https://github.com/Ntarekp/Mbarira-Watermeter-model/issues/new?" + urllib.parse.urlencode(
        {"title": issue_title, "body": issue_body}
    )
    discuss_url = "https://github.com/Ntarekp/Mbarira-Watermeter-model/discussions"

    with st.container(border=True):
        st.markdown('<div class="mb-card-title">Feedback &amp; Community</div>', unsafe_allow_html=True)
        st.markdown(
            "Spotted a bad reading, a bug, or have an idea? Every report helps improve the model."
        )
        col_a, col_b = st.columns(2)
        with col_a:
            st.link_button("Report an issue", issue_url, use_container_width=True)
        with col_b:
            st.link_button("Join the discussion", discuss_url, use_container_width=True)


def render_home():
    inject_css()
    render_topnav("Home")

    st.markdown(
        """
        <div class="mb-hero">
            <h1>Intelligence for Infrastructure</h1>
            <p>Upload a photo of a mechanical water meter and get an accurate digit reading in
            under a second -- a two-stage YOLOv8-OBB pipeline built for Rwanda water infrastructure.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        if st.button("Try the Demo", type="primary", use_container_width=True):
            st.switch_page(demo_page)
    with col_b:
        if st.button("View Technology", use_container_width=True):
            st.switch_page(tech_page)
    with col_c:
        if st.button("Read Documentation", use_container_width=True):
            st.switch_page(docs_page)

    st.write("")
    st.write("")

    st.markdown('<p class="mb-label-caps" style="text-align:center;">By the numbers</p>', unsafe_allow_html=True)
    stat_cols = st.columns(4)
    stats = [
        ("510", "Labeled training images"),
        ("0.995", "Stage 1 meter mAP50"),
        ("~0.97", "Stage 2 avg digit mAP50"),
        ("~0.1-0.2s", "Inference time, CPU"),
    ]
    for col, (value, label) in zip(stat_cols, stats):
        with col:
            with st.container(border=True):
                st.markdown(
                    f'<div style="text-align:center;">'
                    f'<div class="mb-stat-value">{value}</div>'
                    f'<div class="mb-stat-label">{label}</div></div>',
                    unsafe_allow_html=True,
                )

    st.markdown(
        '<p class="mb-caveat" style="text-align:center;">These are measured results on our test set '
        'and this deployment, not universal guarantees -- performance depends on photo lighting, angle, '
        'and meter condition. See Documentation for known limitations.</p>',
        unsafe_allow_html=True,
    )

    st.write("")
    st.write("")

    st.markdown('<div class="mb-card-title" style="justify-content:center;">How it works</div>', unsafe_allow_html=True)
    step_cols = st.columns(4)
    steps = [
        ("01", "Detect", "Locate the meter and digit window in the full photo (YOLOv8-OBB)."),
        ("02", "Deskew", "Straighten the digit row using the window's own detected rotation."),
        ("03", "Recognize", "Re-detect each digit on the deskewed, upscaled crop."),
        ("04", "Assemble", "Group digits by row and position, then build the final reading."),
    ]
    for col, (num, title, desc) in zip(step_cols, steps):
        with col:
            with st.container(border=True):
                st.markdown(f'<div class="mb-pipeline-num">STAGE {num}</div>', unsafe_allow_html=True)
                st.markdown(f"**{title}**")
                st.caption(desc)

    st.markdown(
        '<div class="mb-footer">Mbarira AI Research Lab &middot; Precision computer vision for '
        'Rwanda water infrastructure.</div>',
        unsafe_allow_html=True,
    )


def render_demo():
    inject_css()
    render_topnav("Demo")

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
                    st.markdown('<p class="mb-label-caps" style="margin-top:14px;">Cropped digit window (deskewed)</p>', unsafe_allow_html=True)
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


def render_technology():
    inject_css()
    render_topnav("Technology")

    st.markdown(
        """
        <div class="mb-header">
            <h1>The Vision Pipeline</h1>
            <p>A four-stage inference pipeline: two YOLOv8-OBB models, a geometric deskew step,
            and rule-based reading assembly -- no black-box heuristics.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    stage_cols = st.columns(2)
    with stage_cols[0]:
        with st.container(border=True):
            st.markdown('<div class="mb-pipeline-num">STAGE 01</div>', unsafe_allow_html=True)
            st.markdown("### Detect Meter + Window")
            st.markdown(
                "A YOLOv8-OBB model (`meter_detector.pt`) scans the full photo and detects the "
                "meter body, the digit display window, and rough digit locations, all as "
                "**oriented** bounding boxes -- meaning rotation is captured, not discarded."
            )
            st.caption("Measured mAP50 / mAP50-95 on held-out test images: meter 0.995 / 0.917, window 0.985 / 0.802.")

    with stage_cols[1]:
        with st.container(border=True):
            st.markdown('<div class="mb-pipeline-num">STAGE 02</div>', unsafe_allow_html=True)
            st.markdown("### Deskew")
            st.markdown(
                "The window's own detected rotation (via `cv2.minAreaRect`) is used to rotate the "
                "full image so the digit row becomes horizontal, then crop it -- instead of naively "
                "cropping a tilted region, which both wastes resolution and can scramble left-to-right "
                "reading order on an un-deskewed crop."
            )
            st.caption("Horizontal padding is sized relative to digit height, not the window box's own width, "
                       "so a tightly-detected box doesn't clip an edge digit.")

    stage_cols2 = st.columns(2)
    with stage_cols2[0]:
        with st.container(border=True):
            st.markdown('<div class="mb-pipeline-num">STAGE 03</div>', unsafe_allow_html=True)
            st.markdown("### Recognize Digits")
            st.markdown(
                "A second, dedicated YOLOv8-OBB model (`digit_detector.pt`) re-detects each digit "
                "on the deskewed, upscaled crop -- a tighter, cleaner input than the full photo, "
                "which is what recovers the tight-localization accuracy the first-stage model alone "
                "couldn't reach."
            )
            st.caption("11 classes: digits 0-9, plus `u` for a digit the model finds but can't classify confidently.")

    with stage_cols2[1]:
        with st.container(border=True):
            st.markdown('<div class="mb-pipeline-num">STAGE 04</div>', unsafe_allow_html=True)
            st.markdown("### Assemble Reading")
            st.markdown(
                "Detected digits are grouped into rows by y-position (supporting meters with a "
                "genuine second row, like a red sub-counter dial), duplicates on the same physical "
                "slot are dropped by confidence, and survivors are sorted left-to-right within each row."
            )
            st.caption("No fixed digit count is assumed -- genuine repeats like \"00\" pass through unaffected.")

    st.write("")
    st.write("")

    st.markdown('<div class="mb-card-title">Measured Performance</div>', unsafe_allow_html=True)
    perf_cols = st.columns(2)
    with perf_cols[0]:
        with st.container(border=True):
            st.markdown("**Stage 1 -- full photo**")
            stage1_summary = pd.DataFrame([
                {"Class": "meter", "mAP50": 0.995, "mAP50-95": 0.917},
                {"Class": "window", "mAP50": 0.985, "mAP50-95": 0.802},
                {"Class": "digits (avg 0-9)", "mAP50": 0.928, "mAP50-95": 0.723},
                {"Class": "u (unclear)", "mAP50": 0.767, "mAP50-95": 0.545},
            ])
            st.dataframe(stage1_summary, use_container_width=True, hide_index=True)

    with perf_cols[1]:
        with st.container(border=True):
            st.markdown("**Stage 2 -- cropped digit window**")
            stage2_summary = pd.DataFrame([
                {"Class": "digits (avg 0-9)", "mAP50": 0.971, "mAP50-95": 0.738},
                {"Class": "u (unclear)", "mAP50": 0.641, "mAP50-95": 0.422},
            ])
            st.dataframe(stage2_summary, use_container_width=True, hide_index=True)

    st.markdown(
        '<p class="mb-caveat">Stage 2 improves average digit mAP50-95 by roughly +9 points over Stage 1 alone, '
        'but the `u` (unclear) class regresses on the crop -- likely crop clipping or augmentation effects on '
        'this rare class. Full per-class numbers and dataset details are on the Documentation page.</p>',
        unsafe_allow_html=True,
    )

    st.write("")
    st.write("")

    st.markdown('<div class="mb-card-title">What this pipeline does NOT do</div>', unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown(
            """
            Being upfront about scope, since overclaiming here would just recreate the same
            false-confidence problem in marketing copy that we've been fixing in the reading logic:

            - **No cloud API yet.** Everything runs in this Streamlit app; there's no
              `api.mbarira.ai` endpoint today.
            - **No perspective/homography correction.** Deskewing is a 2D rotation, not a full
              perspective unwarp -- it fixes in-plane tilt, not photos taken from a sharp angle.
            - **No confirmed handling of a fully upside-down (~180°) photo.** The deskew step
              picks the smaller rotation to straighten a tilted row, which resolves the common
              case, but geometry alone can't tell "top" from "bottom" of a symmetric box beyond that.
            - **510 training images**, not millions -- a solid start for this meter model/region,
              but accuracy will vary on meter brands or conditions not well represented in it.
            """
        )

    st.markdown(
        '<div class="mb-footer">Mbarira AI Research Lab &middot; Precision computer vision for '
        'Rwanda water infrastructure.</div>',
        unsafe_allow_html=True,
    )


def render_documentation():
    inject_css()
    render_topnav("Documentation")

    st.markdown(
        """
        <div class="mb-header">
            <h1>Documentation</h1>
            <p>How the two-stage YOLOv8-OBB pipeline reads a water meter, what it's good at,
            and what to watch out for.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    col_left, col_right = st.columns([3, 2], gap="large")

    with col_left:
        with st.container(border=True):
            st.markdown('<div class="mb-card-title">How it works</div>', unsafe_allow_html=True)
            st.markdown(
                """
                <ul class="mb-timeline">
                    <li>
                        <p class="step-title">1. Detect meter + digit window</p>
                        <p class="step-meta">meter_detector.pt &middot; full photo, YOLOv8-OBB</p>
                    </li>
                    <li>
                        <p class="step-title">2. Deskew</p>
                        <p class="step-meta">rotate using the window's own detected angle, then crop</p>
                    </li>
                    <li>
                        <p class="step-title">3. Recognize digits</p>
                        <p class="step-meta">digit_detector.pt &middot; re-detects on the deskewed, upscaled crop</p>
                    </li>
                    <li>
                        <p class="step-title">4. Clean detections</p>
                        <p class="step-meta">groups by row (y-position), drops same-slot duplicates by confidence --
                        no fixed digit count assumed</p>
                    </li>
                    <li>
                        <p class="step-title">5. Build the reading</p>
                        <p class="step-meta">sorts surviving digits left-to-right within each row</p>
                    </li>
                </ul>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                "`u` marks a digit the model found but couldn't classify confidently -- "
                "flagged as `?` in the reading for manual review rather than guessed. "
                "This is a dedicated trained class (see Stage 2 performance below), not a "
                "confidence-threshold substitution."
            )
            st.caption("See the Technology page for a visual walkthrough of each stage.")

        with st.container(border=True):
            st.markdown('<div class="mb-card-title">Dataset</div>', unsafe_allow_html=True)
            st.markdown(
                "510 labeled water meter images, split 70 / 20 / 10 (train / val / test), "
                "with 13-class OBB annotations: `meter`, `window`, digits `0`-`9`, and `u` (unclear)."
            )

        with st.container(border=True):
            st.markdown('<div class="mb-card-title">Known limitations</div>', unsafe_allow_html=True)
            st.markdown(
                """
                - The `u` (unclear) class has the weakest mAP of any class in both stages --
                  expect occasional false "unclear" flags on worn/faded meter windows.
                - Deskewing corrects in-plane rotation (a tilted photo), not full perspective
                  distortion or a genuinely upside-down (~180°) photo.
                - Performance depends on lighting and camera angle; photograph the digit
                  window straight-on and well-lit for best results.
                - Trained on 510 images total -- a relatively small dataset, so accuracy may
                  drop on meter brands/models not well represented in training.
                """
            )

    with col_right:
        with st.container(border=True):
            st.markdown('<div class="mb-card-title">Model files</div>', unsafe_allow_html=True)
            files_df = pd.DataFrame([
                {"File": "models/meter_detector.pt", "Role": "Stage 1 -- meter body + digit window (OBB)"},
                {"File": "models/digit_detector.pt", "Role": "Stage 2 -- individual digits on cropped window (OBB)"},
            ])
            st.dataframe(files_df, use_container_width=True, hide_index=True)
            st.caption(
                "Both files are committed directly under models/ -- no external hosting required. "
                "If either grows past ~95 MB, switch to Git LFS before pushing."
            )

        with st.container(border=True):
            st.markdown('<div class="mb-card-title">Stage 1 performance</div>', unsafe_allow_html=True)
            stage1_df = pd.DataFrame([
                {"Class": "meter", "mAP50": 0.995, "mAP50-95": 0.917},
                {"Class": "window", "mAP50": 0.985, "mAP50-95": 0.802},
                {"Class": "0", "mAP50": 0.990, "mAP50-95": 0.766},
                {"Class": "1", "mAP50": 0.903, "mAP50-95": 0.711},
                {"Class": "2", "mAP50": 0.956, "mAP50-95": 0.734},
                {"Class": "3", "mAP50": 0.925, "mAP50-95": 0.736},
                {"Class": "4", "mAP50": 0.874, "mAP50-95": 0.677},
                {"Class": "5", "mAP50": 0.926, "mAP50-95": 0.712},
                {"Class": "6", "mAP50": 0.915, "mAP50-95": 0.684},
                {"Class": "7", "mAP50": 0.945, "mAP50-95": 0.754},
                {"Class": "8", "mAP50": 0.941, "mAP50-95": 0.737},
                {"Class": "9", "mAP50": 0.958, "mAP50-95": 0.722},
                {"Class": "u", "mAP50": 0.767, "mAP50-95": 0.545},
            ])
            st.dataframe(stage1_df, use_container_width=True, hide_index=True)

        with st.container(border=True):
            st.markdown('<div class="mb-card-title">Stage 2 performance</div>', unsafe_allow_html=True)
            stage2_df = pd.DataFrame([
                {"Class": "0", "mAP50": 0.994, "mAP50-95": 0.780},
                {"Class": "1", "mAP50": 0.917, "mAP50-95": 0.705},
                {"Class": "2", "mAP50": 0.989, "mAP50-95": 0.769},
                {"Class": "3", "mAP50": 0.960, "mAP50-95": 0.740},
                {"Class": "4", "mAP50": 0.994, "mAP50-95": 0.773},
                {"Class": "5", "mAP50": 0.990, "mAP50-95": 0.717},
                {"Class": "6", "mAP50": 0.978, "mAP50-95": 0.741},
                {"Class": "7", "mAP50": 0.983, "mAP50-95": 0.739},
                {"Class": "8", "mAP50": 0.954, "mAP50-95": 0.725},
                {"Class": "9", "mAP50": 0.952, "mAP50-95": 0.687},
                {"Class": "u", "mAP50": 0.641, "mAP50-95": 0.422},
            ])
            st.dataframe(stage2_df, use_container_width=True, hide_index=True)
            st.caption(
                "Stage 2 improves digit recall by +9 pts on average over Stage 1, but the `u` class "
                "regresses -- likely crop clipping or augmentation effects on this rare class."
            )

    render_feedback_section()

    st.markdown(
        '<div class="mb-footer">Mbarira AI Research Lab &middot; Precision computer vision for '
        'Rwanda water infrastructure.</div>',
        unsafe_allow_html=True,
    )

st.set_page_config(page_title="Mbarira AI - Water Meter Reading", page_icon=":material/water_drop:", layout="wide")

home_page = st.Page(render_home, title="Home", icon=":material/home:", default=True)
demo_page = st.Page(render_demo, title="Demo", icon=":material/science:")
tech_page = st.Page(render_technology, title="Technology", icon=":material/memory:")
docs_page = st.Page(render_documentation, title="Documentation", icon=":material/description:")

# No position= argument -> defaults to "sidebar", so Streamlit's own page
# navigation (with the real Material icons set above) is back, alongside
# the custom top bar rendered by render_topnav() inside each page.
pg = st.navigation([home_page, demo_page, tech_page, docs_page])
pg.run()
