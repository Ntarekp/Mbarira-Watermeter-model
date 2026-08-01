import base64
import hashlib
import io
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

sys.path.append(str(Path(__file__).parent.parent))
from common import inject_css, render_navbar, COLORS

st.set_page_config(page_title="Mbarira AI - Water Meter Reading", page_icon="\U0001F4A7", layout="wide", initial_sidebar_state="collapsed")
inject_css()

APP_DIR = Path(__file__).parent.parent
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

EXPECTED_DIGITS = None


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
            "t_span": (float(pts[:, 0].min()), float(pts[:, 0].max())),
            "t": cx,
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
