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

from theme import COLORS, BASE_CSS, render_navbar

st.set_page_config(page_title="Mbarira AI - Water Meter Reading", page_icon="\U0001F4A7", layout="wide")

APP_DIR = Path(__file__).parent
STAGE1_PATH = APP_DIR / "models" / "meter_detector.pt"
STAGE2_PATH = APP_DIR / "models" / "digit_detector.pt"

STAGE2_CLASSES = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "u"]
CROP_IMGSZ = 416
CROP_PAD = 0.12

STAGE2_CONF = 0.35
STAGE2_IOU = 0.45

ROW_Y_TOLERANCE = 0.6
ROW_HEIGHT_TOLERANCE = 3.0
SPAN_OVERLAP_THRESH = 0.4

# Optional: purely a UI heads-up, never trims/pads the actual reading.
EXPECTED_DIGITS = None  # e.g. 8

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


# ---------------------------------------------------------------------------
# Digit-detection cleanup
# ---------------------------------------------------------------------------
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


def clean_digit_detections(xyxyxyxy, cls_ids, confs):
    """Returns (kept, discarded) -- both lists of dicts with poly/label/conf/
    geometry. `kept` is what becomes the actual reading. `discarded` is kept
    around purely so the UI can optionally show what got filtered and why,
    for auditing -- it is never used to build the reading itself.

    Filtering uses only the geometry of what the model detected (median
    height/position of the row, and physical-slot overlap) -- never a fixed
    digit count, so genuine repeats like "00" pass straight through."""
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

    heights = np.array([c["h"] for c in candidates])
    cys = np.array([c["cy"] for c in candidates])
    median_h = float(np.median(heights))
    median_cy = float(np.median(cys))
    mad_h = float(np.median(np.abs(heights - median_h))) or (median_h * 0.15 or 1.0)

    in_row, off_row = [], []
    for c in candidates:
        if (abs(c["h"] - median_h) <= ROW_HEIGHT_TOLERANCE * mad_h
                and abs(c["cy"] - median_cy) <= ROW_Y_TOLERANCE * median_h):
            in_row.append(c)
        else:
            c["reason"] = "off digit row"
            off_row.append(c)

    if not in_row:
        in_row, off_row = candidates, []

    in_row.sort(key=lambda c: c["conf"], reverse=True)
    kept, slot_discarded = [], []
    for c in in_row:
        overlap_with = next((k for k in kept if _spans_overlap(c["span"], k["span"])), None)
        if overlap_with is None:
            kept.append(c)
        else:
            c["reason"] = f'duplicate of "{overlap_with["label"]}" on same slot'
            slot_discarded.append(c)

    kept.sort(key=lambda c: c["cx"])
    discarded = off_row + slot_discarded
    return kept, discarded


def draw_digit_boxes(canvas_bgr, kept, discarded, show_discarded=False):
    """Draws ONLY the boxes that made it into the final reading (green),
    matching the digit count exactly. If show_discarded=True, also draws
    filtered-out boxes in gray with an 'x' prefix and the reason, for
    debugging/auditing -- never shown by default."""
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
        status, message = "warn", "Contains an unclear digit (marked `?`) -- recommend manual check."
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


# ---------------------------------------------------------------------------
st.markdown(render_navbar("Demo"), unsafe_allow_html=True)

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
                        "Gray boxes = detections filtered out before the reading was built "
                        "(shown for debugging only, never counted)."
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
