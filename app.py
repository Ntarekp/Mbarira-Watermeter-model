import streamlit as st
import numpy as np
from PIL import Image
import cv2
import pandas as pd
from pathlib import Path
from ultralytics import YOLO

st.set_page_config(page_title="Water Meter Reading System", page_icon="\U0001F4A7", layout="wide")

APP_DIR = Path(__file__).parent
STAGE1_PATH = APP_DIR / "models" / "meter_detector.pt"
STAGE2_PATH = APP_DIR / "models" / "digit_detector.pt"

STAGE2_CLASSES = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "u"]
CROP_IMGSZ = 416
CROP_PAD = 0.12


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


def read_meter(image, stage1_model, stage2_model):
    img = np.array(image.convert("RGB"))[:, :, ::-1].copy()
    h, w = img.shape[:2]

    r1 = stage1_model.predict(img, imgsz=640, conf=0.25, verbose=False)[0]
    annotated_full = Image.fromarray(r1.plot()[:, :, ::-1])

    window_box = find_window_box(r1)
    if window_box is None:
        return annotated_full, None, "No digit window detected -- try a clearer, closer photo.", None

    x0, y0, x1, y1 = pad_clip(window_box, w, h, CROP_PAD)
    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        return annotated_full, None, "Crop failed -- try a different photo.", None

    canvas = letterbox(crop, CROP_IMGSZ)
    r2 = stage2_model.predict(canvas, imgsz=CROP_IMGSZ, conf=0.25, verbose=False)[0]
    annotated_crop = Image.fromarray(r2.plot()[:, :, ::-1])

    if r2.obb is None or len(r2.obb) == 0:
        return annotated_full, annotated_crop, "Window found, but no digits detected.", None

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

    if has_unclear:
        status = f"Reading: **{reading}** -- contains an unclear digit (marked `?`), recommend manual check."
    elif low_conf:
        status = f"Reading: **{reading}** -- low confidence on one or more digits, recommend manual check."
    else:
        status = f"Reading: **{reading}** -- all digits detected with good confidence."

    table = pd.DataFrame(
        [{"Position": i + 1, "Digit": d[1], "Confidence": f"{d[2]:.1%}"} for i, d in enumerate(digits)]
    )
    return annotated_full, annotated_crop, status, table


st.title("\U0001F4A7 Water Meter Reading System (WMRS)")
st.caption(
    "Two-stage YOLOv8-OBB pipeline -- detects the digit display, crops it, then reads each digit. "
    "Built for Rwanda water infrastructure meter reading automation."
)

with st.spinner("Loading models..."):
    stage1_model, stage2_model = load_models()

uploaded = st.file_uploader("Upload a water meter photo", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    image = Image.open(uploaded)
    with st.spinner("Reading meter..."):
        annotated_full, annotated_crop, status, table = read_meter(image, stage1_model, stage2_model)

    col1, col2 = st.columns(2)
    with col1:
        st.image(annotated_full, caption="Full detection", use_container_width=True)
    with col2:
        if annotated_crop is not None:
            st.image(annotated_crop, caption="Cropped digit window", use_container_width=True)

    st.markdown(f"### {status}")
    if table is not None:
        st.dataframe(table, use_container_width=True, hide_index=True)
else:
    st.info("Upload a photo to get started. For best results, photograph the digit window "
            "straight-on, in good lighting, filling as much of the frame as possible.")

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
