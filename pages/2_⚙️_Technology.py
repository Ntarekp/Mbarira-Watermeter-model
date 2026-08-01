import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).parent.parent))
from common import inject_css, render_navbar

st.set_page_config(page_title="Mbarira AI - Technology", page_icon="\u2699\uFE0F", layout="wide")
inject_css()
render_navbar("Technology")

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
        - **No confirmed handling of a fully upside-down (~180\u00b0) photo.** The deskew step
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
