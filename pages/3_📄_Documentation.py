import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).parent.parent))
from common import inject_css, render_navbar

st.set_page_config(page_title="Mbarira AI - Documentation", page_icon="\U0001F4D8", layout="wide", initial_sidebar_state="collapsed")
inject_css()
render_navbar("Documentation")

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
              distortion or a genuinely upside-down (~180\u00b0) photo.
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

st.markdown(
    '<div class="mb-footer">Mbarira AI Research Lab &middot; Precision computer vision for '
    'Rwanda water infrastructure.</div>',
    unsafe_allow_html=True,
)
