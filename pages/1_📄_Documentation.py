import pandas as pd
import streamlit as st

st.set_page_config(page_title="Mbarira AI - Documentation", page_icon="\U0001F4D8", layout="wide")

COLORS = {
    "primary": "#00543b",
    "primary_container": "#0b6e4f",
    "secondary_container": "#8fdfff",
    "on_secondary_container": "#00647d",
    "background": "#f8f9ff",
    "surface_container_lowest": "#ffffff",
    "surface_container_low": "#eff4ff",
    "on_surface": "#0b1c30",
    "on_surface_variant": "#3f4943",
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
[class*="css"] {{
    font-family: 'Inter', sans-serif !important;
    background-color: {COLORS["background"]} !important;
    color: {COLORS["on_surface"]} !important;
}}
[data-testid="stHeader"] {{ background-color: transparent !important; }}
[data-testid="stSidebar"] {{ background-color: {COLORS["surface_container_low"]} !important; }}
.block-container {{ padding-top: 1.5rem; max-width: 1280px; }}
#MainMenu, footer {{visibility: hidden;}}

.mb-navbar {{
    display: flex;
    align-items: center;
    gap: 20px;
    padding: 12px 4px 4px 4px;
    margin-bottom: 4px;
}}
.mb-navbar .brand {{
    font-family: 'Sora', sans-serif; font-weight: 700; font-size: 22px;
    color: {COLORS["on_surface"]} !important;
}}
.mb-navbar .current {{
    font-size: 15px; font-weight: 700; color: {COLORS["primary"]} !important;
    border-bottom: 2px solid {COLORS["primary"]}; padding-bottom: 4px;
}}
.mb-navlinks {{
    border-bottom: 1px solid {COLORS["outline_variant"]}55;
    margin-bottom: 24px;
    padding-bottom: 4px;
}}
[data-testid="stPageLink"] {{ width: auto !important; }}
[data-testid="stPageLink"] a {{
    text-decoration: none !important;
    background: transparent !important;
    border: none !important;
    padding: 4px 2px !important;
    font-size: 15px !important;
    color: {COLORS["on_surface_variant"]} !important;
}}
[data-testid="stPageLink"] a:hover {{ color: {COLORS["primary"]} !important; }}
[data-testid="stPageLink"] p {{ font-size: 15px !important; }}

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
    margin-bottom: 16px; color: {COLORS["on_surface"]} !important;
}}

.mb-timeline {{
    list-style: none; padding-left: 20px; margin: 0;
    border-left: 1px solid {COLORS["outline_variant"]};
}}
.mb-timeline li {{ position: relative; padding-bottom: 16px; }}
.mb-timeline li::before {{
    content: ""; position: absolute; left: -25px; top: 4px;
    width: 8px; height: 8px; border-radius: 999px;
    background: {COLORS["primary_container"]};
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
</style>
"""

st.markdown(BASE_CSS, unsafe_allow_html=True)


st.markdown('<div class="mb-navbar"><span class="brand">Mbarira AI</span></div>', unsafe_allow_html=True)
st.markdown('<div class="mb-navlinks">', unsafe_allow_html=True)
nav_cols = st.columns([1, 1, 8])
with nav_cols[0]:
    st.page_link("app.py", label="Demo", icon="🏠")
with nav_cols[1]:
    st.markdown('<span class="current">Documentation</span>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

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
                    <p class="step-title">2. Crop + letterbox to the window</p>
                    <p class="step-meta">tight crop, padded and resized to a square canvas</p>
                </li>
                <li>
                    <p class="step-title">3. Recognize digits</p>
                    <p class="step-meta">digit_detector.pt &middot; re-detects on the upscaled crop</p>
                </li>
                <li>
                    <p class="step-title">4. Clean detections</p>
                    <p class="step-meta">fits a line through the digit row (robust to a tilted photo),
                    then drops boxes off that line or overlapping another box along it -- using only
                    the geometry of what the model found, never a fixed digit count</p>
                </li>
                <li>
                    <p class="step-title">5. Build the reading</p>
                    <p class="step-meta">sorts surviving digits by position along the fitted row</p>
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
