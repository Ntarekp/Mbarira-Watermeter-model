import streamlit as st
from common import inject_css, render_navbar

st.set_page_config(page_title="Mbarira AI - Water Meter Reading", page_icon="\U0001F4A7", layout="wide", initial_sidebar_state="collapsed")
inject_css()
render_navbar("Home")

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

col_a, col_b, col_c = st.columns([1, 1, 1])
with col_a:
    if st.button("Try the Demo", type="primary", use_container_width=True):
        st.switch_page("pages/1_\U0001F52C_Demo.py")
with col_b:
    if st.button("View Technology", use_container_width=True):
        st.switch_page("pages/2_\u2699\uFE0F_Technology.py")
with col_c:
    if st.button("Read Documentation", use_container_width=True):
        st.switch_page("pages/3_\U0001F4C4_Documentation.py")

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

st.write("")
if st.button("See the full pipeline \u2192", use_container_width=False):
    st.switch_page("pages/2_\u2699\uFE0F_Technology.py")

st.markdown(
    '<div class="mb-footer">Mbarira AI Research Lab &middot; Precision computer vision for '
    'Rwanda water infrastructure.</div>',
    unsafe_allow_html=True,
)
