import streamlit as st

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
.block-container {{ padding-top: 1.5rem; max-width: 1280px; }}
#MainMenu, footer {{visibility: hidden;}}

.mb-navlinks {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-bottom: 1px solid {COLORS["outline_variant"]}55;
    margin-bottom: 24px;
    padding: 14px 4px;
}}
.mb-navlinks .brand-mark {{
    display: flex;
    align-items: center;
    gap: 8px;
}}
.mb-navlinks .brand {{
    font-family: 'Sora', sans-serif;
    font-weight: 700;
    font-size: 20px;
    color: {COLORS["on_surface"]} !important;
}}
.mb-navlinks .current {{
    font-size: 15px; font-weight: 700; color: {COLORS["primary"]} !important;
}}
[data-testid="stHorizontalBlock"]:has(.brand-mark) {{
    align-items: center !important;
}}
[data-testid="stSidebarNav"] {{ display: none; }}
[data-testid="stPageLink"] a {{ justify-content: flex-end; }}
[data-testid="stPageLink"] p {{ text-align: right; }}
[data-testid="stPageLink"] {{ width: auto !important; }}
[data-testid="stPageLink"] a {{
    text-decoration: none !important; background: transparent !important;
    border: none !important; padding: 4px 2px !important;
    font-size: 15px !important; color: {COLORS["on_surface_variant"]} !important;
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

.mb-hero {{
    text-align: center; max-width: 720px; margin: 0 auto 40px auto; padding-top: 24px;
}}
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


BRAND_MARK_SVG = """
<svg width="22" height="22" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
<path d="M12 2.5C12 2.5 5 11.2 5 15.5C5 19.5 8.13 22.5 12 22.5C15.87 22.5 19 19.5 19 15.5C19 11.2 12 2.5 12 2.5Z"
      fill="#0b6e4f" stroke="#00543b" stroke-width="1.2" stroke-linejoin="round"/>
<path d="M9.2 15.8C9.2 17.5 10.5 18.8 12 18.8" stroke="#98edc6" stroke-width="1.4"
      stroke-linecap="round" fill="none"/>
</svg>
"""


def render_navbar(active="Home"):
    """Single row: brand mark + wordmark on the left, real navigation
    (st.page_link, no icons) right-aligned on the same line. The current
    page renders as plain text instead of a link."""
    st.markdown('<div class="mb-navlinks">', unsafe_allow_html=True)
    cols = st.columns([5, 1, 1, 1, 1.3])
    with cols[0]:
        st.markdown(
            f'<div class="brand-mark">{BRAND_MARK_SVG}<span class="brand">Mbarira AI</span></div>',
            unsafe_allow_html=True,
        )
    pages = [
        ("Home", "app.py"),
        ("Demo", "pages/1_\U0001F52C_Demo.py"),
        ("Technology", "pages/2_\u2699\uFE0F_Technology.py"),
        ("Documentation", "pages/3_\U0001F4C4_Documentation.py"),
    ]
    for i, (label, path) in enumerate(pages, start=1):
        with cols[i]:
            if active == label:
                st.markdown(f'<div style="text-align:right;"><span class="current">{label}</span></div>', unsafe_allow_html=True)
            else:
                st.page_link(path, label=label)
    st.markdown('</div>', unsafe_allow_html=True)
