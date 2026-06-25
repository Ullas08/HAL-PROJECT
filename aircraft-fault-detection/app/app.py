"""
app/app.py
==========
Streamlit dashboard for the Aircraft Equipment Fault Detection system.
Developed for HAL – Helicopter Division (Phase 5).

Run with:
    streamlit run app/app.py
"""

import io
import os
import sys
import json
import time
import base64

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ── Make src importable when launched from project root ──────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ─────────────────────────────────────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Aircraft Fault Detection | HAL",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# Theme state
# ─────────────────────────────────────────────────────────────────────────────
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

is_dark = st.session_state.theme == "dark"

# ─────────────────────────────────────────────────────────────────────────────
# Theme tokens
# ─────────────────────────────────────────────────────────────────────────────
if is_dark:
    T = {
        "bg":            "linear-gradient(135deg, #080C18 0%, #0D1225 50%, #080C18 100%)",
        "bg_solid":      "#0D1225",
        "surface":       "linear-gradient(135deg, #111827 0%, #1A2236 100%)",
        "surface_solid": "#111827",
        "border":        "#1E3A5F",
        "border_hover":  "#3B82F6",
        "text":          "#E2E8F0",
        "text_muted":    "#94A3B8",
        "text_faint":    "#4A5568",
        "accent":        "#3B82F6",
        "accent2":       "#A78BFA",
        "accent3":       "#34D399",
        "header_bg":     "linear-gradient(120deg, #0F3460 0%, #16213E 40%, #0F2D52 100%)",
        "header_border": "#1E4A7A",
        "tab_bg":        "#111827",
        "tab_active":    "linear-gradient(135deg, #1E3A5F, #1A2540)",
        "tab_color":     "#94A3B8",
        "tab_active_c":  "#60A5FA",
        "card_shadow":   "0 4px 24px rgba(0,0,0,0.5)",
        "card_hover_sh": "0 10px 40px rgba(59,130,246,0.25)",
        "upload_bg":     "#0D1830",
        "divider":       "#1E293B",
        "btn_bg":        "linear-gradient(135deg, #1D4ED8, #2563EB)",
        "btn_hover":     "linear-gradient(135deg, #2563EB, #3B82F6)",
        "footer_color":  "#334155",
        "theme_btn_bg":  "#1A2540",
        "theme_btn_border": "#2D4A7A",
        "theme_icon":    "☀️",
        "theme_label":   "Light Mode",
        "info_label_c":  "#60A5FA",
    }
else:
    T = {
        "bg":            "linear-gradient(135deg, #F5EDD8 0%, #FDF6E3 50%, #F5EDD8 100%)",
        "bg_solid":      "#FDF6E3",
        "surface":       "linear-gradient(135deg, #FEF9EE 0%, #FDF4DC 100%)",
        "surface_solid": "#FEF9EE",
        "border":        "#D4B896",
        "border_hover":  "#B8860B",
        "text":          "#2D2010",
        "text_muted":    "#6B5A3E",
        "text_faint":    "#A08060",
        "accent":        "#B8541A",
        "accent2":       "#7C3AED",
        "accent3":       "#059669",
        "header_bg":     "linear-gradient(120deg, #8B4513 0%, #6B3A1F 40%, #7A3D14 100%)",
        "header_border": "#A0522D",
        "tab_bg":        "#FEF9EE",
        "tab_active":    "linear-gradient(135deg, #D4A96A, #C8976A)",
        "tab_color":     "#6B5A3E",
        "tab_active_c":  "#5C2D0A",
        "card_shadow":   "0 4px 24px rgba(160,100,40,0.15)",
        "card_hover_sh": "0 10px 40px rgba(184,84,26,0.2)",
        "upload_bg":     "#FEF3DC",
        "divider":       "#D4B896",
        "btn_bg":        "linear-gradient(135deg, #B8541A, #C8651A)",
        "btn_hover":     "linear-gradient(135deg, #C8651A, #D4761A)",
        "footer_color":  "#8B7355",
        "theme_btn_bg":  "#F0DEB8",
        "theme_btn_border": "#C4A070",
        "theme_icon":    "🌙",
        "theme_label":   "Dark Mode",
        "info_label_c":  "#B8541A",
    }

# ─────────────────────────────────────────────────────────────────────────────
# CSS injection — full themed stylesheet
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500&family=Outfit:wght@400;600;700;800&display=swap');

*, html, body {{
    font-family: 'Inter', sans-serif;
    box-sizing: border-box;
}}

/* ── App background ── */
.stApp {{
    background: {T["bg"]};
    color: {T["text"]};
    transition: background 0.4s ease, color 0.4s ease;
}}

/* ── Remove default padding ── */
[data-testid="stAppViewContainer"] > .main {{
    padding-top: 0.5rem;
}}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background: {T["surface_solid"]};
    border-right: 1px solid {T["border"]};
}}

/* ── Header ── */
.hal-header {{
    background: {T["header_bg"]};
    border: 1px solid {T["header_border"]};
    border-radius: 24px;
    padding: 36px 48px;
    margin-bottom: 28px;
    text-align: center;
    position: relative;
    overflow: hidden;
    box-shadow: 0 8px 40px rgba(0,0,0,0.3);
}}

.hal-header::before {{
    content: '';
    position: absolute;
    top: -60%;
    left: -20%;
    width: 60%;
    height: 200%;
    background: rgba(255,255,255,0.04);
    transform: rotate(-15deg);
    pointer-events: none;
}}

.hal-header::after {{
    content: '';
    position: absolute;
    bottom: -40%;
    right: -10%;
    width: 40%;
    height: 150%;
    background: rgba(255,255,255,0.03);
    transform: rotate(-15deg);
    pointer-events: none;
}}

.hal-badge {{
    display: inline-block;
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.2);
    border-radius: 999px;
    padding: 4px 14px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: rgba(255,255,255,0.85);
    margin-bottom: 16px;
    backdrop-filter: blur(4px);
}}

.hal-title {{
    font-family: 'Outfit', sans-serif;
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(90deg, #FFFFFF 0%, #CBD5E1 60%, #93C5FD 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 10px 0;
    letter-spacing: -1px;
    line-height: 1.1;
}}

.hal-subtitle {{
    color: rgba(255,255,255,0.6);
    font-size: 0.95rem;
    font-weight: 400;
    margin: 0;
    letter-spacing: 0.02em;
}}

.hal-stats-row {{
    display: flex;
    gap: 24px;
    justify-content: center;
    margin-top: 24px;
    flex-wrap: wrap;
}}

.hal-stat-pill {{
    background: rgba(255,255,255,0.1);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 12px;
    padding: 10px 20px;
    backdrop-filter: blur(6px);
    text-align: center;
    min-width: 100px;
}}

.hal-stat-pill .val {{
    font-family: 'Outfit', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: #FFFFFF;
    display: block;
}}

.hal-stat-pill .lbl {{
    font-size: 0.65rem;
    color: rgba(255,255,255,0.55);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 2px;
    display: block;
}}

/* ── Theme toggle button ── */
.theme-toggle-wrap {{
    position: fixed;
    top: 18px;
    right: 24px;
    z-index: 9999;
}}

.theme-toggle-btn {{
    display: flex;
    align-items: center;
    gap: 8px;
    background: {T["theme_btn_bg"]};
    border: 1px solid {T["theme_btn_border"]};
    border-radius: 999px;
    padding: 8px 18px;
    cursor: pointer;
    font-size: 0.85rem;
    font-weight: 600;
    color: {T["text"]};
    box-shadow: 0 4px 16px rgba(0,0,0,0.2);
    transition: all 0.3s ease;
    text-decoration: none;
    white-space: nowrap;
    font-family: 'Inter', sans-serif;
}}

.theme-toggle-btn:hover {{
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(0,0,0,0.3);
    border-color: {T["accent"]};
}}

/* ── Glass cards ── */
.glass-card {{
    background: {T["surface"]};
    border: 1px solid {T["border"]};
    border-radius: 18px;
    padding: 24px 28px;
    margin: 10px 0;
    transition: all 0.3s ease;
    box-shadow: {T["card_shadow"]};
    position: relative;
    overflow: hidden;
}}

.glass-card:hover {{
    border-color: {T["border_hover"]};
    box-shadow: {T["card_hover_sh"]};
    transform: translateY(-2px);
}}

.metric-card {{
    background: {T["surface"]};
    border: 1px solid {T["border"]};
    border-radius: 16px;
    padding: 20px 24px;
    margin: 8px 0;
    transition: all 0.3s ease;
    box-shadow: {T["card_shadow"]};
}}

.metric-card:hover {{
    border-color: {T["border_hover"]};
    box-shadow: {T["card_hover_sh"]};
    transform: translateY(-2px);
}}

/* ── KPI cards ── */
.kpi-card {{
    background: {T["surface"]};
    border: 1px solid {T["border"]};
    border-radius: 18px;
    padding: 22px 16px;
    text-align: center;
    transition: all 0.3s ease;
    box-shadow: {T["card_shadow"]};
}}

.kpi-card:hover {{
    transform: translateY(-3px);
    box-shadow: {T["card_hover_sh"]};
}}

.kpi-card .kpi-label {{
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: {T["text_muted"]};
    margin-bottom: 8px;
}}

.kpi-card .kpi-value {{
    font-family: 'Outfit', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    line-height: 1;
}}

/* ── Section label ── */
.section-label {{
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: {T["accent"]};
    margin-bottom: 14px;
    padding-bottom: 8px;
    border-bottom: 2px solid {T["border"]};
    display: flex;
    align-items: center;
    gap: 8px;
}}

/* ── Section heading ── */
.section-heading {{
    font-family: 'Outfit', sans-serif;
    font-size: 1.35rem;
    font-weight: 700;
    color: {T["text"]};
    margin: 0 0 4px 0;
}}

.section-sub {{
    color: {T["text_muted"]};
    font-size: 0.875rem;
    margin-bottom: 20px;
}}

/* ── Alert boxes ── */
.alert-fault {{
    background: linear-gradient(135deg, #2D0A0A 0%, #3D1010 100%);
    border: 2px solid #EF4444;
    border-radius: 18px;
    padding: 28px;
    text-align: center;
    animation: pulse-red 2.5s ease-in-out infinite;
}}

.alert-normal {{
    background: linear-gradient(135deg, #0A1F0A 0%, #102B10 100%);
    border: 2px solid #22C55E;
    border-radius: 18px;
    padding: 28px;
    text-align: center;
    animation: pulse-green 2.5s ease-in-out infinite;
}}

@keyframes pulse-red {{
    0%, 100% {{ box-shadow: 0 0 20px rgba(239,68,68,0.3); }}
    50%       {{ box-shadow: 0 0 50px rgba(239,68,68,0.65); }}
}}

@keyframes pulse-green {{
    0%, 100% {{ box-shadow: 0 0 20px rgba(34,197,94,0.3); }}
    50%       {{ box-shadow: 0 0 50px rgba(34,197,94,0.65); }}
}}

/* ── Confidence bar ── */
.confidence-bar-bg {{
    background: {T["divider"]};
    border-radius: 999px;
    height: 10px;
    margin: 8px 0;
    overflow: hidden;
}}

.confidence-bar-fill-green {{
    background: linear-gradient(90deg, #22C55E, #4ADE80);
    border-radius: 999px;
    height: 100%;
    transition: width 1.2s cubic-bezier(0.4,0,0.2,1);
}}

.confidence-bar-fill-red {{
    background: linear-gradient(90deg, #EF4444, #F87171);
    border-radius: 999px;
    height: 100%;
    transition: width 1.2s cubic-bezier(0.4,0,0.2,1);
}}

/* ── Tabs ── */
[data-testid="stTabs"] [role="tablist"] {{
    gap: 6px;
    border-bottom: 2px solid {T["border"]};
    padding-bottom: 0;
}}

[data-testid="stTabs"] [role="tab"] {{
    background: {T["tab_bg"]};
    color: {T["tab_color"]};
    border-radius: 10px 10px 0 0;
    border: 1px solid {T["border"]};
    border-bottom: none;
    font-weight: 600;
    font-size: 0.85rem;
    padding: 10px 18px;
    transition: all 0.2s ease;
}}

[data-testid="stTabs"] [role="tab"]:hover {{
    color: {T["text"]};
    border-color: {T["border_hover"]};
}}

[data-testid="stTabs"] [role="tab"][aria-selected="true"] {{
    background: {T["tab_active"]};
    color: {T["tab_active_c"]};
    border-bottom-color: transparent;
    font-weight: 700;
}}

/* ── Buttons ── */
.stButton>button {{
    background: {T["btn_bg"]};
    color: white;
    border: none;
    border-radius: 12px;
    padding: 12px 28px;
    font-weight: 700;
    font-size: 0.95rem;
    width: 100%;
    transition: all 0.3s ease;
    box-shadow: 0 4px 16px rgba(0,0,0,0.3);
    letter-spacing: 0.02em;
}}

.stButton>button:hover {{
    background: {T["btn_hover"]};
    box-shadow: 0 8px 28px rgba(0,0,0,0.4);
    transform: translateY(-2px);
}}

/* ── File uploader ── */
[data-testid="stFileUploader"] {{
    background: {T["upload_bg"]};
    border-radius: 16px;
    border: 2px dashed {T["border"]};
    transition: border-color 0.3s ease;
}}

[data-testid="stFileUploader"]:hover {{
    border-color: {T["accent"]};
}}

/* ── Dataframe ── */
.stDataFrame {{
    border-radius: 14px;
    overflow: hidden;
    border: 1px solid {T["border"]};
    box-shadow: {T["card_shadow"]};
}}

/* ── Expander ── */
[data-testid="stExpander"] {{
    background: {T["surface_solid"]};
    border: 1px solid {T["border"]};
    border-radius: 14px;
    overflow: hidden;
}}

/* ── Info boxes ── */
.stInfo, .stWarning, .stError, .stSuccess {{
    border-radius: 12px;
}}

/* ── Divider ── */
hr {{
    border-color: {T["border"]};
    opacity: 0.5;
    margin: 28px 0;
}}

/* ── Code ── */
code {{
    background: {T["divider"]};
    color: {T["accent2"]};
    padding: 2px 6px;
    border-radius: 4px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85em;
}}

/* ── Upload info card list items ── */
.info-item {{
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 6px 0;
    font-size: 0.875rem;
    color: {T["info_label_c"]};
    border-bottom: 1px solid {T["border"]};
}}

.info-item:last-child {{ border-bottom: none; }}

.info-dot {{
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: {T["accent"]};
    flex-shrink: 0;
}}

/* ── Animate in ── */
@keyframes fadeSlideUp {{
    from {{ opacity: 0; transform: translateY(20px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
}}

.animate-in {{
    animation: fadeSlideUp 0.5s ease forwards;
}}

/* ── Scrollbar ── */
::-webkit-scrollbar {{ width: 6px; height: 6px; }}
::-webkit-scrollbar-track {{ background: {T["bg_solid"]}; }}
::-webkit-scrollbar-thumb {{ background: {T["border"]}; border-radius: 999px; }}
::-webkit-scrollbar-thumb:hover {{ background: {T["accent"]}; }}

</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
MODEL_DIR  = os.path.join(os.path.dirname(__file__), "..", "models")
REPORT_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")
DATA_DIR   = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

SENSOR_RANGES = {
    "T2_fan_inlet_temp":       (516,  520,  518.0),
    "T24_lpc_outlet_temp":     (635,  650,  642.0),
    "T30_hpc_outlet_temp":     (1565, 1620, 1590.0),
    "T50_lpt_outlet_temp":     (1365, 1435, 1400.0),
    "P2_fan_inlet_pres":       (14.3, 14.9, 14.62),
    "P15_bypass_duct_pres":    (21.2, 22.0, 21.61),
    "P30_hpc_outlet_pres":     (543,  558,  550.0),
    "Nf_fan_speed":            (2365, 2410, 2388.0),
    "Nc_core_speed":           (8990, 9100, 9046.0),
    "epr_engine_pres_ratio":   (1.28, 1.32, 1.300),
    "Ps30_hpc_static_pres":    (46.5, 48.5, 47.47),
    "phi_ratio_fuel_flow":     (513,  530,  521.0),
    "NRf_corrected_fan_speed": (2365, 2410, 2388.0),
    "NRc_corrected_core_speed":(8100, 8175, 8138.0),
    "BPR_bypass_ratio":        (8.30, 8.50, 8.40),
}

# ─────────────────────────────────────────────────────────────────────────────
# Cached resource loaders
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_model_and_scaler():
    champion_path = os.path.join(MODEL_DIR, "champion_model.pkl")
    scaler_path   = os.path.join(MODEL_DIR, "scaler.pkl")
    meta_path     = os.path.join(MODEL_DIR, "champion_meta.json")
    model  = joblib.load(champion_path) if os.path.exists(champion_path) else None
    scaler = joblib.load(scaler_path)   if os.path.exists(scaler_path)   else None
    meta   = json.load(open(meta_path)) if os.path.exists(meta_path)     else {}
    return model, scaler, meta


@st.cache_data(show_spinner=False)
def load_metrics_table():
    path = os.path.join(REPORT_DIR, "metrics_table.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Load assets
# ─────────────────────────────────────────────────────────────────────────────
model, scaler, meta = load_model_and_scaler()
metrics_df = load_metrics_table()
model_ready = model is not None and scaler is not None

# ─────────────────────────────────────────────────────────────────────────────
# Theme Toggle — floating button via form (top-right)
# ─────────────────────────────────────────────────────────────────────────────
with st.container():
    col_spacer, col_toggle = st.columns([10, 1])
    with col_toggle:
        if st.button(T["theme_icon"], key="theme_toggle", help=T["theme_label"]):
            st.session_state.theme = "light" if is_dark else "dark"
            st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — Header
# ─────────────────────────────────────────────────────────────────────────────
champion_name = meta.get("name", "SVM").replace("_", " ").title() if meta else "N/A"
f1_score      = meta.get("f1", 0) if meta else 0
auc_score     = meta.get("roc_auc", 0) if meta else 0

st.markdown(f"""
<div class="hal-header animate-in">
    <div class="hal-badge">✈️ &nbsp; HAL – Helicopter Division &nbsp; • &nbsp; Predictive Maintenance</div>
    <p class="hal-title">Aircraft Equipment Fault Detection</p>
    <p class="hal-subtitle">ML-Powered Real-time Sensor Anomaly Analysis &nbsp;|&nbsp; Phase 5 Dashboard</p>
    <div class="hal-stats-row">
        <div class="hal-stat-pill">
            <span class="val">{champion_name}</span>
            <span class="lbl">Champion Model</span>
        </div>
        <div class="hal-stat-pill">
            <span class="val" style="color:#4ADE80;">{f1_score:.3f}</span>
            <span class="lbl">F1 Score</span>
        </div>
        <div class="hal-stat-pill">
            <span class="val" style="color:#A78BFA;">{auc_score:.3f}</span>
            <span class="lbl">ROC AUC</span>
        </div>
        <div class="hal-stat-pill">
            <span class="val">15</span>
            <span class="lbl">Sensors</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

if not model_ready:
    st.warning(
        "⚠️ Model or scaler not found. Please run the full pipeline first:\n\n"
        "```bash\n"
        "python src/generate_dataset.py\n"
        "python src/run_pipeline.py\n"
        "```"
    )

# ─────────────────────────────────────────────────────────────────────────────
# Section 2 — Batch File Upload & Prediction
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">📂 &nbsp; Batch Upload & Prediction</div>', unsafe_allow_html=True)

with st.expander("ℹ️  How to use Batch Upload", expanded=False):
    st.markdown("""
    Upload a **CSV file** containing sensor readings. Each row will be analysed independently.

    **Required columns** (any order, extra columns are ignored):
    """
    + ", ".join(f"`{s}`" for s in SENSOR_RANGES.keys()) +
    """

    The system computes the **deviation of each reading from its optimal value**,
    then passes the readings through the trained ML model to predict **FAULT / NORMAL**.

    A sample template CSV is provided below.
    """)
    template_df  = pd.DataFrame([{k: v[2] for k, v in SENSOR_RANGES.items()}])
    csv_template = template_df.to_csv(index=False)
    b64 = base64.b64encode(csv_template.encode()).decode()
    st.markdown(
        f'<a href="data:file/csv;base64,{b64}" download="sensor_template.csv" '
        f'style="color:{T["accent"]};font-weight:600;text-decoration:none;">⬇️ Download CSV Template</a>',
        unsafe_allow_html=True,
    )

upload_col, info_col = st.columns([3, 1], gap="large")

with upload_col:
    uploaded_file = st.file_uploader(
        "Upload sensor readings CSV",
        type=["csv"],
        key="batch_upload",
        help="CSV with sensor columns. Each row = one prediction.",
    )

with info_col:
    st.markdown(f"""
    <div class="glass-card" style="height:100%;min-height:120px;">
        <div style="font-size:0.68rem;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;
                    color:{T["text_muted"]};margin-bottom:14px;">Expected Format</div>
        <div class="info-item"><span class="info-dot"></span>One row per engine cycle</div>
        <div class="info-item"><span class="info-dot"></span>15 sensor columns required</div>
        <div class="info-item"><span class="info-dot"></span><code>fault</code> column optional</div>
        <div class="info-item"><span class="info-dot"></span>Raw unscaled sensor values</div>
        <div style="margin-top:12px;font-size:0.72rem;color:{T["text_faint"]};">Results downloadable as CSV</div>
    </div>
    """, unsafe_allow_html=True)

if uploaded_file is not None:
    try:
        raw_upload = pd.read_csv(uploaded_file)
        st.markdown(
            f'<div class="glass-card" style="margin:14px 0;padding:14px 22px;display:flex;align-items:center;gap:12px;">'
            f'<span style="font-size:1.4rem;">📄</span>'
            f'<div>'
            f'<span style="color:{T["text_muted"]};font-size:0.8rem;">Loaded </span>'
            f'<span style="color:{T["accent"]};font-weight:700;font-size:1.05rem;">{len(raw_upload)}</span>'
            f'<span style="color:{T["text_muted"]};font-size:0.8rem;"> rows × </span>'
            f'<span style="color:{T["accent"]};font-weight:700;font-size:1.05rem;">{len(raw_upload.columns)}</span>'
            f'<span style="color:{T["text_muted"]};font-size:0.8rem;"> columns from </span>'
            f'<strong style="color:{T["text"]};">{uploaded_file.name}</strong>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

        required_cols = list(SENSOR_RANGES.keys())
        missing_cols  = [c for c in required_cols if c not in raw_upload.columns]

        if missing_cols:
            st.error(f"❌ Missing required columns: {missing_cols}\n\nDownload the template above to see the correct format.")
        elif not model_ready:
            st.error("❌ Model not loaded. Run the training pipeline first.")
        else:
            sensor_df_upload = raw_upload[required_cols].copy()

            optimal_vals = {k: v[2] for k, v in SENSOR_RANGES.items()}
            min_vals     = {k: v[0] for k, v in SENSOR_RANGES.items()}
            max_vals     = {k: v[1] for k, v in SENSOR_RANGES.items()}

            deviation_df = pd.DataFrame(index=sensor_df_upload.index)
            for col in required_cols:
                optimal = optimal_vals[col]
                rng     = max_vals[col] - min_vals[col]
                deviation_df[col + "_dev%"] = (
                    (sensor_df_upload[col] - optimal) / rng * 100
                ).round(2)

            scaler_cols       = list(scaler.feature_names_in_) if hasattr(scaler, "feature_names_in_") else required_cols
            upload_for_scaler = sensor_df_upload.reindex(columns=scaler_cols, fill_value=0.0)
            upload_scaled     = scaler.transform(upload_for_scaler)
            upload_scaled_df  = pd.DataFrame(upload_scaled, columns=scaler_cols)

            if hasattr(model, "feature_names_in_"):
                model_cols   = list(model.feature_names_in_)
                upload_final = upload_scaled_df[model_cols]
            else:
                upload_final = upload_scaled_df

            y_preds  = model.predict(upload_final)
            y_probas = model.predict_proba(upload_final)

            results_df = sensor_df_upload.copy()
            results_df["Prediction"]     = ["🚨 FAULT" if p == 1 else "✅ NORMAL" for p in y_preds]
            results_df["Fault Prob %"]   = (y_probas[:, 1] * 100).round(1)
            results_df["Normal Prob %"]  = (y_probas[:, 0] * 100).round(1)
            results_df["Confidence %"]   = results_df[["Fault Prob %", "Normal Prob %"]].max(axis=1)
            results_df["Max Dev Sensor"] = deviation_df.abs().idxmax(axis=1).str.replace("_dev%", "")
            results_df["Max Dev %"]      = deviation_df.abs().max(axis=1).round(2)

            if "fault" in raw_upload.columns:
                results_df.insert(0, "Actual", raw_upload["fault"].map({0: "✅ NORMAL", 1: "🚨 FAULT"}))

            n_fault  = int(sum(y_preds))
            n_normal = len(y_preds) - n_fault
            fault_pct = n_fault / len(y_preds) * 100 if len(y_preds) else 0

            # ── KPI row ──
            st.markdown("<br>", unsafe_allow_html=True)
            k1, k2, k3, k4 = st.columns(4)

            k1.markdown(f"""
            <div class="kpi-card animate-in">
                <div class="kpi-label">Total Rows</div>
                <div class="kpi-value" style="color:{T["accent"]};">{len(y_preds)}</div>
            </div>""", unsafe_allow_html=True)

            k2.markdown(f"""
            <div class="kpi-card animate-in">
                <div class="kpi-label">Faults Detected</div>
                <div class="kpi-value" style="color:#EF4444;">{n_fault}</div>
            </div>""", unsafe_allow_html=True)

            k3.markdown(f"""
            <div class="kpi-card animate-in">
                <div class="kpi-label">Normal Readings</div>
                <div class="kpi-value" style="color:#22C55E;">{n_normal}</div>
            </div>""", unsafe_allow_html=True)

            k4.markdown(f"""
            <div class="kpi-card animate-in">
                <div class="kpi-label">Fault Rate</div>
                <div class="kpi-value" style="color:{'#EF4444' if fault_pct > 30 else '#22C55E'};">{fault_pct:.1f}%</div>
            </div>""", unsafe_allow_html=True)

            # ── Results table ──
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-label">🗂️ &nbsp; Row-by-Row Results</div>', unsafe_allow_html=True)

            def colour_prediction(val):
                if "FAULT" in str(val):
                    return "background-color: #3D1010; color: #EF4444; font-weight: 700;"
                elif "NORMAL" in str(val):
                    return "background-color: #0A2010; color: #22C55E; font-weight: 700;"
                return ""

            def colour_fault_prob(val):
                try:
                    v = float(val)
                    if v >= 70: return "color: #EF4444; font-weight:600;"
                    if v >= 40: return "color: #F59E0B; font-weight:600;"
                    return "color: #22C55E;"
                except:
                    return ""

            display_cols = ["Prediction", "Fault Prob %", "Normal Prob %", "Confidence %", "Max Dev Sensor", "Max Dev %"]
            if "Actual" in results_df.columns:
                display_cols = ["Actual"] + display_cols

            styled_results = (
                results_df[display_cols]
                .style
                .map(colour_prediction, subset=["Prediction"])
                .map(colour_fault_prob, subset=["Fault Prob %"])
            )
            if "Actual" in results_df.columns:
                styled_results = styled_results.map(colour_prediction, subset=["Actual"])

            st.dataframe(styled_results, use_container_width=True, hide_index=False, height=380)

            with st.expander("🌡️  Sensor Deviation Heatmap (% from optimal)", expanded=False):
                dev_display = deviation_df.copy()
                dev_display.columns = [c.replace("_dev%", "") for c in dev_display.columns]
                styled_dev = dev_display.style.background_gradient(
                    cmap="RdYlGn_r", vmin=-20, vmax=20
                ).format("{:+.2f}%")
                st.dataframe(styled_dev, use_container_width=True, height=350)
                st.caption("🔴 Red = above optimal  |  🟢 Green = at/below optimal  |  Values are % deviation from optimal")

            # ── Download button ──
            export_df = results_df.copy()
            export_df["Prediction_raw"] = ["FAULT" if p == 1 else "NORMAL" for p in y_preds]
            csv_out = export_df.to_csv(index=True)
            b64_out = base64.b64encode(csv_out.encode()).decode()
            st.markdown(
                f'<br><a href="data:file/csv;base64,{b64_out}" download="predictions_{uploaded_file.name}" '
                f'style="background:{T["btn_bg"]};color:white;padding:12px 28px;'
                f'border-radius:12px;font-weight:700;text-decoration:none;display:inline-block;'
                f'box-shadow:0 4px 16px rgba(0,0,0,0.3);font-family:Inter,sans-serif;font-size:0.9rem;">'
                f'⬇️ Download Full Results CSV</a>',
                unsafe_allow_html=True,
            )

    except Exception as exc:
        st.error(f"❌ Error processing file: {exc}")

# ─────────────────────────────────────────────────────────────────────────────
# Section 3 — Model Performance Charts (Tabs)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown('<div class="section-label">📊 &nbsp; Model Performance</div>', unsafe_allow_html=True)

tab_roc, tab_cm, tab_fi, tab_pr, tab_metrics = st.tabs([
    "📈  ROC Curves",
    "🟦  Confusion Matrices",
    "🌟  Feature Importance",
    "📉  Precision-Recall",
    "📋  Metrics Table",
])

CHART_PATHS = {
    "roc": os.path.join(REPORT_DIR, "roc_curves.png"),
    "cm":  os.path.join(REPORT_DIR, "confusion_matrices.png"),
    "fi":  os.path.join(REPORT_DIR, "feature_importance.png"),
    "pr":  os.path.join(REPORT_DIR, "pr_curve.png"),
}

def show_chart(tab, path, caption):
    with tab:
        if os.path.exists(path):
            st.image(path, caption=caption, use_container_width=True)
        else:
            st.info(f"Chart not generated yet. Run `python src/evaluate.py` to generate reports.")

show_chart(tab_roc, CHART_PATHS["roc"], "ROC Curves — All 6 Models")
show_chart(tab_cm,  CHART_PATHS["cm"],  "Confusion Matrices — All 6 Models")
show_chart(tab_fi,  CHART_PATHS["fi"],  "Feature Importance — Best Tree Model")
show_chart(tab_pr,  CHART_PATHS["pr"],  "Precision-Recall Curves — Top Models")

with tab_metrics:
    if metrics_df is not None:
        styled_df = metrics_df.copy()
        styled_df.columns = [c.replace("_", " ").title() for c in styled_df.columns]
        st.dataframe(
            styled_df.style.highlight_max(
                subset=[c for c in styled_df.columns if c not in ["Model", "Best Params"]],
                color="#1E3A5F" if is_dark else "#F0DEB8",
            ),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Run the evaluation pipeline to populate metrics.")

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(f"""
<div style="text-align:center;color:{T["footer_color"]};font-size:0.82rem;padding:20px 0 8px;letter-spacing:0.03em;">
    <strong style="color:{T["text_muted"]};">HAL – Helicopter Division</strong>
    &nbsp;|&nbsp; Aircraft Equipment Fault Detection System
    &nbsp;|&nbsp; B.Tech CSE, 2026–27
</div>
""", unsafe_allow_html=True)
