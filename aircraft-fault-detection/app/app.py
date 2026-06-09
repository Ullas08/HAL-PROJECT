"""
app/app.py
==========
Streamlit dashboard for the Aircraft Equipment Fault Detection system.
Developed for HAL – Helicopter Division (Phase 5).

Run with:
    streamlit run app/app.py
"""

import os
import sys
import json
import time

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
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS — Dark, premium aesthetic
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Main background */
.stApp {
    background: linear-gradient(135deg, #0A0D1A 0%, #0F1525 50%, #0A0D1A 100%);
    color: #E2E8F0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D1117 0%, #111827 100%);
    border-right: 1px solid #1E3A5F;
}

/* Cards */
.metric-card {
    background: linear-gradient(135deg, #131B2E 0%, #1A2540 100%);
    border: 1px solid #1E3A5F;
    border-radius: 16px;
    padding: 20px 24px;
    margin: 8px 0;
    transition: all 0.3s ease;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
}

.metric-card:hover {
    border-color: #3B82F6;
    box-shadow: 0 8px 32px rgba(59,130,246,0.2);
    transform: translateY(-2px);
}

/* Alert boxes */
.alert-fault {
    background: linear-gradient(135deg, #2D0A0A 0%, #3D1010 100%);
    border: 2px solid #EF4444;
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    animation: pulse-red 2s infinite;
}

.alert-normal {
    background: linear-gradient(135deg, #0A1F0A 0%, #102B10 100%);
    border: 2px solid #22C55E;
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    animation: pulse-green 2s infinite;
}

@keyframes pulse-red {
    0%, 100% { box-shadow: 0 0 20px rgba(239,68,68,0.3); }
    50%       { box-shadow: 0 0 40px rgba(239,68,68,0.6); }
}

@keyframes pulse-green {
    0%, 100% { box-shadow: 0 0 20px rgba(34,197,94,0.3); }
    50%       { box-shadow: 0 0 40px rgba(34,197,94,0.6); }
}

/* Header */
.header-container {
    background: linear-gradient(90deg, #0F3460 0%, #16213E 50%, #0F3460 100%);
    border: 1px solid #1E3A5F;
    border-radius: 20px;
    padding: 30px 40px;
    margin-bottom: 24px;
    text-align: center;
}

.header-title {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #60A5FA, #A78BFA, #34D399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    letter-spacing: -0.5px;
}

.header-subtitle {
    color: #94A3B8;
    font-size: 1rem;
    margin-top: 8px;
    font-weight: 400;
}

/* Confidence bar */
.confidence-bar-bg {
    background: #1E293B;
    border-radius: 999px;
    height: 12px;
    margin: 8px 0;
    overflow: hidden;
}

.confidence-bar-fill-green {
    background: linear-gradient(90deg, #22C55E, #4ADE80);
    border-radius: 999px;
    height: 100%;
    transition: width 1s ease;
}

.confidence-bar-fill-red {
    background: linear-gradient(90deg, #EF4444, #F87171);
    border-radius: 999px;
    height: 100%;
    transition: width 1s ease;
}

/* Tabs */
[data-testid="stTabs"] [role="tab"] {
    background: #131B2E;
    color: #94A3B8;
    border-radius: 8px 8px 0 0;
    border: 1px solid #1E3A5F;
    font-weight: 500;
}

[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, #1E3A5F, #1A2540);
    color: #60A5FA;
    border-bottom-color: transparent;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(135deg, #1D4ED8, #2563EB);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 12px 24px;
    font-weight: 600;
    font-size: 1rem;
    width: 100%;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(37,99,235,0.4);
}

.stButton>button:hover {
    background: linear-gradient(135deg, #2563EB, #3B82F6);
    box-shadow: 0 8px 25px rgba(37,99,235,0.6);
    transform: translateY(-2px);
}

/* Slider */
[data-testid="stSlider"] > div > div > div { background: #1E3A5F; }
[data-testid="stSlider"] > div > div > div > div { background: #3B82F6; }

/* Number input */
.stNumberInput input {
    background: #131B2E;
    border: 1px solid #1E3A5F;
    color: #E2E8F0;
    border-radius: 8px;
}

/* Dataframe */
.stDataFrame { border-radius: 12px; overflow: hidden; }

/* Section label */
.section-label {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #4A90D9;
    margin-bottom: 12px;
    padding-bottom: 6px;
    border-bottom: 1px solid #1E3A5F;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
MODEL_DIR  = os.path.join(os.path.dirname(__file__), "..", "models")
REPORT_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")
DATA_DIR   = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

SENSOR_RANGES = {
    "T2_fan_inlet_temp":      (516,  520,  518.0),
    "T24_lpc_outlet_temp":    (635,  650,  642.0),
    "T30_hpc_outlet_temp":    (1565, 1620, 1590.0),
    "T50_lpt_outlet_temp":    (1365, 1435, 1400.0),
    "P2_fan_inlet_pres":      (14.3, 14.9, 14.62),
    "P15_bypass_duct_pres":   (21.2, 22.0, 21.61),
    "P30_hpc_outlet_pres":    (543,  558,  550.0),
    "Nf_fan_speed":           (2365, 2410, 2388.0),
    "Nc_core_speed":          (8990, 9100, 9046.0),
    "epr_engine_pres_ratio":  (1.28, 1.32, 1.300),
    "Ps30_hpc_static_pres":   (46.5, 48.5, 47.47),
    "phi_ratio_fuel_flow":    (513,  530,  521.0),
    "NRf_corrected_fan_speed":(2365, 2410, 2388.0),
    "NRc_corrected_core_speed":(8100, 8175, 8138.0),
    "BPR_bypass_ratio":       (8.30, 8.50, 8.40),
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
def load_test_samples():
    test_path = os.path.join(DATA_DIR, "test_clean.csv")
    if os.path.exists(test_path):
        df = pd.read_csv(test_path)
        faults  = df[df["fault"] == 1].drop(columns=["fault"]).head(1)
        normals = df[df["fault"] == 0].drop(columns=["fault"]).head(1)
        return faults.iloc[0].to_dict() if len(faults) else None, \
               normals.iloc[0].to_dict() if len(normals) else None
    return None, None


@st.cache_data(show_spinner=False)
def load_metrics_table():
    path = os.path.join(REPORT_DIR, "metrics_table.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-container">
    <p class="header-title">✈️ Aircraft Equipment Fault Detection</p>
    <p class="header-subtitle">
        HAL – Helicopter Division &nbsp;|&nbsp; ML-Powered Predictive Maintenance Dashboard
        &nbsp;|&nbsp; Real-time Sensor Analysis
    </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Load assets (with error handling)
# ─────────────────────────────────────────────────────────────────────────────
model, scaler, meta = load_model_and_scaler()
fault_sample, normal_sample = load_test_samples()
metrics_df = load_metrics_table()

model_ready = model is not None and scaler is not None

if not model_ready:
    st.warning(
        "⚠️ Model or scaler not found. Please run the full pipeline first:\n\n"
        "```bash\n"
        "python src/generate_dataset.py\n"
        "python src/run_pipeline.py\n"
        "```"
    )

# ─────────────────────────────────────────────────────────────────────────────
# Section 2 — Sidebar Input Panel
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="section-label">🎛️ Sensor Readings</div>', unsafe_allow_html=True)

    if model_ready and meta:
        champion_name = meta.get("name", "Unknown").replace("_", " ").title()
        f1  = meta.get("f1", 0)
        auc = meta.get("roc_auc", 0)
        st.markdown(f"""
        <div class="metric-card" style="margin-bottom:16px;">
            <div style="font-size:0.7rem;color:#94A3B8;text-transform:uppercase;letter-spacing:0.1em;">Active Model</div>
            <div style="font-size:1.1rem;font-weight:700;color:#60A5FA;margin:4px 0;">{champion_name}</div>
            <div style="display:flex;gap:12px;margin-top:6px;">
                <div><span style="color:#94A3B8;font-size:0.75rem;">F1</span>
                     <span style="color:#22C55E;font-weight:600;"> {f1:.3f}</span></div>
                <div><span style="color:#94A3B8;font-size:0.75rem;">AUC</span>
                     <span style="color:#A78BFA;font-weight:600;"> {auc:.3f}</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Sample data buttons
    col1, col2 = st.columns(2)
    load_fault  = col1.button("⚠️ Fault Sample",  key="btn_fault",  use_container_width=True)
    load_normal = col2.button("✅ Normal Sample", key="btn_normal", use_container_width=True)

    st.markdown("---")

    # Determine default values
    if load_fault and fault_sample:
        defaults = fault_sample
    elif load_normal and normal_sample:
        defaults = normal_sample
    else:
        defaults = {k: v[2] for k, v in SENSOR_RANGES.items()}

    sensor_values = {}
    for sensor, (min_val, max_val, default_val) in SENSOR_RANGES.items():
        label = sensor.replace("_", " ").title()
        val = defaults.get(sensor, default_val)
        val = float(np.clip(val, min_val, max_val))
        sensor_values[sensor] = st.slider(
            label,
            min_value=float(min_val),
            max_value=float(max_val),
            value=val,
            step=float((max_val - min_val) / 100),
            key=f"slider_{sensor}",
        )

    st.markdown("---")
    predict_btn = st.button("🔍 Predict Fault Status", key="btn_predict", use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Main content layout
# ─────────────────────────────────────────────────────────────────────────────
col_pred, col_meta = st.columns([3, 2], gap="large")

# ─────────────────────────────────────────────────────────────────────────────
# Section 3 — Prediction Output
# ─────────────────────────────────────────────────────────────────────────────
with col_pred:
    st.markdown('<div class="section-label">🎯 Prediction Result</div>', unsafe_allow_html=True)

    if predict_btn and model_ready:
        # Step 1: Build input DataFrame in scaler's column order
        scaler_cols = list(scaler.feature_names_in_) if hasattr(scaler, "feature_names_in_") else list(sensor_values.keys())
        input_df = pd.DataFrame([sensor_values])[scaler_cols]

        # Step 2: Scale using fitted scaler
        input_scaled = scaler.transform(input_df)
        input_scaled_df = pd.DataFrame(input_scaled, columns=scaler_cols)

        # Step 3: Reorder to match model's expected feature order
        if hasattr(model, "feature_names_in_"):
            model_cols = list(model.feature_names_in_)
            input_final = input_scaled_df[model_cols]
        else:
            input_final = input_scaled_df

        try:
            t0 = time.time()
            y_pred  = model.predict(input_final)
            y_proba = model.predict_proba(input_final)[0]
            elapsed = time.time() - t0

            is_fault       = int(y_pred[0]) == 1
            fault_prob     = float(y_proba[1])
            normal_prob    = float(y_proba[0])
            confidence_pct = int(max(fault_prob, normal_prob) * 100)

            if is_fault:
                bar_fill_class = "confidence-bar-fill-red"
                bar_width = int(fault_prob * 100)
                st.markdown(f"""
                <div class="alert-fault">
                    <div style="font-size:3rem;margin-bottom:8px;">🚨</div>
                    <div style="font-size:2rem;font-weight:800;color:#EF4444;letter-spacing:2px;">FAULT DETECTED</div>
                    <div style="color:#FCA5A5;margin-top:8px;font-size:1rem;">
                        Fault probability: <strong>{fault_prob*100:.1f}%</strong>
                    </div>
                    <div style="color:#94A3B8;font-size:0.8rem;margin-top:4px;">
                        Immediate inspection recommended
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                bar_fill_class = "confidence-bar-fill-green"
                bar_width = int(normal_prob * 100)
                st.markdown(f"""
                <div class="alert-normal">
                    <div style="font-size:3rem;margin-bottom:8px;">✅</div>
                    <div style="font-size:2rem;font-weight:800;color:#22C55E;letter-spacing:2px;">SYSTEM NORMAL</div>
                    <div style="color:#86EFAC;margin-top:8px;font-size:1rem;">
                        Normal probability: <strong>{normal_prob*100:.1f}%</strong>
                    </div>
                    <div style="color:#94A3B8;font-size:0.8rem;margin-top:4px;">
                        All sensors within acceptable range
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown(f"""
            <div style="margin-top:16px;">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <span style="color:#94A3B8;font-size:0.85rem;">Model Confidence</span>
                    <span style="color:#E2E8F0;font-weight:600;">{confidence_pct}%</span>
                </div>
                <div class="confidence-bar-bg">
                    <div class="{bar_fill_class}" style="width:{bar_width}%;"></div>
                </div>
                <div style="display:flex;justify-content:space-between;margin-top:4px;">
                    <span style="color:#22C55E;font-size:0.75rem;">Normal: {normal_prob*100:.1f}%</span>
                    <span style="color:#EF4444;font-size:0.75rem;">Fault: {fault_prob*100:.1f}%</span>
                </div>
            </div>

            <div class="metric-card" style="margin-top:16px;">
                <div style="color:#94A3B8;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.1em;">Inference Details</div>
                <div style="display:flex;gap:24px;margin-top:10px;flex-wrap:wrap;">
                    <div>
                        <div style="color:#94A3B8;font-size:0.7rem;">Response Time</div>
                        <div style="color:#E2E8F0;font-weight:600;font-family:'JetBrains Mono',monospace;">{elapsed*1000:.1f} ms</div>
                    </div>
                    <div>
                        <div style="color:#94A3B8;font-size:0.7rem;">Prediction</div>
                        <div style="color:{'#EF4444' if is_fault else '#22C55E'};font-weight:600;">{'FAULT' if is_fault else 'NORMAL'}</div>
                    </div>
                    <div>
                        <div style="color:#94A3B8;font-size:0.7rem;">Probability</div>
                        <div style="color:#A78BFA;font-weight:600;font-family:'JetBrains Mono',monospace;">{fault_prob:.4f}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Prediction error: {e}")

    elif predict_btn and not model_ready:
        st.error("Models not loaded. Please run the training pipeline first.")
    else:
        st.markdown("""
        <div class="metric-card" style="text-align:center;padding:48px 24px;">
            <div style="font-size:3rem;opacity:0.4;margin-bottom:12px;">🛩️</div>
            <div style="color:#94A3B8;font-size:1rem;">
                Adjust sensor values in the sidebar and click<br>
                <strong style="color:#60A5FA;">Predict Fault Status</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Sensor value preview panel
# ─────────────────────────────────────────────────────────────────────────────
with col_meta:
    st.markdown('<div class="section-label">📡 Live Sensor Values</div>', unsafe_allow_html=True)

    sensor_df = pd.DataFrame([
        {
            "Sensor": k.replace("_", " ").title(),
            "Value": f"{v:.3f}",
            "Range": f"[{SENSOR_RANGES[k][0]:.1f}, {SENSOR_RANGES[k][1]:.1f}]",
        }
        for k, v in sensor_values.items()
    ])
    st.dataframe(sensor_df, use_container_width=True, hide_index=True, height=420)

# ─────────────────────────────────────────────────────────────────────────────
# Section 4 — Model Performance Charts (Tabs)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-label">📊 Model Performance</div>', unsafe_allow_html=True)

tab_roc, tab_cm, tab_fi, tab_pr, tab_metrics = st.tabs([
    "📈 ROC Curves",
    "🟦 Confusion Matrices",
    "🌟 Feature Importance",
    "📉 Precision-Recall",
    "📋 Metrics Table",
])

CHART_PATHS = {
    "roc":        os.path.join(REPORT_DIR, "roc_curves.png"),
    "cm":         os.path.join(REPORT_DIR, "confusion_matrices.png"),
    "fi":         os.path.join(REPORT_DIR, "feature_importance.png"),
    "pr":         os.path.join(REPORT_DIR, "pr_curve.png"),
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
        # Style the table
        styled_df = metrics_df.copy()
        styled_df.columns = [c.replace("_", " ").title() for c in styled_df.columns]
        st.dataframe(
            styled_df.style.highlight_max(
                subset=[c for c in styled_df.columns if c not in ["Model", "Best Params"]],
                color="#1E3A5F",
            ),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Run the evaluation pipeline to populate metrics.")

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#4A5568;font-size:0.85rem;padding:16px 0;">
    HAL – Helicopter Division &nbsp;|&nbsp; Aircraft Equipment Fault Detection System
    &nbsp;|&nbsp; B.Tech CSE, 2026–27 &nbsp;|&nbsp; Developed for Antigravity
</div>
""", unsafe_allow_html=True)
