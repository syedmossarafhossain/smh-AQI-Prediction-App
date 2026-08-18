import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AQI AI | Air Quality Prediction",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS — styled to match the Diabetes AI app
# =========================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 10% 10%, rgba(59,130,246,0.12), transparent 25%),
        radial-gradient(circle at 90% 20%, rgba(139,92,246,0.10), transparent 25%),
        #090d16;
    color: #f8fafc;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1400px;
}

/* =========================================================
   SIDEBAR
   ========================================================= */

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1320 0%, #0a0f19 100%);
    border-right: 1px solid rgba(255,255,255,0.08);
}

section[data-testid="stSidebar"] * {
    color: #e5e7eb;
}

.sidebar-logo {
    text-align: center;
    padding: 10px 0 25px 0;
}

.sidebar-logo-icon {
    font-size: 42px;
    filter: drop-shadow(0 0 16px rgba(96,165,250,0.35));
}

.sidebar-title {
    font-size: 22px;
    font-weight: 800;
    color: #ffffff;
    margin-top: 5px;
}

.sidebar-subtitle {
    font-size: 12px;
    color: #9ca3af;
}

.sidebar-section {
    font-size: 10px;
    color: #64748b;
    font-weight: 700;
    letter-spacing: 1.5px;
    margin-top: 25px;
    margin-bottom: 10px;
}

.sidebar-menu-item {
    padding: 10px 12px;
    margin: 4px 0;
    border-radius: 10px;
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.04);
    color: #cbd5e1;
    font-size: 13px;
}

.sidebar-info {
    background: linear-gradient(145deg, rgba(15,23,42,0.95), rgba(17,24,39,0.82));
    border: 1px solid rgba(148,163,184,0.14);
    border-radius: 14px;
    padding: 14px;
    margin-top: 10px;
    box-shadow: 0 12px 30px rgba(0,0,0,0.15);
}

.sidebar-about {
    color: #94a3b8;
    font-size: 11px;
    line-height: 1.6;
}

/* =========================================================
   HERO
   ========================================================= */

.hero {
    position: relative;
    overflow: hidden;
    margin-top: 25px;
    margin-bottom: 35px;
    padding: 42px 48px;
    min-height: 250px;
    border-radius: 28px;
    background:
        radial-gradient(circle at 85% 20%, rgba(124,58,237,0.28), transparent 32%),
        radial-gradient(circle at 15% 80%, rgba(37,99,235,0.20), transparent 30%),
        linear-gradient(135deg, rgba(15,23,42,0.98), rgba(17,24,39,0.94));
    border: 1px solid rgba(148,163,184,0.14);
    box-shadow: 0 25px 80px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.06);
    backdrop-filter: blur(20px);
}

.hero::before {
    content: "";
    position: absolute;
    width: 280px;
    height: 280px;
    right: -90px;
    top: -120px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(99,102,241,0.30), transparent 70%);
    filter: blur(10px);
    pointer-events: none;
}

.hero::after {
    content: "";
    position: absolute;
    inset: 0;
    background-image:
        linear-gradient(rgba(255,255,255,0.025) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.025) 1px, transparent 1px);
    background-size: 32px 32px;
    mask-image: linear-gradient(to bottom right, black, transparent 70%);
    pointer-events: none;
}

.hero-content {
    position: relative;
    z-index: 2;
}

.badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 8px 14px;
    border-radius: 999px;
    background: rgba(59,130,246,0.10);
    border: 1px solid rgba(96,165,250,0.25);
    color: #93c5fd;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.4px;
    margin-bottom: 18px;
}

.badge-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #60a5fa;
    box-shadow: 0 0 12px rgba(96,165,250,0.8);
}

.hero-title {
    margin: 0;
    max-width: 760px;
    font-size: clamp(34px, 4vw, 54px);
    line-height: 1.05;
    font-weight: 800;
    letter-spacing: -2px;
    color: #f8fafc;
}

.hero-title span {
    background: linear-gradient(90deg, #60a5fa, #818cf8, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.hero-subtitle {
    margin-top: 18px;
    max-width: 680px;
    font-size: 16px;
    line-height: 1.7;
    color: #94a3b8;
}

.hero-meta {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-top: 26px;
    flex-wrap: wrap;
}

.hero-meta-item {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 9px 13px;
    border-radius: 10px;
    background: rgba(255,255,255,0.035);
    border: 1px solid rgba(255,255,255,0.07);
    color: #cbd5e1;
    font-size: 12px;
    font-weight: 500;
}

.hero-visual {
    position: absolute;
    z-index: 2;
    right: 48px;
    top: 50%;
    transform: translateY(-50%);
    width: 210px;
    height: 210px;
    display: flex;
    align-items: center;
    justify-content: center;
}

.hero-orb {
    width: 150px;
    height: 150px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    background: radial-gradient(circle at 35% 30%, rgba(96,165,250,0.35), rgba(124,58,237,0.12) 45%, rgba(15,23,42,0.9) 70%);
    border: 1px solid rgba(129,140,248,0.35);
    box-shadow: 0 0 50px rgba(99,102,241,0.22), inset 0 0 35px rgba(96,165,250,0.08);
    animation: heroPulse 4s ease-in-out infinite;
}

.hero-orb-icon {
    font-size: 58px;
    filter: drop-shadow(0 0 18px rgba(96,165,250,0.45));
}

@keyframes heroPulse {
    0%, 100% { transform: scale(1); }
    50% { transform: scale(1.05); }
}

/* =========================================================
   SECTION HEADER
   ========================================================= */

.section-header {
    width: 100%;
    min-height: 90px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-sizing: border-box;
    margin-top: 30px;
    margin-bottom: 22px;
    padding: 18px 24px;
    border-radius: 18px;
    background: linear-gradient(145deg, rgba(15,23,42,0.95), rgba(17,24,39,0.82));
    border: 1px solid rgba(148,163,184,0.14);
    box-shadow: 0 15px 45px rgba(0,0,0,0.18), inset 0 1px 0 rgba(255,255,255,0.03);
    position: relative;
    overflow: hidden;
}

.section-header::before {
    content: "";
    position: absolute;
    width: 180px;
    height: 180px;
    right: -80px;
    top: -100px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(99,102,241,0.12), transparent 70%);
    pointer-events: none;
}

.section-header-left {
    display: flex;
    align-items: center;
    gap: 16px;
    position: relative;
    z-index: 2;
}

.section-icon {
    width: 52px;
    height: 52px;
    flex-shrink: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 14px;
    background: linear-gradient(135deg, rgba(37,99,235,0.20), rgba(124,58,237,0.20));
    border: 1px solid rgba(96,165,250,0.22);
    font-size: 23px;
    box-shadow: 0 8px 25px rgba(37,99,235,0.12);
}

.section-title {
    margin: 0;
    color: #f8fafc;
    font-size: 22px;
    font-weight: 800;
    line-height: 1.2;
    letter-spacing: -0.5px;
}

.section-description {
    margin-top: 6px;
    color: #64748b;
    font-size: 13px;
    font-weight: 500;
    line-height: 1.5;
}

.section-status {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    flex-shrink: 0;
    padding: 9px 14px;
    border-radius: 999px;
    background: rgba(34,197,94,0.08);
    border: 1px solid rgba(34,197,94,0.18);
    color: #86efac;
    font-size: 11px;
    font-weight: 700;
    white-space: nowrap;
    position: relative;
    z-index: 2;
}

.section-status-dot, .live-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #4ade80;
    box-shadow: 0 0 12px rgba(74,222,128,0.85);
    display: inline-block;
    animation: livePulse 1.8s infinite;
}

@keyframes livePulse {
    0%,100% { opacity: 1; transform: scale(1); }
    50% { opacity: .45; transform: scale(.75); }
}

/* =========================================================
   INPUTS
   ========================================================= */

div[data-testid="stNumberInput"] {
    padding: 16px 18px 14px 18px;
    margin-bottom: 14px;
    border-radius: 18px;
    background: linear-gradient(145deg, rgba(20,26,39,0.92), rgba(15,23,42,0.88));
    border: 1px solid rgba(255,255,255,0.07);
    box-shadow: 0 8px 30px rgba(0,0,0,0.14);
    transition: transform .2s ease, border-color .2s ease, box-shadow .2s ease;
}

div[data-testid="stNumberInput"]:hover {
    transform: translateY(-2px);
    border-color: rgba(96,165,250,0.25);
    box-shadow: 0 12px 35px rgba(0,0,0,0.22);
}

div[data-testid="stNumberInput"] label {
    color: #cbd5e1 !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    margin-bottom: 8px !important;
}

div[data-baseweb="input"] {
    background: rgba(15,23,42,0.95) !important;
    border-radius: 12px !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    min-height: 42px;
}

div[data-baseweb="input"]:focus-within {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.12) !important;
}

div[data-baseweb="input"] input {
    color: #f8fafc !important;
    font-size: 14px !important;
    font-weight: 500 !important;
}

button[data-testid="stNumberInputStepDown"],
button[data-testid="stNumberInputStepUp"] {
    color: #94a3b8 !important;
    background: rgba(255,255,255,0.035) !important;
    border: none !important;
}

/* =========================================================
   BUTTON
   ========================================================= */

.stButton > button {
    width: 100%;
    height: 52px;
    border-radius: 14px;
    border: none;
    background: linear-gradient(135deg, #2563eb, #7c3aed);
    color: white;
    font-size: 16px;
    font-weight: 700;
    box-shadow: 0 10px 30px rgba(37,99,235,0.25);
    transition: all .2s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 15px 35px rgba(37,99,235,0.35);
}

/* =========================================================
   AQI RESULT CARDS
   ========================================================= */

.prediction-result-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-top: 45px;
    margin-bottom: 22px;
}

.prediction-result-title {
    display: flex;
    align-items: center;
    gap: 12px;
}

.prediction-result-icon {
    width: 46px;
    height: 46px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 13px;
    background: linear-gradient(135deg, rgba(37,99,235,0.18), rgba(124,58,237,0.18));
    border: 1px solid rgba(96,165,250,0.20);
    font-size: 22px;
}

.prediction-result-heading {
    color: #f8fafc;
    font-size: 22px;
    font-weight: 800;
}

.prediction-result-subtitle {
    margin-top: 4px;
    color: #64748b;
    font-size: 12px;
}

.result-card, .stat-card, .chart-card, .precaution-card {
    background: linear-gradient(145deg, rgba(15,23,42,0.96), rgba(20,27,42,0.88));
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px;
    box-shadow: 0 15px 45px rgba(0,0,0,0.20);
}

.result-card {
    padding: 30px;
    min-height: 220px;
}

.result-label, .stat-label {
    color: #94a3b8;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: .7px;
    text-transform: uppercase;
}

.result-value {
    margin-top: 13px;
    color: #f8fafc;
    font-size: 52px;
    font-weight: 800;
    line-height: 1.1;
}

.result-unit {
    color: #64748b;
    font-size: 12px;
    margin-top: 5px;
}

.category-badge {
    display: inline-block;
    padding: 7px 15px;
    border-radius: 30px;
    font-size: 13px;
    font-weight: 700;
    margin-top: 12px;
}

.advice-box {
    background: rgba(255,255,255,0.035);
    border-left: 4px solid #3b82f6;
    border-radius: 10px;
    padding: 12px 15px;
    margin-top: 15px;
    color: #cbd5e1;
    font-size: 13px;
}

.stat-card {
    padding: 24px;
    min-height: 150px;
}

.stat-value {
    color: #f8fafc;
    font-size: 28px;
    font-weight: 800;
    margin-top: 10px;
}

.stat-description {
    margin-top: 12px;
    font-size: 12px;
    color: #64748b;
}

/* =========================================================
   CHARTS
   ========================================================= */

[data-testid="stVerticalBlockBorderWrapper"] {
    border-color: rgba(255,255,255,0.08) !important;
    border-radius: 20px !important;
    background: linear-gradient(145deg, rgba(15,23,42,0.96), rgba(17,24,39,0.82));
    box-shadow: 0 20px 60px rgba(0,0,0,0.18);
}

.chart-title {
    color: #cbd5e1;
    font-size: 14px;
    font-weight: 700;
    margin-bottom: 12px;
}

/* Matplotlib dark canvas */
.stPlotlyChart, .stImage {
    border-radius: 16px;
}

/* =========================================================
   HEALTH
   ========================================================= */

.precaution-card {
    padding: 24px;
}

.precaution-title {
    color: #f8fafc;
    font-size: 18px;
    font-weight: 800;
    margin-bottom: 12px;
}

.precaution-item {
    padding: 9px 0;
    color: #cbd5e1;
    font-size: 14px;
    border-bottom: 1px solid rgba(255,255,255,0.04);
}

.precaution-item:last-child {
    border-bottom: none;
}

/* =========================================================
   FOOTER — same visual language as first app
   ========================================================= */

.clinical-dashboard-footer {
    width: 100%;
    min-height: 72px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-sizing: border-box;
    margin-top: 25px;
    margin-bottom: 28px;
    padding: 16px 22px;
    border-radius: 18px;
    background: linear-gradient(145deg, rgba(15,23,42,0.95), rgba(17,24,39,0.82));
    border: 1px solid rgba(148,163,184,0.14);
    box-shadow: 0 15px 45px rgba(0,0,0,0.18), inset 0 1px 0 rgba(255,255,255,0.03);
}

.clinical-dashboard-footer .footer-item {
    display: flex;
    align-items: center;
    gap: 8px;
    color: #b2b4b6;
    font-size: 13px;
    font-weight: 500;
    white-space: nowrap;
}

.clinical-dashboard-footer .footer-icon {
    width: 30px;
    height: 30px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 9px;
    background: linear-gradient(135deg, rgba(37,99,235,0.12), rgba(124,58,237,0.12));
    border: 1px solid rgba(96,165,250,0.15);
    font-size: 14px;
}

.clinical-dashboard-footer b {
    color: #cbd5e1;
}

.footer {
    text-align: center;
    margin-top: 35px;
    padding-top: 20px;
    border-top: 1px solid rgba(255,255,255,0.08);
    color: #64748b;
    font-size: 12px;
}

@media (max-width: 900px) {
    .hero { padding: 34px 30px; }
    .hero-visual { opacity: .25; right: 15px; }
}

@media (max-width: 700px) {
    .section-header { align-items: flex-start; }
    .section-title { font-size: 18px; }
    .section-description { font-size: 11px; }
    .section-status { display: none; }
    .clinical-dashboard-footer { flex-direction: column; align-items: flex-start; gap: 12px; }
    .clinical-dashboard-footer .footer-item { white-space: normal; }
}

@media (max-width: 520px) {
    .hero { padding: 28px 22px; min-height: 300px; }
    .hero-title { font-size: 34px; letter-spacing: -1px; }
    .hero-subtitle { font-size: 14px; }
    .hero-visual { display: none; }
    .section-header { flex-direction: column; gap: 12px; }
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD MODEL — unchanged functionality
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "aqi_model.pkl")

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"Unable to load AQI model: {e}")
    st.stop()

# =========================================================
# SIDEBAR — redesigned to match first app
# =========================================================

with st.sidebar:

    st.markdown(
        '<div class="sidebar-logo">'
        '<div class="sidebar-logo-icon">🌍</div>'
        '<div class="sidebar-title">AQI AI</div>'
        '<div class="sidebar-subtitle">Air Quality Intelligence</div>'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-section">MENU</div>'
        '<div class="sidebar-menu-item">📊 &nbsp; Dashboard</div>'
        '<div class="sidebar-menu-item">🔬 &nbsp; AQI Prediction</div>'
        '<div class="sidebar-menu-item">📈 &nbsp; Pollutant Analysis</div>'
        '<div class="sidebar-menu-item">🩺 &nbsp; Health Guidance</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-section">MODEL</div>'
        '<div class="sidebar-info">'
        '<b>🤖 AI Prediction Model</b><br>'
        '<span style="font-size:11px;color:#94a3b8;line-height:1.6;">'
        'Machine Learning based AQI estimation using pollutant concentrations.'
        '</span>'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-section">ABOUT</div>'
        '<div class="sidebar-about">'
        '🌍 AQI AI analyzes major air pollutants and estimates the Air Quality Index.'
        '</div>',
        unsafe_allow_html=True
    )

# =========================================================
# HERO — visual only
# =========================================================

st.markdown(
    '<div class="hero">'
    '<div class="hero-content">'
    '<div class="badge"><span class="badge-dot"></span>AI-POWERED AIR QUALITY ANALYTICS</div>'
    '<div class="hero-title">Air Quality <span>Prediction</span></div>'
    '<div class="hero-subtitle">'
    'Analyze pollutant concentrations and estimate the current Air Quality Index using a machine learning model.'
    '</div>'
    '<div class="hero-meta">'
    '<div class="hero-meta-item">🤖 Machine Learning</div>'
    '<div class="hero-meta-item">⚡ Instant Prediction</div>'
    '<div class="hero-meta-item">🌫️ 8 Pollutants</div>'
    '</div>'
    '</div>'
    '<div class="hero-visual"><div class="hero-orb"><div class="hero-orb-icon">🌍</div></div></div>'
    '</div>',
    unsafe_allow_html=True
)

# =========================================================
# AQI FUNCTION — unchanged
# =========================================================

def get_aqi_info(aqi):
    if aqi <= 50:
        return (
            "Good",
            "#16a34a",
            "Perfect day for outdoor activities! 🌳"
        )
    elif aqi <= 100:
        return (
            "Moderate",
            "#ca8a04",
            "Air is acceptable, but sensitive people should be cautious."
        )
    elif aqi <= 200:
        return (
            "Poor",
            "#ea580c",
            "Limit outdoor exercise and consider wearing a mask."
        )
    elif aqi <= 300:
        return (
            "Very Poor",
            "#dc2626",
            "Avoid prolonged outdoor activities and keep windows closed."
        )
    else:
        return (
            "Severe",
            "#991b1b",
            "Stay indoors and use an air purifier if available. 🚨"
        )

# =========================================================
# INPUT SECTION — functionality unchanged
# =========================================================

st.markdown(
    '<div class="section-header">'
    '<div class="section-header-left">'
    '<div class="section-icon">🧪</div>'
    '<div>'
    '<div class="section-title">Pollution Parameters</div>'
    '<div class="section-description">Enter the measured pollutant concentrations below.</div>'
    '</div>'
    '</div>'
    '<div class="section-status"><span class="section-status-dot"></span>8 Parameters</div>'
    '</div>',
    unsafe_allow_html=True
)

with st.container(border=True, key="input_card"):

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        pm25 = st.number_input("PM2.5", min_value=0.0, value=0.0, step=0.1)
        pm10 = st.number_input("PM10", min_value=0.0, value=0.0, step=0.1)

    with col2:
        no = st.number_input("NO", min_value=0.0, value=0.0, step=0.1)
        no2 = st.number_input("NO2", min_value=0.0, value=0.0, step=0.1)

    with col3:
        nh3 = st.number_input("NH3", min_value=0.0, value=0.0, step=0.1)
        co = st.number_input("CO", min_value=0.0, value=0.0, step=0.1)

    with col4:
        so2 = st.number_input("SO2", min_value=0.0, value=0.0, step=0.1)
        o3 = st.number_input("O3", min_value=0.0, value=0.0, step=0.1)

pollutants = [pm25, pm10, no, no2, nh3, co, so2, o3]

labels = [
    "PM2.5", "PM10", "NO", "NO2", "NH3", "CO", "SO2", "O3"
]

# =========================================================
# PREDICT BUTTON — unchanged
# =========================================================

st.write("")
predict_col1, predict_col2, predict_col3 = st.columns([1, 2, 1])

with predict_col2:
    predict_button = st.button(
        "🔍 Predict Air Quality Index",
        use_container_width=True
    )

# =========================================================
# PREDICTION — unchanged functionality
# =========================================================

if predict_button:

    input_data = np.array([pollutants])

    try:
        prediction = float(model.predict(input_data)[0])
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    category, color, message = get_aqi_info(prediction)

    # =====================================================
    # RESULT HEADER
    # =====================================================

    st.markdown(
        '<div class="prediction-result-header">'
        '<div class="prediction-result-title">'
        '<div class="prediction-result-icon">📊</div>'
        '<div>'
        '<div class="prediction-result-heading">Prediction Result</div>'
        '<div class="prediction-result-subtitle">AI-generated air quality assessment</div>'
        '</div>'
        '</div>'
        '<div class="section-status"><span class="section-status-dot"></span>ANALYSIS COMPLETE</div>'
        '</div>',
        unsafe_allow_html=True
    )

    # =====================================================
    # RESULT CARDS — values/functionality unchanged
    # =====================================================

    result_col1, result_col2, result_col3 = st.columns([1.4, 1, 1])

    with result_col1:
        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">Predicted Air Quality Index</div>
            <div class="result-value">{prediction:.2f}</div>
            <div class="result-unit">AQI Score</div>
            <div class="category-badge" style="background:{color}20;color:{color};">{category}</div>
            <div class="advice-box">💡 {message}</div>
        </div>
        """, unsafe_allow_html=True)

    with result_col2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">AIR QUALITY STATUS</div>
            <div class="stat-value" style="color:{color};">{category}</div>
            <div class="stat-description">Current estimated air quality</div>
        </div>
        """, unsafe_allow_html=True)

    with result_col3:
        max_index = int(np.argmax(pollutants))
        max_pollutant = labels[max_index]
        max_value = pollutants[max_index]

        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">HIGHEST POLLUTANT</div>
            <div class="stat-value">{max_pollutant}</div>
            <div class="stat-description">Concentration: {max_value:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    # =====================================================
    # CHART SECTION — functionality unchanged
    # =====================================================

    st.markdown(
        '<div class="section-header">'
        '<div class="section-header-left">'
        '<div class="section-icon">📈</div>'
        '<div>'
        '<div class="section-title">Pollutant Analysis</div>'
        '<div class="section-description">Visual analysis of the pollutant measurements used for prediction.</div>'
        '</div>'
        '</div>'
        '<div class="section-status"><span class="section-status-dot"></span>LIVE ANALYSIS</div>'
        '</div>',
        unsafe_allow_html=True
    )

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        with st.container(border=True, key="bar_chart_card"):
            st.markdown('<div class="chart-title">📊 Pollutant Concentration Profile</div>', unsafe_allow_html=True)

            fig, ax = plt.subplots(figsize=(7, 4))
            fig.patch.set_facecolor('#0f172a')
            ax.set_facecolor('#0f172a')
            ax.bar(labels, pollutants)
            ax.set_ylabel("Concentration", color='#cbd5e1')
            ax.set_title("Pollutant Levels", color='#f8fafc')
            ax.tick_params(axis="x", rotation=35, colors='#94a3b8')
            ax.tick_params(axis="y", colors='#94a3b8')
            ax.spines['bottom'].set_color('#334155')
            ax.spines['left'].set_color('#334155')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    with chart_col2:
        with st.container(border=True, key="pie_chart_card"):
            st.markdown('<div class="chart-title">🥧 Pollutant Distribution</div>', unsafe_allow_html=True)

            fig2, ax2 = plt.subplots(figsize=(7, 4))
            fig2.patch.set_facecolor('#0f172a')
            ax2.set_facecolor('#0f172a')

            values = np.array(pollutants)

            if values.sum() > 0:
                ax2.pie(
                    values,
                    labels=labels,
                    autopct="%1.1f%%",
                    startangle=90,
                    textprops={'color': '#cbd5e1'}
                )
                ax2.set_title("Pollutant Contribution", color='#f8fafc')
            else:
                ax2.text(
                    0.5, 0.5,
                    "Enter pollutant values",
                    ha="center",
                    va="center",
                    color='#94a3b8'
                )
                ax2.axis("off")

            plt.tight_layout()
            st.pyplot(fig2, use_container_width=True)
            plt.close(fig2)

    # =====================================================
    # HEALTH PRECAUTIONS — unchanged functionality
    # =====================================================

    st.markdown(
        '<div class="section-header">'
        '<div class="section-header-left">'
        '<div class="section-icon">🩺</div>'
        '<div>'
        '<div class="section-title">Health Recommendations</div>'
        '<div class="section-description">Suggested precautions based on the predicted AQI level.</div>'
        '</div>'
        '</div>'
        '<div class="section-status"><span class="section-status-dot"></span>GUIDANCE</div>'
        '</div>',
        unsafe_allow_html=True
    )

    if prediction > 200:
        precaution_text = """
        <div class="precaution-card">
            <div class="precaution-title">🚨 High Pollution Alert</div>
            <div class="precaution-item">😷 Wear a mask when going outside</div>
            <div class="precaution-item">🏃 Avoid outdoor workouts</div>
            <div class="precaution-item">🏠 Keep windows closed when outdoor pollution is high</div>
            <div class="precaution-item">🌬️ Use an air purifier indoors if available</div>
            <div class="precaution-item">💧 Stay hydrated</div>
        </div>
        """

    elif prediction > 100:
        precaution_text = """
        <div class="precaution-card">
            <div class="precaution-title">⚠️ Moderate Health Concern</div>
            <div class="precaution-item">😷 Consider a mask during prolonged outdoor exposure</div>
            <div class="precaution-item">🏃 Reduce intense outdoor exercise</div>
            <div class="precaution-item">🌳 Prefer cleaner outdoor areas</div>
            <div class="precaution-item">👶 Sensitive individuals should take extra care</div>
        </div>
        """

    else:
        precaution_text = """
        <div class="precaution-card">
            <div class="precaution-title">✅ Air Quality is Relatively Good</div>
            <div class="precaution-item">🌳 Outdoor activities are generally suitable</div>
            <div class="precaution-item">🚶 Walking and cycling can be enjoyed</div>
            <div class="precaution-item">🌬️ Maintain good indoor ventilation</div>
            <div class="precaution-item">💚 Continue healthy outdoor habits</div>
        </div>
        """

    st.markdown(precaution_text, unsafe_allow_html=True)

    # =====================================================
    # STATUS FOOTER — same style as first app
    # =====================================================

    st.markdown(
        '<div class="clinical-dashboard-footer">'
        '<div class="footer-item">'
        '<span class="footer-icon">🌫️</span>'
        '<span><b>8</b> pollutants analyzed</span>'
        '</div>'
        '<div class="footer-item">'
        '<span class="footer-icon">⚡</span>'
        '<span>Ready for machine learning prediction</span>'
        '</div>'
        '<div class="footer-item">'
        '<span class="footer-icon">🔒</span>'
        '<span>Application-level data processing</span>'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )

# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<div class="footer">
    AQI AI • Machine Learning Air Quality Prediction System<br>
    Built with Python • Streamlit • Scikit-learn
</div>
""", unsafe_allow_html=True)
