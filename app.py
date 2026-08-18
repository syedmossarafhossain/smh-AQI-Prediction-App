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
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

    /* -------------------------------
       GLOBAL
    --------------------------------*/

    .stApp {
        background: #f5f7fb;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    /* -------------------------------
       SIDEBAR
    --------------------------------*/

    section[data-testid="stSidebar"] {
        background: #111827;
        border-right: 1px solid #1f2937;
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
    }

    .sidebar-title {
        font-size: 22px;
        font-weight: 700;
        color: white;
        margin-top: 5px;
    }

    .sidebar-subtitle {
        font-size: 12px;
        color: #9ca3af;
    }

    .sidebar-section {
        font-size: 11px;
        color: #6b7280;
        font-weight: 700;
        letter-spacing: 1px;
        margin-top: 25px;
        margin-bottom: 8px;
    }

    .sidebar-info {
        background: #1f2937;
        border: 1px solid #374151;
        border-radius: 12px;
        padding: 14px;
        margin-top: 20px;
    }

    /* -------------------------------
       HEADER
    --------------------------------*/

    .dashboard-header {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 18px;
        padding: 24px 28px;
        margin-bottom: 22px;
        box-shadow: 0 4px 18px rgba(15, 23, 42, 0.05);
    }

    .header-small {
        color: #6b7280;
        font-size: 13px;
        font-weight: 600;
        letter-spacing: .5px;
        text-transform: uppercase;
    }

    .header-title {
        color: #111827;
        font-size: 32px;
        font-weight: 800;
        margin: 4px 0;
    }

    .header-description {
        color: #6b7280;
        font-size: 14px;
    }

    /* -------------------------------
       SECTION TITLES
    --------------------------------*/

    .section-title {
        font-size: 20px;
        font-weight: 750;
        color: #111827;
        margin: 15px 0 12px 0;
    }

    .section-subtitle {
        color: #6b7280;
        font-size: 13px;
        margin-bottom: 15px;
    }

    /* -------------------------------
       INPUT CARD
    --------------------------------*/

    .input-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 18px;
        padding: 20px;
        box-shadow: 0 4px 18px rgba(15, 23, 42, 0.04);
        margin-bottom: 20px;
    }

    /* -------------------------------
       INPUT LABELS
    --------------------------------*/

    label {
        font-weight: 600 !important;
        color: #374151 !important;
    }

    div[data-baseweb="input"] {
        border-radius: 10px;
    }

    /* -------------------------------
       BUTTON
    --------------------------------*/

    .stButton > button {
        width: 100%;
        border-radius: 11px;
        height: 48px;
        border: none;
        font-weight: 700;
        font-size: 15px;
        background: #2563eb;
        color: white;
        transition: all .2s ease;
    }

    .stButton > button:hover {
        background: #1d4ed8;
        transform: translateY(-1px);
    }

    /* -------------------------------
       RESULT CARD
    --------------------------------*/

    .result-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 18px;
        padding: 25px;
        min-height: 220px;
        box-shadow: 0 4px 18px rgba(15, 23, 42, 0.05);
    }

    .result-label {
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #6b7280;
        font-weight: 700;
    }

    .result-value {
        font-size: 48px;
        font-weight: 850;
        color: #111827;
        margin: 5px 0;
    }

    .result-unit {
        color: #6b7280;
        font-size: 13px;
    }

    .category-badge {
        display: inline-block;
        padding: 7px 15px;
        border-radius: 30px;
        font-size: 13px;
        font-weight: 700;
        margin-top: 10px;
    }

    .advice-box {
        background: #f8fafc;
        border-left: 4px solid #2563eb;
        border-radius: 10px;
        padding: 12px 15px;
        margin-top: 15px;
        color: #374151;
        font-size: 13px;
    }

    /* -------------------------------
       STAT CARDS
    --------------------------------*/

    .stat-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 18px;
        box-shadow: 0 3px 15px rgba(15, 23, 42, 0.04);
    }

    .stat-label {
        color: #6b7280;
        font-size: 12px;
        font-weight: 600;
    }

    .stat-value {
        color: #111827;
        font-size: 25px;
        font-weight: 800;
        margin-top: 5px;
    }

    /* -------------------------------
       CHART CARDS
    --------------------------------*/

    .chart-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 18px;
        padding: 20px;
        box-shadow: 0 4px 18px rgba(15, 23, 42, 0.04);
    }

    /* -------------------------------
       PRECAUTION CARD
    --------------------------------*/

    .precaution-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 18px;
        padding: 22px;
        box-shadow: 0 4px 18px rgba(15, 23, 42, 0.04);
    }

    .precaution-title {
        color: #111827;
        font-size: 18px;
        font-weight: 750;
        margin-bottom: 12px;
    }

    .precaution-item {
        padding: 9px 0;
        color: #4b5563;
        font-size: 14px;
    }

    /* -------------------------------
       FOOTER
    --------------------------------*/

    .footer {
        text-align: center;
        color: #9ca3af;
        font-size: 12px;
        padding: 30px 0 10px 0;
    }

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODEL
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "aqi_model.pkl")

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"Unable to load AQI model: {e}")
    st.stop()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    # Logo / Branding
    st.markdown("""
        <div class="sidebar-logo">
            <div class="sidebar-logo-icon">🌫️</div>
            <div class="sidebar-title">AQI AI</div>
            <div class="sidebar-subtitle">
                Air Quality Intelligence
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Menu
    st.markdown(
        '<div class="sidebar-section">MENU</div>',
        unsafe_allow_html=True
    )

    st.markdown("📊 **Dashboard**")
    st.markdown("🔬 **AQI Prediction**")
    st.markdown("📈 **Pollutant Analysis**")
    st.markdown("🩺 **Health Guidance**")

    # Model Section
    st.markdown(
        '<div class="sidebar-section">MODEL</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
        <div class="sidebar-info">
            <b>AI Prediction Model</b><br>
            <span style="font-size:12px;color:#9ca3af;">
                Machine Learning based AQI estimation using
                pollutant concentrations.
            </span>
        </div>
    """, unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================

st.markdown("""
<div class="dashboard-header">

    <div class="header-small">
        🌍 AIR QUALITY INTELLIGENCE
    </div>

    <div class="header-title">
        AQI Prediction Dashboard
    </div>

    <div class="header-description">
        Analyze pollutant concentrations and estimate the current
        Air Quality Index using Machine Learning.
    </div>

</div>
""", unsafe_allow_html=True)


# =========================================================
# AQI FUNCTION
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
# INPUT SECTION
# =========================================================

st.markdown(
    '<div class="section-title">🧪 Pollution Parameters</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-subtitle">'
    'Enter the measured pollutant concentrations below.'
    '</div>',
    unsafe_allow_html=True
)

with st.container(border=True, key="input_card"):

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        pm25 = st.number_input(
            "PM2.5",
            min_value=0.0,
            value=0.0,
            step=0.1
        )

        pm10 = st.number_input(
            "PM10",
            min_value=0.0,
            value=0.0,
            step=0.1
        )

    with col2:
        no = st.number_input(
            "NO",
            min_value=0.0,
            value=0.0,
            step=0.1
        )

        no2 = st.number_input(
            "NO2",
            min_value=0.0,
            value=0.0,
            step=0.1
        )

    with col3:
        nh3 = st.number_input(
            "NH3",
            min_value=0.0,
            value=0.0,
            step=0.1
        )

        co = st.number_input(
            "CO",
            min_value=0.0,
            value=0.0,
            step=0.1
        )

    with col4:
        so2 = st.number_input(
            "SO2",
            min_value=0.0,
            value=0.0,
            step=0.1
        )

        o3 = st.number_input(
            "O3",
            min_value=0.0,
            value=0.0,
            step=0.1
        )


pollutants = [
    pm25,
    pm10,
    no,
    no2,
    nh3,
    co,
    so2,
    o3
]

labels = [
    "PM2.5",
    "PM10",
    "NO",
    "NO2",
    "NH3",
    "CO",
    "SO2",
    "O3"
]


# =========================================================
# PREDICT BUTTON
# =========================================================

st.write("")

predict_col1, predict_col2, predict_col3 = st.columns(
    [1, 2, 1]
)

with predict_col2:

    predict_button = st.button(
        "🔍 Predict Air Quality Index",
        use_container_width=True
    )


# =========================================================
# PREDICTION
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
        '<div class="section-title">📊 Prediction Result</div>',
        unsafe_allow_html=True
    )


    # =====================================================
    # RESULT CARDS
    # =====================================================

    result_col1, result_col2, result_col3 = st.columns(
        [1.4, 1, 1]
    )


    # AQI CARD
    with result_col1:

        st.markdown(f"""
        <div class="result-card">

            <div class="result-label">
                Predicted Air Quality Index
            </div>

            <div class="result-value">
                {prediction:.2f}
            </div>

            <div class="result-unit">
                AQI Score
            </div>

            <div class="category-badge"
                 style="background:{color}20;color:{color};">
                {category}
            </div>

            <div class="advice-box">
                💡 {message}
            </div>

        </div>
        """, unsafe_allow_html=True)


    # CATEGORY CARD
    with result_col2:

        st.markdown(f"""
        <div class="stat-card">

            <div class="stat-label">
                AIR QUALITY STATUS
            </div>

            <div class="stat-value"
                 style="color:{color};">
                {category}
            </div>

            <div style="margin-top:12px;
                        font-size:13px;
                        color:#6b7280;">
                Current estimated air quality
            </div>

        </div>
        """, unsafe_allow_html=True)


    # HIGHEST POLLUTANT CARD
    with result_col3:

        max_index = int(np.argmax(pollutants))
        max_pollutant = labels[max_index]
        max_value = pollutants[max_index]

        st.markdown(f"""
        <div class="stat-card">

            <div class="stat-label">
                HIGHEST POLLUTANT
            </div>

            <div class="stat-value">
                {max_pollutant}
            </div>

            <div style="margin-top:12px;
                        font-size:13px;
                        color:#6b7280;">
                Concentration: {max_value:.2f}
            </div>

        </div>
        """, unsafe_allow_html=True)


    # =====================================================
    # CHART SECTION
    # =====================================================

    st.markdown(
        '<div class="section-title">📈 Clinical Parameter Profile</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Visual analysis of the pollutant measurements used for prediction.'
        '</div>',
        unsafe_allow_html=True
    )


    chart_col1, chart_col2 = st.columns(2)


    # -----------------------------------------------------
    # BAR CHART
    # -----------------------------------------------------

    with chart_col1:

        with st.container(border=True, key="bar_chart_card"):

            st.markdown(
                "**Pollutant Concentration Profile**"
            )

            fig, ax = plt.subplots(
                figsize=(7, 4)
            )

            ax.bar(
                labels,
                pollutants
            )

            ax.set_ylabel(
                "Concentration"
            )

            ax.set_title(
                "Pollutant Levels"
            )

            ax.tick_params(
                axis="x",
                rotation=35
            )

            plt.tight_layout()

            st.pyplot(
                fig,
                use_container_width=True
            )


    # -----------------------------------------------------
    # PIE CHART
    # -----------------------------------------------------

    with chart_col2:

        with st.container(border=True, key="pie_chart_card"):

            st.markdown(
                "**Pollutant Distribution**"
            )

            fig2, ax2 = plt.subplots(
                figsize=(7, 4)
            )

            values = np.array(pollutants)

            if values.sum() > 0:

                ax2.pie(
                    values,
                    labels=labels,
                    autopct="%1.1f%%",
                    startangle=90
                )

                ax2.set_title(
                    "Pollutant Contribution"
                )

            else:

                ax2.text(
                    0.5,
                    0.5,
                    "Enter pollutant values",
                    ha="center",
                    va="center"
                )

                ax2.axis("off")

            plt.tight_layout()

            st.pyplot(
                fig2,
                use_container_width=True
            )


    # =====================================================
    # HEALTH PRECAUTIONS
    # =====================================================

    st.markdown(
        '<div class="section-title">🩺 Health Recommendations</div>',
        unsafe_allow_html=True
    )

    if prediction > 200:

        precaution_text = """
        <div class="precaution-card">

        <div class="precaution-title">
        🚨 High Pollution Alert
        </div>

        <div class="precaution-item">
        😷 Wear a mask when going outside
        </div>

        <div class="precaution-item">
        🏃 Avoid outdoor workouts
        </div>

        <div class="precaution-item">
        🏠 Keep windows closed when outdoor pollution is high
        </div>

        <div class="precaution-item">
        🌬️ Use an air purifier indoors if available
        </div>

        <div class="precaution-item">
        💧 Stay hydrated
        </div>

        </div>
        """

    elif prediction > 100:

        precaution_text = """
        <div class="precaution-card">

        <div class="precaution-title">
        ⚠️ Moderate Health Concern
        </div>

        <div class="precaution-item">
        😷 Consider a mask during prolonged outdoor exposure
        </div>

        <div class="precaution-item">
        🏃 Reduce intense outdoor exercise
        </div>

        <div class="precaution-item">
        🌳 Prefer cleaner outdoor areas
        </div>

        <div class="precaution-item">
        👶 Sensitive individuals should take extra care
        </div>

        </div>
        """

    else:

        precaution_text = """
        <div class="precaution-card">

        <div class="precaution-title">
        ✅ Air Quality is Relatively Good
        </div>

        <div class="precaution-item">
        🌳 Outdoor activities are generally suitable
        </div>

        <div class="precaution-item">
        🚶 Walking and cycling can be enjoyed
        </div>

        <div class="precaution-item">
        🌬️ Maintain good indoor ventilation
        </div>

        <div class="precaution-item">
        💚 Continue healthy outdoor habits
        </div>

        </div>
        """

    st.markdown(
        precaution_text,
        unsafe_allow_html=True
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<div class="footer">
    AQI AI • Machine Learning Air Quality Prediction System
    <br>
    Built with Python • Streamlit • Scikit-learn
</div>
""", unsafe_allow_html=True)
