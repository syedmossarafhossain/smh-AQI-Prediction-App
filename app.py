import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load model
import joblib
model = joblib.load("Model/aqi_model.pkl")

st.title("🌫 Air Quality Index Prediction App")
st.write("Enter pollutant values to predict AQI")

st.divider()

# User Inputs
pm25 = st.number_input("PM2.5", min_value=0.0)
pm10 = st.number_input("PM10", min_value=0.0)
no = st.number_input("NO", min_value=0.0)
no2 = st.number_input("NO2", min_value=0.0)
nh3 = st.number_input("NH3", min_value=0.0)
co = st.number_input("CO", min_value=0.0)
so2 = st.number_input("SO2", min_value=0.0)
o3 = st.number_input("O3", min_value=0.0)

pollutants = [pm25, pm10, no, no2, nh3, co, so2, o3]
labels = ["PM2.5", "PM10", "NO", "NO2", "NH3", "CO", "SO2", "O3"]


# AQI category function
def get_aqi_info(aqi):
    if aqi <= 50:
        return "Good", "green", "Perfect day for outdoor activities! 🌳"

    elif aqi <= 100:
        return "Moderate", "yellow", "Air is acceptable but sensitive people should be cautious."

    elif aqi <= 200:
        return "Poor", "orange", "Limit outdoor exercise and wear a mask."

    elif aqi <= 300:
        return "Very Poor", "red", "Avoid outdoor activities and keep windows closed."

    else:
        return "Severe", "darkred", "Stay indoors! Air purifier recommended. 🚨"


if st.button("Predict AQI"):

    input_data = np.array([pollutants])
    prediction = model.predict(input_data)[0]

    category, color, message = get_aqi_info(prediction)

    st.subheader("Prediction Result")

    st.markdown(
        f"<h2 style='color:{color};'>Predicted AQI: {prediction:.2f}</h2>",
        unsafe_allow_html=True
    )

    st.write(f"### Category: {category}")
    st.write(f"💡 **Advice:** {message}")

    st.divider()

    # Bar Graph
    st.subheader("Pollutant Levels")

    fig, ax = plt.subplots()
    ax.bar(labels, pollutants)
    ax.set_ylabel("Concentration")
    ax.set_title("Pollutant Levels")
    st.pyplot(fig)

    # Pie Chart
    st.subheader("Pollutant Contribution")

    fig2, ax2 = plt.subplots()
    ax2.pie(pollutants, labels=labels, autopct='%1.1f%%')
    ax2.set_title("Pollutant Distribution")
    st.pyplot(fig2)

    st.divider()

    # Precautions
    st.subheader("Health Precautions")

    if prediction > 200:
        st.warning("""
        • Wear a mask when going outside  
        • Avoid outdoor workouts  
        • Use air purifiers indoors  
        • Drink plenty of water  
        """)

    else:
        st.success("""
        • Safe for outdoor activities  
        • Open windows for fresh air  
        • Walking or cycling recommended  

        """)


