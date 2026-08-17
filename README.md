# 🌫 Air Quality Index (AQI) Prediction App

## 📌 Overview

The **AQI Prediction App** is a Machine Learning web application that predicts the **Air Quality Index (AQI)** based on pollutant concentration levels.

The model is trained using environmental pollution data and deployed using **Streamlit** to provide an interactive web interface where users can input pollutant values and instantly receive AQI predictions.

The application also provides **visualizations, AQI categories, and health precautions** to help users understand the air quality conditions better.

---
👉Live demo : 
---

## 🚀 Features

* Predicts **Air Quality Index (AQI)** using a Machine Learning model
* Interactive **Streamlit web interface**
* **Pollutant visualization graphs** (Bar chart and Pie chart)
* Displays **AQI category** (Good, Moderate, Poor, Very Poor, Severe)
* Shows **health precautions and recommendations** based on AQI level
* Clean and simple project structure for easy understanding

---

## 🧠 Machine Learning Model

The project uses a **Random Forest Regressor** trained on air pollution data.

**Input Features**

* PM2.5
* PM10
* NO
* NO2
* NH3
* CO
* SO2
* O3

**Target**

* AQI (Air Quality Index)

---

## 📂 Project Structure

```
aqi_prediction_project/
│
├── dataset/
│   └── city_day.csv
│
├── model/
│   └── aqi_model.pkl
│
├── train_model.py
│
├── app.py
│
├── requirements.txt
│
└── README.md


---

Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Train the Model

Run the following command to train the model and generate the model file.

```bash
python train_model.py
```

This will create:

```
model/aqi_model.pkl
```

---

## 💻 Run the Streamlit App

Start the Streamlit application using:

```bash
streamlit run app.py
```

The app will open automatically in your browser.

---

## 📊 Visualizations Included

The application includes visualizations to help users understand pollution levels:

* **Bar Chart** showing pollutant concentrations
* **Pie Chart** showing pollutant contribution

---

## 🩺 AQI Categories

| AQI Range | Category  |
| --------- | --------- |
| 0 – 50    | Good      |
| 51 – 100  | Moderate  |
| 101 – 200 | Poor      |
| 201 – 300 | Very Poor |
| 301+      | Severe    |

---

## ⚠ Health Precautions

Depending on the predicted AQI, the application provides suggestions such as:

* Wearing masks outdoors
* Avoiding outdoor exercise
* Using air purifiers indoors
* Staying hydrated

---

## 🛠 Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* Streamlit
* Matplotlib

---

## 📊 Dataset

The dataset used for this project contains air pollution measurements from various Indian cities including pollutant concentrations and AQI values.

---

## 🌍 Deployment

This project can be deployed using **Streamlit Cloud** for public access.

---

## 📌 Future Improvements

* Live AQI data integration using APIs
* Interactive AQI trend visualization
* Map visualization for city-wise AQI

---

## 👩‍💻 Author

Syed Mossaraf Hossain

Bachelor of Technology in Computer Science and Engineering

---

⭐ If you found this project helpful, consider giving the repository a star!

