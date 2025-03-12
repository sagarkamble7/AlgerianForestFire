import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Load the trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Load the scaler
with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# Streamlit UI
st.title("🔥 Fire Weather Index Prediction App 🔥")

# Input fields with realistic value ranges
temperature = st.number_input("Temperature (°C)", min_value=-10, max_value=40, value=25)
rh = st.number_input("Relative Humidity (%)", min_value=10, max_value=100, value=50)
ws = st.number_input("Wind Speed (km/h)", min_value=0, max_value=30, value=10)
rain = st.number_input("Rain (mm)", min_value=0.0, max_value=20.0, value=0.0)
ffmc = st.number_input("FFMC", min_value=0.0, max_value=100.0, value=80.0)
dmc = st.number_input("DMC", min_value=0.0, max_value=300.0, value=10.0)
dc = st.number_input("DC", min_value=0.0, max_value=800.0, value=50.0)
isi = st.number_input("ISI", min_value=0.0, max_value=50.0, value=5.0)
bui = st.number_input("BUI", min_value=0.0, max_value=200.0, value=10.0)

# Prediction
if st.button("Predict FWI"):
    input_data = np.array([[temperature, rh, ws, rain, ffmc, dmc, dc, isi, bui]]).astype(float)

    # Scaling and Prediction
    try:
        scaled_data = scaler.transform(input_data)
        prediction = model.predict(scaled_data)[0]

        # Inverse scaling logic (if the model predicts a scaled value)
        fwi_unscaled = prediction * scaler.scale_[-1] + scaler.mean_[-1]

        st.success(f"Predicted FWI: {fwi_unscaled:.2f}")
    except Exception as e:
        st.error(f"Error in prediction: {e}")

st.markdown("---")
st.markdown("**Note:** This app predicts the Fire Weather Index (FWI) based on provided weather conditions.")
