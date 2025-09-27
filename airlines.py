
import streamlit as st
import pandas as pd
import pickle

# Load model & scaler
with open("airlines_delay.pkl", "rb") as f:
    dict1 = pickle.load(f)

model = dict1['model']
columns = dict1['columns']
scaler = dict1['scaler'] 

st.title("Airlines Delay Prediction")

airports = pd.read_csv("Airlines.csv")["AirportFrom"].unique()

# Inputs
airline = st.selectbox("Airline Code:", ['CO', 'US', 'AA', 'AS', 'DL', 'B6', 'HA', 'OO', '9E', 'OH', 'EV','XE', 'YV', 'UA', 'MQ', 'FL', 'F9', 'WN'])
airportfrom = st.selectbox("AirportFrom:", airports)
airportto = st.selectbox("AirportTo:", airports)
day = st.selectbox("Day of the Week:", range(1,8))
time = st.number_input("Take off time (mins from midnight):", 1, 1439)
length = st.number_input("Flight duration (mins):", 0, 655)

# Prediction
if st.button("PREDICT"):
    input_df = pd.DataFrame({
        'Airline': [airline],
        'AirportFrom': [airportfrom],
        'AirportTo': [airportto],
        'Day': [day],
        'Time': [time],
        'Length': [length]
    })
    
    # Align with training features
    input_dummies = pd.get_dummies(input_df)
    input_aligned = input_dummies.reindex(columns=columns, fill_value=0)

    # Scaler
    input_scaled = scaler.transform(input_aligned)

    # Predict
    pred = model.predict(input_scaled)[0]
    prediction_text = "Delayed" if pred == 1 else "On Time"
    st.success(f"Prediction: **{prediction_text}**")
