import streamlit as st
import pandas as pd
import joblib

model = joblib.load("xgb_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Predictive Maintenance System")

machine_type = st.selectbox(
    "Machine Type",
    ["L", "M", "H"]
)

if machine_type == "L":
    machine_type = 0
elif machine_type == "M":
    machine_type = 1
else:
    machine_type = 2

air_temp = st.number_input("Air Temperature [K]", value=300.0)
process_temp = st.number_input("Process Temperature [K]", value=310.0)
rpm = st.number_input("Rotational Speed [rpm]", value=1500)
torque = st.number_input("Torque [Nm]", value=40.0)
tool_wear = st.number_input("Tool Wear [min]", value=20)

if st.button("Predict"):

    sample = pd.DataFrame(
        [[machine_type, air_temp, process_temp, rpm, torque, tool_wear]],
        columns=[
            'Type',
            'Air temperature [K]',
            'Process temperature [K]',
            'Rotational speed [rpm]',
            'Torque [Nm]',
            'Tool wear [min]'
        ]
    )

    sample_scaled = scaler.transform(sample)

    prediction = model.predict(sample_scaled)[0]

    if prediction == 1:
        st.error("Machine Failure Predicted")
    else:
        st.success("Machine Healthy")
