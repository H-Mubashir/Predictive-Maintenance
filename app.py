import streamlit as st
import pandas as pd
import joblib
import json

st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    layout="wide"
)

@st.cache_resource
def load_model():
    return joblib.load(
        "Predictive_Maintenance_Model_20260714_152250.joblib"
    )

@st.cache_data
def load_dashboard():
    return pd.read_csv(
        "Dashboard_Data_20260714_152250.csv"
    )

@st.cache_data
def load_config():
    with open(
        "Model_Config_20260714_152250.json",
        "r"
    ) as f:
        return json.load(f)

model = load_model()
dashboard = load_dashboard()
config = load_config()

st.title(
    "Predictive Maintenance and Reliability Early Warning System"
)

st.markdown("---")

critical = (
    dashboard["Risk Level"] == "Critical"
).sum()

col1,col2,col3,col4 = st.columns(4)

col1.metric(
    "Machines",
    len(dashboard)
)

col2.metric(
    "Average EHI",
    round(
        dashboard[
            "Equipment Health Risk Index"
        ].mean(),
        2
    )
)

col3.metric(
    "Critical Machines",
    critical
)

col4.metric(
    "Decision Threshold",
    round(
        config["Threshold"],
        2
    )
)

st.markdown("---")

st.subheader(
    "Risk Level Distribution"
)

st.bar_chart(
    dashboard[
        "Risk Level"
    ].value_counts()
)

st.markdown("---")

st.subheader(
    "Top 20 Highest Risk Machines"
)

top20 = dashboard.sort_values(
    by="Equipment Health Risk Index",
    ascending=False
).head(20)

st.dataframe(
    top20,
    use_container_width=True
)

st.markdown("---")

st.subheader(
    "Live Prediction"
)

air = st.number_input(
    "Air Temperature (K)",
    value=300.0
)

process = st.number_input(
    "Process Temperature (K)",
    value=310.0
)

speed = st.number_input(
    "Rotational Speed (rpm)",
    value=1500
)

torque = st.number_input(
    "Torque (Nm)",
    value=40.0
)

wear = st.number_input(
    "Tool Wear (min)",
    value=100
)

if st.button("Predict"):

    power = torque * speed

    wear_intensity = torque * wear

    thermal = process * torque

    sample = pd.DataFrame({

        "Air_temperature_K":[air],
        "Process_temperature_K":[process],
        "Rotational_speed_rpm":[speed],
        "Torque_Nm":[torque],
        "Tool_wear_min":[wear],
        "Power_Index":[power],
        "Wear_Intensity":[wear_intensity],
        "Thermal_Stress_Index":[thermal]

    })

    probability = model.predict_proba(
        sample
    )[0][1]

    ehi = probability * 100

    st.metric(
        "Failure Probability",
        f"{probability:.3f}"
    )

    st.metric(
        "Equipment Health Risk Index",
        f"{ehi:.2f}"
    )