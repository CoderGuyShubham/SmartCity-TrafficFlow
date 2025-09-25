import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Traffic Dashboard", layout="wide")

st.title("🚦 Smart Traffic Light Dashboard")

# Load data
@st.cache_data
def load_data():
    try:
        return pd.read_csv("traffic_log.csv")
    except FileNotFoundError:
        return pd.DataFrame(columns=["time", "direction", "vehicle"])

df = load_data()

# Show recent logs
st.subheader("📋 Recent Vehicle Detections")
st.dataframe(df.tail(20))

# Vehicle distribution
st.subheader("🚗 Vehicle Distribution")
fig1, ax1 = plt.subplots()
df["vehicle"].value_counts().plot(kind="bar", ax=ax1)
st.pyplot(fig1)

# Vehicles per direction
st.subheader("🧭 Vehicles Per Direction")
fig2, ax2 = plt.subplots()
df["direction"].value_counts().plot(kind="bar", ax=ax2)
st.pyplot(fig2)

# Timeline
st.subheader("⏳ Traffic Over Time")
df["time"] = pd.to_datetime(df["time"])
fig3, ax3 = plt.subplots()
df.groupby(df["time"].dt.hour)["vehicle"].count().plot(kind="line", ax=ax3, marker="o")
st.pyplot(fig3)
