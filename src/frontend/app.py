# src/frontend/app.py
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import json
import os

# Import our backend modules
from src.backend.processor import DataProcessor
from src.backend.ml_anomaly import AnomalyDetector
from src.backend.payload_monitor import PayloadMonitor

# --- Page Config ---
st.set_page_config(page_title="API Observability Platform", layout="wide")
st.title("🛡️ API Observability & Anomaly Dashboard")
st.markdown("Monitoring API Latency, Error Rates, and Payload Drift using Agentic ML.")

# --- Helper Function to Load Data ---
@st.cache_data
def load_and_process_data():
    """Loads mock data and runs it through our backend engine."""
    # 1. Load Metrics
    with open("mock_es_response.json", "r") as f:
        es_response = json.load(f)
    
    processor = DataProcessor()
    df = processor.process_es_aggregations(es_response)
    
    detector = AnomalyDetector(contamination=0.05)
    df_with_anomalies = detector.detect_anomalies(df, features=["p95_latency", "error_rate"])
    
    # 2. Load Payloads
    with open("payloads_yesterday.json", "r") as f:
        yesterday_data = json.load(f)
    with open("payloads_today.json", "r") as f:
        today_data = json.load(f)
        
    monitor = PayloadMonitor()
    drift_results = monitor.detect_data_drift(yesterday_data, today_data, target_fields=["user.account_balance"])
    
    return df_with_anomalies, drift_results

# --- Load Data ---
try:
    df_with_anomalies, drift_results = load_and_process_data()
    # Convert Polars to Pandas for easier Plotly integration
    df_pd = df_with_anomalies.to_pandas()
except Exception as e:
    st.error(f"Error loading data. Did you run the mock data generators? Details: {e}")
    st.stop()

# --- UI Layout: Top Metrics ---
st.subheader("Live API Metrics (Last 24 Hours)")
col1, col2, col3 = st.columns(3)

total_requests = df_pd["request_count"].sum()
total_anomalies = df_pd["is_anomaly"].sum()
avg_latency = df_pd["p95_latency"].mean()

col1.metric("Total Requests", f"{total_requests:,}")
col2.metric("Anomalies Detected", int(total_anomalies), delta_color="inverse")
col3.metric("Avg P95 Latency", f"{avg_latency:.2f} ms")

# --- UI Layout: Time-Series Chart ---
st.markdown("### 📈 P95 Latency & Anomaly Detection (Isolation Forest)")

# Create a Plotly line chart for latency
fig = px.line(df_pd, x="timestamp", y="p95_latency", 
              labels={"timestamp": "Time", "p95_latency": "P95 Latency (ms)"})

# Filter out the anomalies and overlay them as red dots
anomalies_df = df_pd[df_pd["is_anomaly"] == True]
fig.add_trace(
    go.Scatter(
        x=anomalies_df["timestamp"], 
        y=anomalies_df["p95_latency"],
        mode="markers",
        marker=dict(color="red", size=10, symbol="x"),
        name="Anomaly"
    )
)

st.plotly_chart(fig, use_container_width=True)

# --- UI Layout: Payload Drift ---
st.markdown("### ⚠️ Payload Statistical Drift (Evidently AI)")
st.info("Monitoring nested field: `user.account_balance`")

drift_score = drift_results.get("user.account_balance", {}).get("psi_score", 0)
is_drifted = drift_results.get("user.account_balance", {}).get("is_drifted", False)

drift_col1, drift_col2 = st.columns([1, 3])
with drift_col1:
    st.metric(
        label="Population Stability Index (PSI)", 
        value=f"{drift_score:.4f}",
        delta="Drift Detected!" if is_drifted else "Stable",
        delta_color="inverse" if is_drifted else "normal"
    )
with drift_col2:
    if is_drifted:
        st.error("🚨 **High Data Drift Detected!** The distribution of `user.account_balance` has changed significantly compared to yesterday. This could indicate a change in client behavior or a backend bug.")
    else:
        st.success("✅ Payload distribution is stable.")

# --- UI Layout: Raw Anomaly Data ---
with st.expander("View Raw Anomaly Data"):
    st.dataframe(anomalies_df[["timestamp", "request_count", "p95_latency", "error_rate", "anomaly_score"]])