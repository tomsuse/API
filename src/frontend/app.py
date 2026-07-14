# src/frontend/app.py
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
import yaml
import json

# --- Page Config ---
st.set_page_config(page_title="API Observability (V2 Ensemble)", layout="wide")
st.title("🛡️ API Observability: V2 Ensemble AI")
st.markdown("Comparing Isolation Forest, Amazon Chronos, and IBM TTM.")

# --- Load Config & Data ---
@st.cache_data
def load_data():
    # Load Config
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Load V2 Results
    try:
        df = pl.read_json("v2_ensemble_results.json").to_pandas()
    except Exception:
        st.error("Could not find v2_ensemble_results.json. Did you run main_engine.py?")
        st.stop()
        
    # Load Payload Drift Results
    with open("payloads_yesterday.json", "r") as f:
        yesterday_data = json.load(f)
    with open("payloads_today.json", "r") as f:
        today_data = json.load(f)
        
    from src.backend.payload_monitor import PayloadMonitor
    monitor = PayloadMonitor()
    drift_results = monitor.detect_data_drift(yesterday_data, today_data, target_fields=["user.account_balance"])
        
    return config, df, drift_results

config, df_pd, drift_results = load_data()
models_cfg = config["apis"][0]["anomaly_detection"]["models"]

# --- UI Layout: Top Metrics ---
st.subheader("Ensemble Model Performance")
col1, col2, col3, col4 = st.columns(4)

col1.metric("PyOD (Isolation Forest)", f"{df_pd['anomaly_iforest'].sum()} Alerts")
col2.metric("Amazon Chronos (TSFM)", f"{df_pd['anomaly_chronos'].sum()} Alerts")
col3.metric("IBM TTM (Statistical)", f"{df_pd['anomaly_ttm'].sum()} Alerts")
col4.metric("🚨 True Consensus Alerts", f"{df_pd['consensus_anomaly'].sum()} Alerts", delta="High Confidence", delta_color="inverse")

# --- UI Layout: Time-Series Chart ---
st.markdown("### 📈 Multi-Model Anomaly Comparison")

# Base Line Chart
fig = px.line(df_pd, x="timestamp", y="p95_latency", labels={"timestamp": "Time", "p95_latency": "P95 Latency (ms)"})
fig.update_traces(line_color="lightgrey") # Make the base line subtle

# 1. Plot PyOD (If Enabled)
if models_cfg.get("isolation_forest", {}).get("enabled"):
    anomalies = df_pd[df_pd["anomaly_iforest"] == True]
    fig.add_trace(go.Scatter(x=anomalies["timestamp"], y=anomalies["p95_latency"], 
                             mode="markers", marker=dict(color="blue", size=8, symbol="circle"), name="PyOD"))

# 2. Plot Chronos (If Enabled)
if models_cfg.get("chronos_t5", {}).get("enabled"):
    anomalies = df_pd[df_pd["anomaly_chronos"] == True]
    fig.add_trace(go.Scatter(x=anomalies["timestamp"], y=anomalies["p95_latency"], 
                             mode="markers", marker=dict(color="orange", size=6, symbol="triangle-up"), name="Chronos"))

# 3. Plot TTM (If Enabled)
if models_cfg.get("ibm_ttm", {}).get("enabled"):
    anomalies = df_pd[df_pd["anomaly_ttm"] == True]
    fig.add_trace(go.Scatter(x=anomalies["timestamp"], y=anomalies["p95_latency"], 
                             mode="markers", marker=dict(color="green", size=8, symbol="square"), name="IBM TTM"))

# 4. Plot TRUE CONSENSUS (The big red X)
consensus = df_pd[df_pd["consensus_anomaly"] == True]
fig.add_trace(go.Scatter(x=consensus["timestamp"], y=consensus["p95_latency"], 
                         mode="markers", marker=dict(color="red", size=14, symbol="x", line=dict(width=2, color="darkred")), 
                         name="🔥 CONSENSUS ALERT"))

st.plotly_chart(fig, use_container_width=True)

# --- UI Layout: Payload Drift ---
st.markdown("### ⚠️ Payload Statistical Drift (Evidently AI)")
drift_score = drift_results.get("user.account_balance", {}).get("psi_score", 0)
is_drifted = drift_results.get("user.account_balance", {}).get("is_drifted", False)

drift_col1, drift_col2 = st.columns([1, 3])
with drift_col1:
    st.metric(label="Population Stability Index (PSI)", value=f"{drift_score:.4f}")
with drift_col2:
    if is_drifted:
        st.error("🚨 **High Data Drift Detected!** The distribution of `user.account_balance` has changed significantly.")
    else:
        st.success("✅ Payload distribution is stable.")

# --- UI Layout: Raw Data Table ---
with st.expander("View Raw Ensemble Data"):
    st.dataframe(df_pd[["timestamp", "p95_latency", "anomaly_iforest", "anomaly_chronos", "anomaly_ttm", "consensus_anomaly"]])