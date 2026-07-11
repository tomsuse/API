***

# 🛡️ API Observability & Anomaly Detection Platform (v1)

An enterprise-grade, machine-learning-driven observability platform designed to monitor API performance, detect latency/error anomalies, and identify payload schema changes and statistical data drift. 

This project is built for high-performance, air-gapped environments (like RedHat OCP) and utilizes a highly modular, decoupled architecture.

---

## 🏗️ The Architectural Strategy (Decoupled Design)

To ensure the application is highly configurable, scalable, and performant, the architecture is split into distinct, decoupled components.

### **Design Considerations & Rationale**
1. **Separation of Concerns:** The heavy lifting (data aggregation, ML inference, statistical drift calculation) is completely separated from the UI. 
2. **Performance:** By utilizing **Polars** (Rust-based) instead of Pandas for the core processing layer, we achieve zero-copy memory management and lightning-fast JSON flattening.
3. **Air-Gapped Compatibility:** All dependencies and models are designed to be downloaded via a corporate proxy and executed 100% locally without external API calls.
4. **Configurability:** The pipeline is designed to be driven by a central YAML configuration, allowing new APIs or ML models to be added without rewriting core Python logic.

### **Benefits**
* **Lightning-Fast UI:** The Streamlit frontend only reads pre-calculated anomaly flags from Elasticsearch, ensuring the dashboard loads instantly regardless of the underlying data volume.
* **Scalability:** The backend engine can be scheduled as a Cron/Batch job (e.g., hourly), processing millions of Elasticsearch records in chunks without overwhelming the cluster.
* **Future-Proof:** The modular ML layer allows seamless swapping of algorithms (e.g., moving from Scikit-Learn/PyOD to Hugging Face Foundation Models) without touching the Data or UI layers.

---

## ⚙️ Pipeline Layers & Project Phases

The platform is built across five distinct layers:

1. **Data Layer (Elasticsearch):** Executes `date_histogram` aggregations. Instead of pulling millions of raw logs, it calculates the `p95_latency`, `error_rate`, and `request_count` directly at the database level.
2. **Processing Layer (Polars):** Ingests the deeply nested Elasticsearch JSON responses and flattens them into a fast, in-memory Polars DataFrame for downstream ML tasks.
3. **ML Anomaly Layer (PyOD):** Utilizes the **Isolation Forest** algorithm to analyze multivariate time-series data (latency + errors) and flags outliers (`is_anomaly: True`).
4. **Payload & Drift Layer (DeepDiff & Evidently AI):** 
    * **DeepDiff:** Deterministically compares flattened JSON schemas (Yesterday vs. Today) to identify added or removed payload fields.
    * **Evidently AI:** Calculates the Population Stability Index (PSI) to detect statistical data drift in specific nested business fields (e.g., `user.account_balance`).
5. **UI Layer (Streamlit):** A lightweight, interactive dashboard utilizing Plotly to visualize anomalies and drift alerts.

---

## 📂 Project Directory Structure

```text
api-observability-platform/
│
├── config/
│   └── config.yaml                 # Central configurator for APIs, metrics, and ML models
│
├── src/
│   ├── backend/                    # The Batch/Cron Engine
│   │   ├── __init__.py
│   │   ├── es_client.py            # Data Layer: Elasticsearch connection & query logic
│   │   ├── processor.py            # Processing Layer: Polars JSON flattening & aggregations
│   │   ├── ml_anomaly.py           # ML Layer: PyOD Isolation Forest implementation
│   │   └── payload_monitor.py      # Drift Layer: DeepDiff schema & Evidently AI PSI logic
│   │
│   └── frontend/                   # The UI Layer
│       ├── __init__.py
│       └── app.py                  # Streamlit dashboard entry point & Plotly visualizations
│
├── models/                         # Local storage for air-gapped ML models (e.g., Hugging Face)
│   └── .gitkeep                    
│
├── logs/                           # Application execution logs
│   └── .gitkeep
│
├── mock_es_data.py                 # (Dev) Generates mock Elasticsearch date_histogram JSON
├── mock_payload_data.py            # (Dev) Generates mock payloads for schema/drift testing
├── test_pipeline_p2.py             # (Dev) Execution script for Phase 2 (Polars + PyOD)
├── test_pipeline_p3.py             # (Dev) Execution script for Phase 3 (DeepDiff + Evidently)
│
├── requirements.txt                # Strictly pinned Python dependencies
├── .gitignore                      # Git ignore rules for venv, logs, and models
└── README.md                       # Project documentation
```

---

## 🚀 Setup & Execution Guide

### **1. Prerequisites**
* Python 3.9+
* `uv` or `pip` package manager

### **2. Environment Setup**
Create and activate a virtual environment, then install the strictly pinned dependencies.
```bash
python -m venv venv

# Activate (Windows)
venv\Scripts\activate
# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies (If behind a proxy, use --proxy http://proxy:port)
pip install -r requirements.txt
```

### **3. Generate Mock Data (Local Testing)**
Generate the simulated Elasticsearch aggregations and payload JSONs:
```bash
python mock_es_data.py
python mock_payload_data.py
```

### **4. Run Backend Tests**
Verify the Data, Processing, and ML layers are functioning correctly:
```bash
# Test Polars & PyOD Isolation Forest
python test_pipeline_p2.py

# Test DeepDiff Schema Comparison & Evidently AI Drift
python test_pipeline_p3.py
```

### **5. Launch the Dashboard**
Start the Streamlit UI to visualize the results:
```bash
python -m streamlit run src/frontend/app.py
```

---

## 🔮 The Way Forward (Phase 5)

As the platform matures, the next evolutionary step is to transition from statistical anomaly detection to **Agentic AI and Time-Series Foundation Models (TSFMs)**. This will be developed in a separate branch.

### **1. Hugging Face Time-Series Foundation Models (TSFMs)**
Instead of training traditional ML models (like Isolation Forest) from scratch, we will integrate pre-trained models like **Amazon Chronos** (`chronos-t5-small`) or **IBM TinyTimeMixer (TTM)**.
* **How it works:** These models have been pre-trained on billions of generic time-series data points. We will use them for *Zero-Shot Forecasting*. We feed the model the last 24 hours of latency, and it predicts a confidence band for the next hour. If actual latency falls outside this band, it is flagged as an anomaly.
* **Air-Gapped Strategy:** The model weights will be downloaded once via the corporate proxy (`huggingface-cli download`) and stored in the `/models` directory. The `config.yaml` will be updated to point to this local path, requiring zero code rewrites.
* **Benefits:** Drastic reduction in false positives, better handling of seasonality (e.g., weekend traffic drops), and zero model-training overhead.

### **2. Agentic AI for Contextual Alerting**
Currently, the system deterministically flags anomalies. In Phase 5, we will introduce a local Large Language Model (LLM) to act as an **Agent**.
* **The Workflow:** When the backend engine detects a latency spike *and* a payload schema change simultaneously, it will pass this context to the LLM.
* **The Benefit:** Instead of sending a generic "Latency High" alert, the Agent will reason about the correlation and generate a highly contextual Slack/Email alert: *"🚨 API Latency spiked to 1359ms at 12:27. This highly correlates with the 'region' field being dropped from the payload schema 5 minutes prior. Please investigate backend routing."* 

This transition will upgrade the platform from a passive monitoring tool to an active, intelligent observability agent.