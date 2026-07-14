***

# 🛡️ API Observability & Ensemble AI Platform (v2)

An enterprise-grade, machine-learning-driven observability platform designed to monitor API performance, detect latency/error anomalies, and identify payload schema changes and statistical data drift. 

**Version 2** introduces a highly advanced **Ensemble AI Architecture**, running traditional statistical models alongside Hugging Face Time-Series Foundation Models (TSFMs) locally in an air-gapped environment (e.g., RedHat OCP) to virtually eliminate false-positive alerts.

---

## 🏗️ The Architectural Strategy (Decoupled & Ensemble Design)

To ensure the application is highly configurable, scalable, and performant, the architecture is split into distinct, decoupled components.

### **Design Considerations & Rationale**
1. **Separation of Concerns:** The heavy lifting (data aggregation, PyTorch neural network inference, statistical drift calculation) is completely separated from the UI. 
2. **Ensemble AI (Consensus Logic):** API traffic is naturally noisy. Relying on a single AI model generates false positives. V2 runs three distinct models (Isolation Forest, Amazon Chronos, IBM TTM) concurrently. An alert is only fired if a **2-out-of-3 consensus** is reached.
3. **Performance:** By utilizing **Polars** (Rust-based) instead of Pandas for the core processing layer, we achieve zero-copy memory management and lightning-fast JSON flattening before feeding data to the neural networks.
4. **Air-Gapped Compatibility:** All dependencies and Hugging Face model weights are downloaded via a corporate proxy and executed 100% locally on CPUs/GPUs without external API calls.
5. **Configurability:** The pipeline is driven by a central YAML configuration, allowing developers to toggle specific AI models on/off dynamically without rewriting core Python logic.

### **Benefits**
* **Self-Correcting Alerts:** The consensus logic drastically reduces alert fatigue for DevOps teams.
* **Lightning-Fast UI:** The Streamlit frontend only reads pre-calculated anomaly flags from the backend engine, ensuring the dashboard loads instantly regardless of the underlying data volume.
* **Scalability:** The backend orchestrator can be scheduled as a Cron/Batch job (e.g., hourly), processing millions of Elasticsearch records in chunks.

---

## ⚙️ Pipeline Layers

The platform is built across five distinct layers:

1. **Data Layer (Elasticsearch):** Executes `date_histogram` aggregations. Instead of pulling millions of raw logs, it calculates the `p95_latency`, `error_rate`, and `request_count` directly at the database level.
2. **Processing Layer (Polars):** Ingests the deeply nested Elasticsearch JSON responses and flattens them into a fast, in-memory Polars DataFrame.
3. **Ensemble AI Layer (PyOD + Hugging Face):** 
    * **Model 1 (Statistical):** PyOD Isolation Forest detects multivariate outliers.
    * **Model 2 (Neural Network):** Amazon Chronos (`chronos-t5-small`) performs Zero-Shot forecasting to predict dynamic confidence bands.
    * **Model 3 (Lightweight TSFM):** IBM TinyTimeMixer (TTM) provides a secondary forecasting baseline.
4. **Payload & Drift Layer (DeepDiff & Evidently AI):** 
    * **DeepDiff:** Deterministically compares flattened JSON schemas (Yesterday vs. Today) to identify added or removed payload fields.
    * **Evidently AI:** Calculates the Population Stability Index (PSI) to detect statistical data drift in specific nested business fields (e.g., `user.account_balance`).
5. **UI Layer (Streamlit):** A lightweight dashboard utilizing Plotly to visualize the multi-model comparison, highlighting exactly where the models reached a consensus.

---

## 📂 Project Directory Structure

```text
api-observability-platform/
│
├── config/
│   └── config.yaml                 # Central configurator for APIs, metrics, and ML toggles
│
├── src/
│   ├── backend/                    # The Batch/Cron Engine
│   │   ├── __init__.py
│   │   ├── main_engine.py          # Master Orchestrator: Ties data, ML, and consensus together
│   │   ├── es_client.py            # Data Layer: Elasticsearch connection & query logic
│   │   ├── processor.py            # Processing Layer: Polars JSON flattening & aggregations
│   │   ├── ml_anomaly.py           # AI Layer: PyOD Isolation Forest implementation
│   │   ├── tsfm_inference.py       # AI Layer: Hugging Face PyTorch Inference (Chronos/TTM)
│   │   └── payload_monitor.py      # Drift Layer: DeepDiff schema & Evidently AI PSI logic
│   │
│   └── frontend/                   # The UI Layer
│       ├── __init__.py
│       └── app.py                  # Streamlit dashboard & Multi-Model Plotly visualizations
│
├── models/                         # Air-gapped storage for Hugging Face weights
│   ├── chronos-t5-small/           # Downloaded Amazon Chronos weights
│   └── ibm-ttm-v1/                 # Downloaded IBM TTM weights
│
├── logs/                           # Application execution logs
│
├── mock_es_data.py                 # (Dev) Generates mock Elasticsearch date_histogram JSON
├── mock_payload_data.py            # (Dev) Generates mock payloads for schema/drift testing
├── v2_ensemble_results.json        # Output generated by main_engine.py for the UI
│
├── requirements.txt                # Strictly pinned Python dependencies (Includes PyTorch)
├── .gitignore                      # Git ignore rules for venv, logs, and models
└── README.md                       # Project documentation
```

---

## 🚀 Setup & Execution Guide

### **1. Prerequisites**
* Python 3.9+
* `uv` or `pip` package manager
* Hugging Face CLI (`hf`)

### **2. Environment Setup**
Create and activate a virtual environment, then install the strictly pinned dependencies (including PyTorch and Transformers).
```bash
python -m venv venv

# Activate (Windows)
venv\Scripts\activate
# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies (If behind a proxy, use --proxy http://proxy:port)
uv pip install -r requirements.txt
```

### **3. Air-Gapped Model Download (One-Time Setup)**
Download the Hugging Face Time-Series Foundation Models to your local machine:
```bash
hf download amazon/chronos-t5-small --local-dir ./models/chronos-t5-small
hf download ibm-granite/granite-timeseries-ttm-v1 --local-dir ./models/ibm-ttm-v1
```

### **4. Generate Mock Data (Local Testing)**
Generate the simulated Elasticsearch aggregations and payload JSONs:
```bash
python mock_es_data.py
python mock_payload_data.py
```

### **5. Run the Master Orchestrator**
Execute the backend engine. This will run the data through Polars, execute the PyOD and Hugging Face models, calculate the consensus, and output `v2_ensemble_results.json`:
```bash
python src/backend/main_engine.py
```

### **6. Launch the Dashboard**
Start the Streamlit UI to visualize the multi-model battle and drift alerts:
```bash
python -m streamlit run src/frontend/app.py
```

---

## 🔮 The Way Forward (Phase 3: Agentic AI)

Now that the platform features a highly accurate, self-correcting Ensemble AI backend, the next evolutionary step is to transition from passive monitoring to **Active Agentic AI**.

### **Agentic AI for Contextual Alerting**
Currently, the system deterministically flags anomalies and visualizes them on a dashboard. In Phase 3, we will introduce a local Large Language Model (LLM) (e.g., Llama-3 or Mistral) to act as an **Observability Agent**.

* **The Workflow:** When the Master Orchestrator detects a consensus latency spike *and* the Payload Monitor detects a schema change simultaneously, it will pass this context directly to the LLM via a prompt.
* **The Benefit:** Instead of sending a generic "Latency High" PagerDuty alert, the Agent will reason about the correlation and generate a highly contextual, human-readable Slack/Email alert: 
  > *"🚨 **API Alert:** Latency spiked to 1359ms at 12:27. This highly correlates with the 'region' field being dropped from the payload schema 5 minutes prior. Please investigate backend routing logic."* 

