***

# 📚 Code Documentation

This document provides a detailed explanation of the core Python modules in the API Observability Platform (v1). It is designed to help developers and data scientists understand the data flow, library choices, and algorithmic logic.

---

## 1. The Data Layer: `src/backend/es_client.py`
**Purpose:** Handles all communication with the Elasticsearch cluster. It executes the time-series aggregations and writes the final anomaly results back to the database.

```python
from elasticsearch import Elasticsearch
import logging
```
* **Explanation:** Imports the official Elasticsearch Python client and the standard logging library to track connection status and query execution.

```python
class ESClient:
    def __init__(self, host: str, api_key: str = None):
        if api_key:
            self.client = Elasticsearch(host, api_key=api_key)
        else:
            self.client = Elasticsearch(host)
        logging.info(f"Connected to Elasticsearch at {host}")
```
* **Explanation:** The constructor initializes the connection. It supports both unauthenticated connections (for local testing) and API Key authentication (for OCP/Production environments).

```python
    def fetch_metrics(self, index: str, interval: str = "15m", time_range_hours: int = 24) -> dict:
        query = {
            "size": 0, 
            "query": { "range": { "@timestamp": { "gte": f"now-{time_range_hours}h", "lte": "now" } } },
            "aggregations": {
                "api_traffic_over_time": {
                    "date_histogram": { "field": "@timestamp", "fixed_interval": interval },
                    "aggregations": {
                        "p95_latency": { "percentiles": { "field": "response_time_ms", "percents": [95] } },
                        "error_count": { "filter": { "range": { "status_code": { "gte": 500 } } } }
                    }
                }
            }
        }
        return self.client.search(index=index, body=query)
```
* **Explanation:** This is the most critical database query. 
  * `"size": 0`: Tells Elasticsearch *not* to return the raw logs (which could be millions of rows), saving massive amounts of network bandwidth.
  * `"date_histogram"`: Groups the data into time buckets (e.g., every 15 minutes).
  * `"aggregations"`: For each 15-minute bucket, the database calculates the 95th percentile latency and counts how many HTTP 5xx errors occurred.

```python
    def write_anomalies(self, alerts_index: str, anomalies: list[dict]):
        from elasticsearch.helpers import bulk
        actions = [ { "_index": alerts_index, "_source": record } for record in anomalies ]
        success, _ = bulk(self.client, actions)
```
* **Explanation:** Uses the `bulk` helper from the Elasticsearch library to efficiently write multiple anomaly records back to a new index in a single network request. Streamlit will later read from this index.

---

## 2. The Processing Layer: `src/backend/processor.py`
**Purpose:** Parses the deeply nested JSON returned by Elasticsearch and flattens it into a high-performance Polars DataFrame.

```python
import polars as pl
import logging
```
* **Explanation:** Imports `polars`, a Rust-based DataFrame library that is significantly faster and more memory-efficient than Pandas.

```python
    def process_es_aggregations(self, es_response: dict) -> pl.DataFrame:
        buckets = es_response["aggregations"]["api_traffic_over_time"]["buckets"]
        flattened_data = []
```
* **Explanation:** Navigates the standard Elasticsearch JSON response tree to isolate the `buckets` array (the actual time-series data).

```python
        for bucket in buckets:
            timestamp = bucket.get("key_as_string")
            request_count = bucket.get("doc_count", 0)
            
            p95_latency = bucket.get("p95_latency", {}).get("values", {}).get("95.0", 0.0)
            error_count = bucket.get("error_count", {}).get("value", bucket.get("error_count", {}).get("doc_count", 0))
            error_rate = (error_count / request_count) if request_count > 0 else 0.0
```
* **Explanation:** Iterates through each time bucket. Because Elasticsearch JSONs can be unpredictable if data is missing, we use `.get()` with safe fallbacks (`0.0`). It calculates the `error_rate` dynamically to prevent division-by-zero errors.

```python
            flattened_data.append({
                "timestamp": timestamp, "request_count": request_count,
                "p95_latency": p95_latency, "error_count": error_count, "error_rate": error_rate
            })
```
* **Explanation:** Appends the cleaned, flattened data into a simple Python list of dictionaries.

```python
        df = pl.DataFrame(flattened_data)
        df = df.with_columns(
            pl.col("timestamp").str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%S%.fZ", strict=False)
        )
        return df
```
* **Explanation:** Converts the list directly into a Polars DataFrame. It then casts the string `timestamp` column into a native `Datetime` object using Rust's `chrono` formatting (`%.f` for fractional seconds), which is required for plotting in the UI.

---

## 3. The ML Anomaly Layer: `src/backend/ml_anomaly.py`
**Purpose:** Uses Machine Learning to identify statistical outliers in the time-series data.

```python
import polars as pl
from pyod.models.iforest import IForest
```
* **Explanation:** Imports the `IForest` (Isolation Forest) algorithm from the `pyod` (Python Outlier Detection) library.

```python
    def __init__(self, contamination: float = 0.05):
        self.contamination = contamination
        self.model = IForest(contamination=self.contamination, random_state=42)
```
* **Explanation:** Initializes the model. `contamination` tells the algorithm what percentage of the dataset we expect to be anomalous (5%). `random_state=42` ensures the results are deterministic (reproducible) during testing.

```python
    def detect_anomalies(self, df: pl.DataFrame, features: list[str]) -> pl.DataFrame:
        X = df.select(features).fill_null(0.0).to_numpy()
```
* **Explanation:** Scikit-learn and PyOD cannot read Polars DataFrames directly. This line selects the target features (e.g., latency and error rate), replaces any Null values with `0.0`, and converts it to a 2D NumPy array (`X`) via zero-copy memory transfer.

```python
        self.model.fit(X)
        predictions = self.model.predict(X) 
        scores = self.model.decision_function(X)
```
* **Explanation:** 
  * `.fit(X)`: Trains the Isolation Forest on the dataset to understand what "normal" looks like.
  * `.predict(X)`: Returns an array of `0` (normal) and `1` (anomaly).
  * `.decision_function(X)`: Returns the raw mathematical anomaly score (higher = more anomalous).

```python
        is_anomaly_series = pl.Series("is_anomaly", predictions == 1)
        anomaly_score_series = pl.Series("anomaly_score", scores)
        return df.with_columns([is_anomaly_series, anomaly_score_series])
```
* **Explanation:** Converts the NumPy prediction arrays back into Polars Series (columns). It casts the `0/1` predictions into boolean `True/False` values, appends them to the original DataFrame, and returns it.

---

## 4. The Payload & Drift Layer: `src/backend/payload_monitor.py`
**Purpose:** Detects structural changes in the API JSON schema and statistical drift in the actual data values.

```python
import pandas as pd
from deepdiff import DeepDiff
from evidently.report import Report
from evidently.metrics import ColumnDriftMetric
```
* **Explanation:** Imports `pandas` (required by Evidently), `DeepDiff` for deterministic schema comparison, and `Evidently` (v0.6.7) for Population Stability Index (PSI) calculations.

```python
    def _flatten_json(self, data: list[dict]) -> pd.DataFrame:
        return pd.json_normalize(data)
```
* **Explanation:** A helper function utilizing Pandas' `json_normalize`. If a payload has `{"user": {"balance": 100}}`, it flattens it to a single column named `user.balance`.

```python
    def detect_schema_changes(self, yesterday_data: list[dict], today_data: list[dict]) -> dict:
        schema_yesterday = list(self._flatten_json([yesterday_data[0]]).columns)
        schema_today = list(self._flatten_json([today_data[0]]).columns)
        
        diff = DeepDiff(schema_yesterday, schema_today, ignore_order=True)
```
* **Explanation:** Extracts just the column names (the schema keys) from yesterday's and today's payloads. `DeepDiff` compares these two lists. `ignore_order=True` ensures that if keys just shifted positions in the JSON, it doesn't trigger a false alarm.

```python
    def detect_data_drift(self, yesterday_data: list[dict], today_data: list[dict], target_fields: list[str]) -> dict:
        df_yesterday = self._flatten_json(yesterday_data)
        df_today = self._flatten_json(today_data)
        
        report = Report(metrics=[ColumnDriftMetric(column_name=field, stattest="psi")])
        report.run(reference_data=df_yesterday, current_data=df_today)
```
* **Explanation:** Flattens the full datasets. It initializes an Evidently `Report` specifically targeting the `psi` (Population Stability Index) statistical test. It compares `reference_data` (yesterday) against `current_data` (today) to see if the distribution of values (e.g., account balances) has fundamentally shifted.

---

## 5. The UI Layer: `src/frontend/app.py`
**Purpose:** Renders the interactive dashboard using Streamlit and Plotly.

```python
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
```
* **Explanation:** Imports Streamlit for the web framework and Plotly for highly interactive, JavaScript-based charting.

```python
@st.cache_data
def load_and_process_data():
    # ... calls backend modules ...
```
* **Explanation:** The `@st.cache_data` decorator is crucial. It tells Streamlit to run the heavy backend ML processing exactly *once* and cache the result in memory. Without this, Streamlit would re-run the Isolation Forest every time a user clicked a button on the UI.

```python
fig = px.line(df_pd, x="timestamp", y="p95_latency")
anomalies_df = df_pd[df_pd["is_anomaly"] == True]
fig.add_trace(
    go.Scatter(x=anomalies_df["timestamp"], y=anomalies_df["p95_latency"], mode="markers", marker=dict(color="red", size=10, symbol="x"))
)
st.plotly_chart(fig, use_container_width=True)
```
* **Explanation:** 
  1. Creates a standard line chart for the P95 Latency over time.
  2. Filters the DataFrame to isolate only the rows where PyOD flagged an anomaly.
  3. Uses `fig.add_trace()` to overlay those specific anomaly points as large, red "X" markers on top of the line chart.
  4. Renders the chart dynamically to fit the user's screen width.