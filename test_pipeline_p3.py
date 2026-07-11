# test_pipeline.py
import json
import logging
import polars as pl
from src.backend.processor import DataProcessor
from src.backend.ml_anomaly import AnomalyDetector
from src.backend.payload_monitor import PayloadMonitor # <-- NEW IMPORT

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def run_weekend_test():
    print("\n" + "="*40)
    print(" PHASE 2: METRICS & ANOMALY DETECTION")
    print("="*40)

    with open("mock_es_response.json", "r") as f:
        es_response = json.load(f)

    processor = DataProcessor()
    df = processor.process_es_aggregations(es_response)
    
    detector = AnomalyDetector(contamination=0.05)
    df_with_anomalies = detector.detect_anomalies(df, features=["p95_latency", "error_rate"])
    
    anomalies_only = df_with_anomalies.filter(pl.col("is_anomaly") == True)
    print("\nDetected Anomalies:")
    print(anomalies_only.select(["timestamp", "p95_latency", "error_rate", "anomaly_score"]))

    print("\n" + "="*40)
    print(" PHASE 3: PAYLOAD SCHEMA & DRIFT DETECTION")
    print("="*40)

    # 1. Load the mock payloads
    print("1. Loading yesterday and today's payloads...")
    try:
        with open("payloads_yesterday.json", "r") as f:
            yesterday_data = json.load(f)
        with open("payloads_today.json", "r") as f:
            today_data = json.load(f)
    except FileNotFoundError:
        print("Error: Please run 'python mock_payload_data.py' first!")
        return

    monitor = PayloadMonitor()

    # 2. Schema Comparison (DeepDiff)
    print("\n2. Running Schema Comparison (DeepDiff)...")
    schema_changes = monitor.detect_schema_changes(yesterday_data, today_data)
    print(f"Schema Changes Result: {json.dumps(schema_changes, indent=2)}")

    # 3. Statistical Drift (Evidently AI - PSI)
    print("\n3. Running Statistical Drift Detection (Evidently AI)...")
    # We want to monitor the nested field "user.account_balance"
    drift_results = monitor.detect_data_drift(yesterday_data, today_data, target_fields=["user.account_balance"])
    print(f"Drift Results: {json.dumps(drift_results, indent=2)}")

    print("\n--- Pipeline Test Complete ---")

if __name__ == "__main__":
    run_weekend_test()