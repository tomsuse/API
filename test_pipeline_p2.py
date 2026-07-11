# test_pipeline.py
import json
import logging
import polars as pl
from src.backend.processor import DataProcessor
from src.backend.ml_anomaly import AnomalyDetector

# Setup basic logging to see the output in the console
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def run_weekend_test():
    print("\n--- Starting Pipeline Test ---\n")

    # 1. Load the Mock Data (Simulating ES Client)
    print("1. Loading mock Elasticsearch data...")
    with open("mock_es_response.json", "r") as f:
        es_response = json.load(f)

    # 2. Process Data with Polars
    print("\n2. Processing data with Polars...")
    processor = DataProcessor()
    df = processor.process_es_aggregations(es_response)
    
    print("\nPolars DataFrame (First 5 rows):")
    print(df.head(5))

    # 3. Run ML Anomaly Detection
    print("\n3. Running PyOD Isolation Forest...")
    detector = AnomalyDetector(contamination=0.05)
    
    # We want the model to look for anomalies in latency and error rate
    features_to_analyze = ["p95_latency", "error_rate"]
    df_with_anomalies = detector.detect_anomalies(df, features=features_to_analyze)

    # 4. View the Results!
    print("\n4. Results! Here are the detected anomalies:")
    # Filter the Polars dataframe to only show rows where is_anomaly == True
    anomalies_only = df_with_anomalies.filter(pl.col("is_anomaly") == True)
    
    print(anomalies_only.select(["timestamp", "request_count", "p95_latency", "error_rate", "anomaly_score"]))
    print("\n--- Pipeline Test Complete ---")

if __name__ == "__main__":
    run_weekend_test()