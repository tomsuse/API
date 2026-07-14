# src/backend/main_engine.py
import yaml
import json
import logging
import polars as pl

from src.backend.processor import DataProcessor
from src.backend.ml_anomaly import AnomalyDetector
from src.backend.tsfm_inference import TSFMEnsemble

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class Orchestrator:
    def __init__(self, config_path: str):
        logging.info(f"Loading configuration from {config_path}...")
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

    def run_pipeline(self):
        print("\n" + "="*50)
        print("🚀 STARTING V2 ENSEMBLE PIPELINE")
        print("="*50)

        # 1. Data Layer (Simulated ES Fetch)
        logging.info("Fetching data from Elasticsearch...")
        with open("mock_es_response.json", "r") as f:
            es_response = json.load(f)

        # 2. Processing Layer (Polars)
        processor = DataProcessor()
        df = processor.process_es_aggregations(es_response)

        # 3. ML Layer: PyOD Isolation Forest
        if_config = self.config["apis"][0]["anomaly_detection"]["models"]["isolation_forest"]
        if if_config.get("enabled"):
            logging.info("--- Running Model 1: PyOD Isolation Forest ---")
            detector = AnomalyDetector(contamination=if_config.get("contamination", 0.05))
            df = detector.detect_anomalies(df, features=["p95_latency", "error_rate"])
            # Rename PyOD's default column to match our ensemble naming convention
            df = df.rename({"is_anomaly": "anomaly_iforest"})

        # 4. ML Layer: Hugging Face TSFMs
        logging.info("--- Running Models 2 & 3: Hugging Face TSFMs ---")
        tsfm_ensemble = TSFMEnsemble(self.config["apis"][0])
        df = tsfm_ensemble.detect_anomalies(df, target_col="p95_latency")

        # 5. Consensus Calculation
        logging.info("--- Calculating Ensemble Consensus ---")
        # An anomaly is a consensus if at least 2 out of 3 models flag it
        consensus_series = (
            df["anomaly_iforest"].cast(pl.Int8) + 
            df["anomaly_chronos"].cast(pl.Int8) + 
            df["anomaly_ttm"].cast(pl.Int8)
        ) >= 2
        
        df = df.with_columns(consensus_series.alias("consensus_anomaly"))

        # 6. Results
        print("\n" + "="*50)
        print("📊 ENSEMBLE RESULTS (Anomalies Only)")
        print("="*50)
        
        anomalies_only = df.filter(
            (pl.col("anomaly_iforest") == True) | 
            (pl.col("anomaly_chronos") == True) | 
            (pl.col("anomaly_ttm") == True)
        )
        
        # Select the columns to display
        display_cols = ["timestamp", "p95_latency", "anomaly_iforest", "anomaly_chronos", "anomaly_ttm", "consensus_anomaly"]
        print(anomalies_only.select(display_cols))
        
        # Save the final dataframe for Streamlit to read
        df.write_json("v2_ensemble_results.json")
        logging.info("Pipeline complete. Results saved to v2_ensemble_results.json")

if __name__ == "__main__":
    orchestrator = Orchestrator("config/config.yaml")
    orchestrator.run_pipeline()