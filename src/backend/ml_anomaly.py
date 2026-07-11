# src/backend/ml_anomaly.py
import polars as pl
from pyod.models.iforest import IForest
import logging

class AnomalyDetector:
    def __init__(self, contamination: float = 0.05):
        """
        contamination: The expected percentage of anomalies in the dataset (e.g., 0.05 = 5%)
        """
        self.contamination = contamination
        # Initialize the Isolation Forest model
        self.model = IForest(contamination=self.contamination, random_state=42)
        logging.info(f"Initialized Isolation Forest with contamination={contamination}")

    def detect_anomalies(self, df: pl.DataFrame, features: list[str]) -> pl.DataFrame:
        """
        Runs anomaly detection on the specified features of the Polars DataFrame.
        Returns the DataFrame with new 'is_anomaly' and 'anomaly_score' columns.
        """
        if df.is_empty():
            logging.warning("Empty DataFrame provided to AnomalyDetector.")
            return df

        # 1. Extract the feature columns and convert to a 2D NumPy array
        # Isolation Forest requires numerical matrices without nulls
        X = df.select(features).fill_null(0.0).to_numpy()

        # 2. Fit the model and predict
        logging.info(f"Training Isolation Forest on {len(X)} records using features: {features}...")
        self.model.fit(X)
        
        # predict() returns an array of 0s (normal) and 1s (anomaly)
        predictions = self.model.predict(X) 
        
        # decision_function() returns the raw anomaly scores
        scores = self.model.decision_function(X)

        # 3. Append the results back to the Polars DataFrame
        # Convert 0/1 predictions to boolean (True = Anomaly)
        is_anomaly_series = pl.Series("is_anomaly", predictions == 1)
        anomaly_score_series = pl.Series("anomaly_score", scores)

        df_result = df.with_columns([
            is_anomaly_series,
            anomaly_score_series
        ])

        num_anomalies = df_result["is_anomaly"].sum()
        logging.info(f"Anomaly detection complete. Found {num_anomalies} anomalies.")

        return df_result