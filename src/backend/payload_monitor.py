 # src/backend/payload_monitor.py
import pandas as pd
from deepdiff import DeepDiff
from evidently.report import Report
from evidently.metrics import ColumnDriftMetric
import logging

class PayloadMonitor:
    def __init__(self):
        pass

    def _flatten_json(self, data: list[dict]) -> pd.DataFrame:
        """
        Flattens deeply nested JSON into a tabular format.
        Example: {"user": {"balance": 100}} becomes column "user.balance"
        """
        # pandas json_normalize is perfect for flattening nested dictionaries deterministically
        return pd.json_normalize(data)

    def detect_schema_changes(self, yesterday_data: list[dict], today_data: list[dict]) -> dict:
        """
        Extracts the schema keys from both datasets and uses DeepDiff to find 
        added or removed fields.
        """
        if not yesterday_data or not today_data:
            return {}

        # Flatten to extract all possible nested keys as a single schema representation
        # We only need the columns (schema), not the data, so we check the first record
        df_yesterday = self._flatten_json([yesterday_data[0]])
        df_today = self._flatten_json([today_data[0]])

        schema_yesterday = list(df_yesterday.columns)
        schema_today = list(df_today.columns)

        # Use DeepDiff to compare the lists deterministically, ignoring order
        diff = DeepDiff(schema_yesterday, schema_today, ignore_order=True)
        
        # Clean up the output for our logs/alerts
        changes = {
            "fields_added": [item.strip("'") for item in diff.get("iterable_item_added", {}).values()],
            "fields_removed": [item.strip("'") for item in diff.get("iterable_item_removed", {}).values()]
        }
        
        if changes["fields_added"] or changes["fields_removed"]:
            logging.warning(f"Schema changes detected! {changes}")
        else:
            logging.info("No schema changes detected.")
            
        return changes

    def detect_data_drift(self, yesterday_data: list[dict], today_data: list[dict], target_fields: list[str]) -> dict:
        """
        Uses Evidently AI to calculate Population Stability Index (PSI) on specific fields.
        """
        # Evidently requires Pandas DataFrames
        df_yesterday = self._flatten_json(yesterday_data)
        df_today = self._flatten_json(today_data)

        drift_results = {}

        for field in target_fields:
            if field not in df_yesterday.columns or field not in df_today.columns:
                logging.error(f"Cannot calculate drift: Field '{field}' is missing from the schema.")
                continue

            logging.info(f"Calculating PSI drift for field: {field}...")
            
            # Configure Evidently to use PSI for this specific column
            report = Report(metrics=[
                ColumnDriftMetric(column_name=field, stattest="psi")
            ])
            
            # Run the statistical comparison
            report.run(reference_data=df_yesterday, current_data=df_today)
            
            # Extract the results as a dictionary
            result_dict = report.as_dict()
            metric_result = result_dict["metrics"][0]["result"]
            
            is_drifted = metric_result["drift_detected"]
            drift_score = metric_result["drift_score"] # This is the PSI value
            
            drift_results[field] = {
                "is_drifted": is_drifted,
                "psi_score": round(drift_score, 4)
            }
            
            if is_drifted:
                logging.warning(f"Data Drift Detected in '{field}'! PSI Score: {drift_score:.4f}")

        return drift_results