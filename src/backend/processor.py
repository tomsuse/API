# src/backend/processor.py
import polars as pl
import logging

class DataProcessor:
    def __init__(self):
        pass

    def process_es_aggregations(self, es_response: dict) -> pl.DataFrame:
        """
        Parses the nested Elasticsearch date_histogram response and converts it 
        into a flat Polars DataFrame.
        """
        # Navigate to the buckets in the ES response
        try:
            buckets = es_response["aggregations"]["api_traffic_over_time"]["buckets"]
        except KeyError as e:
            logging.error(f"Invalid ES response structure. Missing key: {e}")
            return pl.DataFrame()

        flattened_data = []

        for bucket in buckets:
            # Extract basic bucket info
            timestamp = bucket.get("key_as_string")
            request_count = bucket.get("doc_count", 0)
            
            # Extract nested p95 latency (ES returns {"values": {"95.0": value}})
            p95_latency = None
            if "p95_latency" in bucket and "values" in bucket["p95_latency"]:
                p95_latency = bucket["p95_latency"]["values"].get("95.0")
            
            # Extract nested error count
            error_count = 0
            if "error_count" in bucket:
                # Depending on ES version/query, it might be in 'value' or 'doc_count'
                error_count = bucket["error_count"].get("value", bucket["error_count"].get("doc_count", 0))

            # Calculate error rate safely
            error_rate = (error_count / request_count) if request_count > 0 else 0.0

            flattened_data.append({
                "timestamp": timestamp,
                "request_count": request_count,
                "p95_latency": p95_latency if p95_latency is not None else 0.0,
                "error_count": error_count,
                "error_rate": error_rate
            })

        # Convert the list of dictionaries directly into a Polars DataFrame (Zero-copy, very fast)
        df = pl.DataFrame(flattened_data)
        
        # Ensure timestamp is cast to a proper Datetime object for future time-series operations
        df = df.with_columns(
            pl.col("timestamp").str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%S%.fZ", strict=False)
        )
        
        logging.info(f"Processed {len(df)} records into Polars DataFrame.")
        return df