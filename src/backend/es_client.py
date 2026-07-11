# src/backend/es_client.py
from elasticsearch import Elasticsearch
import logging

class ESClient:
    def __init__(self, host: str, api_key: str = None):
        """
        Initializes the Elasticsearch client.
        """
        # For production, you might use API keys or basic auth depending on your OCP setup
        if api_key:
            self.client = Elasticsearch(host, api_key=api_key)
        else:
            self.client = Elasticsearch(host)
        logging.info(f"Connected to Elasticsearch at {host}")

    def fetch_metrics(self, index: str, interval: str = "15m", time_range_hours: int = 24) -> dict:
        """
        Executes a date_histogram query to aggregate metrics over time.
        """
        query = {
            "size": 0, # We only want aggregations, not raw hits
            "query": {
                "range": {
                    "@timestamp": {
                        "gte": f"now-{time_range_hours}h",
                        "lte": "now"
                    }
                }
            },
            "aggregations": {
                "api_traffic_over_time": {
                    "date_histogram": {
                        "field": "@timestamp",
                        "fixed_interval": interval
                    },
                    "aggregations": {
                        "p95_latency": {
                            "percentiles": {
                                "field": "response_time_ms",
                                "percents": [95]
                            }
                        },
                        "error_count": {
                            "filter": { "range": { "status_code": { "gte": 500 } } }
                        }
                    }
                }
            }
        }
        
        logging.info(f"Executing aggregation query on {index}")
        response = self.client.search(index=index, body=query)
        return response

    def write_anomalies(self, alerts_index: str, anomalies: list[dict]):
        """
        Writes flagged anomalies back to a new Elasticsearch index for Streamlit to read.
        """
        from elasticsearch.helpers import bulk
        
        actions = [
            {
                "_index": alerts_index,
                "_source": record
            }
            for record in anomalies
        ]
        
        success, _ = bulk(self.client, actions)
        logging.info(f"Successfully wrote {success} anomalies to {alerts_index}")