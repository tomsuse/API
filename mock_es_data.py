# mock_es_data.py
import json
import random
from datetime import datetime, timedelta

def generate_mock_es_response(num_buckets=100):
    """
    Simulates an Elasticsearch date_histogram aggregation response.
    Includes p95_latency, error_rate, and request_count.
    """
    now = datetime.utcnow()
    buckets = []
    
    for i in range(num_buckets):
        # Go back in time, 15 minutes per bucket
        timestamp = now - timedelta(minutes=15 * (num_buckets - i))
        
        # Simulate normal traffic with occasional anomalies (spikes)
        is_anomaly = random.random() < 0.05 # 5% chance of anomaly
        
        req_count = random.randint(1000, 5000)
        p95 = random.uniform(100, 300) if not is_anomaly else random.uniform(800, 1500)
        errors = random.randint(5, 50) if not is_anomaly else random.randint(200, 500)
        
        bucket = {
            "key_as_string": timestamp.isoformat() + "Z",
            "key": int(timestamp.timestamp() * 1000),
            "doc_count": req_count,
            "p95_latency": {
                "values": {
                    "95.0": p95
                }
            },
            "error_count": {
                "value": errors
            }
        }
        buckets.append(bucket)

    # Wrap in standard Elasticsearch response format
    es_response = {
        "took": 45,
        "timed_out": False,
        "_shards": {"total": 5, "successful": 5, "skipped": 0, "failed": 0},
        "hits": {
            "total": {"value": 10000, "relation": "gte"},
            "max_score": None,
            "hits": [] # We don't need raw hits, just aggregations
        },
        "aggregations": {
            "api_traffic_over_time": {
                "buckets": buckets
            }
        }
    }
    
    with open("mock_es_response.json", "w") as f:
        json.dump(es_response, f, indent=2)
        
    print(f"Successfully generated mock_es_response.json with {num_buckets} buckets.")

if __name__ == "__main__":
    generate_mock_es_response()