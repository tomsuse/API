# mock_payload_data.py
import json
import random

def generate_payloads(is_today=False, num_records=500):
    payloads = []
    for i in range(num_records):
        # Base nested JSON structure
        record = {
            "transaction_id": f"txn_{i}",
            "user": {
                "account_type": random.choice(["basic", "premium", "enterprise"]),
                # Yesterday's balance is normally distributed around 1000
                "account_balance": max(0, random.gauss(1000, 200))
            },
            "metadata": {
                "region": "us-east-1"
            }
        }

        if is_today:
            # 1. INJECT STATISTICAL DRIFT: Today, balances suddenly spike (mean shifts to 5000)
            record["user"]["account_balance"] = max(0, random.gauss(5000, 1000))
            
            # 2. INJECT SCHEMA CHANGE: A new field appears in today's payload
            record["user"]["kyc_verified"] = random.choice([True, False])
            
            # 3. INJECT SCHEMA CHANGE: A field is removed (region)
            del record["metadata"]["region"]

        payloads.append(record)
    return payloads

if __name__ == "__main__":
    yesterday = generate_payloads(is_today=False)
    today = generate_payloads(is_today=True)
    
    with open("payloads_yesterday.json", "w") as f:
        json.dump(yesterday, f, indent=2)
    with open("payloads_today.json", "w") as f:
        json.dump(today, f, indent=2)
        
    print("Generated payloads_yesterday.json and payloads_today.json")