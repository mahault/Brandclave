"""Test Genius API with correct headers."""
import httpx
from dotenv import load_dotenv
import os
from datetime import datetime, timezone

load_dotenv()

API_URL = os.getenv("GENIUS_API_URL")
API_KEY = os.getenv("GENIUS_API_KEY")
LICENSE_KEY = os.getenv("GENIUS_LICENSE_KEY")

print(f"API URL: {API_URL}")
print(f"API Key: {API_KEY}")
print(f"License Key: {LICENSE_KEY}")

# RFC7231 datetime format
rfc7231_date = datetime.now(timezone.utc).strftime('%a, %d %b %Y %H:%M:%S GMT')
print(f"RFC7231 Date: {rfc7231_date}")

client = httpx.Client(timeout=30)

# Test 1: License activation with correct schema
print("\n1. Activating license:")
r = client.post(
    f"{API_URL}/license/update",
    json={"license_key": LICENSE_KEY},
    headers={"x-api-key": API_KEY}
)
print(f"   Status: {r.status_code}")
print(f"   Response: {r.text}")

# Test 2: Get graph with conditional header
print("\n2. GET graph with if-none-match:")
r = client.get(
    f"{API_URL}/graph",
    headers={
        "x-api-key": API_KEY,
        "if-none-match": "*"
    }
)
print(f"   Status: {r.status_code}")
print(f"   Response: {r.text[:500] if r.text else 'empty'}")

# Test 3: Set graph with if-none-match (for creating new graph)
print("\n3. PUT graph with if-none-match: *")
vfg = {
    "version": "0.5.0",
    "variables": {
        "category": {
            "elements": ["luxury", "budget", "boutique"]
        }
    },
    "factors": [
        {
            "variables": ["category"],
            "distribution": "categorical",
            "values": [0.33, 0.33, 0.34]
        }
    ],
    "metadata": {
        "model_type": "bayesian_network"
    }
}
r = client.put(
    f"{API_URL}/graph",
    json={"vfg": vfg},
    headers={
        "x-api-key": API_KEY,
        "if-none-match": "*",
        "Content-Type": "application/json"
    }
)
print(f"   Status: {r.status_code}")
print(f"   Response: {r.text[:500] if r.text else 'empty'}")

# Test 4: Try with date header instead
print("\n4. PUT graph with date header:")
r = client.put(
    f"{API_URL}/graph",
    json={"vfg": vfg},
    headers={
        "x-api-key": API_KEY,
        "date": rfc7231_date,
        "Content-Type": "application/json"
    }
)
print(f"   Status: {r.status_code}")
print(f"   Response: {r.text[:500] if r.text else 'empty'}")

# Test 5: Try inference
print("\n5. POST inference:")
r = client.post(
    f"{API_URL}/infer",
    json={
        "variables": ["category"],
        "library": "pgmpy"
    },
    headers={"x-api-key": API_KEY}
)
print(f"   Status: {r.status_code}")
print(f"   Response: {r.text[:500] if r.text else 'empty'}")

client.close()
