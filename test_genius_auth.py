"""Test different auth methods for Genius API."""
import httpx
from dotenv import load_dotenv
import os

load_dotenv()

API_URL = os.getenv("GENIUS_API_URL", "https://agent-3271175-2724605-8366d751a2c4dc.agents.genius.verses.ai")
API_KEY = os.getenv("GENIUS_API_KEY", "")

print(f"API URL: {API_URL}")
print(f"API Key: {API_KEY[:20]}..." if API_KEY else "API Key: MISSING!")

if not API_KEY:
    print("ERROR: No API key found in environment")
    exit(1)

# Test 1: Just x-api-key header
print("\n1. Testing with x-api-key header only:")
with httpx.Client(timeout=30) as client:
    r = client.get(f"{API_URL}/version", headers={"x-api-key": API_KEY})
    print(f"   Status: {r.status_code}")
    if r.status_code == 200:
        print(f"   Response: {r.json()}")

# Test 2: Bearer token
print("\n2. Testing with Bearer token:")
with httpx.Client(timeout=30) as client:
    r = client.get(f"{API_URL}/version", headers={"Authorization": f"Bearer {API_KEY}"})
    print(f"   Status: {r.status_code}")
    if r.status_code == 200:
        print(f"   Response: {r.json()}")

# Test 3: Both headers
print("\n3. Testing with both x-api-key and Bearer token:")
with httpx.Client(timeout=30) as client:
    r = client.get(f"{API_URL}/version", headers={
        "x-api-key": API_KEY,
        "Authorization": f"Bearer {API_KEY}"
    })
    print(f"   Status: {r.status_code}")
    if r.status_code == 200:
        print(f"   Response: {r.json()}")

# Test 4: Check what headers the root endpoint accepts
print("\n4. Testing root endpoint (health):")
with httpx.Client(timeout=30) as client:
    r = client.get(f"{API_URL}/")
    print(f"   Status (no auth): {r.status_code}")
    r = client.get(f"{API_URL}/", headers={"x-api-key": API_KEY})
    print(f"   Status (with key): {r.status_code}")

# Test 5: Try POST to graph with detailed error
print("\n5. Testing POST to /graph:")
vfg = {
    "version": "0.5.0",
    "variables": {"x": {"type": "discrete", "elements": ["a", "b"], "cardinality": 2}},
    "factors": {"p": {"type": "categorical", "variable": "x", "probabilities": [0.5, 0.5]}},
    "metadata": {"model_type": "bayesian_network"}
}
with httpx.Client(timeout=30) as client:
    r = client.put(
        f"{API_URL}/graph",
        json={"vfg": vfg},
        headers={
            "x-api-key": API_KEY,
            "Content-Type": "application/json"
        }
    )
    print(f"   Status: {r.status_code}")
    print(f"   Response: {r.text[:500]}")
