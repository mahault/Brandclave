"""Test Genius license activation."""
import httpx
from dotenv import load_dotenv
import os
import json

load_dotenv()

API_URL = os.getenv("GENIUS_API_URL")
API_KEY = os.getenv("GENIUS_API_KEY")
LICENSE_KEY = os.getenv("GENIUS_LICENSE_KEY")

print(f"API URL: {API_URL}")
print(f"API Key: {API_KEY}")
print(f"License Key: {LICENSE_KEY}")

# Test 1: Try to activate license
print("\n1. Activating license:")
with httpx.Client(timeout=30) as client:
    r = client.post(
        f"{API_URL}/license/update",
        json={"license_key": LICENSE_KEY},
        headers={"x-api-key": API_KEY, "Content-Type": "application/json"}
    )
    print(f"   Status: {r.status_code}")
    print(f"   Response: {r.text}")

# Test 2: Try with just license key in body
print("\n2. Alternative license format:")
with httpx.Client(timeout=30) as client:
    r = client.post(
        f"{API_URL}/license/update",
        json=LICENSE_KEY,  # Just the string
        headers={"x-api-key": API_KEY, "Content-Type": "application/json"}
    )
    print(f"   Status: {r.status_code}")
    print(f"   Response: {r.text}")

# Test 3: Try license refresh
print("\n3. License refresh:")
with httpx.Client(timeout=30) as client:
    r = client.post(
        f"{API_URL}/license/refresh",
        headers={"x-api-key": API_KEY}
    )
    print(f"   Status: {r.status_code}")
    print(f"   Response: {r.text}")

# Test 4: Get license metadata
print("\n4. License metadata:")
with httpx.Client(timeout=30) as client:
    r = client.get(
        f"{API_URL}/license/metadata",
        headers={"x-api-key": API_KEY}
    )
    print(f"   Status: {r.status_code}")
    print(f"   Response: {r.text}")
