"""Test script for Genius API connection."""
import json
from services.active_inference.genius_client import test_genius_connection

result = test_genius_connection()
print(json.dumps(result, indent=2))
