"""Test script for Genius-powered city analyzer."""
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

from services.active_inference.genius_city_analyzer import GeniusCityAnalyzer

print("Starting Genius-powered city analysis for Lisbon...")
print("This will take 1-2 minutes as it actively explores and learns.\n")

with GeniusCityAnalyzer(
    max_iterations=5,  # Reduced for testing
    confidence_threshold=0.6,
) as analyzer:
    result = analyzer.analyze_city("Lisbon", "Portugal")

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
print(json.dumps(result, indent=2, default=str))
