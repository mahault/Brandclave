"""Test PyMDP integration."""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
logging.basicConfig(level=logging.INFO)

from services.active_inference.pymdp_learner import PyMDPStructureLearner, PyMDPObservation, PYMDP_AVAILABLE

print(f"PyMDP available: {PYMDP_AVAILABLE}")

learner = PyMDPStructureLearner()
print(f"Agent initialized: {learner.agent is not None}")

# Test observations
test_obs = [
    PyMDPObservation(
        text="Looking for boutique hotel with good wifi for remote work",
        keywords=["boutique", "wifi", "remote", "work", "coworking"],
        source="reddit",
        sentiment=0.5
    ),
    PyMDPObservation(
        text="Best budget hostels with social atmosphere",
        keywords=["budget", "hostel", "social", "cheap"],
        source="reddit",
        sentiment=0.3
    ),
    PyMDPObservation(
        text="Luxury spa resort for honeymoon",
        keywords=["luxury", "spa", "resort", "honeymoon", "romantic"],
        source="youtube",
        sentiment=0.8
    ),
]

print("\nProcessing observations:")
for obs in test_obs:
    posteriors = learner.observe(obs)
    top_cats = sorted(posteriors.items(), key=lambda x: -x[1])[:3]
    print(f"  {obs.text[:40]}... -> {top_cats}")

print(f"\nStructure: {learner.get_structure()}")
print(f"\nNext action: {learner.select_action()}")
print(f"\nFree energy: {learner.get_free_energy():.3f}")
