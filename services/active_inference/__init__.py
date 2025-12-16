"""Active Inference module for adaptive structure learning.

This module provides two implementations:

1. Local Structure Learner (structure_learner.py)
   - Uses local embeddings and heuristics
   - No external dependencies beyond NLP pipeline
   - Good for offline/testing scenarios

2. Genius Structure Learner (genius_structure_learner.py)
   - Uses VERSES Genius Active Inference API
   - Proper Bayesian inference and POMDP action selection
   - Production-ready with real variational free energy
"""

from .structure_learner import StructureLearner, Category, Observation

# Genius-backed implementations (require API key)
from .genius_client import GeniusClient, VFGBuilder, GeniusConfig
from .genius_structure_learner import GeniusStructureLearner, GeniusObservation

__all__ = [
    # Local implementation
    "StructureLearner",
    "Category",
    "Observation",
    # Genius implementation
    "GeniusClient",
    "VFGBuilder",
    "GeniusConfig",
    "GeniusStructureLearner",
    "GeniusObservation",
]
