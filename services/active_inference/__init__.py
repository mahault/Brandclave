"""Active Inference module for adaptive structure learning.

This module provides three implementations:

1. Local Structure Learner (structure_learner.py)
   - Uses local embeddings and heuristics
   - No external dependencies beyond NLP pipeline
   - Good for offline/testing scenarios

2. Genius Structure Learner (genius_structure_learner.py)
   - Uses VERSES Genius Active Inference API
   - Proper Bayesian inference and POMDP action selection
   - Requires valid license from VERSES

3. PyMDP Structure Learner (pymdp_learner.py)
   - Uses JAX-based pymdp library
   - Local active inference with EFE minimization
   - No external API required, fully open source
"""

from .structure_learner import StructureLearner, Category, Observation

# Genius-backed implementations (require API key)
from .genius_client import GeniusClient, VFGBuilder, GeniusConfig
from .genius_structure_learner import GeniusStructureLearner, GeniusObservation

# PyMDP-backed implementations (local, JAX-based)
from .pymdp_learner import PyMDPStructureLearner, PyMDPObservation, PYMDP_AVAILABLE

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
    # PyMDP implementation
    "PyMDPStructureLearner",
    "PyMDPObservation",
    "PYMDP_AVAILABLE",
]
