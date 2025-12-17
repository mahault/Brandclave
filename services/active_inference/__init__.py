"""Active Inference module for adaptive decision-making.

This module provides multiple implementations for different use cases:

## Structure Learning (Category Discovery)

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

## Component POMDPs (Adaptive Decision Making)

4. Scraping POMDP (scraping_pomdp.py)
   - Adaptive source selection for web scraping
   - Learns which sources are most productive

5. Clustering POMDP (clustering_pomdp.py)
   - Adaptive HDBSCAN parameter selection
   - Learns optimal clustering parameters for data

6. Move Extraction POMDP (move_extraction_pomdp.py)
   - Adaptive extraction method selection (skip/regex/llm)
   - Optimizes quality vs cost trade-off

7. Coordinator POMDP (coordinator_pomdp.py)
   - Cross-component orchestration
   - Detects correlations and optimization opportunities

8. User-Adaptive POMDP (user_adaptive_pomdp.py)
   - Personalization based on user interaction history
   - Learns user preferences and profiles

## Hybrid Controller

9. Hybrid Controller (hybrid_controller.py)
   - Unified interface for PyMDP and Genius backends
   - Automatic fallback and backend selection
"""

from .structure_learner import StructureLearner, Category, Observation

# Genius-backed implementations (require API key)
from .genius_client import GeniusClient, VFGBuilder, GeniusConfig
from .genius_structure_learner import GeniusStructureLearner, GeniusObservation

# PyMDP-backed implementations (local, JAX-based)
from .pymdp_learner import PyMDPStructureLearner, PyMDPObservation, PYMDP_AVAILABLE

# Component POMDPs
from .scraping_pomdp import ScrapingPOMDP, get_scraping_pomdp
from .clustering_pomdp import ClusteringPOMDP, get_clustering_pomdp
from .move_extraction_pomdp import MoveExtractionPOMDP, get_extraction_pomdp
from .coordinator_pomdp import CoordinatorPOMDP, get_coordinator_pomdp
from .user_adaptive_pomdp import UserAdaptivePOMDP, get_user_adaptive_pomdp

# Hybrid controller
from .hybrid_controller import (
    HybridActiveInferenceController,
    HybridScrapingController,
    HybridClusteringController,
    HybridExtractionController,
    HybridCoordinatorController,
    InferenceBackend,
    InferenceResult,
    get_hybrid_controller,
    PYMDP_AVAILABLE as HYBRID_PYMDP_AVAILABLE,
    GENIUS_AVAILABLE as HYBRID_GENIUS_AVAILABLE,
)

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
    # Component POMDPs
    "ScrapingPOMDP",
    "get_scraping_pomdp",
    "ClusteringPOMDP",
    "get_clustering_pomdp",
    "MoveExtractionPOMDP",
    "get_extraction_pomdp",
    "CoordinatorPOMDP",
    "get_coordinator_pomdp",
    "UserAdaptivePOMDP",
    "get_user_adaptive_pomdp",
    # Hybrid controller
    "HybridActiveInferenceController",
    "HybridScrapingController",
    "HybridClusteringController",
    "HybridExtractionController",
    "HybridCoordinatorController",
    "InferenceBackend",
    "InferenceResult",
    "get_hybrid_controller",
    "HYBRID_PYMDP_AVAILABLE",
    "HYBRID_GENIUS_AVAILABLE",
]
