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

NOTE: JAX/PyMDP imports are lazy-loaded to reduce memory usage on low-memory
environments (e.g., Render free tier with 512MB limit).
"""

# Only import lightweight modules at module level
from .structure_learner import StructureLearner, Category, Observation

# Lazy loading for heavy JAX/PyMDP modules to save memory
def __getattr__(name):
    """Lazy load heavy modules only when accessed."""
    # Genius client (lightweight, no JAX)
    if name in ("GeniusClient", "VFGBuilder", "GeniusConfig"):
        from .genius_client import GeniusClient, VFGBuilder, GeniusConfig
        return {"GeniusClient": GeniusClient, "VFGBuilder": VFGBuilder, "GeniusConfig": GeniusConfig}[name]

    if name in ("GeniusStructureLearner", "GeniusObservation"):
        from .genius_structure_learner import GeniusStructureLearner, GeniusObservation
        return {"GeniusStructureLearner": GeniusStructureLearner, "GeniusObservation": GeniusObservation}[name]

    # PyMDP modules (heavy, loads JAX)
    if name in ("PyMDPStructureLearner", "PyMDPObservation", "PYMDP_AVAILABLE"):
        from .pymdp_learner import PyMDPStructureLearner, PyMDPObservation, PYMDP_AVAILABLE
        return {"PyMDPStructureLearner": PyMDPStructureLearner, "PyMDPObservation": PyMDPObservation, "PYMDP_AVAILABLE": PYMDP_AVAILABLE}[name]

    # Component POMDPs (heavy, load JAX)
    if name in ("ScrapingPOMDP", "get_scraping_pomdp"):
        from .scraping_pomdp import ScrapingPOMDP, get_scraping_pomdp
        return {"ScrapingPOMDP": ScrapingPOMDP, "get_scraping_pomdp": get_scraping_pomdp}[name]

    if name in ("ClusteringPOMDP", "get_clustering_pomdp"):
        from .clustering_pomdp import ClusteringPOMDP, get_clustering_pomdp
        return {"ClusteringPOMDP": ClusteringPOMDP, "get_clustering_pomdp": get_clustering_pomdp}[name]

    if name in ("MoveExtractionPOMDP", "get_extraction_pomdp"):
        from .move_extraction_pomdp import MoveExtractionPOMDP, get_extraction_pomdp
        return {"MoveExtractionPOMDP": MoveExtractionPOMDP, "get_extraction_pomdp": get_extraction_pomdp}[name]

    if name in ("CoordinatorPOMDP", "get_coordinator_pomdp"):
        from .coordinator_pomdp import CoordinatorPOMDP, get_coordinator_pomdp
        return {"CoordinatorPOMDP": CoordinatorPOMDP, "get_coordinator_pomdp": get_coordinator_pomdp}[name]

    if name in ("UserAdaptivePOMDP", "get_user_adaptive_pomdp"):
        from .user_adaptive_pomdp import UserAdaptivePOMDP, get_user_adaptive_pomdp
        return {"UserAdaptivePOMDP": UserAdaptivePOMDP, "get_user_adaptive_pomdp": get_user_adaptive_pomdp}[name]

    # Hybrid controller
    if name in ("HybridActiveInferenceController", "HybridScrapingController",
                "HybridClusteringController", "HybridExtractionController",
                "HybridCoordinatorController", "InferenceBackend", "InferenceResult",
                "get_hybrid_controller", "HYBRID_PYMDP_AVAILABLE", "HYBRID_GENIUS_AVAILABLE"):
        from .hybrid_controller import (
            HybridActiveInferenceController, HybridScrapingController,
            HybridClusteringController, HybridExtractionController,
            HybridCoordinatorController, InferenceBackend, InferenceResult,
            get_hybrid_controller, PYMDP_AVAILABLE as HYBRID_PYMDP_AVAILABLE,
            GENIUS_AVAILABLE as HYBRID_GENIUS_AVAILABLE,
        )
        mapping = {
            "HybridActiveInferenceController": HybridActiveInferenceController,
            "HybridScrapingController": HybridScrapingController,
            "HybridClusteringController": HybridClusteringController,
            "HybridExtractionController": HybridExtractionController,
            "HybridCoordinatorController": HybridCoordinatorController,
            "InferenceBackend": InferenceBackend,
            "InferenceResult": InferenceResult,
            "get_hybrid_controller": get_hybrid_controller,
            "HYBRID_PYMDP_AVAILABLE": HYBRID_PYMDP_AVAILABLE,
            "HYBRID_GENIUS_AVAILABLE": HYBRID_GENIUS_AVAILABLE,
        }
        return mapping[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

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
