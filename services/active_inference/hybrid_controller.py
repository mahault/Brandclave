"""
Hybrid Active Inference Controller.

Provides a unified interface that can use either:
- PyMDP (local JAX-based inference) - fast, offline, open source
- VERSES Genius (cloud API) - proper Bayesian inference, requires license

The controller automatically falls back to PyMDP if Genius is unavailable,
and can be configured to prefer one backend over the other.
"""

import logging
import os
from enum import Enum
from typing import Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

logger = logging.getLogger(__name__)


class InferenceBackend(str, Enum):
    """Available inference backends."""
    PYMDP = "pymdp"      # Local JAX-based
    GENIUS = "genius"    # VERSES cloud API
    AUTO = "auto"        # Auto-select based on availability


@dataclass
class InferenceResult:
    """Result from any inference backend."""
    action: str
    action_index: int
    beliefs: dict[str, float]
    expected_free_energy: float
    confidence: float
    backend: str
    latency_ms: float
    metadata: dict = field(default_factory=dict)


# Check backend availability
PYMDP_AVAILABLE = False
GENIUS_AVAILABLE = False

try:
    from .scraping_pomdp import ScrapingPOMDP, get_scraping_pomdp
    from .clustering_pomdp import ClusteringPOMDP, get_clustering_pomdp
    from .move_extraction_pomdp import MoveExtractionPOMDP, get_extraction_pomdp
    from .coordinator_pomdp import CoordinatorPOMDP, get_coordinator_pomdp
    PYMDP_AVAILABLE = True
    logger.info("PyMDP backend available")
except ImportError as e:
    logger.warning(f"PyMDP backend not available: {e}")

try:
    from .genius_client import GeniusClient, VFGBuilder, GeniusConfig
    # Check if API key is configured
    if os.getenv("GENIUS_API_KEY"):
        GENIUS_AVAILABLE = True
        logger.info("Genius backend available")
    else:
        logger.info("Genius API key not configured")
except ImportError as e:
    logger.warning(f"Genius client not available: {e}")


class HybridScrapingController:
    """
    Hybrid controller for scraping decisions.

    Uses either PyMDP or Genius backend for action selection.
    """

    def __init__(
        self,
        preferred_backend: InferenceBackend = InferenceBackend.AUTO,
        genius_config: Optional[GeniusConfig] = None,
    ):
        self.preferred_backend = preferred_backend
        self._pymdp_agent: Optional[ScrapingPOMDP] = None
        self._genius_client: Optional[GeniusClient] = None
        self._genius_graph_initialized = False

        # Initialize backends
        if PYMDP_AVAILABLE:
            try:
                self._pymdp_agent = get_scraping_pomdp()
                logger.info("PyMDP scraping agent initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize PyMDP scraping agent: {e}")

        if GENIUS_AVAILABLE and preferred_backend in [InferenceBackend.GENIUS, InferenceBackend.AUTO]:
            try:
                self._genius_client = GeniusClient(genius_config)
                logger.info("Genius client initialized for scraping")
            except Exception as e:
                logger.warning(f"Failed to initialize Genius client: {e}")

    def _select_backend(self) -> str:
        """Select which backend to use based on preference and availability."""
        if self.preferred_backend == InferenceBackend.GENIUS:
            if self._genius_client is not None:
                return "genius"
            elif self._pymdp_agent is not None:
                logger.warning("Genius preferred but unavailable, falling back to PyMDP")
                return "pymdp"
        elif self.preferred_backend == InferenceBackend.PYMDP:
            if self._pymdp_agent is not None:
                return "pymdp"
            elif self._genius_client is not None:
                logger.warning("PyMDP preferred but unavailable, falling back to Genius")
                return "genius"
        else:  # AUTO - prefer PyMDP for speed
            if self._pymdp_agent is not None:
                return "pymdp"
            elif self._genius_client is not None:
                return "genius"

        return "fallback"

    def _init_genius_graph(self) -> bool:
        """Initialize the Genius VFG for scraping decisions."""
        if self._genius_client is None or self._genius_graph_initialized:
            return self._genius_graph_initialized

        try:
            # Build VFG for scraping POMDP
            sources = [
                "skift", "hospitalitynet", "hoteldive", "hotelmanagement",
                "phocuswire", "travelweekly", "hotelnewsresource", "traveldailynews",
                "businesstravelnews", "boutiquehotelier", "hotelonline", "hoteltechreport",
                "tophotelnews", "siteminder", "ehlinsights", "cbrehotels",
                "cushmanwakefield", "costar", "traveldaily",
                "reddit", "youtube", "quora", "tripadvisor", "booking",
            ]

            vfg = (
                VFGBuilder()
                .add_variable("productivity", ["high", "medium", "low", "stale"])
                .add_variable("source_action", sources + ["wait"])
                .add_variable("observed_items", ["many", "some", "few"])
                .add_variable("observed_freshness", ["fresh", "moderate", "stale"])
                .add_variable("observed_errors", ["low", "medium", "high"])
                .add_categorical_factor(
                    "productivity_prior",
                    "productivity",
                    [0.25, 0.25, 0.25, 0.25]  # Uniform prior
                )
                .set_model_type("pomdp")
                .build()
            )

            self._genius_client.set_graph(vfg)
            self._genius_graph_initialized = True
            logger.info("Genius scraping VFG initialized")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize Genius scraping graph: {e}")
            return False

    def select_next_source(self) -> InferenceResult:
        """
        Select the next source to scrape.

        Returns:
            InferenceResult with selected source and metadata
        """
        backend = self._select_backend()
        start_time = datetime.utcnow()

        if backend == "pymdp" and self._pymdp_agent is not None:
            result = self._pymdp_agent.select_next_source()
            latency = (datetime.utcnow() - start_time).total_seconds() * 1000

            return InferenceResult(
                action=result.get("source", "wait"),
                action_index=result.get("action_index", -1),
                beliefs=result.get("efe_values", {}),
                expected_free_energy=result.get("priority", 0),
                confidence=1 - abs(result.get("priority", 0)),
                backend="pymdp",
                latency_ms=latency,
                metadata=result,
            )

        elif backend == "genius" and self._genius_client is not None:
            self._init_genius_graph()
            try:
                result = self._genius_client.select_action(wait=True)
                latency = (datetime.utcnow() - start_time).total_seconds() * 1000

                action = result.get("action", {})
                return InferenceResult(
                    action=action.get("source_action", "wait"),
                    action_index=-1,
                    beliefs=result.get("belief_state", {}),
                    expected_free_energy=result.get("efe_components", {}).get("total", 0),
                    confidence=result.get("confidence", 0.5),
                    backend="genius",
                    latency_ms=latency,
                    metadata=result,
                )
            except Exception as e:
                logger.error(f"Genius action selection failed: {e}")
                # Fall back to PyMDP
                if self._pymdp_agent is not None:
                    return self.select_next_source()

        # Fallback
        latency = (datetime.utcnow() - start_time).total_seconds() * 1000
        return InferenceResult(
            action="skift",  # Default to a reliable source
            action_index=0,
            beliefs={},
            expected_free_energy=0,
            confidence=0.3,
            backend="fallback",
            latency_ms=latency,
            metadata={"reason": "No backend available"},
        )

    def observe_result(
        self,
        source: str,
        items: int,
        errors: int,
        novelty: float,
    ) -> dict:
        """Record observation for learning."""
        results = {}

        if self._pymdp_agent is not None:
            pymdp_result = self._pymdp_agent.observe_scrape_result(
                source=source,
                items_scraped=items,
                errors=errors,
                novelty_ratio=novelty,
            )
            results["pymdp"] = pymdp_result

        if self._genius_client is not None and self._genius_graph_initialized:
            try:
                # Encode observation
                items_obs = "many" if items > 30 else ("some" if items > 10 else "few")
                fresh_obs = "fresh" if novelty > 0.7 else ("moderate" if novelty > 0.3 else "stale")
                error_obs = "low" if errors == 0 else ("medium" if errors < 3 else "high")

                genius_result = self._genius_client.learn(
                    data=[{
                        "observed_items": items_obs,
                        "observed_freshness": fresh_obs,
                        "observed_errors": error_obs,
                    }],
                    wait=True,
                )
                results["genius"] = genius_result
            except Exception as e:
                logger.warning(f"Genius learning failed: {e}")

        return results

    def get_status(self) -> dict:
        """Get status of both backends."""
        status = {
            "preferred_backend": self.preferred_backend.value,
            "active_backend": self._select_backend(),
            "pymdp_available": self._pymdp_agent is not None,
            "genius_available": self._genius_client is not None,
            "genius_graph_initialized": self._genius_graph_initialized,
        }

        if self._pymdp_agent is not None:
            status["pymdp_status"] = self._pymdp_agent.get_status()

        return status


class HybridClusteringController:
    """
    Hybrid controller for clustering parameter selection.
    """

    def __init__(
        self,
        preferred_backend: InferenceBackend = InferenceBackend.AUTO,
        genius_config: Optional[GeniusConfig] = None,
    ):
        self.preferred_backend = preferred_backend
        self._pymdp_agent: Optional[ClusteringPOMDP] = None
        self._genius_client: Optional[GeniusClient] = None

        if PYMDP_AVAILABLE:
            try:
                self._pymdp_agent = get_clustering_pomdp()
            except Exception as e:
                logger.warning(f"Failed to initialize PyMDP clustering agent: {e}")

        if GENIUS_AVAILABLE and preferred_backend in [InferenceBackend.GENIUS, InferenceBackend.AUTO]:
            try:
                self._genius_client = GeniusClient(genius_config)
            except Exception as e:
                logger.warning(f"Failed to initialize Genius client: {e}")

    def select_parameters(self, embeddings: np.ndarray) -> InferenceResult:
        """Select clustering parameters."""
        start_time = datetime.utcnow()

        if self._pymdp_agent is not None:
            result = self._pymdp_agent.select_parameters(embeddings)
            latency = (datetime.utcnow() - start_time).total_seconds() * 1000

            return InferenceResult(
                action=f"mcs={result['min_cluster_size']},ms={result['min_samples']}",
                action_index=result.get("action_index", -1),
                beliefs={},
                expected_free_energy=result.get("efe", 0),
                confidence=result.get("confidence", 0.5),
                backend=result.get("method", "pymdp"),
                latency_ms=latency,
                metadata=result,
            )

        # Fallback
        latency = (datetime.utcnow() - start_time).total_seconds() * 1000
        return InferenceResult(
            action="mcs=3,ms=2",
            action_index=-1,
            beliefs={},
            expected_free_energy=0,
            confidence=0.3,
            backend="fallback",
            latency_ms=latency,
            metadata={"min_cluster_size": 3, "min_samples": 2},
        )

    def observe_result(
        self,
        params: dict,
        labels: np.ndarray,
        embeddings: np.ndarray,
    ) -> dict:
        """Record clustering result for learning."""
        if self._pymdp_agent is not None:
            return self._pymdp_agent.observe_clustering_result(params, labels, embeddings)
        return {}

    def get_status(self) -> dict:
        """Get status."""
        status = {
            "preferred_backend": self.preferred_backend.value,
            "pymdp_available": self._pymdp_agent is not None,
            "genius_available": self._genius_client is not None,
        }
        if self._pymdp_agent is not None:
            status["pymdp_status"] = self._pymdp_agent.get_status()
        return status


class HybridExtractionController:
    """
    Hybrid controller for move extraction method selection.
    """

    def __init__(
        self,
        preferred_backend: InferenceBackend = InferenceBackend.AUTO,
        genius_config: Optional[GeniusConfig] = None,
    ):
        self.preferred_backend = preferred_backend
        self._pymdp_agent: Optional[MoveExtractionPOMDP] = None
        self._genius_client: Optional[GeniusClient] = None

        if PYMDP_AVAILABLE:
            try:
                self._pymdp_agent = get_extraction_pomdp()
            except Exception as e:
                logger.warning(f"Failed to initialize PyMDP extraction agent: {e}")

        if GENIUS_AVAILABLE and preferred_backend in [InferenceBackend.GENIUS, InferenceBackend.AUTO]:
            try:
                self._genius_client = GeniusClient(genius_config)
            except Exception as e:
                logger.warning(f"Failed to initialize Genius client: {e}")

    def select_method(self, article: dict) -> InferenceResult:
        """Select extraction method for an article."""
        start_time = datetime.utcnow()

        if self._pymdp_agent is not None:
            result = self._pymdp_agent.select_extraction_method(article)
            latency = (datetime.utcnow() - start_time).total_seconds() * 1000

            return InferenceResult(
                action=result.get("method", "llm_single"),
                action_index=result.get("action_index", -1),
                beliefs=result.get("beliefs", {}),
                expected_free_energy=result.get("efe", 0),
                confidence=result.get("expected_quality", 0.5),
                backend=result.get("method_detail", "pymdp"),
                latency_ms=latency,
                metadata=result,
            )

        # Fallback
        latency = (datetime.utcnow() - start_time).total_seconds() * 1000
        return InferenceResult(
            action="llm_single",
            action_index=2,
            beliefs={},
            expected_free_energy=0,
            confidence=0.5,
            backend="fallback",
            latency_ms=latency,
            metadata={},
        )

    def observe_result(
        self,
        article: dict,
        method: str,
        result: Optional[dict],
    ) -> dict:
        """Record extraction result for learning."""
        if self._pymdp_agent is not None:
            return self._pymdp_agent.observe_extraction_result(article, method, result)
        return {}

    def get_status(self) -> dict:
        """Get status."""
        status = {
            "preferred_backend": self.preferred_backend.value,
            "pymdp_available": self._pymdp_agent is not None,
            "genius_available": self._genius_client is not None,
        }
        if self._pymdp_agent is not None:
            status["pymdp_status"] = self._pymdp_agent.get_status()
        return status


class HybridCoordinatorController:
    """
    Hybrid controller for cross-component coordination.
    """

    def __init__(
        self,
        preferred_backend: InferenceBackend = InferenceBackend.AUTO,
        genius_config: Optional[GeniusConfig] = None,
    ):
        self.preferred_backend = preferred_backend
        self._pymdp_agent: Optional[CoordinatorPOMDP] = None
        self._genius_client: Optional[GeniusClient] = None

        if PYMDP_AVAILABLE:
            try:
                self._pymdp_agent = get_coordinator_pomdp()
            except Exception as e:
                logger.warning(f"Failed to initialize PyMDP coordinator agent: {e}")

        if GENIUS_AVAILABLE and preferred_backend in [InferenceBackend.GENIUS, InferenceBackend.AUTO]:
            try:
                self._genius_client = GeniusClient(genius_config)
            except Exception as e:
                logger.warning(f"Failed to initialize Genius client: {e}")

    def coordinate(self) -> InferenceResult:
        """Run coordination logic."""
        start_time = datetime.utcnow()

        if self._pymdp_agent is not None:
            result = self._pymdp_agent.coordinate()
            latency = (datetime.utcnow() - start_time).total_seconds() * 1000

            return InferenceResult(
                action=result.get("action", "balance_all"),
                action_index=result.get("action_index", -1),
                beliefs=result.get("beliefs", {}),
                expected_free_energy=0,
                confidence=0.5,
                backend=result.get("method", "pymdp"),
                latency_ms=latency,
                metadata=result,
            )

        # Fallback
        latency = (datetime.utcnow() - start_time).total_seconds() * 1000
        return InferenceResult(
            action="balance_all",
            action_index=3,
            beliefs={},
            expected_free_energy=0,
            confidence=0.3,
            backend="fallback",
            latency_ms=latency,
            metadata={},
        )

    def record_scraping_result(self, source: str, items: int, errors: int, novelty: float):
        """Record scraping result."""
        if self._pymdp_agent is not None:
            self._pymdp_agent.record_scraping_result(source, items, errors, novelty)

    def record_clustering_result(self, silhouette: float, num_clusters: int, noise_ratio: float):
        """Record clustering result."""
        if self._pymdp_agent is not None:
            self._pymdp_agent.record_clustering_result(silhouette, num_clusters, noise_ratio)

    def record_extraction_result(self, method: str, quality: float, cost: float, success: bool):
        """Record extraction result."""
        if self._pymdp_agent is not None:
            self._pymdp_agent.record_extraction_result(method, quality, cost, success)

    def get_focus_recommendation(self, component: str) -> float:
        """Get focus recommendation for a component."""
        if self._pymdp_agent is not None:
            return self._pymdp_agent.get_focus_recommendation(component)
        return 0.33

    def get_status(self) -> dict:
        """Get status."""
        status = {
            "preferred_backend": self.preferred_backend.value,
            "pymdp_available": self._pymdp_agent is not None,
            "genius_available": self._genius_client is not None,
        }
        if self._pymdp_agent is not None:
            status["pymdp_status"] = self._pymdp_agent.get_status()
        return status


# Master controller that manages all hybrid controllers
class HybridActiveInferenceController:
    """
    Master controller managing all POMDP agents.

    Provides a unified interface for:
    - Scraping decisions
    - Clustering parameter selection
    - Move extraction method selection
    - Cross-component coordination

    Can use either PyMDP (local) or Genius (cloud) backends.
    """

    def __init__(
        self,
        preferred_backend: InferenceBackend = InferenceBackend.AUTO,
        genius_config: Optional[GeniusConfig] = None,
    ):
        self.preferred_backend = preferred_backend
        self.genius_config = genius_config

        # Initialize all controllers
        self.scraping = HybridScrapingController(preferred_backend, genius_config)
        self.clustering = HybridClusteringController(preferred_backend, genius_config)
        self.extraction = HybridExtractionController(preferred_backend, genius_config)
        self.coordinator = HybridCoordinatorController(preferred_backend, genius_config)

        logger.info(f"Hybrid Active Inference Controller initialized (preferred: {preferred_backend.value})")

    def get_status(self) -> dict:
        """Get status of all controllers."""
        return {
            "preferred_backend": self.preferred_backend.value,
            "pymdp_available": PYMDP_AVAILABLE,
            "genius_available": GENIUS_AVAILABLE,
            "scraping": self.scraping.get_status(),
            "clustering": self.clustering.get_status(),
            "extraction": self.extraction.get_status(),
            "coordinator": self.coordinator.get_status(),
        }


# Singleton instance
_hybrid_controller: Optional[HybridActiveInferenceController] = None


def get_hybrid_controller(
    preferred_backend: InferenceBackend = InferenceBackend.AUTO,
) -> HybridActiveInferenceController:
    """Get or create the hybrid controller instance."""
    global _hybrid_controller
    if _hybrid_controller is None:
        _hybrid_controller = HybridActiveInferenceController(preferred_backend)
    return _hybrid_controller
