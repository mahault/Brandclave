"""
Genius-backed POMDP implementations.

These classes mirror the PyMDP POMDPs but use VERSES Genius API for inference.
Provides proper Bayesian inference via cloud API with VFG (Variable Factor Graph) models.
"""

import logging
import os
from datetime import datetime, timedelta
from typing import Optional, Any
from dataclasses import dataclass, field

import numpy as np

from .genius_client import GeniusClient, VFGBuilder, GeniusConfig

logger = logging.getLogger(__name__)


# Check if Genius is available
GENIUS_AVAILABLE = False
try:
    if os.getenv("GENIUS_API_KEY"):
        GENIUS_AVAILABLE = True
except Exception:
    pass


class GeniusScrapingPOMDP:
    """
    Genius-backed POMDP for adaptive scraping decisions.

    Uses VERSES Genius API for proper Bayesian inference and EFE minimization.
    """

    SOURCES = [
        "skift", "hospitalitynet", "hoteldive", "hotelmanagement",
        "phocuswire", "travelweekly", "hotelnewsresource", "traveldailynews",
        "businesstravelnews", "boutiquehotelier", "hotelonline", "hoteltechreport",
        "tophotelnews", "siteminder", "ehlinsights", "cbrehotels",
        "cushmanwakefield", "costar", "traveldaily",
        "reddit", "youtube", "quora", "tripadvisor", "booking",
    ]

    PRODUCTIVITY_LEVELS = ["high", "medium", "low", "stale"]

    def __init__(self, config: Optional[GeniusConfig] = None):
        """Initialize Genius Scraping POMDP."""
        self.config = config or GeniusConfig()
        self.client: Optional[GeniusClient] = None
        self.graph_initialized = False

        # Fallback beliefs (used when Genius unavailable)
        self.source_beliefs = {s: 0.5 for s in self.SOURCES}

        # Initialize client
        if GENIUS_AVAILABLE:
            try:
                self.client = GeniusClient(self.config)
                self._initialize_graph()
            except Exception as e:
                logger.error(f"Failed to initialize Genius client: {e}")

    def _initialize_graph(self) -> bool:
        """Initialize the VFG for scraping decisions."""
        if self.client is None:
            return False

        try:
            vfg = (
                VFGBuilder()
                .add_variable("productivity", self.PRODUCTIVITY_LEVELS)
                .add_variable("source_action", self.SOURCES + ["wait"])
                .add_variable("obs_items", ["many", "some", "few"])
                .add_variable("obs_freshness", ["fresh", "moderate", "stale"])
                .add_variable("obs_errors", ["low", "medium", "high"])
                .add_categorical_factor(
                    "productivity_prior",
                    "productivity",
                    [0.25, 0.25, 0.25, 0.25]
                )
                .set_model_type("pomdp")
                .build()
            )

            self.client.set_graph(vfg)
            self.graph_initialized = True
            logger.info("Genius Scraping VFG initialized")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize Genius scraping graph: {e}")
            return False

    def select_next_source(self) -> dict:
        """Select next source using Genius EFE minimization."""
        if self.client is not None and self.graph_initialized:
            try:
                result = self.client.select_action(wait=True)

                action = result.get("action", {})
                source = action.get("source_action", "wait")

                return {
                    "source": source,
                    "beliefs": result.get("belief_state", {}),
                    "efe_components": result.get("efe_components", {}),
                    "confidence": result.get("confidence", 0.5),
                    "method": "genius_api",
                }

            except Exception as e:
                logger.warning(f"Genius action selection failed: {e}")

        # Fallback: select by belief
        return self._fallback_select_source()

    def _fallback_select_source(self) -> dict:
        """Fallback source selection."""
        best_source = max(self.source_beliefs, key=self.source_beliefs.get)
        return {
            "source": best_source,
            "beliefs": self.source_beliefs,
            "confidence": 0.5,
            "method": "fallback",
        }

    def observe_scrape_result(
        self,
        source: str,
        items_scraped: int,
        errors: int,
        novelty_ratio: float,
    ) -> dict:
        """Update beliefs after scraping result."""
        # Encode observations
        items_obs = "many" if items_scraped > 30 else ("some" if items_scraped > 10 else "few")
        fresh_obs = "fresh" if novelty_ratio > 0.7 else ("moderate" if novelty_ratio > 0.3 else "stale")
        error_obs = "low" if errors == 0 else ("medium" if errors < 3 else "high")

        # Update local beliefs
        productivity = items_scraped / max(items_scraped + errors * 5, 1)
        if source in self.source_beliefs:
            self.source_beliefs[source] = 0.8 * self.source_beliefs[source] + 0.2 * productivity

        # Learn via Genius if available
        if self.client is not None and self.graph_initialized:
            try:
                result = self.client.learn(
                    data=[{
                        "obs_items": items_obs,
                        "obs_freshness": fresh_obs,
                        "obs_errors": error_obs,
                    }],
                    wait=True,
                )
                return {
                    "source": source,
                    "learned": True,
                    "genius_result": result,
                    "method": "genius_learn",
                }
            except Exception as e:
                logger.warning(f"Genius learning failed: {e}")

        return {
            "source": source,
            "learned": False,
            "productivity": productivity,
            "method": "fallback",
        }

    def get_status(self) -> dict:
        """Get POMDP status."""
        return {
            "enabled": True,
            "genius_available": self.client is not None,
            "graph_initialized": self.graph_initialized,
            "num_sources": len(self.SOURCES),
            "source_beliefs": self.source_beliefs,
        }


class GeniusClusteringPOMDP:
    """
    Genius-backed POMDP for adaptive clustering parameter selection.
    """

    MIN_CLUSTER_SIZES = [2, 3, 4, 5]
    MIN_SAMPLES = [1, 2, 3]
    QUALITY_LEVELS = ["poor", "fair", "good", "excellent"]

    def __init__(self, config: Optional[GeniusConfig] = None):
        """Initialize Genius Clustering POMDP."""
        self.config = config or GeniusConfig()
        self.client: Optional[GeniusClient] = None
        self.graph_initialized = False

        # Build action space
        self.actions = []
        for mcs in self.MIN_CLUSTER_SIZES:
            for ms in self.MIN_SAMPLES:
                self.actions.append({"min_cluster_size": mcs, "min_samples": ms})

        # Fallback beliefs
        self.param_beliefs = {
            f"{a['min_cluster_size']}_{a['min_samples']}": 0.5
            for a in self.actions
        }

        if GENIUS_AVAILABLE:
            try:
                self.client = GeniusClient(self.config)
                self._initialize_graph()
            except Exception as e:
                logger.error(f"Failed to initialize Genius client: {e}")

    def _initialize_graph(self) -> bool:
        """Initialize VFG for clustering decisions."""
        if self.client is None:
            return False

        try:
            # Create action labels
            action_labels = [f"mcs{a['min_cluster_size']}_ms{a['min_samples']}" for a in self.actions]

            vfg = (
                VFGBuilder()
                .add_variable("quality", self.QUALITY_LEVELS)
                .add_variable("param_action", action_labels)
                .add_variable("obs_silhouette", ["negative", "low", "medium", "high"])
                .add_variable("obs_clusters", ["few", "moderate", "many", "too_many"])
                .add_variable("obs_noise", ["low", "moderate", "high"])
                .add_categorical_factor(
                    "quality_prior",
                    "quality",
                    [0.25, 0.25, 0.25, 0.25]
                )
                .set_model_type("pomdp")
                .build()
            )

            self.client.set_graph(vfg)
            self.graph_initialized = True
            logger.info("Genius Clustering VFG initialized")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize Genius clustering graph: {e}")
            return False

    def select_parameters(self, embeddings: np.ndarray) -> dict:
        """Select clustering parameters using Genius."""
        if self.client is not None and self.graph_initialized:
            try:
                result = self.client.select_action(wait=True)

                action = result.get("action", {})
                param_str = action.get("param_action", "mcs3_ms2")

                # Parse params from action string
                parts = param_str.replace("mcs", "").replace("ms", "_").split("_")
                mcs = int(parts[0]) if parts[0].isdigit() else 3
                ms = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 2

                return {
                    "min_cluster_size": mcs,
                    "min_samples": ms,
                    "beliefs": result.get("belief_state", {}),
                    "efe_components": result.get("efe_components", {}),
                    "confidence": result.get("confidence", 0.5),
                    "method": "genius_api",
                }

            except Exception as e:
                logger.warning(f"Genius action selection failed: {e}")

        # Fallback
        return self._fallback_select_parameters(embeddings)

    def _fallback_select_parameters(self, embeddings: np.ndarray) -> dict:
        """Fallback parameter selection."""
        # Use data density heuristic
        n_samples = min(100, len(embeddings))
        if n_samples < 2:
            return {"min_cluster_size": 3, "min_samples": 2, "method": "fallback"}

        indices = np.random.choice(len(embeddings), n_samples, replace=False)
        sample = embeddings[indices]

        diffs = sample[:, None, :] - sample[None, :, :]
        distances = np.sqrt(np.sum(diffs ** 2, axis=-1))
        mean_dist = np.mean(distances)
        density = 1.0 / (1.0 + mean_dist)

        if density > 0.6:
            mcs, ms = 2, 1
        elif density > 0.4:
            mcs, ms = 3, 2
        elif density > 0.2:
            mcs, ms = 4, 2
        else:
            mcs, ms = 5, 3

        return {
            "min_cluster_size": mcs,
            "min_samples": ms,
            "density": density,
            "method": "fallback",
        }

    def observe_clustering_result(
        self,
        params: dict,
        labels: np.ndarray,
        embeddings: np.ndarray,
    ) -> dict:
        """Update beliefs after clustering result."""
        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_ratio = np.sum(labels == -1) / len(labels)

        # Compute silhouette (simplified)
        try:
            from sklearn.metrics import silhouette_score
            mask = labels != -1
            if np.sum(mask) > num_clusters >= 2:
                sil_score = silhouette_score(embeddings[mask], labels[mask])
            else:
                sil_score = -1.0
        except Exception:
            sil_score = 0.0

        # Encode observations
        if sil_score < 0:
            sil_obs = "negative"
        elif sil_score < 0.25:
            sil_obs = "low"
        elif sil_score < 0.5:
            sil_obs = "medium"
        else:
            sil_obs = "high"

        if num_clusters <= 3:
            cnt_obs = "few"
        elif num_clusters <= 10:
            cnt_obs = "moderate"
        elif num_clusters <= 20:
            cnt_obs = "many"
        else:
            cnt_obs = "too_many"

        if noise_ratio < 0.1:
            noise_obs = "low"
        elif noise_ratio < 0.3:
            noise_obs = "moderate"
        else:
            noise_obs = "high"

        # Update local beliefs
        key = f"{params['min_cluster_size']}_{params['min_samples']}"
        quality = (sil_score + 1) / 2
        if key in self.param_beliefs:
            self.param_beliefs[key] = 0.8 * self.param_beliefs[key] + 0.2 * quality

        # Learn via Genius
        if self.client is not None and self.graph_initialized:
            try:
                result = self.client.learn(
                    data=[{
                        "obs_silhouette": sil_obs,
                        "obs_clusters": cnt_obs,
                        "obs_noise": noise_obs,
                    }],
                    wait=True,
                )
                return {
                    "silhouette": sil_score,
                    "num_clusters": num_clusters,
                    "noise_ratio": noise_ratio,
                    "learned": True,
                    "method": "genius_learn",
                }
            except Exception as e:
                logger.warning(f"Genius learning failed: {e}")

        return {
            "silhouette": sil_score,
            "num_clusters": num_clusters,
            "noise_ratio": noise_ratio,
            "learned": False,
            "method": "fallback",
        }

    def get_status(self) -> dict:
        """Get POMDP status."""
        return {
            "enabled": True,
            "genius_available": self.client is not None,
            "graph_initialized": self.graph_initialized,
            "num_actions": len(self.actions),
            "param_beliefs": self.param_beliefs,
        }


class GeniusExtractionPOMDP:
    """
    Genius-backed POMDP for adaptive move extraction method selection.
    """

    METHODS = ["skip", "regex", "llm_single", "llm_multi"]
    ACTION_COSTS = {"skip": 0.0, "regex": 0.01, "llm_single": 1.0, "llm_multi": 3.0}
    QUALITY_LEVELS = ["noise", "weak_signal", "clear_signal", "strong_signal"]

    MOVE_KEYWORDS = [
        "acquisition", "merger", "partnership", "expansion", "investment",
        "launch", "rebrand", "renovation", "opening", "closing",
        "deal", "agreement", "joint venture", "stake",
        "million", "billion", "property", "portfolio", "brand",
    ]

    def __init__(self, config: Optional[GeniusConfig] = None):
        """Initialize Genius Extraction POMDP."""
        self.config = config or GeniusConfig()
        self.client: Optional[GeniusClient] = None
        self.graph_initialized = False

        # Fallback beliefs
        self.method_beliefs = {
            m: {"success_rate": 0.5, "avg_quality": 0.5, "total_uses": 0}
            for m in self.METHODS
        }
        self.llm_calls_saved = 0
        self.total_extractions = 0

        if GENIUS_AVAILABLE:
            try:
                self.client = GeniusClient(self.config)
                self._initialize_graph()
            except Exception as e:
                logger.error(f"Failed to initialize Genius client: {e}")

    def _initialize_graph(self) -> bool:
        """Initialize VFG for extraction decisions."""
        if self.client is None:
            return False

        try:
            vfg = (
                VFGBuilder()
                .add_variable("article_quality", self.QUALITY_LEVELS)
                .add_variable("method_action", self.METHODS)
                .add_variable("obs_title", ["none", "weak", "strong"])
                .add_variable("obs_length", ["short", "medium", "long"])
                .add_variable("obs_source", ["low_rep", "medium_rep", "high_rep"])
                .add_categorical_factor(
                    "quality_prior",
                    "article_quality",
                    [0.3, 0.3, 0.25, 0.15]
                )
                .set_model_type("pomdp")
                .build()
            )

            self.client.set_graph(vfg)
            self.graph_initialized = True
            logger.info("Genius Extraction VFG initialized")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize Genius extraction graph: {e}")
            return False

    def _encode_article(self, article: dict) -> tuple[str, str, str]:
        """Encode article features for Genius."""
        title = (article.get("title") or "").lower()
        content = article.get("content") or ""
        source = (article.get("source") or "").lower()

        # Title keywords
        keyword_count = sum(1 for kw in self.MOVE_KEYWORDS if kw in title)
        if keyword_count == 0:
            title_obs = "none"
        elif keyword_count <= 2:
            title_obs = "weak"
        else:
            title_obs = "strong"

        # Content length
        if len(content) < 500:
            length_obs = "short"
        elif len(content) < 2000:
            length_obs = "medium"
        else:
            length_obs = "long"

        # Source reputation
        high_rep = ["skift", "hospitalitynet", "costar", "phocuswire"]
        low_rep = ["reddit", "quora"]
        if source in high_rep:
            source_obs = "high_rep"
        elif source in low_rep:
            source_obs = "low_rep"
        else:
            source_obs = "medium_rep"

        return title_obs, length_obs, source_obs

    def select_extraction_method(self, article: dict) -> dict:
        """Select extraction method using Genius."""
        title_obs, length_obs, source_obs = self._encode_article(article)

        if self.client is not None and self.graph_initialized:
            try:
                result = self.client.select_action(
                    observations={
                        "obs_title": title_obs,
                        "obs_length": length_obs,
                        "obs_source": source_obs,
                    },
                    wait=True,
                )

                action = result.get("action", {})
                method = action.get("method_action", "llm_single")

                return {
                    "method": method,
                    "expected_cost": self.ACTION_COSTS.get(method, 1.0),
                    "beliefs": result.get("belief_state", {}),
                    "efe_components": result.get("efe_components", {}),
                    "confidence": result.get("confidence", 0.5),
                    "reason": "Genius EFE minimization",
                    "method_detail": "genius_api",
                }

            except Exception as e:
                logger.warning(f"Genius action selection failed: {e}")

        # Fallback
        return self._fallback_select_method(title_obs, length_obs, source_obs)

    def _fallback_select_method(
        self,
        title_obs: str,
        length_obs: str,
        source_obs: str,
    ) -> dict:
        """Fallback method selection."""
        # Score based on observations
        title_score = {"none": 0, "weak": 0.5, "strong": 1}[title_obs]
        length_score = {"short": 0, "medium": 0.5, "long": 1}[length_obs]
        source_score = {"low_rep": 0, "medium_rep": 0.5, "high_rep": 1}[source_obs]

        score = title_score * 0.5 + length_score * 0.3 + source_score * 0.2

        if score < 0.2:
            method = "skip"
        elif score < 0.4:
            method = "regex"
        elif score < 0.7:
            method = "llm_single"
        else:
            method = "llm_multi"

        return {
            "method": method,
            "expected_cost": self.ACTION_COSTS[method],
            "confidence": score,
            "reason": f"Fallback heuristic (score={score:.2f})",
            "method_detail": "fallback",
        }

    def observe_extraction_result(
        self,
        article: dict,
        method: str,
        result: Optional[dict],
    ) -> dict:
        """Update beliefs after extraction result."""
        self.total_extractions += 1

        quality = result.get("confidence_score", 0.5) if result else 0.0
        success = quality >= 0.5

        if method in ["skip", "regex"]:
            self.llm_calls_saved += 1

        if method in self.method_beliefs:
            mb = self.method_beliefs[method]
            mb["total_uses"] += 1
            n = mb["total_uses"]
            mb["success_rate"] = (mb["success_rate"] * (n - 1) + (1.0 if success else 0.0)) / n
            mb["avg_quality"] = (mb["avg_quality"] * (n - 1) + quality) / n

        # Learn via Genius
        if self.client is not None and self.graph_initialized:
            try:
                title_obs, length_obs, source_obs = self._encode_article(article)
                self.client.learn(
                    data=[{
                        "obs_title": title_obs,
                        "obs_length": length_obs,
                        "obs_source": source_obs,
                        "method_action": method,
                    }],
                    wait=False,  # Async learning
                )
            except Exception as e:
                logger.warning(f"Genius learning failed: {e}")

        return {
            "method": method,
            "quality": quality,
            "success": success,
            "llm_calls_saved": self.llm_calls_saved,
            "total_extractions": self.total_extractions,
        }

    def get_status(self) -> dict:
        """Get POMDP status."""
        return {
            "enabled": True,
            "genius_available": self.client is not None,
            "graph_initialized": self.graph_initialized,
            "num_methods": len(self.METHODS),
            "total_extractions": self.total_extractions,
            "llm_calls_saved": self.llm_calls_saved,
            "savings_rate": self.llm_calls_saved / max(self.total_extractions, 1),
            "method_beliefs": self.method_beliefs,
        }


class GeniusCoordinatorPOMDP:
    """
    Genius-backed POMDP for cross-component coordination.
    """

    OPTIMIZATION_LEVELS = ["suboptimal", "baseline", "good", "optimal"]
    ACTIONS = ["focus_scraping", "focus_clustering", "focus_extraction", "balance_all", "investigate"]

    def __init__(self, config: Optional[GeniusConfig] = None):
        """Initialize Genius Coordinator POMDP."""
        self.config = config or GeniusConfig()
        self.client: Optional[GeniusClient] = None
        self.graph_initialized = False

        self.focus_weights = {
            "scraping": 0.33,
            "clustering": 0.33,
            "extraction": 0.34,
        }

        self.signal_history = {
            "scraping": [],
            "clustering": [],
            "extraction": [],
        }

        if GENIUS_AVAILABLE:
            try:
                self.client = GeniusClient(self.config)
                self._initialize_graph()
            except Exception as e:
                logger.error(f"Failed to initialize Genius client: {e}")

    def _initialize_graph(self) -> bool:
        """Initialize VFG for coordination."""
        if self.client is None:
            return False

        try:
            vfg = (
                VFGBuilder()
                .add_variable("optimization", self.OPTIMIZATION_LEVELS)
                .add_variable("action", self.ACTIONS)
                .add_variable("obs_scraping", ["low", "medium", "high"])
                .add_variable("obs_clustering", ["poor", "fair", "good"])
                .add_variable("obs_extraction", ["low", "medium", "high"])
                .add_variable("obs_correlation", ["none", "weak", "strong"])
                .add_categorical_factor(
                    "optimization_prior",
                    "optimization",
                    [0.2, 0.4, 0.3, 0.1]
                )
                .set_model_type("pomdp")
                .build()
            )

            self.client.set_graph(vfg)
            self.graph_initialized = True
            logger.info("Genius Coordinator VFG initialized")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize Genius coordinator graph: {e}")
            return False

    def record_scraping_result(self, source: str, items: int, errors: int, novelty: float):
        """Record scraping result."""
        productivity = items / max(items + errors * 5, 1)
        self.signal_history["scraping"].append({
            "value": productivity,
            "timestamp": datetime.utcnow(),
        })
        # Keep last 100
        self.signal_history["scraping"] = self.signal_history["scraping"][-100:]

    def record_clustering_result(self, silhouette: float, num_clusters: int, noise_ratio: float):
        """Record clustering result."""
        quality = max(0, (silhouette + 1) / 2) * (1 - noise_ratio)
        self.signal_history["clustering"].append({
            "value": quality,
            "timestamp": datetime.utcnow(),
        })
        self.signal_history["clustering"] = self.signal_history["clustering"][-100:]

    def record_extraction_result(self, method: str, quality: float, cost: float, success: bool):
        """Record extraction result."""
        efficiency = quality / max(cost, 0.1) if success else 0
        self.signal_history["extraction"].append({
            "value": efficiency,
            "timestamp": datetime.utcnow(),
        })
        self.signal_history["extraction"] = self.signal_history["extraction"][-100:]

    def _encode_observations(self) -> dict[str, str]:
        """Encode current state into observation dict."""
        obs = {}

        # Scraping
        scraping_vals = [s["value"] for s in self.signal_history["scraping"][-20:]]
        avg = np.mean(scraping_vals) if scraping_vals else 0.5
        obs["obs_scraping"] = "low" if avg < 0.3 else ("medium" if avg < 0.6 else "high")

        # Clustering
        cluster_vals = [s["value"] for s in self.signal_history["clustering"][-20:]]
        avg = np.mean(cluster_vals) if cluster_vals else 0.5
        obs["obs_clustering"] = "poor" if avg < 0.3 else ("fair" if avg < 0.6 else "good")

        # Extraction
        extract_vals = [s["value"] for s in self.signal_history["extraction"][-20:]]
        avg = np.mean(extract_vals) if extract_vals else 0.5
        obs["obs_extraction"] = "low" if avg < 0.3 else ("medium" if avg < 0.6 else "high")

        # Correlation (simplified)
        obs["obs_correlation"] = "weak"

        return obs

    def coordinate(self) -> dict:
        """Run coordination using Genius."""
        obs = self._encode_observations()

        if self.client is not None and self.graph_initialized:
            try:
                result = self.client.select_action(observations=obs, wait=True)

                action = result.get("action", {})
                selected = action.get("action", "balance_all")

                self._update_focus_weights(selected)

                return {
                    "action": selected,
                    "beliefs": result.get("belief_state", {}),
                    "efe_components": result.get("efe_components", {}),
                    "focus_weights": self.focus_weights.copy(),
                    "method": "genius_api",
                }

            except Exception as e:
                logger.warning(f"Genius coordination failed: {e}")

        # Fallback
        return self._fallback_coordinate()

    def _fallback_coordinate(self) -> dict:
        """Fallback coordination."""
        component_scores = {}
        for comp, signals in self.signal_history.items():
            vals = [s["value"] for s in signals[-20:]]
            component_scores[comp] = np.mean(vals) if vals else 0.5

        weakest = min(component_scores, key=component_scores.get)
        action = f"focus_{weakest}"

        self._update_focus_weights(action)

        return {
            "action": action,
            "component_scores": component_scores,
            "focus_weights": self.focus_weights.copy(),
            "method": "fallback",
        }

    def _update_focus_weights(self, action: str):
        """Update focus weights based on action."""
        decay = 0.9

        if "focus_" in action:
            component = action.replace("focus_", "")
            if component in self.focus_weights:
                self.focus_weights[component] = min(0.5, self.focus_weights[component] * decay + 0.1)
                for other in self.focus_weights:
                    if other != component:
                        self.focus_weights[other] *= decay
        elif action == "balance_all":
            self.focus_weights = {"scraping": 0.33, "clustering": 0.33, "extraction": 0.34}

        # Normalize
        total = sum(self.focus_weights.values())
        for k in self.focus_weights:
            self.focus_weights[k] /= total

    def get_focus_recommendation(self, component: str) -> float:
        """Get focus recommendation."""
        return self.focus_weights.get(component, 0.33)

    def get_status(self) -> dict:
        """Get POMDP status."""
        return {
            "enabled": True,
            "genius_available": self.client is not None,
            "graph_initialized": self.graph_initialized,
            "focus_weights": self.focus_weights,
            "signal_counts": {k: len(v) for k, v in self.signal_history.items()},
        }


# Factory functions for easy instantiation
def create_genius_scraping_pomdp(config: Optional[GeniusConfig] = None) -> GeniusScrapingPOMDP:
    """Create a Genius-backed scraping POMDP."""
    return GeniusScrapingPOMDP(config)


def create_genius_clustering_pomdp(config: Optional[GeniusConfig] = None) -> GeniusClusteringPOMDP:
    """Create a Genius-backed clustering POMDP."""
    return GeniusClusteringPOMDP(config)


def create_genius_extraction_pomdp(config: Optional[GeniusConfig] = None) -> GeniusExtractionPOMDP:
    """Create a Genius-backed extraction POMDP."""
    return GeniusExtractionPOMDP(config)


def create_genius_coordinator_pomdp(config: Optional[GeniusConfig] = None) -> GeniusCoordinatorPOMDP:
    """Create a Genius-backed coordinator POMDP."""
    return GeniusCoordinatorPOMDP(config)
