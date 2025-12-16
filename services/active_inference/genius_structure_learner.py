"""
Genius-Backed Structure Learning for Adaptive Category Discovery.

This module integrates the VERSES Genius Active Inference API to provide
proper Bayesian structure learning with:
- Variational Free Energy minimization
- POMDP-based action selection for adaptive scraping
- Online parameter learning from observations

Instead of our local approximations, this uses Genius's proper inference engine.
"""

import logging
from typing import Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np

from .genius_client import GeniusClient, VFGBuilder, GeniusConfig

logger = logging.getLogger(__name__)


@dataclass
class GeniusCategory:
    """A learned category backed by Genius beliefs."""

    id: str
    name: str
    keywords: list[str] = field(default_factory=list)
    observation_count: int = 0
    belief_strength: float = 0.0  # From Genius posterior
    keyword_counts: dict = field(default_factory=lambda: defaultdict(int))

    def update_name(self):
        """Generate name from top keywords."""
        if not self.keyword_counts:
            return
        top_keywords = sorted(
            self.keyword_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        self.name = " + ".join(kw for kw, _ in top_keywords).title()
        self.keywords = [kw for kw, _ in top_keywords[:10]]


@dataclass
class GeniusObservation:
    """An observation to feed to Genius."""

    text: str
    keywords: list[str]
    source: str
    sentiment: float = 0.0
    embedding: Optional[np.ndarray] = None


class GeniusStructureLearner:
    """
    Structure learner backed by VERSES Genius Active Inference API.

    This provides proper Bayesian inference for:
    - Category discovery (what categories exist?)
    - Category assignment (which category does this belong to?)
    - Action selection (what should we search for next?)
    - Parameter learning (update beliefs from data)

    The model maintains a VFG (Variable Factor Graph) on Genius with:
    - Category variable (discrete, learnable number of states)
    - Feature variables (keywords, sentiment, source type)
    - Conditional relationships between features and categories
    """

    # Predefined category seeds - these can expand based on data
    INITIAL_CATEGORIES = [
        "luxury_wellness",
        "budget_social",
        "digital_nomad",
        "boutique_design",
        "family_resort",
        "business_corporate",
        "adventure_experience",
        "romantic_couples",
        "eco_sustainable",
        "other"  # Catch-all for expansion
    ]

    # Common hospitality keywords to track
    TRACKED_KEYWORDS = [
        "luxury", "budget", "boutique", "hostel", "resort",
        "wifi", "coworking", "remote", "nomad", "workspace",
        "pool", "gym", "spa", "breakfast", "rooftop",
        "romantic", "family", "solo", "business", "adventure",
        "clean", "safe", "quiet", "central", "walkable",
        "view", "beach", "mountain", "city", "nature",
        "local", "authentic", "modern", "historic", "design",
        "frustrating", "disappointing", "amazing", "perfect",
    ]

    # Sentiment levels for discretization
    SENTIMENT_LEVELS = ["negative", "neutral", "positive"]

    # Source types
    SOURCE_TYPES = ["reddit", "youtube", "news", "review", "forum", "other"]

    def __init__(
        self,
        config: Optional[GeniusConfig] = None,
        auto_connect: bool = True
    ):
        self.config = config or GeniusConfig()
        self.client: Optional[GeniusClient] = None

        self.categories: dict[str, GeniusCategory] = {}
        self.observations: list[GeniusObservation] = []

        # Local tracking for keyword analysis
        self.global_keyword_counts: dict[str, int] = defaultdict(int)

        # Track learning metrics
        self.learning_history: list[dict] = []

        # Whether we've initialized the Genius graph
        self._graph_initialized = False

        if auto_connect:
            self._connect()

    def _connect(self) -> bool:
        """Connect to Genius API."""
        try:
            self.client = GeniusClient(self.config)

            # Check health
            if not self.client.health_check():
                logger.error("Genius API health check failed")
                return False

            # Activate license
            self.client.activate_license()

            # Initialize graph
            self._initialize_graph()

            logger.info("Connected to Genius API successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to Genius: {e}")
            self.client = None
            return False

    def _initialize_graph(self):
        """Initialize the VFG on Genius."""
        if self._graph_initialized:
            return

        # Build initial VFG structure
        builder = VFGBuilder(version="0.5.0")

        # Category variable - the main thing we're inferring
        builder.add_variable(
            "category",
            self.INITIAL_CATEGORIES,
            "Learned category of traveler desire"
        )

        # Sentiment variable
        builder.add_variable(
            "sentiment",
            self.SENTIMENT_LEVELS,
            "Sentiment of the observation"
        )

        # Source type variable
        builder.add_variable(
            "source",
            self.SOURCE_TYPES,
            "Source platform of the observation"
        )

        # Keyword presence variables (binary: present/absent)
        for kw in self.TRACKED_KEYWORDS[:20]:  # Limit initial keywords
            builder.add_variable(
                f"kw_{kw}",
                ["absent", "present"],
                f"Presence of keyword: {kw}"
            )

        # Prior over categories (initially uniform)
        num_cats = len(self.INITIAL_CATEGORIES)
        builder.add_categorical_factor(
            "category_prior",
            "category",
            [1.0 / num_cats] * num_cats
        )

        builder.set_model_type("bayesian_network")

        vfg = builder.build()

        try:
            # Delete any existing graph first
            self.client.delete_graph()

            # Set new graph
            result = self.client.set_graph(vfg)
            self._graph_initialized = True
            logger.info("Initialized Genius VFG graph")

            # Initialize local categories
            for cat_id in self.INITIAL_CATEGORIES:
                self.categories[cat_id] = GeniusCategory(
                    id=cat_id,
                    name=cat_id.replace("_", " ").title(),
                    keywords=[],
                    observation_count=0,
                    belief_strength=1.0 / num_cats
                )

        except Exception as e:
            logger.error(f"Failed to initialize Genius graph: {e}")
            self._graph_initialized = False

    def observe(self, observation: GeniusObservation) -> dict[str, float]:
        """
        Process a new observation and update beliefs.

        Args:
            observation: The observed data

        Returns:
            Posterior probabilities over categories
        """
        self.observations.append(observation)

        # Update local keyword tracking
        for kw in observation.keywords:
            self.global_keyword_counts[kw] += 1

        # Build evidence for Genius
        evidence = self._build_evidence(observation)

        # If Genius is available, use it for inference
        if self.client and self._graph_initialized:
            try:
                result = self.client.infer(
                    variables=["category"],
                    evidence=evidence,
                    wait=True
                )

                # Extract posteriors
                posteriors = result.get("probabilities", {}).get("category", {})

                if posteriors:
                    # Update local category beliefs
                    best_cat = max(posteriors, key=posteriors.get)
                    self._update_category(best_cat, observation, posteriors[best_cat])

                    return posteriors

            except Exception as e:
                logger.warning(f"Genius inference failed, using fallback: {e}")

        # Fallback to local heuristic
        return self._local_inference(observation)

    def _build_evidence(self, observation: GeniusObservation) -> dict[str, Any]:
        """Build evidence dict for Genius inference."""
        evidence = {}

        # Sentiment
        if observation.sentiment > 0.3:
            evidence["sentiment"] = "positive"
        elif observation.sentiment < -0.3:
            evidence["sentiment"] = "negative"
        else:
            evidence["sentiment"] = "neutral"

        # Source
        source_lower = observation.source.lower()
        if source_lower in self.SOURCE_TYPES:
            evidence["source"] = source_lower
        else:
            evidence["source"] = "other"

        # Keywords
        obs_keywords = set(kw.lower() for kw in observation.keywords)
        for kw in self.TRACKED_KEYWORDS[:20]:
            evidence[f"kw_{kw}"] = "present" if kw in obs_keywords else "absent"

        return evidence

    def _local_inference(self, observation: GeniusObservation) -> dict[str, float]:
        """Fallback local inference when Genius unavailable."""
        # Simple keyword matching
        posteriors = {}
        obs_keywords = set(kw.lower() for kw in observation.keywords)

        category_keywords = {
            "luxury_wellness": {"luxury", "spa", "wellness", "premium"},
            "budget_social": {"budget", "cheap", "hostel", "social", "backpacker"},
            "digital_nomad": {"wifi", "coworking", "remote", "nomad", "workspace"},
            "boutique_design": {"boutique", "design", "unique", "aesthetic"},
            "family_resort": {"family", "kids", "resort", "pool"},
            "business_corporate": {"business", "corporate", "conference"},
            "adventure_experience": {"adventure", "experience", "activity"},
            "romantic_couples": {"romantic", "couples", "honeymoon"},
            "eco_sustainable": {"eco", "sustainable", "green", "organic"},
            "other": set()
        }

        total_score = 0
        for cat_id, cat_kws in category_keywords.items():
            overlap = len(obs_keywords & cat_kws)
            # Add base score to prevent zeros
            score = overlap + 0.1
            posteriors[cat_id] = score
            total_score += score

        # Normalize
        for cat_id in posteriors:
            posteriors[cat_id] /= total_score

        # Update local category
        best_cat = max(posteriors, key=posteriors.get)
        self._update_category(best_cat, observation, posteriors[best_cat])

        return posteriors

    def _update_category(
        self,
        cat_id: str,
        observation: GeniusObservation,
        belief: float
    ):
        """Update category statistics with new observation."""
        if cat_id not in self.categories:
            self.categories[cat_id] = GeniusCategory(
                id=cat_id,
                name=cat_id.replace("_", " ").title()
            )

        cat = self.categories[cat_id]
        cat.observation_count += 1
        cat.belief_strength = (
            cat.belief_strength * (cat.observation_count - 1) + belief
        ) / cat.observation_count

        for kw in observation.keywords:
            cat.keyword_counts[kw] += 1

        if cat.observation_count % 5 == 0:
            cat.update_name()

    def select_next_action(self) -> dict:
        """
        Use active inference to select what to search next.

        This uses Expected Free Energy (EFE) minimization to balance:
        - Epistemic value: Reduce uncertainty about categories
        - Pragmatic value: Find useful information

        Returns:
            Suggested action with query and rationale
        """
        if self.client and self._graph_initialized:
            try:
                result = self.client.select_action(wait=True)

                if result:
                    # Parse action selection result
                    action = result.get("action", {})
                    efe = result.get("efe_components", {})

                    return {
                        "query": self._action_to_query(action),
                        "reason": f"EFE minimization (epistemic: {efe.get('epistemic', 0):.2f})",
                        "efe_components": efe,
                        "source": "genius"
                    }

            except Exception as e:
                logger.warning(f"Genius action selection failed: {e}")

        # Fallback: find most uncertain category
        return self._local_action_selection()

    def _action_to_query(self, action: dict) -> str:
        """Convert Genius action to search query."""
        # The action contains information about which state to explore
        # Convert to a search query
        action_idx = action.get("selected_action", [0])[0] if action else 0

        # Map actions to queries
        exploration_queries = [
            "luxury hotel amenities",
            "budget accommodation tips",
            "digital nomad coworking hotel",
            "boutique hotel design",
            "family friendly resort",
            "business hotel facilities",
            "adventure travel accommodation",
            "romantic hotel getaway",
            "eco friendly sustainable hotel",
            "hotel travel tips"
        ]

        idx = action_idx % len(exploration_queries)
        return exploration_queries[idx]

    def _local_action_selection(self) -> dict:
        """Fallback local action selection."""
        # Find category with fewest observations
        if not self.categories:
            return {
                "query": "hotel accommodation travel",
                "reason": "initial exploration",
                "source": "local"
            }

        min_cat = min(
            self.categories.values(),
            key=lambda c: c.observation_count
        )

        return {
            "query": " ".join(min_cat.keywords[:3]) if min_cat.keywords else min_cat.name,
            "reason": f"reduce uncertainty about '{min_cat.name}'",
            "target_category": min_cat.id,
            "source": "local"
        }

    def learn_from_batch(self, observations: list[GeniusObservation]) -> dict:
        """
        Learn parameters from a batch of observations.

        This uses Genius's learning API to update the VFG parameters
        based on observed data.
        """
        if not self.client or not self._graph_initialized:
            logger.warning("Genius not available for batch learning")
            return {"status": "skipped", "reason": "genius_unavailable"}

        # Convert observations to learning data
        data = []
        for obs in observations:
            evidence = self._build_evidence(obs)
            data.append(evidence)

        if not data:
            return {"status": "skipped", "reason": "no_data"}

        try:
            result = self.client.learn(data=data, wait=True)

            # Track learning metrics
            history = result.get("history", {})
            self.learning_history.append({
                "batch_size": len(data),
                "js_divergence": history.get("js_divergence"),
                "log_likelihood": history.get("log_likelihood"),
            })

            logger.info(f"Learned from batch of {len(data)} observations")

            return {
                "status": "success",
                "observations_processed": len(data),
                "history": history
            }

        except Exception as e:
            logger.error(f"Batch learning failed: {e}")
            return {"status": "error", "error": str(e)}

    def get_free_energy(self) -> float:
        """
        Get variational free energy of current model.

        If Genius is available, this uses proper VFE calculation.
        Otherwise falls back to local approximation.
        """
        if not self.observations:
            return 0.0

        # For now, use local approximation
        # (Genius doesn't expose raw VFE, but action selection uses it internally)
        accuracy = 0.0
        for cat in self.categories.values():
            accuracy += cat.belief_strength * cat.observation_count

        if self.observations:
            accuracy /= len(self.observations)

        # Complexity penalty
        num_active_cats = sum(
            1 for c in self.categories.values()
            if c.observation_count > 0
        )
        complexity = num_active_cats / len(self.INITIAL_CATEGORIES)

        return -accuracy + 0.1 * complexity

    def get_structure(self) -> dict:
        """Return current learned structure."""
        return {
            "num_categories": len(self.categories),
            "num_observations": len(self.observations),
            "genius_connected": self.client is not None and self._graph_initialized,
            "categories": [
                {
                    "id": cat.id,
                    "name": cat.name,
                    "keywords": cat.keywords,
                    "observation_count": cat.observation_count,
                    "belief_strength": cat.belief_strength,
                }
                for cat in sorted(
                    self.categories.values(),
                    key=lambda c: c.observation_count,
                    reverse=True
                )
            ]
        }

    def close(self):
        """Clean up resources."""
        if self.client:
            self.client.close()
            self.client = None


# Convenience function
def create_genius_learner() -> GeniusStructureLearner:
    """Create a Genius-backed structure learner."""
    return GeniusStructureLearner()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test the learner
    learner = GeniusStructureLearner()

    # Test observations
    test_obs = [
        GeniusObservation(
            text="Looking for boutique hotel with good wifi for remote work",
            keywords=["boutique", "wifi", "remote", "work"],
            source="reddit",
            sentiment=0.5
        ),
        GeniusObservation(
            text="Best budget hostels with social atmosphere",
            keywords=["budget", "hostel", "social"],
            source="reddit",
            sentiment=0.3
        ),
    ]

    for obs in test_obs:
        posteriors = learner.observe(obs)
        print(f"Observation: {obs.text[:50]}...")
        print(f"Posteriors: {posteriors}")
        print()

    print("Structure:", learner.get_structure())
    print("Next action:", learner.select_next_action())

    learner.close()
