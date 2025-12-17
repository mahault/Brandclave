"""
Structure Learning with Active Inference for Category Discovery.

Instead of fixed categories, we learn them from data and adapt the structure
when new observations don't fit well.

Key concepts:
- Categories are latent clusters that explain observed traveler desires
- We maintain uncertainty over both:
  1. Which category each observation belongs to (assignment)
  2. What categories exist (structure)
- When observations don't fit existing categories well, we expand
- When categories become redundant, we merge

This uses a Dirichlet Process-like approach where:
- Categories can be created as needed
- The model prefers simpler explanations (fewer categories)
- But will add complexity when data demands it

Fit computation:
- Primary: Cosine similarity of embeddings (from Mistral or local model)
- Secondary: Keyword/semantic overlap
- Threshold-based decision for category assignment
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Callable
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class Category:
    """A learned category of traveler desires."""

    id: str
    name: str  # Human-readable name (generated from keywords)
    keywords: list[str] = field(default_factory=list)
    centroid: Optional[np.ndarray] = None  # Embedding centroid
    observation_count: int = 0
    total_fit_score: float = 0.0  # How well observations fit this category

    # Sufficient statistics for online updates
    keyword_counts: dict = field(default_factory=lambda: defaultdict(int))

    @property
    def avg_fit(self) -> float:
        if self.observation_count == 0:
            return 0.0
        return self.total_fit_score / self.observation_count

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
class Observation:
    """An observed signal from scraped content."""

    text: str
    embedding: np.ndarray
    keywords: list[str]
    source: str
    sentiment: float = 0.0
    engagement: float = 0.0


class StructureLearner:
    """
    Learns category structure from observations using active inference principles.

    The model maintains:
    - A set of learned categories (can grow/shrink)
    - Assignment probabilities for each observation
    - Uncertainty about the structure itself

    Key parameters:
    - alpha: Concentration parameter (higher = more categories)
    - fit_threshold: How well must an observation fit to not trigger expansion
    - merge_threshold: How similar must categories be to merge
    - embedding_fn: Function to compute embeddings (uses NLP pipeline)

    Fit Thresholds (cosine similarity):
    - > 0.7: Strong fit - definitely belongs to this category
    - 0.5 - 0.7: Moderate fit - probably belongs, update category
    - 0.3 - 0.5: Weak fit - might belong, or might need new category
    - < 0.3: Poor fit - likely needs a new category
    """

    def __init__(
        self,
        alpha: float = 1.0,  # Dirichlet Process concentration
        fit_threshold: float = 0.5,  # Min cosine similarity to existing category
        merge_threshold: float = 0.80,  # Cosine similarity to trigger merge
        min_observations_to_merge: int = 10,
        embedding_fn: Optional[Callable[[str], list[float]]] = None,
    ):
        self.alpha = alpha
        self.fit_threshold = fit_threshold
        self.merge_threshold = merge_threshold
        self.min_observations_to_merge = min_observations_to_merge

        # Set up embedding function - use real NLP pipeline if available
        if embedding_fn is not None:
            self.embedding_fn = embedding_fn
        else:
            self.embedding_fn = self._get_default_embedding_fn()

        self.categories: dict[str, Category] = {}
        self.observations: list[Observation] = []
        self.assignments: dict[int, dict[str, float]] = {}  # obs_idx -> {cat_id: prob}

        # Track fit statistics for analysis
        self.fit_history: list[dict] = []

        self._next_category_id = 0

    def _get_default_embedding_fn(self) -> Callable[[str], list[float]]:
        """Get the default embedding function from the NLP pipeline."""
        try:
            from data_models.embeddings import get_default_provider
            provider = get_default_provider()
            logger.info(f"Using embedding provider: {type(provider).__name__} (dim={provider.dimension})")
            return provider.embed
        except Exception as e:
            logger.warning(f"Could not load embedding provider: {e}. Using fallback.")
            return self._fallback_embedding

    def _fallback_embedding(self, text: str) -> list[float]:
        """Fallback embedding using simple TF-IDF-like approach."""
        # This is a deterministic fallback when no embedding model is available
        import hashlib

        # Create pseudo-embedding from character n-grams
        text_lower = text.lower()
        ngrams = [text_lower[i:i+3] for i in range(len(text_lower)-2)]

        # Hash each n-gram to a dimension
        dim = 384
        embedding = np.zeros(dim)

        for ngram in ngrams:
            h = int(hashlib.md5(ngram.encode()).hexdigest(), 16)
            idx = h % dim
            embedding[idx] += 1

        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding.tolist()

    def observe(self, observation: Observation) -> dict[str, float]:
        """
        Process a new observation and update structure if needed.

        Returns:
            Assignment probabilities over categories
        """
        obs_idx = len(self.observations)
        self.observations.append(observation)

        if not self.categories:
            # First observation - create initial category
            cat = self._create_category(observation)
            self.assignments[obs_idx] = {cat.id: 1.0}
            return self.assignments[obs_idx]

        # Compute fit to each existing category
        fits = self._compute_fits(observation)

        # Check if observation fits well enough
        max_fit = max(fits.values()) if fits else 0.0

        if max_fit < self.fit_threshold:
            # Observation doesn't fit well - consider creating new category
            # But first, check if it might indicate we should SPLIT an existing one

            if self._should_create_new_category(observation, fits):
                cat = self._create_category(observation)
                fits[cat.id] = 1.0
                logger.info(f"Created new category: {cat.name} (max_fit was {max_fit:.2f})")

        # Compute assignment probabilities (softmax over fits)
        assignments = self._compute_assignments(fits)
        self.assignments[obs_idx] = assignments

        # Update category statistics
        for cat_id, prob in assignments.items():
            if prob > 0.1:  # Only update if meaningful assignment
                self._update_category(cat_id, observation, prob)

        # Periodically check for merge opportunities
        if len(self.observations) % 50 == 0:
            self._check_for_merges()

        return assignments

    def _compute_fits(self, observation: Observation) -> dict[str, float]:
        """
        Compute how well observation fits each category.

        Uses a weighted combination of:
        1. Embedding cosine similarity (primary signal, 70% weight)
        2. Keyword overlap (secondary signal, 30% weight)

        Returns:
            Dict mapping category_id -> fit_score (0 to 1)

        Interpretation of fit scores:
        - 0.7+: Strong fit, high confidence assignment
        - 0.5-0.7: Moderate fit, reasonable assignment
        - 0.3-0.5: Weak fit, consider creating new category
        - <0.3: Poor fit, likely needs new category
        """
        fits = {}
        fit_details = {}  # For logging/debugging

        for cat_id, cat in self.categories.items():
            embedding_sim = 0.0
            keyword_sim = 0.0

            # Primary: Embedding cosine similarity
            if cat.centroid is not None and observation.embedding is not None:
                embedding_sim = self._cosine_similarity(
                    np.array(observation.embedding),
                    cat.centroid
                )
                # Cosine similarity is in [-1, 1], normalize to [0, 1]
                embedding_sim = (embedding_sim + 1) / 2

            # Secondary: Keyword Jaccard similarity
            if cat.keywords and observation.keywords:
                cat_kw = set(cat.keywords)
                obs_kw = set(observation.keywords)
                union = cat_kw | obs_kw
                if union:
                    keyword_sim = len(cat_kw & obs_kw) / len(union)

            # Weighted combination
            # Embedding similarity is more reliable for semantic matching
            fit = 0.7 * embedding_sim + 0.3 * keyword_sim

            fits[cat_id] = fit
            fit_details[cat_id] = {
                "embedding_sim": embedding_sim,
                "keyword_sim": keyword_sim,
                "combined": fit,
            }

        # Log fit details for debugging
        if fits:
            best_cat = max(fits, key=fits.get)
            best_fit = fits[best_cat]
            logger.debug(
                f"Fit scores - Best: {best_cat} ({best_fit:.3f}), "
                f"Details: emb={fit_details[best_cat]['embedding_sim']:.3f}, "
                f"kw={fit_details[best_cat]['keyword_sim']:.3f}"
            )

        # Track for analysis
        self.fit_history.append({
            "observation_idx": len(self.observations) - 1,
            "fits": fit_details,
            "max_fit": max(fits.values()) if fits else 0,
        })

        return fits

    def _compute_assignments(self, fits: dict[str, float]) -> dict[str, float]:
        """Convert fits to probability distribution (softmax)."""
        if not fits:
            return {}

        # Temperature-scaled softmax
        temperature = 0.5
        values = np.array(list(fits.values()))
        exp_values = np.exp(values / temperature)
        probs = exp_values / exp_values.sum()

        return {cat_id: float(prob) for cat_id, prob in zip(fits.keys(), probs)}

    def _should_create_new_category(
        self,
        observation: Observation,
        fits: dict[str, float]
    ) -> bool:
        """
        Decide whether to create a new category.

        Uses a CRP-like probability:
        P(new) ∝ alpha / (n + alpha)

        But also considers:
        - How poorly the observation fits existing categories
        - Whether we have enough observations to justify expansion
        """
        n = len(self.observations)

        # Base probability from CRP
        p_new_base = self.alpha / (n + self.alpha)

        # Adjust based on fit quality
        max_fit = max(fits.values()) if fits else 0.0
        fit_factor = 1.0 - max_fit  # Higher when fit is poor

        p_new = p_new_base * (1 + fit_factor)

        # Stochastic decision (or deterministic if fit is very poor)
        if max_fit < 0.1:
            return True  # Definitely doesn't fit

        return np.random.random() < p_new

    def _create_category(self, observation: Observation) -> Category:
        """Create a new category from an observation."""
        cat_id = f"cat_{self._next_category_id}"
        self._next_category_id += 1

        cat = Category(
            id=cat_id,
            name="New Category",
            keywords=observation.keywords[:5],
            centroid=observation.embedding.copy() if observation.embedding is not None else None,
            observation_count=1,
            total_fit_score=1.0,
        )

        for kw in observation.keywords:
            cat.keyword_counts[kw] += 1

        cat.update_name()
        self.categories[cat_id] = cat

        return cat

    def _update_category(
        self,
        cat_id: str,
        observation: Observation,
        weight: float
    ):
        """Update category statistics with new observation."""
        cat = self.categories[cat_id]

        # Update centroid (online mean)
        if observation.embedding is not None:
            if cat.centroid is None:
                cat.centroid = observation.embedding.copy()
            else:
                # Weighted running average
                n = cat.observation_count
                cat.centroid = (n * cat.centroid + weight * observation.embedding) / (n + weight)

        # Update keyword counts
        for kw in observation.keywords:
            cat.keyword_counts[kw] += weight

        # Update counts
        cat.observation_count += weight
        cat.total_fit_score += weight * self._compute_fits(observation).get(cat_id, 0)

        # Periodically refresh name
        if int(cat.observation_count) % 10 == 0:
            cat.update_name()

    def _check_for_merges(self):
        """Check if any categories should be merged."""
        if len(self.categories) < 2:
            return

        cat_ids = list(self.categories.keys())
        merged = set()

        for i, cat1_id in enumerate(cat_ids):
            if cat1_id in merged:
                continue

            cat1 = self.categories[cat1_id]

            if cat1.observation_count < self.min_observations_to_merge:
                continue

            for cat2_id in cat_ids[i+1:]:
                if cat2_id in merged:
                    continue

                cat2 = self.categories[cat2_id]

                if cat2.observation_count < self.min_observations_to_merge:
                    continue

                similarity = self._category_similarity(cat1, cat2)

                if similarity > self.merge_threshold:
                    self._merge_categories(cat1_id, cat2_id)
                    merged.add(cat2_id)
                    logger.info(f"Merged categories: {cat1.name} + {cat2.name}")

    def _category_similarity(self, cat1: Category, cat2: Category) -> float:
        """Compute similarity between two categories."""
        sim = 0.0

        # Centroid similarity
        if cat1.centroid is not None and cat2.centroid is not None:
            sim += 0.6 * self._cosine_similarity(cat1.centroid, cat2.centroid)

        # Keyword overlap
        kw1 = set(cat1.keywords)
        kw2 = set(cat2.keywords)
        if kw1 and kw2:
            jaccard = len(kw1 & kw2) / len(kw1 | kw2)
            sim += 0.4 * jaccard

        return sim

    def _merge_categories(self, keep_id: str, remove_id: str):
        """Merge remove_id into keep_id."""
        keep = self.categories[keep_id]
        remove = self.categories[remove_id]

        # Merge centroids (weighted average)
        total = keep.observation_count + remove.observation_count
        if keep.centroid is not None and remove.centroid is not None:
            keep.centroid = (
                keep.observation_count * keep.centroid +
                remove.observation_count * remove.centroid
            ) / total

        # Merge keyword counts
        for kw, count in remove.keyword_counts.items():
            keep.keyword_counts[kw] += count

        # Update counts
        keep.observation_count = total
        keep.total_fit_score += remove.total_fit_score

        # Update name
        keep.update_name()

        # Update assignments
        for obs_idx, assignments in self.assignments.items():
            if remove_id in assignments:
                prob = assignments.pop(remove_id)
                assignments[keep_id] = assignments.get(keep_id, 0) + prob

        # Remove merged category
        del self.categories[remove_id]

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def get_structure(self) -> dict:
        """Return current learned structure."""
        return {
            "num_categories": len(self.categories),
            "num_observations": len(self.observations),
            "categories": [
                {
                    "id": cat.id,
                    "name": cat.name,
                    "keywords": cat.keywords,
                    "observation_count": cat.observation_count,
                    "avg_fit": cat.avg_fit,
                }
                for cat in sorted(
                    self.categories.values(),
                    key=lambda c: c.observation_count,
                    reverse=True
                )
            ]
        }

    def get_fit_statistics(self) -> dict:
        """Return statistics about fit scores for analysis."""
        if not self.fit_history:
            return {"message": "No fit history yet"}

        max_fits = [h["max_fit"] for h in self.fit_history if h["max_fit"] > 0]

        if not max_fits:
            return {"message": "No valid fits recorded"}

        return {
            "total_observations": len(self.fit_history),
            "avg_max_fit": float(np.mean(max_fits)),
            "min_max_fit": float(np.min(max_fits)),
            "max_max_fit": float(np.max(max_fits)),
            "std_max_fit": float(np.std(max_fits)),
            "fits_above_threshold": sum(1 for f in max_fits if f >= self.fit_threshold),
            "fits_below_threshold": sum(1 for f in max_fits if f < self.fit_threshold),
            "fit_threshold": self.fit_threshold,
            "interpretation": {
                "strong_fits": sum(1 for f in max_fits if f >= 0.7),
                "moderate_fits": sum(1 for f in max_fits if 0.5 <= f < 0.7),
                "weak_fits": sum(1 for f in max_fits if 0.3 <= f < 0.5),
                "poor_fits": sum(1 for f in max_fits if f < 0.3),
            }
        }

    def get_free_energy(self) -> float:
        """
        Compute variational free energy of current model.

        F = -E[log P(observations | structure)] + KL(Q(assignments) || P(assignments))

        Lower is better. This measures how well our structure explains the data
        while penalizing complexity.
        """
        if not self.observations:
            return 0.0

        # Accuracy term: How well do observations fit their assigned categories?
        accuracy = 0.0
        for obs_idx, assignments in self.assignments.items():
            obs = self.observations[obs_idx]
            fits = self._compute_fits(obs)
            expected_fit = sum(
                prob * fits.get(cat_id, 0)
                for cat_id, prob in assignments.items()
            )
            accuracy += expected_fit

        accuracy /= len(self.observations)

        # Complexity penalty: More categories = higher complexity
        # Using CRP prior
        n = len(self.observations)
        expected_categories = self.alpha * np.log(1 + n / self.alpha)
        complexity = len(self.categories) / max(expected_categories, 1)

        # Free energy (lower is better)
        free_energy = -accuracy + 0.1 * complexity

        return free_energy

    def suggest_next_query(self) -> dict:
        """
        Suggest what to search for next to reduce uncertainty.

        This is the "active" part of active inference - we don't just
        passively observe, we choose what to observe to maximize
        information gain.
        """
        if not self.categories:
            return {"query": "hotel travel accommodation", "reason": "initial exploration"}

        # Find categories with high uncertainty (few observations, low fit)
        uncertain_cats = [
            cat for cat in self.categories.values()
            if cat.observation_count < 20 or cat.avg_fit < 0.5
        ]

        if uncertain_cats:
            # Explore uncertain category
            cat = min(uncertain_cats, key=lambda c: c.observation_count)
            return {
                "query": " ".join(cat.keywords[:3]),
                "reason": f"reduce uncertainty about '{cat.name}'",
                "target_category": cat.id,
            }

        # All categories are well-established - look for potential new ones
        # by searching for things NOT covered by current keywords
        all_keywords = set()
        for cat in self.categories.values():
            all_keywords.update(cat.keywords)

        exploration_queries = [
            "unique hotel experience",
            "travel frustration",
            "accommodation problem",
            "hotel wish list",
            "travel disappointment",
        ]

        return {
            "query": np.random.choice(exploration_queries),
            "reason": "explore for potential new categories",
            "target_category": None,
        }


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    learner = StructureLearner(alpha=1.0, fit_threshold=0.3)

    # Simulate some observations
    test_observations = [
        Observation(
            text="Looking for a boutique hotel with good wifi for remote work",
            embedding=np.random.randn(384),
            keywords=["boutique", "wifi", "remote", "work", "hotel"],
            source="reddit",
        ),
        Observation(
            text="Best luxury spa resorts for honeymoon",
            embedding=np.random.randn(384),
            keywords=["luxury", "spa", "resort", "honeymoon", "romantic"],
            source="reddit",
        ),
        Observation(
            text="Budget hostels with good atmosphere in Barcelona",
            embedding=np.random.randn(384),
            keywords=["budget", "hostel", "atmosphere", "barcelona", "social"],
            source="reddit",
        ),
    ]

    for obs in test_observations:
        assignments = learner.observe(obs)
        print(f"Observed: {obs.text[:50]}...")
        print(f"Assignments: {assignments}")
        print()

    print("Learned structure:")
    print(learner.get_structure())

    print(f"\nFree energy: {learner.get_free_energy():.3f}")
    print(f"Next query suggestion: {learner.suggest_next_query()}")
