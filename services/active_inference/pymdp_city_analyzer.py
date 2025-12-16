"""
PyMDP-Powered Adaptive City Analyzer.

Uses the JAX implementation of pymdp for active inference with:
- Expected Free Energy (EFE) minimization for query selection
- Variational inference for belief updates over categories
- Online learning of observation likelihoods

This provides a fully local active inference solution without
requiring external APIs.
"""

import logging
import time
from datetime import datetime
from typing import Optional

import httpx
import numpy as np

from .pymdp_learner import (
    PyMDPStructureLearner,
    PyMDPObservation,
    PYMDP_AVAILABLE
)

logger = logging.getLogger(__name__)


class PyMDPCityAnalyzer:
    """
    City desire analyzer powered by pymdp active inference.

    Uses a POMDP formulation where:
    - Hidden states: True category of traveler desires
    - Observations: Content features (keywords, source, sentiment)
    - Actions: Search queries to execute
    - Rewards: Information gain about categories

    The agent balances exploration (learning about categories)
    with exploitation (finding relevant content).
    """

    def __init__(
        self,
        max_iterations: int = 10,
        confidence_threshold: float = 0.7,
        use_learning: bool = True,
        policy_len: int = 1,
    ):
        """
        Initialize the pymdp city analyzer.

        Args:
            max_iterations: Maximum active inference iterations
            confidence_threshold: Confidence level to stop exploration
            use_learning: Whether to update model parameters online
            policy_len: Planning horizon for action selection
        """
        self.max_iterations = max_iterations
        self.confidence_threshold = confidence_threshold

        # Initialize pymdp structure learner
        self.structure_learner = PyMDPStructureLearner(
            use_learning=use_learning,
            policy_len=policy_len,
        )

        # HTTP client for scraping
        self.client = httpx.Client(
            timeout=30,
            follow_redirects=True,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            },
        )

        # Track searches
        self.search_history: list[dict] = []

        # Embedding provider (optional, for enhanced keyword extraction)
        self._embedding_provider = None
        self._init_embedding_provider()

    def _init_embedding_provider(self):
        """Initialize embedding provider for semantic analysis."""
        try:
            from data_models.embeddings import get_default_provider
            self._embedding_provider = get_default_provider()
            logger.info(f"Initialized embedding provider: {type(self._embedding_provider).__name__}")
        except Exception as e:
            logger.debug(f"Embedding provider not available: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """Clean up resources."""
        self.client.close()

    def analyze_city(self, city: str, country: str = "") -> dict:
        """
        Analyze a city using pymdp-powered active inference.

        The analysis proceeds in phases:
        1. Initial broad exploration with predefined queries
        2. Active inference loop (query selection via EFE)
        3. Profile generation from learned beliefs

        Args:
            city: City name
            country: Country (optional)

        Returns:
            City desire profile with learned categories
        """
        logger.info(f"Starting PyMDP analysis for {city}, {country}")
        logger.info(f"PyMDP available: {PYMDP_AVAILABLE}")

        location = f"{city} {country}".strip()

        # Phase 1: Initial exploration
        initial_queries = [
            f"hotel {city} recommendation",
            f"where to stay {city}",
            f"{city} accommodation review",
            f"best area stay {city}",
        ]

        for query in initial_queries:
            self._search_and_observe(query, location)
            time.sleep(1)

        # Phase 2: Active inference loop
        iteration = 0
        while iteration < self.max_iterations:
            iteration += 1

            # Compute confidence
            confidence = self._compute_confidence()
            logger.info(f"Iteration {iteration}, confidence: {confidence:.2f}")

            if confidence >= self.confidence_threshold:
                logger.info("Confidence threshold reached")
                break

            # Use pymdp to select next action
            action = self.structure_learner.select_action()
            query = f"{action['query']} {city}"

            logger.info(f"PyMDP action: {query} (source: {action.get('source', 'unknown')})")

            # Execute and observe
            new_obs = self._search_and_observe(query, location)

            if new_obs == 0:
                # Try alternative query
                alt_query = f"{city} travel accommodation tips"
                self._search_and_observe(alt_query, location)

            time.sleep(1)

        # Phase 3: Build profile
        return self._build_profile(city, country)

    def _search_and_observe(self, query: str, location: str) -> int:
        """Execute search and create observations."""
        observations_added = 0

        self.search_history.append({
            "query": query,
            "timestamp": datetime.utcnow().isoformat(),
        })

        # Search Reddit
        observations_added += self._search_reddit(query, location)

        # Search YouTube
        observations_added += self._search_youtube(query, location)

        return observations_added

    def _search_reddit(self, query: str, location: str) -> int:
        """Search Reddit and create observations."""
        count = 0
        subreddits = ["travel", "solotravel", "hotels", "digitalnomad"]

        for subreddit in subreddits[:2]:
            try:
                url = f"https://old.reddit.com/r/{subreddit}/search.json"
                params = {
                    "q": query,
                    "restrict_sr": "on",
                    "sort": "relevance",
                    "t": "year",
                    "limit": 10,
                }

                response = self.client.get(url, params=params)
                if response.status_code != 200:
                    continue

                data = response.json()
                posts = data.get("data", {}).get("children", [])

                for post in posts:
                    post_data = post.get("data", {})
                    title = post_data.get("title", "")
                    selftext = post_data.get("selftext", "")

                    if not title:
                        continue

                    full_text = f"{title} {selftext}".lower()
                    location_lower = location.lower().split()[0]

                    if location_lower not in full_text:
                        continue

                    # Create observation
                    text = f"{title}\n{selftext[:500]}"
                    obs = PyMDPObservation(
                        text=text,
                        keywords=self._extract_keywords(text),
                        source="reddit",
                        sentiment=self._simple_sentiment(text),
                    )

                    # Feed to learner
                    self.structure_learner.observe(obs)
                    count += 1

                time.sleep(1)

            except Exception as e:
                logger.debug(f"Reddit search error: {e}")
                continue

        return count

    def _search_youtube(self, query: str, location: str) -> int:
        """Search YouTube and create observations."""
        count = 0

        try:
            search_url = f"https://www.youtube.com/results?search_query={query.replace(' ', '+')}"
            response = self.client.get(search_url)

            if response.status_code != 200:
                return 0

            html = response.text

            import re
            import json

            match = re.search(r'var ytInitialData = ({.*?});', html)
            if not match:
                return 0

            try:
                data = json.loads(match.group(1))
                contents = data.get("contents", {}).get(
                    "twoColumnSearchResultsRenderer", {}
                ).get("primaryContents", {}).get(
                    "sectionListRenderer", {}
                ).get("contents", [])

                for section in contents:
                    items = section.get("itemSectionRenderer", {}).get("contents", [])
                    for item in items[:5]:
                        video = item.get("videoRenderer", {})
                        if not video:
                            continue

                        title = video.get("title", {}).get("runs", [{}])[0].get("text", "")
                        if not title:
                            continue

                        location_lower = location.lower().split()[0]
                        if location_lower not in title.lower():
                            continue

                        obs = PyMDPObservation(
                            text=title,
                            keywords=self._extract_keywords(title),
                            source="youtube",
                            sentiment=0.0,
                        )

                        self.structure_learner.observe(obs)
                        count += 1

            except json.JSONDecodeError:
                pass

        except Exception as e:
            logger.debug(f"YouTube search error: {e}")

        return count

    def _extract_keywords(self, text: str) -> list[str]:
        """Extract keywords from text."""
        text_lower = text.lower()

        terms = [
            "hotel", "hostel", "airbnb", "accommodation", "stay", "room",
            "boutique", "luxury", "budget", "affordable", "cheap",
            "central", "walkable", "quiet", "safe", "clean",
            "wifi", "workspace", "coworking", "remote", "digital nomad",
            "pool", "gym", "spa", "breakfast", "rooftop",
            "romantic", "family", "solo", "backpacker",
            "modern", "historic", "design", "view", "beach", "mountain",
            "nightlife", "restaurant", "local", "authentic",
            "frustrating", "disappointing", "amazing", "perfect", "worst",
            "eco", "sustainable", "green", "adventure", "experience",
            "business", "corporate", "conference", "resort",
        ]

        found = [term for term in terms if term in text_lower]
        return found[:15]

    def _simple_sentiment(self, text: str) -> float:
        """Simple sentiment analysis."""
        text_lower = text.lower()

        positive = ["love", "amazing", "perfect", "great", "best", "beautiful", "recommend", "fantastic"]
        negative = ["hate", "awful", "terrible", "worst", "disappointing", "frustrating", "avoid", "bad"]

        pos_count = sum(1 for word in positive if word in text_lower)
        neg_count = sum(1 for word in negative if word in text_lower)

        if pos_count + neg_count == 0:
            return 0.0

        return (pos_count - neg_count) / (pos_count + neg_count)

    def _compute_confidence(self) -> float:
        """Compute confidence in current understanding."""
        structure = self.structure_learner.get_structure()

        if structure["num_observations"] < 5:
            return 0.0

        # More observations = more confidence
        obs_confidence = 1 - np.exp(-structure["num_observations"] / 30)

        # Higher belief strengths = more confidence
        active_cats = [c for c in structure["categories"] if c["observation_count"] > 0]
        if active_cats:
            avg_belief = np.mean([c["belief_strength"] for c in active_cats])
            belief_confidence = avg_belief
        else:
            belief_confidence = 0.0

        # Lower free energy = better model
        free_energy = self.structure_learner.get_free_energy()
        fe_confidence = 1 / (1 + np.exp(free_energy))

        confidence = 0.4 * obs_confidence + 0.3 * belief_confidence + 0.3 * fe_confidence

        return confidence

    def _build_profile(self, city: str, country: str) -> dict:
        """Build final city desire profile."""
        structure = self.structure_learner.get_structure()

        # Convert categories to themes
        themes = []
        for cat in structure["categories"]:
            if cat["observation_count"] == 0:
                continue

            theme = {
                "theme_name": cat["name"],
                "description": f"Travelers discussing {', '.join(cat['keywords'][:3]) if cat['keywords'] else cat['name']}",
                "intensity_score": min(cat["observation_count"] / 20, 1.0),
                "belief_strength": cat["belief_strength"],
                "frequency": cat["observation_count"],
                "keywords": cat["keywords"],
                "category_id": cat["id"],
                "is_learned": True,
            }
            themes.append(theme)

        themes.sort(key=lambda t: t["frequency"], reverse=True)

        # Identify opportunities
        opportunities = []
        for theme in themes:
            if theme["intensity_score"] > 0.3 and theme["frequency"] < 10:
                opportunities.append(f"Emerging interest: {theme['theme_name']}")

        return {
            "city": city,
            "country": country,
            "total_signals": structure["num_observations"],
            "num_learned_categories": len([c for c in structure["categories"] if c["observation_count"] > 0]),
            "pymdp_available": structure["pymdp_available"],
            "agent_initialized": structure["agent_initialized"],
            "top_desires": themes[:10],
            "white_space_opportunities": opportunities,
            "model_confidence": self._compute_confidence(),
            "free_energy": self.structure_learner.get_free_energy(),
            "search_history": self.search_history,
            "generated_at": datetime.utcnow().isoformat(),
            "method": "pymdp_active_inference",
        }


def analyze_city_pymdp(city: str, country: str = "") -> dict:
    """Convenience function for PyMDP-powered city analysis."""
    with PyMDPCityAnalyzer() as analyzer:
        return analyzer.analyze_city(city, country)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print(f"PyMDP available: {PYMDP_AVAILABLE}")

    result = analyze_city_pymdp("Lisbon", "Portugal")

    import json
    print(json.dumps(result, indent=2, default=str))
