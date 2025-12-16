"""
Genius-Powered Adaptive City Analyzer.

Uses VERSES Genius Active Inference API for:
- Bayesian category inference
- POMDP-based action selection (what to search next)
- Online learning from scraped observations
- Expected Free Energy minimization for adaptive exploration

This is the production version that uses proper active inference
instead of our local approximations.
"""

import logging
import time
from datetime import datetime
from typing import Optional

import httpx
import numpy as np

from .genius_structure_learner import (
    GeniusStructureLearner,
    GeniusObservation,
    GeniusConfig
)

logger = logging.getLogger(__name__)


class GeniusCityAnalyzer:
    """
    City desire analyzer powered by VERSES Genius Active Inference.

    Key features:
    - Categories are learned from data using Bayesian inference
    - Action selection uses Expected Free Energy minimization
    - Structure adapts as more data is observed
    - Proper uncertainty quantification via posteriors
    """

    def __init__(
        self,
        max_iterations: int = 10,
        confidence_threshold: float = 0.7,
        genius_config: Optional[GeniusConfig] = None,
    ):
        self.max_iterations = max_iterations
        self.confidence_threshold = confidence_threshold

        # Initialize Genius-backed structure learner
        self.structure_learner = GeniusStructureLearner(
            config=genius_config,
            auto_connect=True
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

        # Track observations for batch learning
        self._observation_batch: list[GeniusObservation] = []

        # Embedding provider (optional)
        self._embedding_provider = None
        self._init_embedding_provider()

    def _init_embedding_provider(self):
        """Initialize embedding provider for semantic analysis."""
        try:
            from data_models.embeddings import get_default_provider
            self._embedding_provider = get_default_provider()
            logger.info(f"Initialized embedding provider: {type(self._embedding_provider).__name__}")
        except Exception as e:
            logger.warning(f"Could not initialize embedding provider: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """Clean up resources."""
        self.client.close()
        self.structure_learner.close()

    def analyze_city(self, city: str, country: str = "") -> dict:
        """
        Analyze a city using Genius-powered active inference.

        The analysis proceeds in phases:
        1. Initial broad exploration
        2. Active inference loop (query selection via EFE minimization)
        3. Batch learning from observations
        4. Profile generation

        Args:
            city: City name
            country: Country (optional)

        Returns:
            City desire profile with learned categories
        """
        logger.info(f"Starting Genius-powered analysis for {city}, {country}")

        location = f"{city} {country}".strip()

        # Check Genius connection status
        structure = self.structure_learner.get_structure()
        logger.info(f"Genius connected: {structure.get('genius_connected', False)}")

        # Phase 1: Initial exploration with broad queries
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

            # Compute confidence in current understanding
            confidence = self._compute_confidence()
            logger.info(f"Iteration {iteration}, confidence: {confidence:.2f}")

            if confidence >= self.confidence_threshold:
                logger.info("Confidence threshold reached")
                break

            # Use Genius to select next action
            action = self.structure_learner.select_next_action()
            query = f"{action['query']} {city}"

            logger.info(f"Genius action: {query} (source: {action.get('source', 'unknown')})")

            # Execute and observe
            new_obs = self._search_and_observe(query, location)

            if new_obs == 0:
                # No new observations - try alternative
                alt_query = f"{city} travel accommodation tips"
                self._search_and_observe(alt_query, location)

            time.sleep(1)

        # Phase 3: Batch learning (if we have accumulated observations)
        if self._observation_batch:
            logger.info(f"Running batch learning on {len(self._observation_batch)} observations")
            learn_result = self.structure_learner.learn_from_batch(self._observation_batch)
            logger.info(f"Learning result: {learn_result.get('status')}")

        # Phase 4: Build final profile
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

                    # Create Genius observation
                    text = f"{title}\n{selftext[:500]}"
                    obs = GeniusObservation(
                        text=text,
                        keywords=self._extract_keywords(text),
                        source="reddit",
                        sentiment=self._simple_sentiment(text),
                        embedding=self._get_embedding(text) if self._embedding_provider else None
                    )

                    # Feed to structure learner
                    self.structure_learner.observe(obs)
                    self._observation_batch.append(obs)
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

                        obs = GeniusObservation(
                            text=title,
                            keywords=self._extract_keywords(title),
                            source="youtube",
                            sentiment=0.0,
                            embedding=self._get_embedding(title) if self._embedding_provider else None
                        )

                        self.structure_learner.observe(obs)
                        self._observation_batch.append(obs)
                        count += 1

            except json.JSONDecodeError:
                pass

        except Exception as e:
            logger.debug(f"YouTube search error: {e}")

        return count

    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding for text."""
        if self._embedding_provider is None:
            return None

        try:
            truncated = text[:2000] if len(text) > 2000 else text
            embedding_list = self._embedding_provider.embed(truncated)
            embedding = np.array(embedding_list)

            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            return embedding
        except Exception as e:
            logger.debug(f"Embedding failed: {e}")
            return None

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
        ]

        found = [term for term in terms if term in text_lower]
        return found[:10]

    def _simple_sentiment(self, text: str) -> float:
        """Simple sentiment analysis."""
        text_lower = text.lower()

        positive = ["love", "amazing", "perfect", "great", "best", "beautiful", "recommend"]
        negative = ["hate", "awful", "terrible", "worst", "disappointing", "frustrating", "avoid"]

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

        # Higher average belief strength = more confidence
        if structure["categories"]:
            active_cats = [c for c in structure["categories"] if c["observation_count"] > 0]
            if active_cats:
                avg_belief = np.mean([c["belief_strength"] for c in active_cats])
                belief_confidence = avg_belief
            else:
                belief_confidence = 0.0
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

        # Convert learned categories to themes
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

        # Sort by frequency
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
            "genius_connected": structure.get("genius_connected", False),
            "top_desires": themes[:10],
            "white_space_opportunities": opportunities,
            "model_confidence": self._compute_confidence(),
            "free_energy": self.structure_learner.get_free_energy(),
            "search_history": self.search_history,
            "learning_history": self.structure_learner.learning_history,
            "generated_at": datetime.utcnow().isoformat(),
            "method": "genius_active_inference",
        }


def analyze_city_genius(city: str, country: str = "") -> dict:
    """Convenience function for Genius-powered city analysis."""
    with GeniusCityAnalyzer() as analyzer:
        return analyzer.analyze_city(city, country)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    result = analyze_city_genius("Lisbon", "Portugal")

    import json
    print(json.dumps(result, indent=2))
