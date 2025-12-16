"""
Adaptive City Analyzer using Active Inference and Structure Learning.

Instead of using fixed categories, this analyzer:
1. Starts with minimal assumptions about what travelers want
2. Learns category structure from scraped data
3. Actively decides what to search for next to reduce uncertainty
4. Expands/merges categories as evidence accumulates

This replaces the fixed DesireCategory enum with learned, adaptive categories.
"""

import logging
import time
from datetime import datetime
from typing import Optional

import httpx
import numpy as np
from bs4 import BeautifulSoup

from .structure_learner import StructureLearner, Observation, Category

logger = logging.getLogger(__name__)


class AdaptiveCityAnalyzer:
    """
    Active inference-based city desire analyzer.

    Key differences from the original CityDesireEngine:
    - Categories are learned, not fixed
    - The analyzer actively chooses what to search for
    - Structure evolves as more data is observed
    - Maintains uncertainty and can express confidence
    """

    def __init__(
        self,
        alpha: float = 1.0,  # Category creation tendency
        fit_threshold: float = 0.5,  # Cosine similarity threshold
        max_iterations: int = 10,  # Max active inference loops
        confidence_threshold: float = 0.7,  # When to stop exploring
    ):
        # Initialize embedding provider from NLP pipeline
        self._embedding_provider = None
        self._init_embedding_provider()

        self.structure_learner = StructureLearner(
            alpha=alpha,
            fit_threshold=fit_threshold,
            embedding_fn=self._get_embedding_list,
        )
        self.max_iterations = max_iterations
        self.confidence_threshold = confidence_threshold

        self.client = httpx.Client(
            timeout=30,
            follow_redirects=True,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            },
        )

        # Track what we've searched
        self.search_history: list[dict] = []

        # Embedding cache to avoid redundant API calls
        self._embedding_cache: dict[str, np.ndarray] = {}

    def _init_embedding_provider(self):
        """Initialize the embedding provider from the NLP pipeline."""
        try:
            from data_models.embeddings import get_default_provider
            self._embedding_provider = get_default_provider()
            logger.info(f"Initialized embedding provider: {type(self._embedding_provider).__name__}")
        except Exception as e:
            logger.warning(f"Could not initialize embedding provider: {e}")
            self._embedding_provider = None

    def _get_embedding_list(self, text: str) -> list[float]:
        """Get embedding as list (for structure learner)."""
        embedding = self._get_embedding(text)
        return embedding.tolist()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.client.close()

    def analyze_city(self, city: str, country: str = "") -> dict:
        """
        Analyze a city using active inference.

        Instead of scraping everything at once, we:
        1. Start with broad exploration
        2. Observe results and update beliefs
        3. Actively choose what to search next
        4. Stop when confident enough or max iterations reached
        """
        logger.info(f"Starting adaptive analysis for {city}, {country}")

        location = f"{city} {country}".strip()

        # Phase 1: Initial broad exploration
        initial_queries = [
            f"where to stay {city}",
            f"hotel {city} recommendation",
            f"{city} accommodation",
            f"best area to stay {city}",
        ]

        for query in initial_queries:
            self._search_and_observe(query, location)

        # Phase 2: Active inference loop
        iteration = 0
        while iteration < self.max_iterations:
            iteration += 1

            # Check if we're confident enough
            confidence = self._compute_confidence()
            logger.info(f"Iteration {iteration}, confidence: {confidence:.2f}")

            if confidence >= self.confidence_threshold:
                logger.info("Confidence threshold reached, stopping exploration")
                break

            # Get suggestion for what to search next
            suggestion = self.structure_learner.suggest_next_query()
            query = f"{suggestion['query']} {city}"

            logger.info(f"Active query: {query} (reason: {suggestion['reason']})")

            # Execute search and observe
            new_observations = self._search_and_observe(query, location)

            # If no new observations, try a different approach
            if new_observations == 0:
                logger.info("No new observations, trying alternative query")
                alt_query = f"{city} travel tips hotel"
                self._search_and_observe(alt_query, location)

            time.sleep(1)  # Rate limiting

        # Build final profile
        return self._build_profile(city, country)

    def _search_and_observe(self, query: str, location: str) -> int:
        """Execute search and feed observations to structure learner."""
        observations_added = 0

        # Track search
        self.search_history.append({
            "query": query,
            "timestamp": datetime.utcnow().isoformat(),
        })

        # Search Reddit
        observations_added += self._search_reddit(query, location)

        # Search YouTube (simplified)
        observations_added += self._search_youtube(query, location)

        return observations_added

    def _search_reddit(self, query: str, location: str) -> int:
        """Search Reddit and create observations."""
        count = 0

        subreddits = ["travel", "solotravel", "hotels", "digitalnomad"]

        for subreddit in subreddits[:2]:  # Limit for speed
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

                    # Check relevance to location
                    full_text = f"{title} {selftext}".lower()
                    location_lower = location.lower().split()[0]  # City name

                    if location_lower not in full_text:
                        continue

                    # Create observation
                    text = f"{title}\n{selftext[:500]}"
                    obs = Observation(
                        text=text,
                        embedding=self._get_embedding(text),
                        keywords=self._extract_keywords(text),
                        source="reddit",
                        sentiment=self._simple_sentiment(text),
                    )

                    # Feed to structure learner
                    self.structure_learner.observe(obs)
                    count += 1

                time.sleep(1)  # Rate limiting

            except Exception as e:
                logger.debug(f"Reddit search error: {e}")
                continue

        return count

    def _search_youtube(self, query: str, location: str) -> int:
        """Search YouTube for relevant content."""
        count = 0

        try:
            search_url = f"https://www.youtube.com/results?search_query={query.replace(' ', '+')}"
            response = self.client.get(search_url)

            if response.status_code != 200:
                return 0

            # Extract video titles from page
            html = response.text

            # Simple extraction (video titles are in the page)
            import re
            match = re.search(r'var ytInitialData = ({.*?});', html)
            if not match:
                return 0

            import json
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

                        # Check relevance
                        location_lower = location.lower().split()[0]
                        if location_lower not in title.lower():
                            continue

                        obs = Observation(
                            text=title,
                            embedding=self._get_embedding(title),
                            keywords=self._extract_keywords(title),
                            source="youtube",
                        )

                        self.structure_learner.observe(obs)
                        count += 1

            except json.JSONDecodeError:
                pass

        except Exception as e:
            logger.debug(f"YouTube search error: {e}")

        return count

    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text using the NLP pipeline."""
        # Check cache first
        cache_key = text[:500]  # Truncate for cache key
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]

        # Use real embedding provider if available
        if self._embedding_provider is not None:
            try:
                # Truncate text to avoid token limits
                truncated = text[:2000] if len(text) > 2000 else text
                embedding_list = self._embedding_provider.embed(truncated)
                embedding = np.array(embedding_list)

                # Normalize
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm

                self._embedding_cache[cache_key] = embedding
                return embedding

            except Exception as e:
                logger.warning(f"Embedding failed, using fallback: {e}")

        # Fallback: deterministic pseudo-embedding from text features
        embedding = self._fallback_embedding(text)
        self._embedding_cache[cache_key] = embedding
        return embedding

    def _fallback_embedding(self, text: str) -> np.ndarray:
        """Fallback embedding when provider unavailable."""
        import hashlib

        text_lower = text.lower()
        ngrams = [text_lower[i:i+3] for i in range(max(0, len(text_lower)-2))]

        dim = 384
        embedding = np.zeros(dim)

        for ngram in ngrams:
            h = int(hashlib.md5(ngram.encode()).hexdigest(), 16)
            idx = h % dim
            embedding[idx] += 1

        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def _extract_keywords(self, text: str) -> list[str]:
        """Extract keywords from text."""
        # Simple keyword extraction
        text_lower = text.lower()

        # Travel/hospitality terms to look for
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
        """
        Compute overall confidence in our understanding.

        Based on:
        - Number of observations
        - Average category fit
        - Free energy of the model
        """
        structure = self.structure_learner.get_structure()

        if structure["num_observations"] < 5:
            return 0.0

        # More observations = more confidence (diminishing returns)
        obs_confidence = 1 - np.exp(-structure["num_observations"] / 30)

        # Better average fit = more confidence
        if structure["categories"]:
            avg_fit = np.mean([c["avg_fit"] for c in structure["categories"]])
            fit_confidence = avg_fit
        else:
            fit_confidence = 0.0

        # Lower free energy = better model = more confidence
        free_energy = self.structure_learner.get_free_energy()
        fe_confidence = 1 / (1 + np.exp(free_energy))  # Sigmoid

        # Combine
        confidence = 0.4 * obs_confidence + 0.3 * fit_confidence + 0.3 * fe_confidence

        return confidence

    def _build_profile(self, city: str, country: str) -> dict:
        """Build final city desire profile from learned structure."""
        structure = self.structure_learner.get_structure()

        # Convert learned categories to desire themes
        themes = []
        for cat in structure["categories"]:
            theme = {
                "theme_name": cat["name"],
                "description": f"Travelers discussing {', '.join(cat['keywords'][:3])}",
                "intensity_score": min(cat["observation_count"] / 20, 1.0),
                "frequency": cat["observation_count"],
                "keywords": cat["keywords"],
                "category": cat["id"],
                "is_learned": True,  # Flag that this was learned, not predefined
            }
            themes.append(theme)

        # Sort by frequency
        themes.sort(key=lambda t: t["frequency"], reverse=True)

        # Identify potential opportunities (categories with high intensity but few observations)
        opportunities = []
        for theme in themes:
            if theme["intensity_score"] > 0.3 and theme["frequency"] < 10:
                opportunities.append(f"Emerging interest in: {theme['theme_name']}")

        return {
            "city": city,
            "country": country,
            "total_signals": structure["num_observations"],
            "num_learned_categories": structure["num_categories"],
            "top_desires": themes[:10],
            "white_space_opportunities": opportunities,
            "model_confidence": self._compute_confidence(),
            "free_energy": self.structure_learner.get_free_energy(),
            "search_history": self.search_history,
            "generated_at": datetime.utcnow().isoformat(),
            "method": "active_inference_structure_learning",
        }


def analyze_city_adaptive(city: str, country: str = "") -> dict:
    """Convenience function for adaptive city analysis."""
    with AdaptiveCityAnalyzer() as analyzer:
        return analyzer.analyze_city(city, country)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    result = analyze_city_adaptive("Lisbon", "Portugal")

    import json
    print(json.dumps(result, indent=2))
