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

from ingestion.http_client import resilient_get

from .structure_learner import StructureLearner, Observation, Category

logger = logging.getLogger(__name__)


def _strip_markdown_json(response: str) -> str:
    """Strip markdown code blocks from LLM response."""
    json_str = response.strip()
    if json_str.startswith("```"):
        first_newline = json_str.find("\n")
        if first_newline != -1:
            json_str = json_str[first_newline + 1:]
        if json_str.endswith("```"):
            json_str = json_str[:-3].strip()
    return json_str


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
        use_llm: bool = True,  # Whether to use LLM for insight synthesis
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
        self.use_llm = use_llm
        self._llm = None

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

    @property
    def llm(self):
        """Lazy-load LLM client."""
        if not self.use_llm:
            return None
        if self._llm is None:
            try:
                from processing.llm_utils import MistralLLM
                self._llm = MistralLLM()
            except (ValueError, ImportError) as e:
                logger.warning(f"LLM not available: {e}")
                self.use_llm = False
        return self._llm

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

                response = resilient_get(url, params=params, timeout=None, client=self.client)
                if response is None or response.status_code != 200:
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
            response = resilient_get(search_url, timeout=None, client=self.client)

            if response is None or response.status_code != 200:
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

        # Synthesize insights with LLM if available
        llm_insights = {}
        if self.llm and structure["categories"]:
            llm_insights = self._synthesize_insights_batch(city, country, structure["categories"])

        for cat in structure["categories"]:
            # Format sources for display
            sources = cat.get("sources", {})
            source_list = [
                {"name": src, "count": int(count)}
                for src, count in sorted(sources.items(), key=lambda x: x[1], reverse=True)
            ]

            # Get example snippets with attribution
            examples = cat.get("example_texts", [])

            # Get LLM-synthesized insights for this category
            cat_insights = llm_insights.get(cat["id"], {})

            # Create rich theme with insights
            theme_name = cat_insights.get("theme_name", cat["name"])
            unmet_need = cat_insights.get("unmet_need", "")
            why_supply_fails = cat_insights.get("why_supply_fails", "")
            solving_features = cat_insights.get("solving_features", [])
            target_guest = cat_insights.get("target_guest", "")

            # Create description - either from LLM or fallback
            if unmet_need:
                description = unmet_need
            else:
                # Fallback description based on keywords
                description = self._generate_fallback_description(cat["keywords"], examples)

            theme = {
                "theme_name": theme_name,
                "description": description,
                "unmet_need": unmet_need,
                "why_supply_fails": why_supply_fails,
                "solving_features": solving_features,
                "target_guest": target_guest,
                "intensity_score": min(cat["observation_count"] / 20, 1.0),
                "frequency": cat["observation_count"],
                "keywords": cat["keywords"],
                "category": cat["id"],
                "is_learned": True,
                "sources": source_list,
                "example_snippets": examples,
            }
            themes.append(theme)

        # Sort by frequency
        themes.sort(key=lambda t: t["frequency"], reverse=True)

        # Identify potential opportunities (categories with high intensity but few observations)
        opportunities = []
        for theme in themes:
            if theme["intensity_score"] > 0.3 and theme["frequency"] < 10:
                opportunities.append(f"Emerging interest in: {theme['theme_name']}")

        # Aggregate all sources across themes
        all_sources = {}
        for theme in themes:
            for src in theme["sources"]:
                all_sources[src["name"]] = all_sources.get(src["name"], 0) + src["count"]

        # Generate concept lanes with LLM
        concept_lanes = []
        if self.llm and themes:
            concept_lanes = self._synthesize_concept_lanes(city, country, themes[:5])

        return {
            "city": city,
            "country": country,
            "total_signals": structure["num_observations"],
            "num_learned_categories": structure["num_categories"],
            "top_desires": themes[:10],
            "white_space_opportunities": opportunities,
            "concept_lanes": concept_lanes,
            "model_confidence": self._compute_confidence(),
            "free_energy": self.structure_learner.get_free_energy(),
            "search_history": self.search_history,
            "sources_summary": all_sources,
            "generated_at": datetime.utcnow().isoformat(),
            "method": "active_inference_structure_learning",
        }

    def _generate_fallback_description(self, keywords: list[str], examples: list[dict]) -> str:
        """Generate a meaningful description without LLM."""
        kw_lower = [k.lower() for k in keywords]

        # Context-aware descriptions based on keywords
        if any(k in kw_lower for k in ["rooftop", "terrace", "view", "balcony"]):
            return "Travelers seeking accommodations with outdoor spaces and scenic views. Many express frustration at limited rooftop access in urban properties."

        if any(k in kw_lower for k in ["workspace", "coworking", "wifi", "remote", "digital nomad"]):
            return "Remote workers and digital nomads looking for hotels with dedicated work spaces and reliable wifi. Current options often treat this as an afterthought."

        if any(k in kw_lower for k in ["walkable", "central", "location"]):
            return "Location is a top priority - travelers want walkable access to attractions and transit. Budget options in central areas are particularly scarce."

        if any(k in kw_lower for k in ["boutique", "design", "unique", "authentic"]):
            return "Travelers want properties with personality and local character. They're seeking authentic experiences, not generic chain hotels."

        if any(k in kw_lower for k in ["affordable", "budget", "cheap"]):
            return "Budget-conscious travelers struggling to find quality at reasonable prices. They want value without sacrificing cleanliness and comfort."

        if any(k in kw_lower for k in ["spa", "wellness", "yoga"]):
            return "Wellness-focused travelers seeking holistic experiences beyond just a spa - yoga, meditation, healthy dining, and fitness facilities."

        if any(k in kw_lower for k in ["safe", "solo", "female"]):
            return "Safety-conscious travelers, especially solo and female travelers, prioritizing secure neighborhoods and well-reviewed properties."

        if any(k in kw_lower for k in ["family", "kid"]):
            return "Families seeking kid-friendly amenities, spacious rooms, and convenient locations for traveling with children."

        # Generic but still useful fallback
        if keywords:
            return f"Travelers expressing interest in {', '.join(keywords[:3])}. This represents an opportunity for properties that can deliver on these specific needs."
        return "Travelers have expressed needs that current accommodations aren't fully meeting."

    def _synthesize_insights_batch(self, city: str, country: str, categories: list[dict]) -> dict:
        """Use LLM to synthesize insights for multiple categories at once."""
        import json

        if not self.llm or not categories:
            return {}

        # Build clusters for batch synthesis
        clusters = []
        for i, cat in enumerate(categories[:8]):  # Limit to top 8
            examples = cat.get("example_texts", [])
            sample_quotes = [ex.get("text", "")[:200] for ex in examples[:3]] if examples else []

            clusters.append({
                "cluster_id": i,
                "category_id": cat["id"],
                "keywords": cat["keywords"][:5],
                "frequency": cat["observation_count"],
                "sample_quotes": sample_quotes,
            })

        system_prompt = """You are a hospitality industry analyst. Analyze traveler signals and transform keyword clusters into meaningful, actionable insights.

For each cluster, provide:
1. A compelling theme name (not keyword concatenation)
2. The underlying unmet need (what travelers want but can't find)
3. Why current supply fails to meet this need
4. Specific features that would solve the problem

Be specific and actionable. Focus on human insights, not data summaries.

Output format: JSON array only, no markdown."""

        user_prompt = f"""Analyze these {len(clusters)} theme clusters for {city}, {country}:

{json.dumps(clusters, indent=2)}

For each cluster, provide:

[
    {{
        "cluster_id": 0,
        "theme_name": "Compelling 3-6 word name",
        "unmet_need": "What travelers want but can't find",
        "why_supply_fails": "Why current options don't work",
        "solving_features": ["Feature 1", "Feature 2", "Feature 3"],
        "target_guest": "Specific traveler type"
    }},
    ...
]

Transform these keyword lists into human insights. What's the story?"""

        try:
            response = self.llm.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                max_tokens=2000,
                temperature=0.6,
            )

            json_str = _strip_markdown_json(response)
            results = json.loads(json_str)

            # Map results back to category IDs
            insights = {}
            for result in results:
                cluster_id = result.get("cluster_id", -1)
                if 0 <= cluster_id < len(clusters):
                    cat_id = clusters[cluster_id]["category_id"]
                    insights[cat_id] = result

            return insights

        except Exception as e:
            logger.warning(f"Failed to synthesize insights: {e}")
            return {}

    def _synthesize_concept_lanes(self, city: str, country: str, themes: list[dict]) -> list[dict]:
        """Use LLM to generate hotel concept recommendations."""
        import json

        if not self.llm or not themes:
            return []

        # Build themes summary
        themes_summary = ""
        for i, theme in enumerate(themes[:5], 1):
            themes_summary += f"{i}. **{theme['theme_name']}**\n"
            themes_summary += f"   - Unmet need: {theme.get('unmet_need', theme['description'][:100])}\n"
            themes_summary += f"   - Frequency: {theme['frequency']} mentions\n\n"

        system_prompt = """You are a hospitality brand strategist who turns market insights into hotel concepts.

Given desire themes for a city, create actionable hotel concept recommendations that would capture the identified white space.

Rules:
1. Concepts should be differentiated and ownable
2. Include specific positioning, not generic descriptions
3. Connect features to the underlying desires they solve
4. Be creative but grounded in the data

Output format: JSON only, no markdown."""

        user_prompt = f"""Based on these desire themes for {city}, {country}, recommend hotel concepts:

**Top Desire Themes:**
{themes_summary}

Provide 2-3 hotel concept recommendations as JSON:

{{
    "concepts": [
        {{
            "name": "Concept name (creative, memorable)",
            "positioning": "One sentence positioning statement",
            "solves": "The core unmet desire this addresses",
            "key_differentiators": ["Differentiator 1", "Differentiator 2", "Differentiator 3"],
            "target_guest": "Specific guest profile",
            "price_position": "Budget/Midscale/Upscale/Luxury",
            "why_it_wins": "Why this concept would outperform existing options"
        }}
    ]
}}"""

        try:
            response = self.llm.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                max_tokens=1000,
                temperature=0.7,
            )

            json_str = _strip_markdown_json(response)
            result = json.loads(json_str)
            return result.get("concepts", [])

        except Exception as e:
            logger.warning(f"Failed to synthesize concept lanes: {e}")
            return []


def analyze_city_adaptive(city: str, country: str = "") -> dict:
    """Convenience function for adaptive city analysis."""
    with AdaptiveCityAnalyzer() as analyzer:
        return analyzer.analyze_city(city, country)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    result = analyze_city_adaptive("Lisbon", "Portugal")

    import json
    print(json.dumps(result, indent=2))
