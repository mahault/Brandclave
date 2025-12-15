"""City Desire Engine - Scrape and analyze what travelers want in a city.

Type a city → See what people want but can't get.
Identifies white space opportunities for hotel concepts.
"""

import logging
import re
import time
from datetime import datetime, timedelta
from typing import Optional

import httpx
from bs4 import BeautifulSoup

from data_models.city_desires import (
    CityDesireProfile,
    DesireCategory,
    DesireSignal,
    DesireTheme,
    SentimentType,
    TravelerSegment,
    DESIRE_PATTERNS,
    SEGMENT_KEYWORDS,
    CATEGORY_KEYWORDS,
)

logger = logging.getLogger(__name__)


class CityDesireEngine:
    """Engine for scraping and analyzing city-specific traveler desires."""

    def __init__(self):
        """Initialize the engine."""
        self.client = httpx.Client(
            timeout=30,
            follow_redirects=True,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            },
        )
        self.signals: list[DesireSignal] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.client.close()

    def analyze_city(self, city: str, country: str = "") -> CityDesireProfile:
        """Analyze a city and return its desire profile.

        Args:
            city: City name (e.g., "Lisbon")
            country: Optional country name for disambiguation

        Returns:
            CityDesireProfile with desires, themes, and opportunities
        """
        logger.info(f"Analyzing desires for {city}, {country}")

        self.signals = []

        # Scrape from multiple sources
        self._scrape_reddit(city, country)
        self._scrape_youtube(city, country)
        self._scrape_travel_forums(city, country)

        # Extract desires from signals
        self._classify_signals()

        # Cluster into themes
        themes = self._cluster_into_themes(city, country)

        # Build profile
        profile = self._build_profile(city, country, themes)

        return profile

    def _scrape_reddit(self, city: str, country: str) -> None:
        """Scrape Reddit for city-specific hotel discussions."""
        logger.info(f"Scraping Reddit for {city}")

        subreddits = ["travel", "solotravel", "digitalnomad", "hotels", "TravelHacks"]
        queries = [
            f"where to stay {city}",
            f"hotel {city}",
            f"accommodation {city}",
            f"{city} hostel",
            f"{city} airbnb vs hotel",
        ]

        for subreddit in subreddits:
            for query in queries[:2]:  # Limit queries per subreddit
                try:
                    url = f"https://old.reddit.com/r/{subreddit}/search.json"
                    params = {
                        "q": query,
                        "restrict_sr": "on",
                        "sort": "relevance",
                        "t": "year",
                        "limit": 25,
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
                        permalink = post_data.get("permalink", "")

                        # Check if actually about the city
                        full_text = f"{title} {selftext}".lower()
                        if city.lower() not in full_text:
                            continue

                        # Create signal from post
                        if selftext:
                            self.signals.append(DesireSignal(
                                text=f"{title}\n{selftext[:1000]}",
                                source="reddit",
                                source_url=f"https://reddit.com{permalink}",
                                city=city,
                                country=country,
                            ))

                        # Also fetch comments for richer data
                        self._fetch_reddit_comments(permalink, city, country)

                    time.sleep(2)  # Rate limiting

                except Exception as e:
                    logger.error(f"Reddit scrape error: {e}")
                    continue

    def _fetch_reddit_comments(self, permalink: str, city: str, country: str) -> None:
        """Fetch comments from a Reddit post."""
        try:
            url = f"https://old.reddit.com{permalink}.json"
            response = self.client.get(url)
            if response.status_code != 200:
                return

            data = response.json()
            if len(data) < 2:
                return

            comments = data[1].get("data", {}).get("children", [])

            for comment in comments[:20]:  # Limit comments
                comment_data = comment.get("data", {})
                body = comment_data.get("body", "")

                if len(body) > 50 and city.lower() in body.lower():
                    self.signals.append(DesireSignal(
                        text=body[:800],
                        source="reddit",
                        source_url=f"https://reddit.com{permalink}",
                        city=city,
                        country=country,
                    ))

        except Exception as e:
            logger.debug(f"Comment fetch error: {e}")

    def _scrape_youtube(self, city: str, country: str) -> None:
        """Scrape YouTube for city hotel content."""
        logger.info(f"Scraping YouTube for {city}")

        queries = [
            f"where to stay in {city}",
            f"best hotels {city}",
            f"{city} hotel review",
            f"{city} accommodation guide",
        ]

        for query in queries:
            try:
                search_url = f"https://www.youtube.com/results?search_query={query.replace(' ', '+')}"
                response = self.client.get(search_url)
                if response.status_code != 200:
                    continue

                # Extract video data from page
                html = response.text

                # Look for ytInitialData JSON
                match = re.search(r'var ytInitialData = ({.*?});', html)
                if not match:
                    continue

                import json
                try:
                    data = json.loads(match.group(1))
                except json.JSONDecodeError:
                    continue

                # Navigate to video results
                try:
                    contents = data["contents"]["twoColumnSearchResultsRenderer"]["primaryContents"]["sectionListRenderer"]["contents"]
                    for section in contents:
                        items = section.get("itemSectionRenderer", {}).get("contents", [])
                        for item in items[:10]:
                            video = item.get("videoRenderer", {})
                            if not video:
                                continue

                            title = video.get("title", {}).get("runs", [{}])[0].get("text", "")
                            video_id = video.get("videoId", "")
                            description = ""
                            for snippet in video.get("detailedMetadataSnippets", []):
                                for run in snippet.get("snippetText", {}).get("runs", []):
                                    description += run.get("text", "")

                            if title and city.lower() in title.lower():
                                self.signals.append(DesireSignal(
                                    text=f"{title}\n{description}",
                                    source="youtube",
                                    source_url=f"https://youtube.com/watch?v={video_id}",
                                    city=city,
                                    country=country,
                                ))
                except (KeyError, IndexError):
                    pass

                time.sleep(2)

            except Exception as e:
                logger.error(f"YouTube scrape error: {e}")
                continue

    def _scrape_travel_forums(self, city: str, country: str) -> None:
        """Scrape travel forums and Q&A sites."""
        logger.info(f"Scraping travel forums for {city}")

        # TripAdvisor forum search
        try:
            url = f"https://www.tripadvisor.com/Search?q={city}+hotel+where+to+stay"
            response = self.client.get(url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")
                # Extract forum posts and reviews
                for result in soup.select(".result-title, .review-container")[:15]:
                    text = result.get_text(strip=True)
                    if len(text) > 50:
                        self.signals.append(DesireSignal(
                            text=text[:600],
                            source="tripadvisor",
                            city=city,
                            country=country,
                        ))
        except Exception as e:
            logger.debug(f"TripAdvisor error: {e}")

        time.sleep(2)

    def _classify_signals(self) -> None:
        """Classify each signal with sentiment, category, and segments."""
        for signal in self.signals:
            text_lower = signal.text.lower()

            # Detect sentiment
            signal.sentiment = self._detect_sentiment(text_lower)

            # Detect category
            signal.category = self._detect_category(text_lower)

            # Detect segments
            signal.segments = self._detect_segments(text_lower)

            # Extract keywords
            signal.keywords = self._extract_keywords(text_lower)

    def _detect_sentiment(self, text: str) -> SentimentType:
        """Detect the sentiment/type of desire expression."""
        for pattern in DESIRE_PATTERNS["frustration"]:
            if re.search(pattern, text):
                return SentimentType.FRUSTRATION

        for pattern in DESIRE_PATTERNS["complaint"]:
            if re.search(pattern, text):
                return SentimentType.COMPLAINT

        for pattern in DESIRE_PATTERNS["desire"]:
            if re.search(pattern, text):
                return SentimentType.DESIRE

        return SentimentType.QUESTION

    def _detect_category(self, text: str) -> DesireCategory:
        """Detect the category of the desire."""
        scores = {}
        for category, keywords in CATEGORY_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text)
            if score > 0:
                scores[category] = score

        if scores:
            return max(scores, key=scores.get)
        return DesireCategory.ACCOMMODATION

    def _detect_segments(self, text: str) -> list[TravelerSegment]:
        """Detect which traveler segments are mentioned."""
        segments = []
        for segment, keywords in SEGMENT_KEYWORDS.items():
            if any(kw in text for kw in keywords):
                segments.append(segment)
        return segments

    def _extract_keywords(self, text: str) -> list[str]:
        """Extract relevant keywords from text."""
        # Simple keyword extraction - look for common hotel/travel terms
        travel_terms = [
            "boutique", "design", "modern", "luxury", "budget", "affordable",
            "central", "walkable", "quiet", "rooftop", "pool", "breakfast",
            "wifi", "workspace", "gym", "spa", "view", "balcony", "kitchen",
            "historic", "trendy", "hip", "safe", "clean", "friendly",
            "authentic", "local", "nightlife", "beach", "mountain",
        ]
        found = [term for term in travel_terms if term in text]
        return found[:10]

    def _cluster_into_themes(self, city: str, country: str) -> list[DesireTheme]:
        """Cluster signals into desire themes."""
        # Group signals by category and keywords
        theme_groups = {}

        for signal in self.signals:
            # Skip non-desire signals
            if signal.sentiment == SentimentType.DELIGHT:
                continue

            # Create theme key from category + top keywords
            key_parts = [signal.category.value]
            if signal.keywords:
                key_parts.extend(signal.keywords[:2])
            theme_key = "_".join(key_parts)

            if theme_key not in theme_groups:
                theme_groups[theme_key] = {
                    "signals": [],
                    "category": signal.category,
                    "keywords": set(),
                    "segments": set(),
                    "frustration_count": 0,
                }

            theme_groups[theme_key]["signals"].append(signal)
            theme_groups[theme_key]["keywords"].update(signal.keywords)
            theme_groups[theme_key]["segments"].update(signal.segments)
            if signal.sentiment in [SentimentType.FRUSTRATION, SentimentType.COMPLAINT]:
                theme_groups[theme_key]["frustration_count"] += 1

        # Convert to DesireTheme objects
        themes = []
        for theme_key, group in theme_groups.items():
            if len(group["signals"]) < 2:  # Skip themes with only 1 signal
                continue

            # Generate theme name from keywords
            keywords = list(group["keywords"])[:5]
            theme_name = self._generate_theme_name(group["category"], keywords)

            # Calculate scores
            total = len(group["signals"])
            frustration_score = group["frustration_count"] / total if total > 0 else 0
            intensity_score = min(total / 10, 1.0)  # Normalize to 0-1

            # Example snippets
            snippets = [s.text[:200] for s in group["signals"][:5]]

            theme = DesireTheme(
                theme_name=theme_name,
                description=self._generate_theme_description(theme_name, keywords),
                city=city,
                country=country,
                intensity_score=round(intensity_score, 2),
                frustration_score=round(frustration_score, 2),
                frequency=total,
                category=group["category"],
                segments=list(group["segments"]),
                keywords=keywords,
                example_snippets=snippets,
                supply_gap=round(frustration_score * 0.8, 2),  # Estimate
                opportunity_score=round((intensity_score + frustration_score) / 2, 2),
            )
            themes.append(theme)

        # Sort by opportunity score
        themes.sort(key=lambda t: t.opportunity_score, reverse=True)
        return themes[:15]  # Top 15 themes

    def _generate_theme_name(self, category: DesireCategory, keywords: list[str]) -> str:
        """Generate a readable theme name."""
        if not keywords:
            return f"{category.value.title()} Needs"

        # Combine top keywords into a theme name
        kw_str = " + ".join(keywords[:3]).title()
        return f"{kw_str} {category.value.title()}"

    def _generate_theme_description(self, theme_name: str, keywords: list[str]) -> str:
        """Generate a description for the theme."""
        return f"Travelers are looking for {', '.join(keywords[:4])} options. This represents an opportunity for hotels that can deliver on these needs."

    def _build_profile(
        self, city: str, country: str, themes: list[DesireTheme]
    ) -> CityDesireProfile:
        """Build the complete city desire profile."""
        # Calculate aggregates
        total_signals = len(self.signals)
        frustration_signals = [
            s for s in self.signals
            if s.sentiment in [SentimentType.FRUSTRATION, SentimentType.COMPLAINT]
        ]
        avg_frustration = len(frustration_signals) / total_signals if total_signals > 0 else 0

        # Get unique sources
        sources = set(s.source for s in self.signals)

        # Identify underserved segments
        segment_frustration = {}
        for signal in frustration_signals:
            for segment in signal.segments:
                segment_frustration[segment.value] = segment_frustration.get(segment.value, 0) + 1

        underserved = sorted(segment_frustration.keys(), key=lambda k: segment_frustration[k], reverse=True)[:5]

        # Generate white space opportunities
        white_space = []
        for theme in themes[:5]:
            if theme.frustration_score > 0.3:
                white_space.append(f"{theme.theme_name} (frustration: {theme.frustration_score:.0%})")

        # Generate concept lane recommendations
        concept_lanes = self._generate_concept_lanes(themes, city)

        return CityDesireProfile(
            city=city,
            country=country,
            total_signals=total_signals,
            total_sources=len(sources),
            avg_frustration=round(avg_frustration, 2),
            top_desires=themes,
            underserved_segments=underserved,
            white_space_opportunities=white_space,
            concept_lanes=concept_lanes,
            generated_at=datetime.utcnow(),
        )

    def _generate_concept_lanes(self, themes: list[DesireTheme], city: str) -> list[dict]:
        """Generate hotel concept recommendations based on themes."""
        lanes = []

        for theme in themes[:5]:
            if theme.opportunity_score < 0.3:
                continue

            lane = {
                "concept": f"{theme.theme_name} Hotel",
                "target_segments": [s.value for s in theme.segments[:3]],
                "key_features": theme.keywords[:5],
                "opportunity_score": theme.opportunity_score,
                "rationale": f"High demand ({theme.frequency} mentions) with {theme.frustration_score:.0%} frustration rate indicates underserved market.",
            }
            lanes.append(lane)

        return lanes


def analyze_city_desires(city: str, country: str = "") -> dict:
    """Convenience function to analyze a city.

    Args:
        city: City name
        country: Optional country

    Returns:
        City desire profile as dict
    """
    with CityDesireEngine() as engine:
        profile = engine.analyze_city(city, country)
        return profile.to_dict()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = analyze_city_desires("Lisbon", "Portugal")
    import json
    print(json.dumps(result, indent=2))
