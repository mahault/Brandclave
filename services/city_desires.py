"""City Desire Engine - Scrape and analyze what travelers want in a city.

Type a city → See what people want but can't get.
Identifies white space opportunities for hotel concepts.

Now with LLM synthesis for actionable insights instead of keyword concatenation.
"""

import asyncio
import json
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
from processing.llm_utils import MistralLLM
from services.city_desires_prompts import (
    THEME_SYNTHESIS_SYSTEM,
    THEME_SYNTHESIS_USER,
    BATCH_SYNTHESIS_SYSTEM,
    BATCH_SYNTHESIS_USER,
    CONCEPT_LANE_SYSTEM,
    CONCEPT_LANE_USER,
)

logger = logging.getLogger(__name__)


class CityDesireEngine:
    """Engine for scraping and analyzing city-specific traveler desires."""

    def __init__(self, use_llm: bool = True):
        """Initialize the engine.

        Args:
            use_llm: Whether to use LLM for synthesis. Defaults to True.
        """
        self.client = httpx.Client(
            timeout=30,
            follow_redirects=True,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            },
        )
        self.signals: list[DesireSignal] = []
        self.use_llm = use_llm
        self._llm: MistralLLM | None = None

    @property
    def llm(self) -> MistralLLM | None:
        """Lazy-load Mistral LLM client."""
        if not self.use_llm:
            return None
        if self._llm is None:
            try:
                self._llm = MistralLLM()
            except ValueError:
                logger.warning("Mistral API key not found, falling back to keyword synthesis")
                self.use_llm = False
        return self._llm

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
        """Extract relevant keywords from text that represent actual desires.

        Focuses on experience-related terms, not generic words like 'hotel' or 'stay'.
        """
        # Prioritized keyword categories (most insightful first)
        desire_keywords = {
            # Specific experiences
            "rooftop": 3, "terrace": 3, "pool": 2, "spa": 2, "sauna": 2,
            "yoga": 3, "meditation": 3, "wellness": 2, "massage": 2,
            "coworking": 3, "workspace": 2, "remote work": 3,
            "brunch": 2, "restaurant": 1, "bar": 1, "cocktails": 2,

            # Location qualities
            "walkable": 3, "central": 2, "downtown": 2, "neighborhood": 2,
            "quiet": 2, "peaceful": 2, "lively": 2, "nightlife": 2,
            "old town": 3, "historic center": 3, "waterfront": 3,

            # Style/vibe
            "boutique": 3, "design": 2, "modern": 2, "minimalist": 3,
            "cozy": 2, "charming": 2, "trendy": 2, "hip": 2,
            "authentic": 3, "local": 2, "unique": 2, "character": 3,
            "artsy": 3, "eclectic": 3, "instagram": 2,

            # Value propositions
            "affordable": 2, "value": 2, "cheap": 1, "free breakfast": 3,
            "all-inclusive": 3, "deals": 1,

            # Practical needs
            "kitchen": 2, "kitchenette": 2, "laundry": 2,
            "parking": 2, "family-friendly": 3, "pet-friendly": 3,
            "late checkout": 3, "early checkin": 3,

            # Atmosphere
            "views": 2, "balcony": 2, "terrace": 2, "garden": 2,
            "rooftop bar": 3, "infinity pool": 3,
        }

        found = []
        text_lower = text.lower()
        for term, priority in desire_keywords.items():
            if term in text_lower:
                found.append((term, priority))

        # Sort by priority (higher = more insightful) and return terms
        found.sort(key=lambda x: x[1], reverse=True)
        return [term for term, _ in found[:10]]

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

        # Filter out groups with too few signals
        valid_groups = {k: v for k, v in theme_groups.items() if len(v["signals"]) >= 2}

        if not valid_groups:
            return []

        # Try batch LLM synthesis for efficiency
        llm_results = {}
        if self.llm:
            logger.info(f"Synthesizing {len(valid_groups)} themes with LLM...")
            batch_results = self._synthesize_themes_batch(city, country, valid_groups)

            # Map results back to group keys
            group_keys = list(valid_groups.keys())
            for result in batch_results:
                cluster_id = result.get("cluster_id", -1)
                if 0 <= cluster_id < len(group_keys):
                    llm_results[group_keys[cluster_id]] = result

        # Convert to DesireTheme objects
        themes = []
        for theme_key, group in valid_groups.items():
            keywords = list(group["keywords"])[:5]
            snippets = [s.text[:200] for s in group["signals"][:5]]

            # Calculate scores
            total = len(group["signals"])
            frustration_score = group["frustration_count"] / total if total > 0 else 0
            intensity_score = min(total / 10, 1.0)  # Normalize to 0-1

            # Use LLM results if available, otherwise fallback
            llm_data = llm_results.get(theme_key)
            if llm_data:
                theme_name = llm_data.get("theme_name", self._generate_theme_name(group["category"], keywords))
                unmet_need = llm_data.get("unmet_need", "")
                why_supply_fails = llm_data.get("why_supply_fails", "")
                solving_features = llm_data.get("solving_features", [])
                target_guest = llm_data.get("target_guest", "")

                # Create a rich description combining the insights
                description = unmet_need
                if why_supply_fails:
                    description = f"{description}\n\n**Why current supply fails:** {why_supply_fails}"
                if solving_features:
                    features_str = ", ".join(solving_features[:3])
                    description = f"{description}\n\n**What would solve this:** {features_str}"
            else:
                theme_name = self._generate_theme_name(group["category"], keywords)
                description = self._generate_theme_description(theme_name, keywords)
                unmet_need = description
                why_supply_fails = ""
                solving_features = []
                target_guest = ""

            theme = DesireTheme(
                theme_name=theme_name,
                description=description,
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
                # LLM-synthesized insight fields
                unmet_need=unmet_need,
                why_supply_fails=why_supply_fails,
                solving_features=solving_features,
                target_guest=target_guest,
            )

            themes.append(theme)

        # Sort by opportunity score
        themes.sort(key=lambda t: t.opportunity_score, reverse=True)
        return themes[:15]  # Top 15 themes

    def _generate_theme_name(self, category: DesireCategory, keywords: list[str]) -> str:
        """Generate a readable theme name from category and keywords."""
        if not keywords:
            return f"Unmet {category.value.title()} Needs"

        # Create more descriptive theme names based on keywords
        keyword_themes = {
            "rooftop": "Rooftop Experience Seekers",
            "workspace": "Remote Work Travelers",
            "coworking": "Digital Nomad Demand",
            "walkable": "Walkability Priority",
            "boutique": "Boutique Character Demand",
            "authentic": "Authenticity Seekers",
            "wellness": "Wellness-Focused Travelers",
            "nightlife": "Nightlife Proximity Demand",
            "affordable": "Budget-Conscious Explorers",
            "family": "Family Travel Needs",
            "quiet": "Tranquility Seekers",
            "central": "Central Location Priority",
            "local": "Local Experience Demand",
            "pool": "Pool Access Priority",
            "views": "Scenic Views Demand",
        }

        # Try to match a keyword theme
        for kw in keywords[:3]:
            for theme_kw, theme_name in keyword_themes.items():
                if theme_kw in kw.lower():
                    return theme_name

        # Fallback: create a descriptive name from category
        category_descriptions = {
            DesireCategory.ACCOMMODATION: "Accommodation Gap",
            DesireCategory.EXPERIENCE: "Experience Demand",
            DesireCategory.AMENITY: "Amenity Expectations",
            DesireCategory.LOCATION: "Location Priorities",
            DesireCategory.SERVICE: "Service Standards Gap",
            DesireCategory.VIBE: "Atmosphere Preferences",
            DesireCategory.VALUE: "Value Expectations",
            DesireCategory.SAFETY: "Safety & Trust Concerns",
        }
        base_name = category_descriptions.get(category, "Traveler Needs")
        if keywords:
            return f"{keywords[0].title()} {base_name}"
        return base_name

    def _generate_theme_description(self, theme_name: str, keywords: list[str], snippets: list[str] = None) -> str:
        """Generate a meaningful description for the theme (fallback without LLM).

        Creates insight based on actual content patterns, not just keyword concatenation.
        """
        if not keywords:
            return "Travelers in this city have expressed needs that current accommodation options aren't meeting."

        # Create context-aware descriptions
        kw_lower = [k.lower() for k in keywords]

        if any(k in kw_lower for k in ["rooftop", "terrace", "views", "balcony"]):
            return f"Travelers are seeking accommodations with outdoor spaces and scenic views. Many express frustration at limited rooftop or terrace access, especially in urban areas where properties rarely offer this despite demand."

        if any(k in kw_lower for k in ["workspace", "coworking", "wifi", "remote work"]):
            return f"Remote workers and digital nomads are looking for hotels with dedicated work spaces, reliable high-speed wifi, and flexible check-in/out times. Current options often treat this as an afterthought rather than a core offering."

        if any(k in kw_lower for k in ["walkable", "central", "downtown", "location"]):
            return f"Location convenience is a top priority, with travelers seeking walkable access to attractions, restaurants, and public transit. Budget options in central areas are particularly scarce, forcing a trade-off between location and price."

        if any(k in kw_lower for k in ["boutique", "unique", "character", "authentic"]):
            return f"Travelers want properties with personality and local character, not generic chain hotels. They're seeking authentic experiences that reflect the destination's culture, but supply is dominated by standardized offerings."

        if any(k in kw_lower for k in ["affordable", "budget", "cheap", "value"]):
            return f"Budget-conscious travelers are struggling to find quality accommodations at reasonable prices. They want clean, comfortable stays without paying for amenities they won't use, but mid-range options are limited."

        if any(k in kw_lower for k in ["wellness", "spa", "yoga", "meditation"]):
            return f"Wellness-focused travelers want more than just a spa - they're seeking holistic experiences including yoga, meditation spaces, healthy dining, and fitness facilities. True wellness properties remain rare."

        if any(k in kw_lower for k in ["quiet", "peaceful", "tranquil"]):
            return f"Travelers seeking tranquility have difficulty finding genuinely quiet properties. Noise from streets, neighboring rooms, and common areas is a frequent complaint, with soundproofing being an undervalued differentiator."

        if any(k in kw_lower for k in ["nightlife", "bar", "lively"]):
            return f"Travelers looking for vibrant nightlife experiences want properties in lively neighborhoods with on-site bars or easy access to entertainment. Many complain about areas that 'shut down' after dark."

        # Generic fallback with keywords
        kw_str = ", ".join(keywords[:3])
        return f"Travelers discussing {kw_str} are expressing unmet needs in this city. There's an opportunity for properties that can deliver on these specific expectations."

    def _synthesize_theme_with_llm(
        self,
        city: str,
        country: str,
        category: DesireCategory,
        keywords: list[str],
        segments: list[TravelerSegment],
        frustration_pct: float,
        frequency: int,
        snippets: list[str],
    ) -> dict | None:
        """Use LLM to synthesize a theme from raw signals.

        Returns:
            Dict with theme_name, unmet_need, why_supply_fails, solving_features,
            target_description, opportunity_insight. Or None if LLM unavailable.
        """
        if not self.llm:
            return None

        # Format quotes for the prompt
        quotes = "\n".join(f"- \"{s[:300]}\"" for s in snippets[:5])
        segments_str = ", ".join(s.value for s in segments) if segments else "general travelers"

        user_prompt = THEME_SYNTHESIS_USER.format(
            city=city,
            country=country or "Unknown",
            category=category.value,
            keywords=", ".join(keywords),
            segments=segments_str,
            frustration_pct=int(frustration_pct * 100),
            frequency=frequency,
            quotes=quotes,
        )

        try:
            response = self.llm.generate(
                prompt=user_prompt,
                system_prompt=THEME_SYNTHESIS_SYSTEM,
                max_tokens=500,
                temperature=0.6,
            )

            # Strip markdown code blocks if present
            json_str = response.strip()
            if json_str.startswith("```"):
                first_newline = json_str.find("\n")
                if first_newline != -1:
                    json_str = json_str[first_newline + 1:]
                if json_str.endswith("```"):
                    json_str = json_str[:-3].strip()

            # Parse JSON response
            result = json.loads(json_str)
            return result

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM theme response: {e}")
            return None
        except Exception as e:
            logger.error(f"LLM theme synthesis failed: {e}")
            return None

    def _synthesize_themes_batch(
        self,
        city: str,
        country: str,
        theme_groups: dict,
    ) -> list[dict]:
        """Synthesize multiple themes in a single LLM call for efficiency.

        Args:
            city: City name
            country: Country name
            theme_groups: Dict of theme_key -> group data

        Returns:
            List of synthesized theme dicts
        """
        if not self.llm or len(theme_groups) == 0:
            return []

        # Build clusters JSON for batch synthesis
        clusters = []
        group_keys = list(theme_groups.keys())

        for i, key in enumerate(group_keys):
            group = theme_groups[key]
            snippets = [s.text[:200] for s in group["signals"][:3]]
            clusters.append({
                "cluster_id": i,
                "category": group["category"].value,
                "keywords": list(group["keywords"])[:5],
                "segments": [s.value for s in group["segments"]],
                "frustration_pct": int(group["frustration_count"] / len(group["signals"]) * 100) if group["signals"] else 0,
                "frequency": len(group["signals"]),
                "sample_quotes": snippets,
            })

        user_prompt = BATCH_SYNTHESIS_USER.format(
            count=len(clusters),
            city=city,
            country=country or "Unknown",
            clusters_json=json.dumps(clusters, indent=2),
        )

        try:
            response = self.llm.generate(
                prompt=user_prompt,
                system_prompt=BATCH_SYNTHESIS_SYSTEM,
                max_tokens=1500,
                temperature=0.6,
            )

            # Strip markdown code blocks if present
            json_str = response.strip()
            if json_str.startswith("```"):
                # Remove opening fence (```json or ```)
                first_newline = json_str.find("\n")
                if first_newline != -1:
                    json_str = json_str[first_newline + 1:]
                # Remove closing fence
                if json_str.endswith("```"):
                    json_str = json_str[:-3].strip()

            # Parse JSON array response
            results = json.loads(json_str)
            return results

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse batch LLM response: {e}")
            logger.debug(f"Raw response was: {response[:500] if response else 'empty'}")
            return []
        except Exception as e:
            logger.error(f"Batch LLM synthesis failed: {e}")
            return []

    def _synthesize_concept_lanes_with_llm(
        self,
        themes: list[DesireTheme],
        city: str,
        country: str,
        underserved_segments: list[str],
        avg_frustration: float,
    ) -> list[dict]:
        """Use LLM to generate hotel concept recommendations.

        Returns:
            List of concept dicts with name, positioning, differentiators, etc.
        """
        if not self.llm or not themes:
            return []

        # Build themes summary for the prompt
        themes_summary = ""
        for i, theme in enumerate(themes[:5], 1):
            themes_summary += f"{i}. **{theme.theme_name}**\n"
            themes_summary += f"   - Unmet need: {theme.description}\n"
            themes_summary += f"   - Frustration: {theme.frustration_score:.0%}\n"
            themes_summary += f"   - Frequency: {theme.frequency} mentions\n"
            themes_summary += f"   - Keywords: {', '.join(theme.keywords[:5])}\n\n"

        user_prompt = CONCEPT_LANE_USER.format(
            city=city,
            country=country or "Unknown",
            themes_summary=themes_summary,
            underserved_segments=", ".join(underserved_segments) if underserved_segments else "various",
            avg_frustration=int(avg_frustration * 100),
        )

        try:
            response = self.llm.generate(
                prompt=user_prompt,
                system_prompt=CONCEPT_LANE_SYSTEM,
                max_tokens=1000,
                temperature=0.7,
            )

            # Strip markdown code blocks if present
            json_str = response.strip()
            if json_str.startswith("```"):
                first_newline = json_str.find("\n")
                if first_newline != -1:
                    json_str = json_str[first_newline + 1:]
                if json_str.endswith("```"):
                    json_str = json_str[:-3].strip()

            # Parse JSON response
            result = json.loads(json_str)
            return result.get("concepts", [])

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse concept lanes response: {e}")
            logger.debug(f"Raw response was: {response[:500] if response else 'empty'}")
            return []
        except Exception as e:
            logger.error(f"Concept lane synthesis failed: {e}")
            return []

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

        # Generate concept lane recommendations with LLM if available
        concept_lanes = []
        if self.llm:
            logger.info("Generating concept lanes with LLM...")
            concept_lanes = self._synthesize_concept_lanes_with_llm(
                themes, city, country, underserved, avg_frustration
            )

        # Fallback to basic generation if LLM failed or unavailable
        if not concept_lanes:
            concept_lanes = self._generate_concept_lanes_basic(themes, city)

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

    def _generate_concept_lanes_basic(self, themes: list[DesireTheme], city: str) -> list[dict]:
        """Generate basic hotel concept recommendations (fallback without LLM)."""
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
