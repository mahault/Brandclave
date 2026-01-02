"""Social Pulse service - Trend generation and management."""

import logging
from datetime import datetime, timedelta
from typing import Any

from db.database import SessionLocal
from db.models import TrendSignalModel, RawContentModel
from processing.clustering import ContentClusterer, Cluster, get_content_for_cluster
from processing.scoring import TrendScorer, get_metrics_dict
from processing.llm_utils import generate_trend_insights, get_llm

logger = logging.getLogger(__name__)


class SocialPulseService:
    """Service for generating and managing Social Pulse trends.

    Integrates with Clustering POMDP for adaptive parameter selection.
    """

    def __init__(
        self,
        min_cluster_size: int = 3,
        days_back: int = 30,
        use_llm: bool = True,
        use_adaptive: bool = True,
    ):
        """Initialize Social Pulse service.

        Args:
            min_cluster_size: Minimum content items to form a trend (fallback)
            days_back: Days of content to analyze
            use_llm: Whether to use LLM for generating insights
            use_adaptive: Whether to use POMDP for adaptive clustering
        """
        self.clusterer = ContentClusterer(
            min_cluster_size=min_cluster_size,
            use_adaptive=use_adaptive,
        )
        self.scorer = TrendScorer()
        self.days_back = days_back
        self.use_llm = use_llm
        self.use_adaptive = use_adaptive

    def generate_trends(
        self,
        source_types: list[str] | None = None,
        max_trends: int = 20,
    ) -> list[dict]:
        """Generate trend signals from clustered content.

        Args:
            source_types: Filter by source types (e.g., ['social'])
            max_trends: Maximum number of trends to generate

        Returns:
            List of generated trend dicts
        """
        logger.info(f"Generating trends (source_types={source_types}, max={max_trends})")

        # Run clustering (returns clusters and params used)
        clusters, cluster_params = self.clusterer.cluster_content(
            source_types=source_types,
            days_back=self.days_back,
        )

        if cluster_params:
            logger.info(f"Clustering used params: {cluster_params.get('method', 'unknown')}, "
                       f"mcs={cluster_params.get('min_cluster_size')}, ms={cluster_params.get('min_samples')}")

        if not clusters:
            logger.warning("No clusters found")
            return []

        trends = []
        for cluster in clusters[:max_trends]:
            try:
                trend = self._process_cluster(cluster)
                if trend and self._is_quality_trend(trend):
                    trends.append(trend)
            except Exception as e:
                logger.error(f"Error processing cluster {cluster.cluster_id}: {e}")
                continue

        logger.info(f"Generated {len(trends)} trends")
        return trends

    def _is_quality_trend(self, trend: dict) -> bool:
        """Check if a trend meets quality standards.

        Filters out trends with garbage names or descriptions.

        Args:
            trend: Trend dict

        Returns:
            True if trend is acceptable quality
        """
        name = trend.get("name", "").lower()

        # Reject trends with garbage patterns
        garbage_patterns = [
            " there trend",
            " with trend",
            "trend based on",
            "discussion around",
            "pattern identified",
            " & in hospitality",
            "hospitality movement",
            "refund",
            "receptionists trend",
        ]

        for pattern in garbage_patterns:
            if pattern in name:
                logger.debug(f"Filtering out low-quality trend: {trend.get('name')}")
                return False

        # Reject very short or very long names
        if len(name) < 5 or len(name) > 80:
            return False

        # Reject names that are just numbers or generic
        if name.replace(" ", "").isdigit():
            return False

        return True

    def _process_cluster(self, cluster: Cluster) -> dict | None:
        """Process a single cluster into a trend.

        Args:
            cluster: Cluster object

        Returns:
            Trend dict or None
        """
        # Get full content for cluster
        content_items = get_content_for_cluster(cluster)
        if not content_items:
            return None

        # Skip clusters that are just list articles (e.g., "Top 10 trends for 2025")
        if self._is_list_cluster(content_items):
            logger.debug(f"Skipping list-style cluster {cluster.cluster_id}")
            return None

        # Calculate metrics
        metrics = self.scorer.calculate_metrics(content_items)

        # Skip weak trends
        if metrics.strength_score < 0.1:
            return None

        # Extract sample texts for LLM
        sample_texts = [
            f"{item.get('title', '')}\n{item.get('content', '')[:500]}"
            for item in content_items[:10]
        ]

        # Generate insights
        if self.use_llm:
            try:
                insights = generate_trend_insights(
                    sample_texts=sample_texts,
                    metrics=get_metrics_dict(metrics),
                )
            except Exception as e:
                logger.warning(f"LLM generation failed, using fallback: {e}")
                insights = self._fallback_insights(content_items)
        else:
            insights = self._fallback_insights(content_items)

        # Extract sample quotes
        sample_quotes = [
            item.get("content", "")[:200]
            for item in content_items[:3]
            if item.get("content")
        ]

        # Determine region from content
        region = self._extract_region(content_items)

        # Determine audience segment
        audience_segment = self._extract_audience_segment(content_items, insights.get("topics", []))

        # Ensure we have a meaningful name
        trend_name = insights.get("name", "").strip()
        if not trend_name or trend_name.lower() in ["unnamed trend", "untitled", ""]:
            # Generate name from cluster data
            trend_name = self._generate_fallback_name(content_items, cluster)

        return {
            "cluster_id": str(cluster.cluster_id),
            "name": trend_name,
            "description": insights.get("description", ""),
            "why_it_matters": insights.get("why_it_matters", ""),
            "strength_score": metrics.strength_score,
            "white_space_score": metrics.white_space_score,
            "volume": metrics.volume,
            "engagement_score": metrics.avg_engagement,
            "sentiment_delta": metrics.avg_sentiment,
            "region": region,
            "audience_segment": audience_segment,
            "topics": insights.get("topics", []),
            "sample_quotes": sample_quotes,
            "source_content_ids": cluster.content_ids,
            "metrics": get_metrics_dict(metrics),
        }

    def _is_list_cluster(self, content_items: list[dict]) -> bool:
        """Check if cluster is just aggregated list articles.

        Args:
            content_items: Content items in cluster

        Returns:
            True if this looks like a list aggregation, not a real trend
        """
        list_indicators = [
            "top 10", "top 5", "top 20", "best 10", "best 5",
            "trends for 2024", "trends for 2025", "trends in 2024", "trends in 2025",
            "things to", "ways to", "tips for", "guide to",
            "list of", "roundup", "compilation"
        ]

        titles = [item.get("title", "").lower() for item in content_items]
        list_count = 0

        for title in titles:
            if any(indicator in title for indicator in list_indicators):
                list_count += 1

        # If more than half the cluster is list-style content, skip it
        if list_count > len(content_items) / 2:
            return True

        # Check if all content is from the same source (likely just one article)
        sources = set(item.get("source", "") for item in content_items)
        if len(sources) == 1 and len(content_items) < 5:
            return True

        return False

    def _fallback_insights(self, content_items: list[dict]) -> dict:
        """Generate fallback insights without LLM.

        Args:
            content_items: List of content dicts

        Returns:
            Dict with name, description, why_it_matters, topics
        """
        # Extract themes from content
        all_text = " ".join(
            f"{item.get('title', '')} {item.get('content', '')[:300]}"
            for item in content_items[:10]
        ).lower()

        # Hospitality-specific theme detection
        theme_patterns = {
            "Remote Work & Travel": ["digital nomad", "remote work", "workation", "work from", "coworking", "laptop"],
            "Sustainable Tourism": ["sustainable", "eco-friendly", "green hotel", "environment", "carbon", "eco-tourism"],
            "Wellness & Retreat": ["wellness", "spa", "meditation", "yoga", "retreat", "mindfulness", "health"],
            "Luxury Experiences": ["luxury", "five star", "premium", "exclusive", "high-end", "upscale"],
            "Budget Travel": ["budget", "affordable", "cheap", "hostel", "backpack", "low cost"],
            "Solo Travel": ["solo travel", "traveling alone", "solo trip", "single traveler"],
            "Family Vacations": ["family travel", "kids", "children", "family-friendly", "family vacation"],
            "Food & Culinary": ["restaurant", "food", "culinary", "dining", "cuisine", "chef", "foodie"],
            "Adventure Tourism": ["adventure", "hiking", "outdoor", "extreme", "safari", "trekking"],
            "City Exploration": ["city break", "urban", "sightseeing", "downtown", "nightlife"],
            "Beach & Resort": ["beach", "resort", "tropical", "island", "seaside", "coastal"],
            "Cultural Immersion": ["culture", "local experience", "authentic", "heritage", "tradition"],
            "Tech in Travel": ["app", "booking", "technology", "ai", "digital", "mobile"],
            "Short Getaways": ["weekend", "short trip", "quick getaway", "day trip", "mini break"],
        }

        # Find matching themes
        matched_themes = []
        for theme_name, keywords in theme_patterns.items():
            matches = sum(1 for kw in keywords if kw in all_text)
            if matches >= 2:
                matched_themes.append((theme_name, matches))

        # Sort by match count
        matched_themes.sort(key=lambda x: x[1], reverse=True)

        # Generate name from best matching theme
        if matched_themes:
            name = matched_themes[0][0]
            topics = [t[0].split()[0].lower() for t in matched_themes[:5]]
        else:
            # Generic but reasonable fallback
            num_items = len(content_items)
            name = f"Emerging Travel Pattern ({num_items} discussions)"
            topics = ["travel", "hospitality"]

        # Generate description
        if matched_themes:
            description = f"Trend identified from {len(content_items)} social discussions focusing on {name.lower()}. This pattern shows growing traveler interest in this area."
        else:
            description = f"Trend based on {len(content_items)} social mentions."

        why_it_matters = "This emerging pattern indicates shifting traveler preferences that hospitality businesses should monitor for potential opportunities."

        return {
            "name": name,
            "description": description,
            "why_it_matters": why_it_matters,
            "topics": topics,
        }

    def _generate_fallback_name(self, content_items: list[dict], cluster: Cluster) -> str:
        """Generate a fallback trend name when LLM fails.

        Args:
            content_items: List of content dicts
            cluster: Cluster object

        Returns:
            Meaningful trend name
        """
        # Try to extract from titles
        titles = [item.get("title", "") for item in content_items[:5] if item.get("title")]

        if titles:
            # Use first meaningful title, cleaned up
            first_title = titles[0][:50].strip()
            if len(first_title) > 10:
                # Clean up and format as trend name
                return f"{first_title}... Trend"

        # Extract keywords from content
        all_text = " ".join(
            item.get("title", "") + " " + item.get("content", "")[:200]
            for item in content_items[:5]
        ).lower()

        # Look for hospitality-specific themes
        themes = {
            "wellness": ["wellness", "spa", "meditation", "yoga", "health"],
            "luxury": ["luxury", "premium", "five star", "exclusive", "high-end"],
            "budget": ["budget", "affordable", "cheap", "hostel", "backpacker"],
            "sustainable": ["sustainable", "eco", "green", "environment"],
            "digital nomad": ["remote work", "digital nomad", "workation", "coworking"],
            "boutique": ["boutique", "design", "unique", "artisan"],
            "family travel": ["family", "kids", "children", "family-friendly"],
            "solo travel": ["solo", "solo travel", "alone", "single traveler"],
        }

        for theme_name, keywords in themes.items():
            if any(kw in all_text for kw in keywords):
                return f"{theme_name.title()} Travel Trend"

        # Use cluster size as a last resort
        return f"Travel Discussion Cluster ({cluster.size} sources)"

    def _extract_region(self, content_items: list[dict]) -> str | None:
        """Extract geographic region from content.

        Args:
            content_items: List of content dicts

        Returns:
            Region string or None
        """
        region_keywords = {
            "europe": ["europe", "paris", "london", "barcelona", "lisbon", "amsterdam", "rome", "berlin"],
            "asia": ["asia", "tokyo", "bangkok", "bali", "singapore", "hong kong", "seoul", "vietnam"],
            "north_america": ["usa", "new york", "los angeles", "miami", "san francisco", "canada", "mexico"],
            "south_america": ["brazil", "argentina", "colombia", "peru", "chile"],
            "middle_east": ["dubai", "abu dhabi", "qatar", "saudi"],
            "oceania": ["australia", "sydney", "melbourne", "new zealand"],
        }

        text = " ".join(
            f"{item.get('title', '')} {item.get('content', '')}"
            for item in content_items
        ).lower()

        region_scores = {}
        for region, keywords in region_keywords.items():
            score = sum(1 for kw in keywords if kw in text)
            if score > 0:
                region_scores[region] = score

        if region_scores:
            return max(region_scores, key=region_scores.get)
        return None

    def _extract_audience_segment(
        self,
        content_items: list[dict],
        topics: list[str],
    ) -> str:
        """Extract audience segment from content and topics.

        Args:
            content_items: List of content dicts
            topics: Extracted topics

        Returns:
            Audience segment string
        """
        segment_keywords = {
            "luxury": ["luxury", "premium", "five star", "5 star", "high-end", "exclusive"],
            "boutique": ["boutique", "design", "unique", "artisan", "independent"],
            "budget": ["budget", "cheap", "affordable", "hostel", "backpacker"],
            "business": ["business", "corporate", "conference", "meeting"],
            "family": ["family", "kids", "children", "resort"],
            "wellness": ["wellness", "spa", "health", "retreat", "yoga"],
            "adventure": ["adventure", "hiking", "outdoor", "safari", "extreme"],
            "eco": ["eco", "sustainable", "green", "environment", "nature"],
        }

        text = " ".join(
            f"{item.get('title', '')} {item.get('content', '')} {' '.join(topics)}"
            for item in content_items
        ).lower()

        segment_scores = {}
        for segment, keywords in segment_keywords.items():
            score = sum(1 for kw in keywords if kw in text)
            if score > 0:
                segment_scores[segment] = score

        if segment_scores:
            return max(segment_scores, key=segment_scores.get)
        return "general"

    def save_trends(self, trends: list[dict]) -> int:
        """Save generated trends to database.

        Args:
            trends: List of trend dicts

        Returns:
            Number of trends saved
        """
        db = SessionLocal()
        saved = 0

        try:
            for trend in trends:
                db_trend = TrendSignalModel(
                    name=trend["name"],
                    description=trend["description"],
                    why_it_matters=trend["why_it_matters"],
                    strength_score=trend["strength_score"],
                    white_space_score=trend["white_space_score"],
                    volume=trend["volume"],
                    engagement_score=trend["engagement_score"],
                    sentiment_delta=trend["sentiment_delta"],
                    region=trend.get("region"),
                    audience_segment=trend.get("audience_segment", "general"),
                    topics=trend.get("topics", []),
                    source_content_ids=trend.get("source_content_ids", []),
                    sample_quotes=trend.get("sample_quotes", []),
                    cluster_id=trend.get("cluster_id"),
                    metadata_json=trend.get("metrics", {}),
                )
                db.add(db_trend)
                saved += 1

            db.commit()
            logger.info(f"Saved {saved} trends to database")

        except Exception as e:
            db.rollback()
            logger.error(f"Error saving trends: {e}")
            raise

        finally:
            db.close()

        return saved

    def get_trends(
        self,
        limit: int = 20,
        region: str | None = None,
        audience_segment: str | None = None,
        min_strength: float = 0,
        days_back: int = 7,
    ) -> list[dict]:
        """Get trends from database with filters.

        Prefers recent trends, but falls back to latest available if none recent.

        Args:
            limit: Maximum trends to return
            region: Filter by region
            audience_segment: Filter by audience
            min_strength: Minimum strength score
            days_back: Prefer trends updated within this many days (default 7)

        Returns:
            List of trend dicts
        """
        db = SessionLocal()
        try:
            # Try recent trends first
            cutoff_date = datetime.utcnow() - timedelta(days=days_back)

            query = db.query(TrendSignalModel).filter(
                TrendSignalModel.strength_score >= min_strength,
                TrendSignalModel.last_updated >= cutoff_date,
            )

            if region:
                query = query.filter(TrendSignalModel.region == region)
            if audience_segment:
                query = query.filter(TrendSignalModel.audience_segment == audience_segment)

            trends = query.order_by(
                TrendSignalModel.last_updated.desc(),
                TrendSignalModel.strength_score.desc(),
            ).limit(limit).all()

            # Fallback: if no recent trends, get latest regardless of age
            if not trends:
                query = db.query(TrendSignalModel).filter(
                    TrendSignalModel.strength_score >= min_strength,
                )
                if region:
                    query = query.filter(TrendSignalModel.region == region)
                if audience_segment:
                    query = query.filter(TrendSignalModel.audience_segment == audience_segment)

                trends = query.order_by(
                    TrendSignalModel.last_updated.desc(),
                    TrendSignalModel.strength_score.desc(),
                ).limit(limit).all()

            return [self._model_to_dict(t) for t in trends]

        finally:
            db.close()

    def _model_to_dict(self, model: TrendSignalModel) -> dict:
        """Convert TrendSignalModel to dict."""
        return {
            "id": model.id,
            "name": model.name,
            "description": model.description,
            "why_it_matters": model.why_it_matters,
            "strength_score": model.strength_score,
            "white_space_score": model.white_space_score,
            "volume": model.volume,
            "engagement_score": model.engagement_score,
            "sentiment_delta": model.sentiment_delta,
            "region": model.region,
            "audience_segment": model.audience_segment,
            "topics": model.topics or [],
            "sample_quotes": model.sample_quotes or [],
            "source_content_ids": model.source_content_ids or [],
            "first_seen": model.first_seen.isoformat() if model.first_seen else None,
            "last_updated": model.last_updated.isoformat() if model.last_updated else None,
        }


def generate_social_pulse(
    source_types: list[str] | None = None,
    days_back: int = 30,
    save: bool = True,
) -> list[dict]:
    """Convenience function to generate Social Pulse trends.

    Args:
        source_types: Filter by source types
        days_back: Days of content to analyze
        save: Whether to save to database

    Returns:
        List of generated trends
    """
    service = SocialPulseService(days_back=days_back)
    trends = service.generate_trends(source_types=source_types)

    if save and trends:
        service.save_trends(trends)

    return trends
