"""Social Pulse service - Trend generation and management."""

import logging
from datetime import datetime
from typing import Any

from db.database import SessionLocal
from db.models import TrendSignalModel, RawContentModel
from processing.clustering import ContentClusterer, Cluster, get_content_for_cluster
from processing.scoring import TrendScorer, get_metrics_dict
from processing.llm_utils import generate_trend_insights, get_llm

logger = logging.getLogger(__name__)


class SocialPulseService:
    """Service for generating and managing Social Pulse trends."""

    def __init__(
        self,
        min_cluster_size: int = 3,
        days_back: int = 30,
        use_llm: bool = True,
    ):
        """Initialize Social Pulse service.

        Args:
            min_cluster_size: Minimum content items to form a trend
            days_back: Days of content to analyze
            use_llm: Whether to use LLM for generating insights
        """
        self.clusterer = ContentClusterer(min_cluster_size=min_cluster_size)
        self.scorer = TrendScorer()
        self.days_back = days_back
        self.use_llm = use_llm

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

        # Run clustering
        clusters = self.clusterer.cluster_content(
            source_types=source_types,
            days_back=self.days_back,
        )

        if not clusters:
            logger.warning("No clusters found")
            return []

        trends = []
        for cluster in clusters[:max_trends]:
            try:
                trend = self._process_cluster(cluster)
                if trend:
                    trends.append(trend)
            except Exception as e:
                logger.error(f"Error processing cluster {cluster.cluster_id}: {e}")
                continue

        logger.info(f"Generated {len(trends)} trends")
        return trends

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

        return {
            "cluster_id": str(cluster.cluster_id),
            "name": insights.get("name", "Unnamed Trend"),
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

    def _fallback_insights(self, content_items: list[dict]) -> dict:
        """Generate fallback insights without LLM.

        Args:
            content_items: List of content dicts

        Returns:
            Dict with name, description, why_it_matters, topics
        """
        # Extract common words from titles for name
        titles = [item.get("title", "") for item in content_items if item.get("title")]
        words = " ".join(titles).lower().split()

        # Simple word frequency
        word_freq = {}
        stop_words = {"the", "a", "an", "in", "on", "at", "to", "for", "of", "and", "or", "is", "are", "was", "were"}
        for word in words:
            word = word.strip(".,!?\"'()[]")
            if len(word) > 3 and word not in stop_words:
                word_freq[word] = word_freq.get(word, 0) + 1

        # Top words as topics
        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        topics = [w[0] for w in top_words]

        # Generate name from top words
        name = " ".join(topics[:3]).title() if topics else "Emerging Trend"

        return {
            "name": name,
            "description": f"Trend based on {len(content_items)} social mentions.",
            "why_it_matters": "This emerging topic shows growing interest in the hospitality space.",
            "topics": topics,
        }

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
    ) -> list[dict]:
        """Get trends from database with filters.

        Args:
            limit: Maximum trends to return
            region: Filter by region
            audience_segment: Filter by audience
            min_strength: Minimum strength score

        Returns:
            List of trend dicts
        """
        db = SessionLocal()
        try:
            query = db.query(TrendSignalModel).filter(
                TrendSignalModel.strength_score >= min_strength
            )

            if region:
                query = query.filter(TrendSignalModel.region == region)
            if audience_segment:
                query = query.filter(TrendSignalModel.audience_segment == audience_segment)

            trends = query.order_by(
                TrendSignalModel.strength_score.desc()
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
