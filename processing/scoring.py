"""Trend scoring module for calculating strength and white-space scores."""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

from db.database import SessionLocal
from db.models import RawContentModel

logger = logging.getLogger(__name__)


@dataclass
class TrendMetrics:
    """Calculated metrics for a trend cluster."""

    # Volume metrics
    volume: int
    unique_sources: int

    # Engagement metrics
    total_score: int  # Reddit score, YouTube views
    total_comments: int
    avg_engagement: float

    # Sentiment metrics
    avg_sentiment: float
    sentiment_std: float
    positive_ratio: float

    # Recency metrics
    avg_age_hours: float
    newest_age_hours: float

    # Composite scores
    strength_score: float  # 0-1 overall trend strength
    white_space_score: float  # 0-1 demand vs supply gap


class TrendScorer:
    """Calculate trend strength and white-space scores."""

    def __init__(
        self,
        volume_weight: float = 0.25,
        engagement_weight: float = 0.25,
        sentiment_weight: float = 0.20,
        recency_weight: float = 0.30,
    ):
        """Initialize scorer with configurable weights.

        Args:
            volume_weight: Weight for volume in strength score
            engagement_weight: Weight for engagement
            sentiment_weight: Weight for sentiment
            recency_weight: Weight for recency
        """
        self.volume_weight = volume_weight
        self.engagement_weight = engagement_weight
        self.sentiment_weight = sentiment_weight
        self.recency_weight = recency_weight

    def calculate_metrics(self, content_items: list[dict]) -> TrendMetrics:
        """Calculate all metrics for a cluster of content.

        Args:
            content_items: List of content dicts with metadata

        Returns:
            TrendMetrics object
        """
        if not content_items:
            return self._empty_metrics()

        now = datetime.utcnow()

        # Volume metrics
        volume = len(content_items)
        unique_sources = len(set(item.get("source") for item in content_items))

        # Engagement metrics
        scores = []
        comments = []
        for item in content_items:
            meta = item.get("metadata", {})
            # Reddit metrics
            if "score" in meta:
                scores.append(meta.get("score", 0))
            if "num_comments" in meta:
                comments.append(meta.get("num_comments", 0))
            # YouTube metrics
            if "views" in meta:
                scores.append(meta.get("views", 0))

        total_score = sum(scores) if scores else 0
        total_comments = sum(comments) if comments else 0
        avg_engagement = (total_score + total_comments * 10) / volume if volume > 0 else 0

        # Sentiment metrics
        sentiments = [
            item.get("sentiment_score", 0)
            for item in content_items
            if item.get("sentiment_score") is not None
        ]

        avg_sentiment = np.mean(sentiments) if sentiments else 0
        sentiment_std = np.std(sentiments) if len(sentiments) > 1 else 0
        positive_ratio = sum(1 for s in sentiments if s > 0.1) / len(sentiments) if sentiments else 0.5

        # Recency metrics
        ages = []
        for item in content_items:
            scraped_at = item.get("scraped_at")
            if scraped_at:
                age = (now - scraped_at).total_seconds() / 3600  # hours
                ages.append(age)

        avg_age_hours = np.mean(ages) if ages else 168  # default 1 week
        newest_age_hours = min(ages) if ages else 168

        # Calculate composite scores
        strength_score = self._calculate_strength(
            volume=volume,
            avg_engagement=avg_engagement,
            avg_sentiment=avg_sentiment,
            avg_age_hours=avg_age_hours,
        )

        white_space_score = self._calculate_white_space(
            volume=volume,
            avg_sentiment=avg_sentiment,
            unique_sources=unique_sources,
        )

        return TrendMetrics(
            volume=volume,
            unique_sources=unique_sources,
            total_score=total_score,
            total_comments=total_comments,
            avg_engagement=avg_engagement,
            avg_sentiment=avg_sentiment,
            sentiment_std=sentiment_std,
            positive_ratio=positive_ratio,
            avg_age_hours=avg_age_hours,
            newest_age_hours=newest_age_hours,
            strength_score=strength_score,
            white_space_score=white_space_score,
        )

    def _calculate_strength(
        self,
        volume: int,
        avg_engagement: float,
        avg_sentiment: float,
        avg_age_hours: float,
    ) -> float:
        """Calculate overall trend strength score (0-1).

        Args:
            volume: Number of items in cluster
            avg_engagement: Average engagement score
            avg_sentiment: Average sentiment (-1 to 1)
            avg_age_hours: Average age in hours

        Returns:
            Strength score 0-1
        """
        # Volume score: log scale, max at ~100 items
        volume_score = min(1.0, np.log1p(volume) / np.log1p(100))

        # Engagement score: log scale
        engagement_score = min(1.0, np.log1p(avg_engagement) / np.log1p(10000))

        # Sentiment score: transform from (-1,1) to (0,1), with neutral = 0.5
        sentiment_score = (avg_sentiment + 1) / 2

        # Recency score: exponential decay, half-life of 48 hours
        recency_score = np.exp(-avg_age_hours / 96)  # ~50% at 48h, ~25% at 96h

        # Weighted combination
        strength = (
            self.volume_weight * volume_score +
            self.engagement_weight * engagement_score +
            self.sentiment_weight * sentiment_score +
            self.recency_weight * recency_score
        )

        return min(1.0, max(0.0, strength))

    def _calculate_white_space(
        self,
        volume: int,
        avg_sentiment: float,
        unique_sources: int,
    ) -> float:
        """Calculate white-space score (demand vs supply gap).

        High white-space = high demand signals but low supply.

        Args:
            volume: Number of demand signals
            avg_sentiment: Sentiment (positive = desire/demand)
            unique_sources: Source diversity

        Returns:
            White-space score 0-1
        """
        # Demand signal: volume * positive sentiment
        demand_factor = min(1.0, volume / 50) * max(0, (avg_sentiment + 1) / 2)

        # Source diversity factor: more sources = stronger signal
        diversity_factor = min(1.0, unique_sources / 5)

        # For now, assume supply is inversely related to how niche the topic is
        # (fewer sources = more niche = potential white space)
        supply_factor = 1 - diversity_factor * 0.3

        # White space = high demand + limited supply
        white_space = demand_factor * (0.7 + 0.3 * supply_factor)

        return min(1.0, max(0.0, white_space))

    def _empty_metrics(self) -> TrendMetrics:
        """Return empty metrics object."""
        return TrendMetrics(
            volume=0,
            unique_sources=0,
            total_score=0,
            total_comments=0,
            avg_engagement=0,
            avg_sentiment=0,
            sentiment_std=0,
            positive_ratio=0.5,
            avg_age_hours=168,
            newest_age_hours=168,
            strength_score=0,
            white_space_score=0,
        )


def score_cluster(content_items: list[dict]) -> TrendMetrics:
    """Convenience function to score a cluster.

    Args:
        content_items: List of content dicts

    Returns:
        TrendMetrics object
    """
    scorer = TrendScorer()
    return scorer.calculate_metrics(content_items)


def get_metrics_dict(metrics: TrendMetrics) -> dict:
    """Convert TrendMetrics to dictionary.

    Args:
        metrics: TrendMetrics object

    Returns:
        Dictionary representation
    """
    return {
        "volume": metrics.volume,
        "unique_sources": metrics.unique_sources,
        "total_score": metrics.total_score,
        "total_comments": metrics.total_comments,
        "avg_engagement": round(metrics.avg_engagement, 2),
        "avg_sentiment": round(metrics.avg_sentiment, 3),
        "sentiment_std": round(metrics.sentiment_std, 3),
        "positive_ratio": round(metrics.positive_ratio, 2),
        "avg_age_hours": round(metrics.avg_age_hours, 1),
        "newest_age_hours": round(metrics.newest_age_hours, 1),
        "strength_score": round(metrics.strength_score, 3),
        "white_space_score": round(metrics.white_space_score, 3),
    }
