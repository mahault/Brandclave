"""Metrics collection for BrandClave Aggregator.

Provides system metrics, scraper status, and error tracking.
"""

import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy import func

from db.database import SessionLocal
from db.models import (
    RawContentModel,
    ProcessingJobModel,
    TrendSignalModel,
    HotelierMoveModel,
    PropertyFeaturesModel,
)

logger = logging.getLogger(__name__)


@dataclass
class ScraperMetrics:
    """Metrics for a single scraper."""

    source: str
    total_items: int
    items_last_24h: int
    items_last_7d: int
    last_run_at: Optional[str]
    last_run_status: Optional[str]
    last_run_items: int
    error_rate_24h: float


@dataclass
class SystemMetrics:
    """Overall system metrics."""

    total_content: int
    content_by_source: dict
    content_by_type: dict
    processed_content: int
    unprocessed_content: int
    embeddings_count: int
    trends_count: int
    moves_count: int
    properties_count: int
    cache_stats: dict
    scheduler_stats: dict


class MetricsCollector:
    """Collects and aggregates system metrics."""

    def __init__(self):
        """Initialize metrics collector."""
        self.db = None

    def __enter__(self):
        """Context manager entry."""
        self.db = SessionLocal()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.db:
            self.db.close()

    def _ensure_db(self):
        """Ensure database session exists."""
        if self.db is None:
            self.db = SessionLocal()

    def get_scraper_metrics(self, source: str) -> ScraperMetrics:
        """Get metrics for a specific scraper.

        Args:
            source: Source name

        Returns:
            ScraperMetrics object
        """
        self._ensure_db()

        now = datetime.utcnow()
        day_ago = now - timedelta(days=1)
        week_ago = now - timedelta(days=7)

        # Total items from this source
        total = self.db.query(func.count(RawContentModel.id)).filter(
            RawContentModel.source == source
        ).scalar() or 0

        # Items in last 24h
        last_24h = self.db.query(func.count(RawContentModel.id)).filter(
            RawContentModel.source == source,
            RawContentModel.scraped_at >= day_ago,
        ).scalar() or 0

        # Items in last 7d
        last_7d = self.db.query(func.count(RawContentModel.id)).filter(
            RawContentModel.source == source,
            RawContentModel.scraped_at >= week_ago,
        ).scalar() or 0

        # Last job for this source
        last_job = self.db.query(ProcessingJobModel).filter(
            ProcessingJobModel.source == source,
            ProcessingJobModel.job_type == "scrape",
        ).order_by(ProcessingJobModel.started_at.desc()).first()

        # Error rate in last 24h
        recent_jobs = self.db.query(ProcessingJobModel).filter(
            ProcessingJobModel.source == source,
            ProcessingJobModel.job_type == "scrape",
            ProcessingJobModel.started_at >= day_ago,
        ).all()

        error_count = sum(1 for j in recent_jobs if j.status == "failed")
        error_rate = error_count / len(recent_jobs) if recent_jobs else 0.0

        return ScraperMetrics(
            source=source,
            total_items=total,
            items_last_24h=last_24h,
            items_last_7d=last_7d,
            last_run_at=last_job.completed_at.isoformat() if last_job and last_job.completed_at else None,
            last_run_status=last_job.status if last_job else None,
            last_run_items=last_job.items_processed if last_job else 0,
            error_rate_24h=round(error_rate, 2),
        )

    def get_all_scraper_metrics(self) -> list[ScraperMetrics]:
        """Get metrics for all scrapers.

        Returns:
            List of ScraperMetrics
        """
        self._ensure_db()

        # Get unique sources
        sources = self.db.query(RawContentModel.source).distinct().all()
        return [self.get_scraper_metrics(s[0]) for s in sources]

    def get_system_metrics(self) -> SystemMetrics:
        """Get overall system metrics.

        Returns:
            SystemMetrics object
        """
        self._ensure_db()

        # Content counts
        total = self.db.query(func.count(RawContentModel.id)).scalar() or 0
        processed = self.db.query(func.count(RawContentModel.id)).filter(
            RawContentModel.is_processed == True
        ).scalar() or 0

        # By source
        by_source = {}
        source_counts = self.db.query(
            RawContentModel.source,
            func.count(RawContentModel.id),
        ).group_by(RawContentModel.source).all()
        for source, count in source_counts:
            by_source[source] = count

        # By type
        by_type = {}
        type_counts = self.db.query(
            RawContentModel.source_type,
            func.count(RawContentModel.id),
        ).group_by(RawContentModel.source_type).all()
        for stype, count in type_counts:
            by_type[stype] = count

        # Trends, moves, properties
        trends = self.db.query(func.count(TrendSignalModel.id)).scalar() or 0
        moves = self.db.query(func.count(HotelierMoveModel.id)).scalar() or 0
        properties = self.db.query(func.count(PropertyFeaturesModel.id)).scalar() or 0

        # Embeddings count (items with embedding_id)
        embeddings = self.db.query(func.count(RawContentModel.id)).filter(
            RawContentModel.embedding_id.isnot(None)
        ).scalar() or 0

        # Cache stats
        cache_stats = self._get_cache_stats()

        # Scheduler stats
        scheduler_stats = self._get_scheduler_stats()

        return SystemMetrics(
            total_content=total,
            content_by_source=by_source,
            content_by_type=by_type,
            processed_content=processed,
            unprocessed_content=total - processed,
            embeddings_count=embeddings,
            trends_count=trends,
            moves_count=moves,
            properties_count=properties,
            cache_stats=cache_stats,
            scheduler_stats=scheduler_stats,
        )

    def _get_cache_stats(self) -> dict:
        """Get cache statistics."""
        try:
            from cache.redis_cache import get_cache
            cache = get_cache()
            return cache.get_stats()
        except Exception as e:
            return {"status": "unavailable", "error": str(e)}

    def _get_scheduler_stats(self) -> dict:
        """Get scheduler statistics."""
        try:
            from scheduler.scheduler import get_scheduler
            scheduler = get_scheduler()
            if not scheduler.is_available:
                return {"status": "unavailable"}

            jobs = scheduler.get_jobs()
            return {
                "status": "running" if scheduler.is_running else "stopped",
                "available": scheduler.is_available,
                "total_jobs": len(jobs),
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def get_recent_errors(self, limit: int = 20) -> list[dict]:
        """Get recent job errors.

        Args:
            limit: Maximum errors to return

        Returns:
            List of error dicts
        """
        self._ensure_db()

        errors = self.db.query(ProcessingJobModel).filter(
            ProcessingJobModel.status == "failed",
        ).order_by(ProcessingJobModel.completed_at.desc()).limit(limit).all()

        return [
            {
                "id": e.id,
                "job_type": e.job_type,
                "source": e.source,
                "error": e.error_message,
                "timestamp": e.completed_at.isoformat() if e.completed_at else None,
            }
            for e in errors
        ]

    def get_recent_activity(self, hours: int = 24) -> dict:
        """Get recent activity summary.

        Args:
            hours: Hours to look back

        Returns:
            Activity summary dict
        """
        self._ensure_db()

        cutoff = datetime.utcnow() - timedelta(hours=hours)

        # New content
        new_content = self.db.query(func.count(RawContentModel.id)).filter(
            RawContentModel.scraped_at >= cutoff
        ).scalar() or 0

        # Processed content
        processed = self.db.query(func.count(RawContentModel.id)).filter(
            RawContentModel.is_processed == True,
            RawContentModel.scraped_at >= cutoff,
        ).scalar() or 0

        # Jobs run
        jobs = self.db.query(ProcessingJobModel).filter(
            ProcessingJobModel.started_at >= cutoff
        ).all()

        successful_jobs = sum(1 for j in jobs if j.status == "completed")
        failed_jobs = sum(1 for j in jobs if j.status == "failed")

        return {
            "period_hours": hours,
            "new_content": new_content,
            "processed_content": processed,
            "total_jobs": len(jobs),
            "successful_jobs": successful_jobs,
            "failed_jobs": failed_jobs,
        }
