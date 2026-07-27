"""Base class for metric sources.

A metric scraper produces DemandMetricCreate points (a numeric time series per
city) instead of RawContent text. It reuses BaseScraper.run() — job tracking,
error isolation and the summary contract stay identical — by overriding what
scrape() returns and how save_content() persists it.
"""

import logging

from data_models.demand_metric import DemandMetricCreate
from data_models.raw_content import SourceType
from db.database import SessionLocal
from db.models import DemandMetricModel
from ingestion.base_scraper import BaseScraper

logger = logging.getLogger(__name__)


class MetricScraper(BaseScraper):
    """Base for sources that yield demand metric points."""

    source_name = "base_metric"
    source_type = SourceType.METRIC

    def scrape(self) -> list[DemandMetricCreate]:  # type: ignore[override]
        """Collect metric points. Subclasses must implement."""
        raise NotImplementedError

    def save_content(self, items: list[DemandMetricCreate]) -> int:  # type: ignore[override]
        """Upsert metric points on (source, city, metric, date)."""
        if not items:
            return 0

        db = SessionLocal()
        saved = 0
        try:
            for item in items:
                existing = (
                    db.query(DemandMetricModel)
                    .filter(
                        DemandMetricModel.source == item.source,
                        DemandMetricModel.city == item.city,
                        DemandMetricModel.metric == item.metric,
                        DemandMetricModel.date == item.date,
                    )
                    .first()
                )
                if existing is not None:
                    if existing.value != item.value:
                        existing.value = item.value
                        existing.metadata_json = item.metadata
                        saved += 1
                    continue

                db.add(
                    DemandMetricModel(
                        source=item.source,
                        city=item.city,
                        country=item.country,
                        metric=item.metric,
                        date=item.date,
                        value=item.value,
                        metadata_json=item.metadata,
                    )
                )
                saved += 1

            db.commit()
            logger.info(f"{self.source_name}: upserted {saved} of {len(items)} metric points")
            return saved
        except Exception as exc:
            db.rollback()
            logger.error(f"{self.source_name}: failed to save metrics: {exc}")
            return saved
        finally:
            db.close()
