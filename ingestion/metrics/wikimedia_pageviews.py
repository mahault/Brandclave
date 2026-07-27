"""Wikimedia pageviews — daily destination interest per city.

Free, no key, data back to 2015; the best available substitute for Google
Trends (plan §4.3). Wikimedia asks for a descriptive User-Agent with contact
info, so this scraper overrides the browser UA. Cities and lookback come from
configs/sources.yaml.
"""

import logging
from datetime import datetime, timedelta
from urllib.parse import quote

from data_models.demand_metric import DemandMetricCreate
from ingestion.metrics.base import MetricScraper
from ingestion.registry import get_source_config

logger = logging.getLogger(__name__)

API_BASE = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article"
USER_AGENT = "BrandClaveDemandBot/1.0 (https://github.com/mahault/Brandclave; mahault.albarracin@gmail.com)"

# Pageview aggregates lag real time; stay two days behind to avoid empty tails
DATA_LAG_DAYS = 2


class WikimediaPageviewsScraper(MetricScraper):
    """Daily Wikipedia pageviews per destination article."""

    source_name = "wikimedia_pageviews"

    def scrape(self) -> list[DemandMetricCreate]:
        cfg = get_source_config(self.source_name)
        cities = cfg.get("cities", [])
        days_back = int(cfg.get("days_back", 30))
        metric_name = cfg.get("metric", "wikipedia_pageviews")

        end = datetime.utcnow().date() - timedelta(days=DATA_LAG_DAYS)
        start = end - timedelta(days=days_back)

        points: list[DemandMetricCreate] = []
        for entry in cities:
            city, country, article = entry.get("city"), entry.get("country"), entry.get("article")
            if not city or not article:
                continue

            url = (
                f"{API_BASE}/en.wikipedia/all-access/user/{quote(article, safe='')}"
                f"/daily/{start:%Y%m%d}/{end:%Y%m%d}"
            )
            response = self.fetch(url, headers={"User-Agent": USER_AGENT})
            if response is None:
                logger.warning(f"Wikimedia pageviews failed for {city}, continuing")
                continue

            try:
                rows = response.json().get("items", [])
            except ValueError:
                logger.warning(f"Wikimedia returned non-JSON for {city}")
                continue

            for row in rows:
                timestamp = row.get("timestamp", "")  # YYYYMMDDHH
                try:
                    day = datetime.strptime(timestamp[:8], "%Y%m%d")
                except ValueError:
                    continue
                points.append(
                    DemandMetricCreate(
                        source=self.source_name,
                        city=city,
                        country=country,
                        metric=metric_name,
                        date=day,
                        value=float(row.get("views", 0)),
                        metadata={"article": article, "project": "en.wikipedia"},
                    )
                )

        logger.info(f"Wikimedia: collected {len(points)} daily points for {len(cities)} cities")
        return points


def scrape_wikimedia_pageviews() -> dict:
    """Run the Wikimedia pageviews scraper standalone."""
    with WikimediaPageviewsScraper() as scraper:
        return scraper.run()
