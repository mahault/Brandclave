"""Inside Airbnb — short-term-rental supply and activity per city.

Inside Airbnb publishes quarterly snapshots per city under a Creative Commons
BY 4.0 licence (attribution required; commercial reuse is permitted by the
licence text, but the project asks per-city — the registry note keeps that
flag visible). The `visualisations/listings.csv` summary (a few MB) carries
what we need without the full listings dump:

- listing count, and the share that are entire homes
- median nightly price
- mean reviews per month, the community's standard occupancy proxy

The snapshot date in the URL becomes the metric date, so repeated runs upsert
rather than duplicate. Cities are matched by their URL slug.
"""

import csv
import io
import logging
import re
import statistics
from datetime import datetime

from data_models.demand_metric import DemandMetricCreate
from ingestion.metrics.base import MetricScraper
from ingestion.registry import get_source_config

logger = logging.getLogger(__name__)

INDEX_URL = "https://insideairbnb.com/get-the-data/"
USER_AGENT = "BrandClaveDemandBot/1.0 (https://github.com/mahault/Brandclave; mahault.albarracin@gmail.com)"
LINK_RE = re.compile(r"https://data\.insideairbnb\.com/([^\"']+?)/(\d{4}-\d{2}-\d{2})/visualisations/listings\.csv")


class InsideAirbnbScraper(MetricScraper):
    """Quarterly Airbnb supply/activity metrics for configured cities."""

    source_name = "inside_airbnb"

    def scrape(self) -> list[DemandMetricCreate]:
        cfg = get_source_config(self.source_name)
        wanted = cfg.get("cities", [])  # [{city, country, slug}]
        if not wanted:
            logger.warning("inside_airbnb: no cities configured")
            return []

        index = self.fetch(INDEX_URL, headers={"User-Agent": USER_AGENT})
        if index is None:
            logger.warning("Inside Airbnb index unreachable, skipping run")
            return []

        # latest snapshot per path (the index lists every archived date)
        latest: dict[str, tuple[str, str]] = {}
        for match in LINK_RE.finditer(index.text):
            path, snapshot = match.group(1), match.group(2)
            if path not in latest or snapshot > latest[path][0]:
                latest[path] = (snapshot, match.group(0))

        points: list[DemandMetricCreate] = []
        for entry in wanted:
            slug = entry.get("slug", "").lower()
            city, country = entry.get("city"), entry.get("country")
            hit = next(((snap, url) for path, (snap, url) in latest.items() if path.lower().endswith("/" + slug)), None)
            if hit is None:
                logger.warning(f"Inside Airbnb: no snapshot found for slug '{slug}'")
                continue
            snapshot, url = hit
            response = self.fetch(url, headers={"User-Agent": USER_AGENT})
            if response is None:
                logger.warning(f"Inside Airbnb: download failed for {city}")
                continue
            metrics = self._summarise(response.text)
            if not metrics:
                continue
            day = datetime.strptime(snapshot, "%Y-%m-%d")
            for metric, value in metrics.items():
                points.append(
                    DemandMetricCreate(
                        source=self.source_name,
                        city=city,
                        country=country,
                        metric=metric,
                        date=day,
                        value=value,
                        metadata={"snapshot": snapshot, "url": url, "licence": "CC BY 4.0 (Inside Airbnb)"},
                    )
                )
        logger.info(f"Inside Airbnb: {len(points)} metric points for {len(wanted)} cities")
        return points

    @staticmethod
    def _summarise(csv_text: str) -> dict[str, float]:
        reader = csv.DictReader(io.StringIO(csv_text))
        n = entire = 0
        prices: list[float] = []
        rpm: list[float] = []
        for row in reader:
            n += 1
            if (row.get("room_type") or "").startswith("Entire"):
                entire += 1
            try:
                price = float(str(row.get("price", "")).replace("$", "").replace(",", ""))
                if 0 < price < 5000:
                    prices.append(price)
            except ValueError:
                pass
            try:
                value = float(row.get("reviews_per_month") or 0)
                if value > 0:
                    rpm.append(value)
            except ValueError:
                pass
        if n == 0:
            return {}
        return {
            "airbnb_listings": float(n),
            "airbnb_entire_home_share": round(entire / n, 4),
            "airbnb_median_price": float(statistics.median(prices)) if prices else 0.0,
            "airbnb_reviews_per_month": round(statistics.fmean(rpm), 4) if rpm else 0.0,
        }


def scrape_inside_airbnb() -> dict:
    with InsideAirbnbScraper() as scraper:
        return scraper.run()
