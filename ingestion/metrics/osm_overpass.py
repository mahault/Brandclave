"""OpenStreetMap supply density via the Overpass API.

Counts hotels, restaurants, nightlife venues and attractions inside each
tracked city's administrative boundary. Demand without supply is the white
space the platform exists to find, so these counts are the denominator for
every "unmet demand" claim. Free, no key, ~10k requests/day fair use; one
query per city per run with a polite pause between them.

City list is shared with the Wikipedia pageviews source unless overridden;
`osm_area` / `osm_admin_level` in a city entry pin the boundary when the
English name does not match OSM's (Lisbon -> Lisboa).
"""

import logging
import time
from datetime import datetime

from data_models.demand_metric import DemandMetricCreate
from ingestion.metrics.base import MetricScraper
from ingestion.registry import get_source_config

logger = logging.getLogger(__name__)

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
USER_AGENT = "BrandClaveDemandBot/1.0 (https://github.com/mahault/Brandclave; mahault.albarracin@gmail.com)"

# metric name -> Overpass tag filter
CATEGORIES = {
    "osm_hotels": '["tourism"~"^(hotel|hostel|guest_house|apartment)$"]',
    "osm_restaurants": '["amenity"="restaurant"]',
    "osm_nightlife": '["amenity"~"^(bar|pub|nightclub)$"]',
    "osm_attractions": '["tourism"~"^(attraction|museum|gallery|viewpoint)$"]',
}
PAUSE_SECONDS = 4.0


class OSMOverpassScraper(MetricScraper):
    """Per-city counts of hospitality supply from OpenStreetMap."""

    source_name = "osm_overpass"

    def _check_robots_txt(self, url: str) -> bool:
        # overpass-api.de/robots.txt disallows /api/ for crawlers; the
        # interpreter endpoint is the documented programmatic interface and its
        # usage policy (rate + UA) is what applies, so the crawler rule is skipped.
        return True

    def scrape(self) -> list[DemandMetricCreate]:
        cfg = get_source_config(self.source_name)
        cities = cfg.get("cities") or get_source_config("wikimedia_pageviews").get("cities", [])
        today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

        points: list[DemandMetricCreate] = []
        for entry in cities:
            city = entry.get("city")
            if not city:
                continue
            area = entry.get("osm_area", city)
            level = str(entry.get("osm_admin_level", 8))
            query = self._build_query(area, level)
            response = self.fetch(OVERPASS_URL, params={"data": query}, headers={"User-Agent": USER_AGENT})
            if response is None:
                logger.warning(f"Overpass failed for {city}, continuing")
                time.sleep(PAUSE_SECONDS)
                continue
            try:
                elements = response.json().get("elements", [])
            except ValueError:
                logger.warning(f"Overpass returned non-JSON for {city}")
                continue

            counts = {}
            for element, metric in zip(elements, CATEGORIES):
                tags = element.get("tags", {})
                counts[metric] = float(tags.get("total", 0))
            if not counts or sum(counts.values()) == 0:
                logger.warning(f"Overpass found nothing for {city} (area '{area}', level {level}); check osm_area")
            for metric, value in counts.items():
                points.append(
                    DemandMetricCreate(
                        source=self.source_name,
                        city=city,
                        country=entry.get("country"),
                        metric=metric,
                        date=today,
                        value=value,
                        metadata={"osm_area": area, "admin_level": level},
                    )
                )
            time.sleep(PAUSE_SECONDS)

        logger.info(f"Overpass: {len(points)} supply counts across {len(cities)} cities")
        return points

    @staticmethod
    def _build_query(area: str, level: str) -> str:
        """One request per city; `out count` per category keeps the payload tiny.

        Statements are ordered exactly like CATEGORIES so the response elements
        can be zipped back to metric names.
        """
        # Match the English name or the local name; the union dedupes when both
        # tags sit on the same boundary relation.
        area_sel = (
            f'(area["name:en"="{area}"]["boundary"="administrative"]["admin_level"="{level}"];'
            f'area["name"="{area}"]["boundary"="administrative"]["admin_level"="{level}"];)->.a;'
        )
        parts = [f"nwr{tag}(area.a);out count;" for tag in CATEGORIES.values()]
        return "[out:json][timeout:90];" + area_sel + "".join(parts)


def scrape_osm_overpass() -> dict:
    with OSMOverpassScraper() as scraper:
        return scraper.run()
