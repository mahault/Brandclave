"""Eurostat tourism statistics — official monthly nights spent at tourist
accommodation, by country.

Dataset `tour_occ_nim` (nights spent, monthly), free JSON-stat API, no key.
This is hard demand: what the Wikipedia attention curve is a leading proxy
for, Eurostat reports as realised stays. Countries and lookback come from
configs/sources.yaml; each country is stored as a "city" row named after the
country so the same demand-metric table and charts apply.
"""

import logging
from datetime import datetime

from data_models.demand_metric import DemandMetricCreate
from ingestion.metrics.base import MetricScraper
from ingestion.registry import get_source_config

logger = logging.getLogger(__name__)

API_URL = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/tour_occ_nim"
USER_AGENT = "BrandClaveDemandBot/1.0 (https://github.com/mahault/Brandclave; mahault.albarracin@gmail.com)"

COUNTRY_NAMES = {
    "AT": "Austria", "BE": "Belgium", "BG": "Bulgaria", "HR": "Croatia", "CY": "Cyprus",
    "CZ": "Czechia", "DK": "Denmark", "EE": "Estonia", "FI": "Finland", "FR": "France",
    "DE": "Germany", "EL": "Greece", "HU": "Hungary", "IS": "Iceland", "IE": "Ireland",
    "IT": "Italy", "LV": "Latvia", "LT": "Lithuania", "LU": "Luxembourg", "MT": "Malta",
    "NL": "Netherlands", "NO": "Norway", "PL": "Poland", "PT": "Portugal", "RO": "Romania",
    "SK": "Slovakia", "SI": "Slovenia", "ES": "Spain", "SE": "Sweden", "CH": "Switzerland",
    "TR": "Türkiye", "ME": "Montenegro", "RS": "Serbia",
}


class EurostatTourismScraper(MetricScraper):
    """Monthly nights spent at hotels and similar accommodation, per country."""

    source_name = "eurostat_tourism"

    def scrape(self) -> list[DemandMetricCreate]:
        cfg = get_source_config(self.source_name)
        geos = cfg.get("geo", ["ES", "FR", "IT", "PT", "EL", "DE", "NL", "AT", "HR"])
        since = cfg.get("since", "2023-01")
        metric_name = cfg.get("metric", "eurostat_nights_spent")
        nace = cfg.get("nace_r2", "I551-I553")  # hotels + holiday/short-stay + camping

        params = [("format", "JSON"), ("lang", "EN"), ("unit", "NR"), ("c_resid", "TOTAL"),
                  ("nace_r2", nace), ("sinceTimePeriod", since)]
        params += [("geo", g) for g in geos]

        response = self.fetch(API_URL, params=params, headers={"User-Agent": USER_AGENT})
        if response is None:
            logger.warning("Eurostat request failed, skipping this run")
            return []
        try:
            data = response.json()
        except ValueError:
            logger.warning("Eurostat returned non-JSON")
            return []

        return self._parse_jsonstat(data, metric_name, nace)

    def _parse_jsonstat(self, data: dict, metric_name: str, nace: str) -> list[DemandMetricCreate]:
        """Decode the JSON-stat cube into (country, month, value) points.

        JSON-stat stores values in a flat dict keyed by the row-major index over
        `id`-ordered dimensions with sizes `size`; only geo and time vary here,
        but the decoder walks the general case so a changed query still parses.
        """
        dims = data.get("id", [])
        sizes = data.get("size", [])
        categories = {d: data["dimension"][d]["category"]["index"] for d in dims}
        labels = {d: data["dimension"][d]["category"].get("label", {}) for d in dims}
        values = data.get("value", {})
        if not dims or not values:
            return []

        # position -> code for each dimension
        codes = {d: sorted(categories[d], key=lambda k: categories[d][k]) for d in dims}
        strides = []
        acc = 1
        for s in reversed(sizes):
            strides.insert(0, acc)
            acc *= s

        points: list[DemandMetricCreate] = []
        for flat_index, value in values.items():
            idx = int(flat_index)
            coords = {}
            for d, stride, size in zip(dims, strides, sizes):
                coords[d] = codes[d][(idx // stride) % size]
            geo = coords.get("geo")
            period = coords.get("time", "")
            try:
                day = datetime.strptime(period, "%Y-%m")
            except ValueError:
                continue
            country = COUNTRY_NAMES.get(geo, labels["geo"].get(geo, geo))
            points.append(
                DemandMetricCreate(
                    source=self.source_name,
                    city=country,
                    country=country,
                    metric=metric_name,
                    date=day,
                    value=float(value),
                    metadata={"geo": geo, "nace_r2": nace, "dataset": "tour_occ_nim", "granularity": "month"},
                )
            )
        logger.info(f"Eurostat: {len(points)} monthly points across {len(codes.get('geo', []))} countries")
        return points


def scrape_eurostat_tourism() -> dict:
    with EurostatTourismScraper() as scraper:
        return scraper.run()
