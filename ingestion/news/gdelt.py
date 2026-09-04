"""GDELT DOC 2.0 — a free, global, timestamped index of online news.

Nine hand-picked trade feeds see the industry from the inside; GDELT sees
every outlet that mentions hotels, resorts or hospitality, in every market,
with a machine timestamp. It is the breadth layer for trend detection and the
early-warning layer for operator moves that trade press covers a week later.

Rules of the road: one request every five seconds (the API says so in its
429 body), descriptive User-Agent. The article list carries titles only, so
the scraper fetches the page for the top N articles and keeps the paragraph
text; the rest are stored as headline-only items, still useful for clustering.
"""

import logging
import re
import time
from datetime import datetime

from bs4 import BeautifulSoup

from data_models.raw_content import RawContentCreate, SourceType
from ingestion.base_scraper import BaseScraper
from ingestion.registry import get_source_config

logger = logging.getLogger(__name__)

API_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
USER_AGENT = "BrandClaveDemandBot/1.0 (https://github.com/mahault/Brandclave; mahault.albarracin@gmail.com)"
MIN_INTERVAL_SECONDS = 5.5
DEFAULT_QUERIES = [
    '(hotel OR hotels OR resort) (opens OR opening OR acquires OR acquisition OR launches OR rebrand) sourcelang:english',
    '("boutique hotel" OR "lifestyle hotel" OR "hospitality brand") sourcelang:english',
    '("digital nomad" OR "workcation" OR "wellness retreat" OR "hotel demand") sourcelang:english',
]


class GDELTScraper(BaseScraper):
    """Hospitality news breadth via the GDELT DOC API."""

    source_name = "gdelt"
    source_type = SourceType.NEWS

    def scrape(self) -> list[RawContentCreate]:
        cfg = get_source_config(self.source_name)
        queries = cfg.get("queries", DEFAULT_QUERIES)
        timespan = cfg.get("timespan", "3d")
        max_records = min(int(cfg.get("max_records", 75)), 250)
        fetch_bodies = int(cfg.get("fetch_bodies", 30))

        articles: dict[str, dict] = {}
        last_call = 0.0
        for query in queries:
            wait = MIN_INTERVAL_SECONDS - (time.time() - last_call)
            if wait > 0:
                time.sleep(wait)
            last_call = time.time()
            response = self.fetch(
                API_URL,
                params={"query": query, "mode": "artlist", "maxrecords": max_records, "format": "json", "timespan": timespan, "sort": "datedesc"},
                headers={"User-Agent": USER_AGENT},
            )
            if response is None:
                logger.warning(f"GDELT query failed: {query[:50]}")
                continue
            try:
                for article in response.json().get("articles", []):
                    url = article.get("url")
                    if url and url not in articles:
                        article["_query"] = query
                        articles[url] = article
            except ValueError:
                logger.warning("GDELT returned non-JSON (likely throttled); continuing")

        items: list[RawContentCreate] = []
        for i, (url, article) in enumerate(articles.items()):
            title = (article.get("title") or "").strip()
            if not title:
                continue
            body = ""
            if i < fetch_bodies:
                body = self._fetch_body(url)
            content = body or title
            items.append(
                RawContentCreate(
                    source=self.source_name,
                    source_type=self.source_type,
                    url=url,
                    title=title,
                    content=content,
                    author=article.get("domain"),
                    published_at=self._parse_seendate(article.get("seendate")),
                    metadata={
                        "domain": article.get("domain"),
                        "language": article.get("language"),
                        "sourcecountry": article.get("sourcecountry"),
                        "query": article.get("_query"),
                        "body_fetched": bool(body),
                    },
                )
            )
        logger.info(f"GDELT: {len(items)} articles ({min(fetch_bodies, len(items))} with bodies) from {len(queries)} queries")
        return items

    def _fetch_body(self, url: str) -> str:
        """Paragraph text of the article page, capped; empty on any failure."""
        response = self.fetch(url, headers={"User-Agent": USER_AGENT})
        if response is None or "html" not in (response.headers.get("content-type") or ""):
            return ""
        try:
            soup = BeautifulSoup(response.text, "html.parser")
            for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
                tag.decompose()
            paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
            text = " ".join(p for p in paragraphs if len(p) > 40)
            text = re.sub(r"\s+", " ", text).strip()
            return text[:6000]
        except Exception:
            return ""

    @staticmethod
    def _parse_seendate(value: str | None) -> datetime | None:
        if not value:
            return None
        try:
            return datetime.strptime(value, "%Y%m%dT%H%M%SZ")
        except ValueError:
            return None


def scrape_gdelt() -> dict:
    with GDELTScraper() as scraper:
        return scraper.run()
