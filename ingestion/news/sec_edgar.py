"""SEC EDGAR filings from public hotel operators, REITs and travel platforms.

Primary-source operator moves: an 8-K is the company telling the market what
it just did (acquisition, disposition, financing, leadership change) before
any trade outlet rewrites it. EDGAR's per-company Atom feeds are official,
free and stable; the full-text search endpoint is WAF-gated for scripted
clients, so this scraper walks the company feeds instead.

SEC asks for a "Company contact@email" User-Agent and no more than ten
requests per second. The registry config lists companies by CIK.
"""

import logging
import re
from datetime import datetime

import feedparser
from bs4 import BeautifulSoup

from data_models.raw_content import RawContentCreate, SourceType
from ingestion.base_scraper import BaseScraper
from ingestion.registry import get_source_config

logger = logging.getLogger(__name__)

FEED_URL = "https://www.sec.gov/cgi-bin/browse-edgar"
USER_AGENT = "BrandClave mahault.albarracin@gmail.com"
MAX_BODY_CHARS = 8000

DEFAULT_COMPANIES = [
    {"cik": "0001048286", "name": "Marriott International"},
    {"cik": "0001585689", "name": "Hilton Worldwide"},
    {"cik": "0001468174", "name": "Hyatt Hotels"},
    {"cik": "0001070750", "name": "Host Hotels & Resorts"},
    {"cik": "0001617406", "name": "Park Hotels & Resorts"},
    {"cik": "0001474098", "name": "Pebblebrook Hotel Trust"},
    {"cik": "0001418121", "name": "Apple Hospitality REIT"},
    {"cik": "0001295810", "name": "Sunstone Hotel Investors"},
    {"cik": "0001040829", "name": "Ryman Hospitality Properties"},
    {"cik": "0001046311", "name": "Choice Hotels International"},
    {"cik": "0001722684", "name": "Wyndham Hotels & Resorts"},
    {"cik": "0001559720", "name": "Airbnb"},
    {"cik": "0001324424", "name": "Expedia Group"},
    {"cik": "0001075531", "name": "Booking Holdings"},
]


class SECEdgarScraper(BaseScraper):
    """Recent 8-K/10-Q/10-K filings for tracked hospitality companies."""

    source_name = "sec_edgar"
    source_type = SourceType.NEWS

    def scrape(self) -> list[RawContentCreate]:
        cfg = get_source_config(self.source_name)
        companies = cfg.get("companies", DEFAULT_COMPANIES)
        forms = cfg.get("forms", ["8-K"])
        per_company = int(cfg.get("per_company", 5))
        headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"}

        items: list[RawContentCreate] = []
        for company in companies:
            cik, name = company.get("cik"), company.get("name", company.get("cik"))
            for form in forms:
                response = self.fetch(
                    FEED_URL,
                    params={"action": "getcompany", "CIK": cik, "type": form, "dateb": "", "owner": "include", "count": per_company, "output": "atom"},
                    headers=headers,
                )
                if response is None:
                    logger.warning(f"EDGAR feed failed for {name} {form}")
                    continue
                feed = feedparser.parse(response.text)
                for entry in feed.entries[:per_company]:
                    item = self._entry_to_item(entry, name, cik, form, headers)
                    if item is not None:
                        items.append(item)
        logger.info(f"EDGAR: {len(items)} filings across {len(companies)} companies")
        return items

    def _entry_to_item(self, entry, name: str, cik: str, form: str, headers: dict) -> RawContentCreate | None:
        index_url = entry.get("link")
        if not index_url:
            return None
        title = f"{name}: {entry.get('title', form).strip()}"
        summary = re.sub(r"\s+", " ", BeautifulSoup(entry.get("summary", ""), "html.parser").get_text(" ")).strip()
        published = self._parse_date(entry.get("updated") or entry.get("published"))

        body = self._fetch_primary_document(index_url, headers)
        content = body or summary or title

        return RawContentCreate(
            source=self.source_name,
            source_type=self.source_type,
            url=index_url,
            title=title,
            content=content,
            author=name,
            published_at=published,
            metadata={"cik": cik, "form": form, "company": name, "body_fetched": bool(body), "filing_summary": summary[:300]},
        )

    def _fetch_primary_document(self, index_url: str, headers: dict) -> str:
        """Follow the filing index to its primary document and return plain text."""
        index = self.fetch(index_url, headers=headers)
        if index is None:
            return ""
        try:
            soup = BeautifulSoup(index.text, "html.parser")
            doc_href = None
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if re.search(r"/Archives/edgar/data/.+\.htm$", href) and "-index" not in href:
                    doc_href = href
                    break
            if not doc_href:
                return ""
            if doc_href.startswith("/"):
                doc_href = "https://www.sec.gov" + doc_href
            # inline XBRL viewer links wrap the real document path
            doc_href = doc_href.replace("/ix?doc=", "")
            doc = self.fetch(doc_href, headers=headers)
            if doc is None:
                return ""
            text = BeautifulSoup(doc.text, "html.parser").get_text(" ")
            text = re.sub(r"\s+", " ", text).strip()
            return text[:MAX_BODY_CHARS]
        except Exception as exc:
            logger.debug(f"EDGAR primary document parse failed: {exc}")
            return ""

    @staticmethod
    def _parse_date(value: str | None) -> datetime | None:
        if not value:
            return None
        try:
            return datetime.fromisoformat(value).replace(tzinfo=None)
        except ValueError:
            return None


def scrape_sec_edgar() -> dict:
    with SECEdgarScraper() as scraper:
        return scraper.run()
