"""Skift RSS feed scraper."""

import logging
from datetime import datetime
from email.utils import parsedate_to_datetime

import feedparser

from data_models.raw_content import RawContentCreate, SourceType
from ingestion.base_scraper import BaseScraper

logger = logging.getLogger(__name__)


class SkiftScraper(BaseScraper):
    """Scraper for Skift RSS feed."""

    source_name = "skift"
    source_type = SourceType.NEWS

    RSS_URL = "https://skift.com/feed/"

    def __init__(self, config_path: str = "configs/scraping.yaml"):
        super().__init__(config_path)

    def _parse_date(self, date_str: str) -> datetime | None:
        """Parse RSS date string to datetime."""
        if not date_str:
            return None
        try:
            return parsedate_to_datetime(date_str)
        except (ValueError, TypeError):
            logger.warning(f"Could not parse date: {date_str}")
            return None

    def scrape(self) -> list[RawContentCreate]:
        """Scrape articles from Skift RSS feed.

        Returns:
            List of RawContentCreate objects
        """
        logger.info(f"Fetching RSS feed from {self.RSS_URL}")

        # feedparser handles the HTTP request internally
        feed = feedparser.parse(self.RSS_URL)

        if feed.bozo and not feed.entries:
            logger.error(f"Feed parsing error: {feed.bozo_exception}")
            return []

        items = []
        for entry in feed.entries:
            try:
                # Extract content - try different fields
                content = ""
                if hasattr(entry, "content") and entry.content:
                    content = entry.content[0].value
                elif hasattr(entry, "summary"):
                    content = entry.summary
                elif hasattr(entry, "description"):
                    content = entry.description

                if not content:
                    logger.debug(f"Skipping entry without content: {entry.get('title', 'Unknown')}")
                    continue

                # Extract categories/tags
                categories = []
                if hasattr(entry, "tags"):
                    categories = [tag.term for tag in entry.tags]

                item = RawContentCreate(
                    source=self.source_name,
                    source_type=self.source_type,
                    url=entry.link,
                    title=entry.get("title"),
                    content=content,
                    author=entry.get("author") or entry.get("dc_creator"),
                    published_at=self._parse_date(entry.get("published")),
                    metadata={
                        "feed_title": feed.feed.get("title"),
                        "categories": categories,
                        "guid": entry.get("id"),
                    },
                )
                items.append(item)

            except Exception as e:
                logger.error(f"Error parsing entry: {e}")
                continue

        logger.info(f"Scraped {len(items)} articles from Skift")
        return items


def scrape_skift() -> dict:
    """Convenience function to run the scraper.

    Returns:
        Scrape summary dict
    """
    with SkiftScraper() as scraper:
        return scraper.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = scrape_skift()
    print(result)
