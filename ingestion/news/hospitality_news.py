"""Multi-source hospitality news scrapers.

Scrapers for various hospitality industry news sources.
"""

import logging
from datetime import datetime
from email.utils import parsedate_to_datetime

import feedparser
from bs4 import BeautifulSoup

from data_models.raw_content import RawContentCreate, SourceType
from ingestion.base_scraper import BaseScraper

logger = logging.getLogger(__name__)


class RSSNewsScraper(BaseScraper):
    """Base scraper for RSS-based news sites."""

    source_name = "generic_rss"
    source_type = SourceType.NEWS
    RSS_URL = ""

    def _parse_date(self, date_str: str) -> datetime | None:
        """Parse RSS date string to datetime."""
        if not date_str:
            return None
        try:
            return parsedate_to_datetime(date_str)
        except (ValueError, TypeError):
            return None

    def scrape(self) -> list[RawContentCreate]:
        """Scrape articles from RSS feed."""
        if not self.RSS_URL:
            logger.error(f"{self.source_name}: No RSS URL configured")
            return []

        logger.info(f"Fetching RSS feed from {self.RSS_URL}")
        feed = feedparser.parse(self.RSS_URL)

        if feed.bozo and not feed.entries:
            logger.error(f"Feed parsing error: {feed.bozo_exception}")
            return []

        items = []
        for entry in feed.entries:
            try:
                content = ""
                if hasattr(entry, "content") and entry.content:
                    content = entry.content[0].value
                elif hasattr(entry, "summary"):
                    content = entry.summary
                elif hasattr(entry, "description"):
                    content = entry.description

                if not content:
                    continue

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
                    },
                )
                items.append(item)
            except Exception as e:
                logger.error(f"Error parsing entry: {e}")
                continue

        logger.info(f"Scraped {len(items)} articles from {self.source_name}")
        return items


class HotelDiveScraper(RSSNewsScraper):
    """Scraper for HotelDive news."""

    source_name = "hoteldive"
    RSS_URL = "https://www.hoteldive.com/feeds/news/"


class HotelManagementScraper(RSSNewsScraper):
    """Scraper for Hotel Management news."""

    source_name = "hotelmanagement"
    RSS_URL = "https://www.hotelmanagement.net/rss.xml"


class PhocusWireScraper(RSSNewsScraper):
    """Scraper for PhocusWire travel tech news."""

    source_name = "phocuswire"
    RSS_URL = "https://www.phocuswire.com/rss.xml"


class TravelWeeklyScraper(RSSNewsScraper):
    """Scraper for Travel Weekly news."""

    source_name = "travelweekly"
    RSS_URL = "https://www.travelweekly.com/rss/news"


class HotelNewsResourceScraper(RSSNewsScraper):
    """Scraper for Hotel News Resource."""

    source_name = "hotelnewsresource"
    RSS_URL = "https://www.hotelnewsresource.com/rss.xml"


class TravelDailyNewsScraper(RSSNewsScraper):
    """Scraper for Travel Daily News."""

    source_name = "traveldailynews"
    RSS_URL = "https://www.traveldailynews.com/feed/"


class BusinessTravelNewsScraper(RSSNewsScraper):
    """Scraper for Business Travel News."""

    source_name = "businesstravelnews"
    RSS_URL = "https://www.businesstravelnews.com/rss"


class BoutiqueHotelierScraper(BaseScraper):
    """Scraper for Boutique Hotelier news (HTML scraping)."""

    source_name = "boutiquehotelier"
    source_type = SourceType.NEWS
    BASE_URL = "https://www.boutiquehotelier.com/news/"

    def scrape(self) -> list[RawContentCreate]:
        """Scrape articles from Boutique Hotelier."""
        logger.info(f"Scraping {self.BASE_URL}")

        response = self.fetch(self.BASE_URL)
        if not response:
            return []

        soup = BeautifulSoup(response.text, "html.parser")
        items = []

        # Find article elements
        articles = soup.select("article, .post, .article-item, .news-item")[:20]

        for article in articles:
            try:
                # Find title and link
                title_elem = article.select_one("h2 a, h3 a, .title a, a.title")
                if not title_elem:
                    continue

                title = title_elem.get_text(strip=True)
                url = title_elem.get("href", "")
                if url and not url.startswith("http"):
                    url = f"https://www.boutiquehotelier.com{url}"

                # Find excerpt/content
                excerpt_elem = article.select_one(".excerpt, .summary, p")
                content = excerpt_elem.get_text(strip=True) if excerpt_elem else ""

                if not title or not url:
                    continue

                item = RawContentCreate(
                    source=self.source_name,
                    source_type=self.source_type,
                    url=url,
                    title=title,
                    content=content,
                    metadata={"scraped_from": "listing_page"},
                )
                items.append(item)
            except Exception as e:
                logger.error(f"Error parsing article: {e}")
                continue

        logger.info(f"Scraped {len(items)} articles from {self.source_name}")
        return items


class HotelOnlineScraper(BaseScraper):
    """Scraper for Hotel-Online news."""

    source_name = "hotelonline"
    source_type = SourceType.NEWS
    BASE_URL = "https://www.hotel-online.com/press_releases/"

    def scrape(self) -> list[RawContentCreate]:
        """Scrape press releases from Hotel-Online."""
        logger.info(f"Scraping {self.BASE_URL}")

        response = self.fetch(self.BASE_URL)
        if not response:
            return []

        soup = BeautifulSoup(response.text, "html.parser")
        items = []

        # Find article links
        links = soup.select("a[href*='press_releases']")[:30]

        for link in links:
            try:
                title = link.get_text(strip=True)
                url = link.get("href", "")

                if not title or len(title) < 20:
                    continue

                if url and not url.startswith("http"):
                    url = f"https://www.hotel-online.com{url}"

                item = RawContentCreate(
                    source=self.source_name,
                    source_type=self.source_type,
                    url=url,
                    title=title,
                    content="",
                    metadata={"type": "press_release"},
                )
                items.append(item)
            except Exception as e:
                logger.error(f"Error parsing link: {e}")
                continue

        logger.info(f"Scraped {len(items)} articles from {self.source_name}")
        return items


class HotelTechReportScraper(RSSNewsScraper):
    """Scraper for Hotel Tech Report."""

    source_name = "hoteltechreport"
    RSS_URL = "https://hoteltechreport.com/feed"


class TopHotelNewsScraper(BaseScraper):
    """Scraper for Top Hotel News."""

    source_name = "tophotelnews"
    source_type = SourceType.NEWS
    BASE_URL = "https://tophotel.news/"

    def scrape(self) -> list[RawContentCreate]:
        """Scrape articles from Top Hotel News."""
        logger.info(f"Scraping {self.BASE_URL}")

        response = self.fetch(self.BASE_URL)
        if not response:
            return []

        soup = BeautifulSoup(response.text, "html.parser")
        items = []

        # Find article elements
        articles = soup.select("article, .post, .entry")[:20]

        for article in articles:
            try:
                title_elem = article.select_one("h2 a, h3 a, .entry-title a")
                if not title_elem:
                    continue

                title = title_elem.get_text(strip=True)
                url = title_elem.get("href", "")

                excerpt_elem = article.select_one(".excerpt, .entry-summary, p")
                content = excerpt_elem.get_text(strip=True) if excerpt_elem else ""

                if not title or not url:
                    continue

                item = RawContentCreate(
                    source=self.source_name,
                    source_type=self.source_type,
                    url=url,
                    title=title,
                    content=content,
                    metadata={},
                )
                items.append(item)
            except Exception as e:
                logger.error(f"Error parsing article: {e}")
                continue

        logger.info(f"Scraped {len(items)} articles from {self.source_name}")
        return items


# Additional news sources from the full list

class SiteMinderScraper(RSSNewsScraper):
    """Scraper for SiteMinder blog/news."""

    source_name = "siteminder"
    RSS_URL = "https://www.siteminder.com/r/feed/"


class EHLInsightsScraper(BaseScraper):
    """Scraper for EHL Hospitality Insights."""

    source_name = "ehlinsights"
    source_type = SourceType.NEWS
    BASE_URL = "https://hospitalityinsights.ehl.edu/hospitality-industry-trends"

    def scrape(self) -> list[RawContentCreate]:
        """Scrape articles from EHL Hospitality Insights."""
        logger.info(f"Scraping {self.BASE_URL}")

        response = self.fetch(self.BASE_URL)
        if not response:
            return []

        soup = BeautifulSoup(response.text, "html.parser")
        items = []

        articles = soup.select("article, .blog-post, .post-item, .card")[:20]

        for article in articles:
            try:
                title_elem = article.select_one("h2 a, h3 a, .title a, a.title, h2, h3")
                if not title_elem:
                    continue

                title = title_elem.get_text(strip=True)

                link_elem = article.select_one("a[href]")
                url = link_elem.get("href", "") if link_elem else ""

                if url and not url.startswith("http"):
                    url = f"https://hospitalityinsights.ehl.edu{url}"

                excerpt_elem = article.select_one(".excerpt, .summary, .description, p")
                content = excerpt_elem.get_text(strip=True) if excerpt_elem else ""

                if not title or len(title) < 10:
                    continue

                item = RawContentCreate(
                    source=self.source_name,
                    source_type=self.source_type,
                    url=url or self.BASE_URL,
                    title=title,
                    content=content,
                    metadata={"type": "insight_article"},
                )
                items.append(item)
            except Exception as e:
                logger.error(f"Error parsing article: {e}")
                continue

        logger.info(f"Scraped {len(items)} articles from {self.source_name}")
        return items


class CBREHotelsScraper(BaseScraper):
    """Scraper for CBRE Hotels research."""

    source_name = "cbrehotels"
    source_type = SourceType.NEWS
    BASE_URL = "https://www.cbre.com/insights/sectors/hotels"

    def scrape(self) -> list[RawContentCreate]:
        """Scrape research from CBRE Hotels."""
        logger.info(f"Scraping {self.BASE_URL}")

        response = self.fetch(self.BASE_URL)
        if not response:
            return []

        soup = BeautifulSoup(response.text, "html.parser")
        items = []

        articles = soup.select("article, .insight-card, .research-item, .card, [class*='insight']")[:20]

        for article in articles:
            try:
                title_elem = article.select_one("h2, h3, h4, .title, a")
                if not title_elem:
                    continue

                title = title_elem.get_text(strip=True)

                link_elem = article.select_one("a[href]")
                url = link_elem.get("href", "") if link_elem else ""

                if url and not url.startswith("http"):
                    url = f"https://www.cbre.com{url}"

                excerpt_elem = article.select_one("p, .excerpt, .description")
                content = excerpt_elem.get_text(strip=True) if excerpt_elem else ""

                if not title or len(title) < 10:
                    continue

                item = RawContentCreate(
                    source=self.source_name,
                    source_type=self.source_type,
                    url=url or self.BASE_URL,
                    title=title,
                    content=content,
                    metadata={"type": "research"},
                )
                items.append(item)
            except Exception as e:
                logger.error(f"Error parsing article: {e}")
                continue

        logger.info(f"Scraped {len(items)} articles from {self.source_name}")
        return items


class CushmanWakefieldScraper(BaseScraper):
    """Scraper for Cushman & Wakefield hospitality insights."""

    source_name = "cushmanwakefield"
    source_type = SourceType.NEWS
    BASE_URL = "https://www.cushmanwakefield.com/en/insights"

    def scrape(self) -> list[RawContentCreate]:
        """Scrape insights from Cushman & Wakefield."""
        logger.info(f"Scraping {self.BASE_URL}")

        response = self.fetch(self.BASE_URL)
        if not response:
            return []

        soup = BeautifulSoup(response.text, "html.parser")
        items = []

        articles = soup.select("article, .insight-card, .card, [class*='article']")[:20]

        for article in articles:
            try:
                title_elem = article.select_one("h2, h3, h4, .title")
                if not title_elem:
                    continue

                title = title_elem.get_text(strip=True)

                # Filter for hospitality-related content
                title_lower = title.lower()
                if not any(kw in title_lower for kw in ["hotel", "hospitality", "travel", "lodging", "resort"]):
                    continue

                link_elem = article.select_one("a[href]")
                url = link_elem.get("href", "") if link_elem else ""

                if url and not url.startswith("http"):
                    url = f"https://www.cushmanwakefield.com{url}"

                excerpt_elem = article.select_one("p, .excerpt, .description")
                content = excerpt_elem.get_text(strip=True) if excerpt_elem else ""

                if not title or len(title) < 10:
                    continue

                item = RawContentCreate(
                    source=self.source_name,
                    source_type=self.source_type,
                    url=url or self.BASE_URL,
                    title=title,
                    content=content,
                    metadata={"type": "insight"},
                )
                items.append(item)
            except Exception as e:
                logger.error(f"Error parsing article: {e}")
                continue

        logger.info(f"Scraped {len(items)} articles from {self.source_name}")
        return items


class CoStarScraper(BaseScraper):
    """Scraper for CoStar hospitality news."""

    source_name = "costar"
    source_type = SourceType.NEWS
    BASE_URL = "https://www.costar.com/hospitality"

    def scrape(self) -> list[RawContentCreate]:
        """Scrape news from CoStar hospitality."""
        logger.info(f"Scraping {self.BASE_URL}")

        response = self.fetch(self.BASE_URL)
        if not response:
            return []

        soup = BeautifulSoup(response.text, "html.parser")
        items = []

        articles = soup.select("article, .news-item, .story, .card, [class*='article']")[:20]

        for article in articles:
            try:
                title_elem = article.select_one("h2, h3, h4, .headline, .title a, a")
                if not title_elem:
                    continue

                title = title_elem.get_text(strip=True)

                link_elem = article.select_one("a[href]")
                url = link_elem.get("href", "") if link_elem else ""

                if url and not url.startswith("http"):
                    url = f"https://www.costar.com{url}"

                excerpt_elem = article.select_one("p, .excerpt, .summary")
                content = excerpt_elem.get_text(strip=True) if excerpt_elem else ""

                if not title or len(title) < 15:
                    continue

                item = RawContentCreate(
                    source=self.source_name,
                    source_type=self.source_type,
                    url=url or self.BASE_URL,
                    title=title,
                    content=content,
                    metadata={"type": "news"},
                )
                items.append(item)
            except Exception as e:
                logger.error(f"Error parsing article: {e}")
                continue

        logger.info(f"Scraped {len(items)} articles from {self.source_name}")
        return items


class TravelDailyScraper(RSSNewsScraper):
    """Scraper for Travel Daily (different from Travel Daily News)."""

    source_name = "traveldaily"
    RSS_URL = "https://www.traveldaily.com.au/feed"


# Convenience functions
def scrape_hoteldive() -> dict:
    with HotelDiveScraper() as scraper:
        return scraper.run()


def scrape_hotelmanagement() -> dict:
    with HotelManagementScraper() as scraper:
        return scraper.run()


def scrape_phocuswire() -> dict:
    with PhocusWireScraper() as scraper:
        return scraper.run()


def scrape_travelweekly() -> dict:
    with TravelWeeklyScraper() as scraper:
        return scraper.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Test one scraper
    result = scrape_hoteldive()
    print(result)
