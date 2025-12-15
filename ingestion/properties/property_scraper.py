"""Property page scraper for Demand Scan."""

import logging
import re
from urllib.parse import urlparse

from bs4 import BeautifulSoup

from data_models.raw_content import RawContentCreate, SourceType
from ingestion.base_scraper import BaseScraper

logger = logging.getLogger(__name__)


class PropertyScraper(BaseScraper):
    """Scraper for hotel/property websites."""

    source_name = "property"
    source_type = SourceType.PROPERTY

    def __init__(self, config_path: str = "configs/scraping.yaml"):
        """Initialize property scraper.

        Args:
            config_path: Path to scraping config
        """
        super().__init__(config_path)
        # Disable robots.txt for property URLs (user-provided)
        self.config["global_settings"]["respect_robots_txt"] = False

        # Update headers to look like a browser
        self.client.headers.update({
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Cache-Control": "no-cache",
        })

    def scrape_url(self, url: str) -> RawContentCreate | None:
        """Scrape a single property URL.

        Args:
            url: Property website URL

        Returns:
            RawContentCreate or None if failed
        """
        logger.info(f"Scraping property: {url}")

        response = self.fetch(url)
        if not response:
            logger.error(f"Failed to fetch: {url}")
            return None

        try:
            soup = BeautifulSoup(response.text, "html.parser")

            # Extract title
            title = self._extract_title(soup)

            # Extract meta description
            meta_desc = self._extract_meta_description(soup)

            # Extract main content
            body_text = self._extract_body_text(soup)

            # Combine content
            content = f"{title}\n\n{meta_desc}\n\n{body_text}".strip()

            if len(content) < 100:
                logger.warning(f"Insufficient content from: {url}")
                return None

            # Extract domain as source name
            domain = urlparse(url).netloc.replace("www.", "")

            return RawContentCreate(
                source=domain,
                source_type=self.source_type,
                url=url,
                title=title,
                content=content,
                metadata={
                    "meta_description": meta_desc,
                    "word_count": len(content.split()),
                    "has_structured_data": self._has_structured_data(soup),
                },
            )

        except Exception as e:
            logger.error(f"Error parsing {url}: {e}")
            return None

    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract page title."""
        # Try og:title first
        og_title = soup.find("meta", property="og:title")
        if og_title and og_title.get("content"):
            return og_title["content"].strip()

        # Try title tag
        title_tag = soup.find("title")
        if title_tag:
            return title_tag.get_text().strip()

        # Try h1
        h1 = soup.find("h1")
        if h1:
            return h1.get_text().strip()

        return ""

    def _extract_meta_description(self, soup: BeautifulSoup) -> str:
        """Extract meta description."""
        # Try og:description
        og_desc = soup.find("meta", property="og:description")
        if og_desc and og_desc.get("content"):
            return og_desc["content"].strip()

        # Try meta description
        meta_desc = soup.find("meta", attrs={"name": "description"})
        if meta_desc and meta_desc.get("content"):
            return meta_desc["content"].strip()

        return ""

    def _extract_body_text(self, soup: BeautifulSoup) -> str:
        """Extract main body text content."""
        # Remove unwanted elements
        for element in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
            element.decompose()

        # Try to find main content areas
        main_content = None

        # Common content selectors
        content_selectors = [
            "main",
            "article",
            "[role='main']",
            ".content",
            ".main-content",
            "#content",
            "#main",
        ]

        for selector in content_selectors:
            main_content = soup.select_one(selector)
            if main_content:
                break

        # Fall back to body
        if not main_content:
            main_content = soup.find("body")

        if not main_content:
            return ""

        # Get text and clean up
        text = main_content.get_text(separator="\n", strip=True)

        # Clean up whitespace
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        text = "\n".join(lines)

        # Remove excessive whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Truncate if too long (keep first 10000 chars)
        if len(text) > 10000:
            text = text[:10000] + "..."

        return text

    def _has_structured_data(self, soup: BeautifulSoup) -> bool:
        """Check if page has JSON-LD structured data."""
        scripts = soup.find_all("script", type="application/ld+json")
        return len(scripts) > 0

    def scrape(self) -> list[RawContentCreate]:
        """Required by base class but not used for property scraping.

        Property scraping is done per-URL via scrape_url().
        """
        return []


def scrape_property(url: str) -> dict | None:
    """Convenience function to scrape a property URL.

    Args:
        url: Property website URL

    Returns:
        Dict with scraped content or None
    """
    with PropertyScraper() as scraper:
        result = scraper.scrape_url(url)
        if result:
            # Save to database
            content_id = scraper.save_content([result])
            return {
                "url": url,
                "title": result.title,
                "content_length": len(result.content),
                "content_id": content_id[0] if content_id else None,
            }
    return None
