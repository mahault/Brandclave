"""Quora scraper for hospitality-related questions and answers."""

import logging
import time
from datetime import datetime

from bs4 import BeautifulSoup

from data_models.raw_content import RawContentCreate, SourceType
from ingestion.base_scraper import BaseScraper

logger = logging.getLogger(__name__)


class QuoraScraper(BaseScraper):
    """Scraper for Quora hospitality questions."""

    source_name = "quora"
    source_type = SourceType.SOCIAL

    # Topics to search for
    SEARCH_QUERIES = [
        "best hotels",
        "where to stay",
        "hotel recommendation",
        "boutique hotel",
        "luxury hotel experience",
        "budget accommodation",
        "hotel vs airbnb",
        "travel accommodation tips",
        "hotel industry trends",
        "hospitality careers",
    ]

    def scrape(self) -> list[RawContentCreate]:
        """Scrape hospitality questions from Quora."""
        items = []

        for query in self.SEARCH_QUERIES:
            try:
                search_url = f"https://www.quora.com/search?q={query.replace(' ', '+')}"
                logger.info(f"Searching Quora: {query}")

                response = self.fetch(search_url)
                if not response:
                    continue

                soup = BeautifulSoup(response.text, "html.parser")

                # Find question elements
                questions = soup.select(
                    "[class*='question'], [class*='Question'], "
                    ".q-box, .qu-wordBreak--break-word"
                )[:10]

                for q in questions:
                    try:
                        # Get question text
                        title_elem = q.select_one("span, a, div")
                        if not title_elem:
                            continue

                        title = title_elem.get_text(strip=True)

                        if not title or len(title) < 20:
                            continue

                        # Skip non-relevant content
                        title_lower = title.lower()
                        if not any(kw in title_lower for kw in [
                            "hotel", "stay", "accommodation", "travel",
                            "resort", "hostel", "airbnb", "booking"
                        ]):
                            continue

                        # Get link if available
                        link_elem = q.select_one("a[href*='/']")
                        url = ""
                        if link_elem:
                            href = link_elem.get("href", "")
                            if href.startswith("/"):
                                url = f"https://www.quora.com{href}"
                            elif href.startswith("http"):
                                url = href

                        item = RawContentCreate(
                            source=self.source_name,
                            source_type=self.source_type,
                            url=url or search_url,
                            title=title[:300],
                            content=title,  # Question itself is the content
                            metadata={
                                "search_query": query,
                                "type": "question",
                            },
                        )
                        items.append(item)

                    except Exception as e:
                        logger.debug(f"Error parsing question: {e}")
                        continue

                # Rate limiting
                time.sleep(3)

            except Exception as e:
                logger.error(f"Quora search error for '{query}': {e}")
                continue

        logger.info(f"Scraped {len(items)} questions from Quora")
        return items


def scrape_quora() -> dict:
    """Convenience function to run Quora scraper."""
    with QuoraScraper() as scraper:
        return scraper.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = scrape_quora()
    print(result)
