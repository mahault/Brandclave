"""TripAdvisor review scraper for BrandClave Aggregator.

Scrapes hotel reviews from TripAdvisor with conservative rate limiting.
"""

import logging
import random
import re
import time
from datetime import datetime
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from data_models.raw_content import RawContentCreate, SourceType
from ingestion.base_scraper import BaseScraper

logger = logging.getLogger(__name__)


class TripAdvisorScraper(BaseScraper):
    """Scraper for TripAdvisor hotel reviews.

    Uses conservative rate limiting to avoid blocking.
    """

    source_name = "tripadvisor"
    source_type = SourceType.REVIEW

    # Default hotels to scrape (TripAdvisor location IDs)
    DEFAULT_HOTELS = [
        # Format: "g{geo_id}-d{detail_id}" from TripAdvisor URLs
        # Example hotels in major markets
        ("g60763-d93450", "The Plaza New York"),
        ("g60763-d99762", "The St. Regis New York"),
        ("g187147-d197572", "Ritz Paris"),
        ("g186338-d192452", "The Savoy London"),
        ("g294217-d302106", "Mandarin Oriental Bangkok"),
    ]

    BASE_URL = "https://www.tripadvisor.com"

    def __init__(
        self,
        config_path: str = "configs/scraping.yaml",
        hotels: list[tuple[str, str]] | None = None,
        max_reviews_per_hotel: int = 30,
    ):
        """Initialize TripAdvisor scraper.

        Args:
            config_path: Path to scraping config
            hotels: List of (hotel_id, hotel_name) tuples
            max_reviews_per_hotel: Maximum reviews to fetch per hotel
        """
        super().__init__(config_path)

        # Disable robots.txt (reviews are public content)
        self.config["global_settings"]["respect_robots_txt"] = False

        # Set conservative delays (10-20 seconds)
        self.min_delay = 10
        self.max_delay = 20

        self.hotels = hotels or self.DEFAULT_HOTELS
        self.max_reviews_per_hotel = max_reviews_per_hotel

        # Update headers to look like a browser
        self.client.headers.update({
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Cache-Control": "no-cache",
            "Upgrade-Insecure-Requests": "1",
        })

    def _build_review_url(self, hotel_id: str, offset: int = 0) -> str:
        """Build review page URL.

        Args:
            hotel_id: TripAdvisor hotel ID (g{geo}-d{detail})
            offset: Review offset for pagination

        Returns:
            Review page URL
        """
        # TripAdvisor review URL pattern
        if offset > 0:
            return f"{self.BASE_URL}/Hotel_Review-{hotel_id}-Reviews-or{offset}.html"
        return f"{self.BASE_URL}/Hotel_Review-{hotel_id}-Reviews.html"

    def _parse_reviews(self, html: str, hotel_name: str) -> list[dict]:
        """Parse reviews from HTML.

        Args:
            html: Page HTML
            hotel_name: Name of the hotel

        Returns:
            List of review dicts
        """
        reviews = []
        soup = BeautifulSoup(html, "html.parser")

        # Find review containers (TripAdvisor structure varies)
        review_containers = soup.find_all("div", {"data-reviewid": True})

        if not review_containers:
            # Alternative selector
            review_containers = soup.find_all("div", class_=re.compile(r"review-container"))

        for container in review_containers:
            try:
                review = self._parse_single_review(container, hotel_name)
                if review:
                    reviews.append(review)
            except Exception as e:
                logger.debug(f"Error parsing review: {e}")
                continue

        return reviews

    def _parse_single_review(self, container, hotel_name: str) -> dict | None:
        """Parse a single review container.

        Args:
            container: BeautifulSoup element
            hotel_name: Name of the hotel

        Returns:
            Review dict or None
        """
        review_id = container.get("data-reviewid", "")

        # Extract rating (typically in a bubble rating element)
        rating = None
        rating_elem = container.find("span", class_=re.compile(r"bubble_rating"))
        if rating_elem:
            # Rating is in class like "bubble_50" for 5 stars
            classes = rating_elem.get("class", [])
            for cls in classes:
                match = re.search(r"bubble_(\d+)", cls)
                if match:
                    rating = int(match.group(1)) / 10  # Convert 50 -> 5.0
                    break

        # Extract title
        title_elem = container.find("a", class_=re.compile(r"title")) or \
                     container.find("span", class_=re.compile(r"noQuotes"))
        title = title_elem.get_text(strip=True) if title_elem else ""

        # Extract review text
        text_elem = container.find("p", class_=re.compile(r"partial_entry")) or \
                    container.find("div", class_=re.compile(r"entry"))
        text = text_elem.get_text(strip=True) if text_elem else ""

        if not text or len(text) < 20:
            return None

        # Extract reviewer name
        reviewer_elem = container.find("a", class_=re.compile(r"username")) or \
                       container.find("span", class_=re.compile(r"scrname"))
        reviewer = reviewer_elem.get_text(strip=True) if reviewer_elem else "Anonymous"

        # Extract date
        date_elem = container.find("span", class_=re.compile(r"ratingDate"))
        date_str = None
        published_at = None
        if date_elem:
            date_str = date_elem.get("title") or date_elem.get_text(strip=True)
            try:
                # Try various date formats
                for fmt in ["%B %d, %Y", "%d %B %Y", "%b %d, %Y"]:
                    try:
                        published_at = datetime.strptime(date_str, fmt)
                        break
                    except ValueError:
                        continue
            except Exception:
                pass

        # Extract trip type
        trip_type_elem = container.find("span", class_=re.compile(r"trip"))
        trip_type = trip_type_elem.get_text(strip=True) if trip_type_elem else None

        return {
            "review_id": review_id,
            "title": title,
            "text": text,
            "rating": rating,
            "reviewer_name": reviewer,
            "date_str": date_str,
            "published_at": published_at,
            "trip_type": trip_type,
            "hotel_name": hotel_name,
        }

    def _review_to_content(self, review: dict, hotel_id: str) -> RawContentCreate:
        """Convert parsed review to RawContentCreate.

        Args:
            review: Parsed review dict
            hotel_id: TripAdvisor hotel ID

        Returns:
            RawContentCreate object
        """
        # Build review URL
        url = f"{self.BASE_URL}/ShowUserReviews-{hotel_id}-r{review['review_id']}.html"

        # Combine title and text
        content = f"{review.get('title', '')}\n\n{review.get('text', '')}".strip()

        return RawContentCreate(
            source=self.source_name,
            source_type=self.source_type,
            url=url,
            title=review.get("title", ""),
            content=content,
            author=review.get("reviewer_name"),
            published_at=review.get("published_at"),
            metadata={
                "hotel_id": hotel_id,
                "hotel_name": review.get("hotel_name"),
                "rating": review.get("rating"),
                "trip_type": review.get("trip_type"),
                "review_id": review.get("review_id"),
            },
        )

    def _wait(self):
        """Wait between requests with randomized delay."""
        delay = random.uniform(self.min_delay, self.max_delay)
        logger.debug(f"Waiting {delay:.1f}s before next request")
        time.sleep(delay)

    def scrape(self) -> list[RawContentCreate]:
        """Scrape reviews from configured hotels.

        Returns:
            List of RawContentCreate objects
        """
        items = []
        seen_urls = set()

        logger.info(f"Starting TripAdvisor scrape for {len(self.hotels)} hotels")

        for hotel_id, hotel_name in self.hotels:
            logger.info(f"Fetching reviews for: {hotel_name}")
            hotel_reviews = 0
            offset = 0

            while hotel_reviews < self.max_reviews_per_hotel:
                url = self._build_review_url(hotel_id, offset)
                logger.debug(f"Fetching: {url}")

                response = self.fetch(url)
                if not response:
                    logger.warning(f"Failed to fetch reviews for {hotel_name}")
                    break

                reviews = self._parse_reviews(response.text, hotel_name)

                if not reviews:
                    logger.debug(f"No more reviews found for {hotel_name}")
                    break

                for review in reviews:
                    item = self._review_to_content(review, hotel_id)
                    if item.url not in seen_urls:
                        items.append(item)
                        seen_urls.add(item.url)
                        hotel_reviews += 1

                        if hotel_reviews >= self.max_reviews_per_hotel:
                            break

                # Move to next page
                offset += 10  # TripAdvisor shows ~10 reviews per page

                # Wait between requests
                self._wait()

            logger.info(f"Collected {hotel_reviews} reviews from {hotel_name}")

        logger.info(f"TripAdvisor scrape complete: {len(items)} total reviews")
        return items
