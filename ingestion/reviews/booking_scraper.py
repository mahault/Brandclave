"""Booking.com review scraper for BrandClave Aggregator.

Scrapes hotel reviews from Booking.com with conservative rate limiting.
"""

import json
import logging
import random
import re
import time
from datetime import datetime

from bs4 import BeautifulSoup

from data_models.raw_content import RawContentCreate, SourceType
from ingestion.base_scraper import BaseScraper

logger = logging.getLogger(__name__)


class BookingScraper(BaseScraper):
    """Scraper for Booking.com hotel reviews.

    Uses conservative rate limiting to avoid blocking.
    """

    source_name = "booking"
    source_type = SourceType.REVIEW

    # Default properties to scrape
    # Format: (property_path, property_name)
    DEFAULT_PROPERTIES = [
        ("hotel/us/the-plaza", "The Plaza New York"),
        ("hotel/gb/the-savoy", "The Savoy London"),
        ("hotel/fr/ritz-paris", "Ritz Paris"),
        ("hotel/th/mandarin-oriental-bangkok", "Mandarin Oriental Bangkok"),
        ("hotel/jp/park-hyatt-tokyo", "Park Hyatt Tokyo"),
    ]

    BASE_URL = "https://www.booking.com"

    def __init__(
        self,
        config_path: str = "configs/scraping.yaml",
        properties: list[tuple[str, str]] | None = None,
        max_reviews_per_property: int = 30,
    ):
        """Initialize Booking.com scraper.

        Args:
            config_path: Path to scraping config
            properties: List of (property_path, property_name) tuples
            max_reviews_per_property: Maximum reviews to fetch per property
        """
        super().__init__(config_path)

        # Disable robots.txt
        self.config["global_settings"]["respect_robots_txt"] = False

        # Set conservative delays (8-15 seconds)
        self.min_delay = 8
        self.max_delay = 15

        self.properties = properties or self.DEFAULT_PROPERTIES
        self.max_reviews_per_property = max_reviews_per_property

        # Update headers
        self.client.headers.update({
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Cache-Control": "no-cache",
        })

    def _build_review_url(self, property_path: str, offset: int = 0) -> str:
        """Build review page URL.

        Args:
            property_path: Booking.com property path
            offset: Review offset for pagination

        Returns:
            Review page URL
        """
        # Booking.com review URL pattern
        base = f"{self.BASE_URL}/{property_path}.html"
        if offset > 0:
            return f"{base}?offset={offset}&tab=3"  # tab=3 is reviews
        return f"{base}?tab=3"

    def _parse_reviews(self, html: str, property_name: str) -> list[dict]:
        """Parse reviews from HTML.

        Args:
            html: Page HTML
            property_name: Name of the property

        Returns:
            List of review dicts
        """
        reviews = []
        soup = BeautifulSoup(html, "html.parser")

        # Find review containers (multiple possible structures)
        review_containers = soup.find_all("div", {"data-review-url": True})

        if not review_containers:
            # Alternative: look for review list items
            review_containers = soup.find_all("li", class_=re.compile(r"review_item"))

        if not review_containers:
            # Try another common pattern
            review_containers = soup.find_all("div", class_=re.compile(r"review_list_new_item"))

        for container in review_containers:
            try:
                review = self._parse_single_review(container, property_name)
                if review:
                    reviews.append(review)
            except Exception as e:
                logger.debug(f"Error parsing review: {e}")
                continue

        return reviews

    def _parse_single_review(self, container, property_name: str) -> dict | None:
        """Parse a single review container.

        Args:
            container: BeautifulSoup element
            property_name: Name of the property

        Returns:
            Review dict or None
        """
        # Extract review ID
        review_id = container.get("data-review-url", "") or \
                   container.get("data-review-id", "") or \
                   str(hash(container.get_text()[:100]))

        # Extract rating (Booking.com uses scores out of 10)
        rating = None
        rating_elem = container.find("div", class_=re.compile(r"review-score-badge")) or \
                     container.find("span", class_=re.compile(r"review-score"))
        if rating_elem:
            try:
                rating_text = rating_elem.get_text(strip=True)
                rating = float(re.search(r"[\d.]+", rating_text).group())
            except (AttributeError, ValueError):
                pass

        # Extract title (Booking.com reviews often don't have titles)
        title_elem = container.find("span", class_=re.compile(r"review_item_header"))
        title = title_elem.get_text(strip=True) if title_elem else ""

        # Extract positive text
        positive_elem = container.find("p", class_=re.compile(r"review_pos")) or \
                       container.find("span", {"data-testid": "review-positive"})
        positive = positive_elem.get_text(strip=True) if positive_elem else ""

        # Extract negative text
        negative_elem = container.find("p", class_=re.compile(r"review_neg")) or \
                       container.find("span", {"data-testid": "review-negative"})
        negative = negative_elem.get_text(strip=True) if negative_elem else ""

        # Combine for content
        content_parts = []
        if positive:
            content_parts.append(f"Positive: {positive}")
        if negative:
            content_parts.append(f"Negative: {negative}")

        if not content_parts:
            # Try generic review text
            text_elem = container.find("div", class_=re.compile(r"review_item_review_content"))
            if text_elem:
                content_parts.append(text_elem.get_text(strip=True))

        if not content_parts or len(" ".join(content_parts)) < 20:
            return None

        # Extract reviewer info
        reviewer_elem = container.find("span", class_=re.compile(r"reviewer_name")) or \
                       container.find("span", class_=re.compile(r"bui-avatar-block__title"))
        reviewer = reviewer_elem.get_text(strip=True) if reviewer_elem else "Anonymous"

        # Extract country
        country_elem = container.find("span", class_=re.compile(r"reviewer_country")) or \
                      container.find("span", class_=re.compile(r"bui-avatar-block__subtitle"))
        country = country_elem.get_text(strip=True) if country_elem else None

        # Extract date
        date_elem = container.find("span", class_=re.compile(r"review_item_date")) or \
                   container.find("span", class_=re.compile(r"c-review-block__date"))
        published_at = None
        date_str = None
        if date_elem:
            date_str = date_elem.get_text(strip=True)
            # Try to parse date
            try:
                # Remove "Reviewed:" prefix if present
                date_str = re.sub(r"^Reviewed:\s*", "", date_str)
                for fmt in ["%B %d, %Y", "%d %B %Y", "%b %d, %Y", "%Y-%m-%d"]:
                    try:
                        published_at = datetime.strptime(date_str, fmt)
                        break
                    except ValueError:
                        continue
            except Exception:
                pass

        # Extract room type
        room_elem = container.find("div", class_=re.compile(r"room_info")) or \
                   container.find("span", class_=re.compile(r"c-review-block__room-link"))
        room_type = room_elem.get_text(strip=True) if room_elem else None

        # Extract traveler type
        traveler_elem = container.find("span", class_=re.compile(r"review_staytype")) or \
                       container.find("span", {"data-testid": "review-traveler-type"})
        traveler_type = traveler_elem.get_text(strip=True) if traveler_elem else None

        return {
            "review_id": review_id,
            "title": title,
            "positive": positive,
            "negative": negative,
            "text": "\n\n".join(content_parts),
            "rating": rating,
            "reviewer_name": reviewer,
            "reviewer_country": country,
            "date_str": date_str,
            "published_at": published_at,
            "room_type": room_type,
            "traveler_type": traveler_type,
            "property_name": property_name,
        }

    def _review_to_content(self, review: dict, property_path: str) -> RawContentCreate:
        """Convert parsed review to RawContentCreate.

        Args:
            review: Parsed review dict
            property_path: Booking.com property path

        Returns:
            RawContentCreate object
        """
        # Build unique URL
        url = f"{self.BASE_URL}/{property_path}.html#review-{review['review_id']}"

        # Use combined positive/negative as content
        content = review.get("text", "")

        return RawContentCreate(
            source=self.source_name,
            source_type=self.source_type,
            url=url,
            title=review.get("title", f"Review of {review.get('property_name', 'Hotel')}"),
            content=content,
            author=review.get("reviewer_name"),
            published_at=review.get("published_at"),
            metadata={
                "property_path": property_path,
                "property_name": review.get("property_name"),
                "rating": review.get("rating"),
                "reviewer_country": review.get("reviewer_country"),
                "room_type": review.get("room_type"),
                "traveler_type": review.get("traveler_type"),
                "positive": review.get("positive"),
                "negative": review.get("negative"),
            },
        )

    def _wait(self):
        """Wait between requests with randomized delay."""
        delay = random.uniform(self.min_delay, self.max_delay)
        logger.debug(f"Waiting {delay:.1f}s before next request")
        time.sleep(delay)

    def scrape(self) -> list[RawContentCreate]:
        """Scrape reviews from configured properties.

        Returns:
            List of RawContentCreate objects
        """
        items = []
        seen_urls = set()

        logger.info(f"Starting Booking.com scrape for {len(self.properties)} properties")

        for property_path, property_name in self.properties:
            logger.info(f"Fetching reviews for: {property_name}")
            property_reviews = 0
            offset = 0

            while property_reviews < self.max_reviews_per_property:
                url = self._build_review_url(property_path, offset)
                logger.debug(f"Fetching: {url}")

                response = self.fetch(url)
                if not response:
                    logger.warning(f"Failed to fetch reviews for {property_name}")
                    break

                reviews = self._parse_reviews(response.text, property_name)

                if not reviews:
                    logger.debug(f"No more reviews found for {property_name}")
                    break

                for review in reviews:
                    item = self._review_to_content(review, property_path)
                    if item.url not in seen_urls:
                        items.append(item)
                        seen_urls.add(item.url)
                        property_reviews += 1

                        if property_reviews >= self.max_reviews_per_property:
                            break

                # Move to next page
                offset += 25  # Booking.com typically shows ~25 reviews per page

                # Wait between requests
                self._wait()

            logger.info(f"Collected {property_reviews} reviews from {property_name}")

        logger.info(f"Booking.com scrape complete: {len(items)} total reviews")
        return items
