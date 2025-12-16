"""Base scraper class with common functionality."""

import logging
import random
import time
from abc import ABC, abstractmethod
from datetime import datetime
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import httpx
import yaml

from data_models.raw_content import RawContent, RawContentCreate, SourceType
from db.database import SessionLocal
from db.models import RawContentModel, ProcessingJobModel

logger = logging.getLogger(__name__)


class BaseScraper(ABC):
    """Abstract base class for all scrapers."""

    source_name: str = "base"
    source_type: SourceType = SourceType.NEWS

    def __init__(self, config_path: str = "configs/scraping.yaml"):
        """Initialize the scraper with configuration.

        Args:
            config_path: Path to scraping configuration YAML
        """
        self.config = self._load_config(config_path)
        self.client = self._create_client()
        self._robots_cache: dict[str, RobotFileParser] = {}
        self._request_count = 0
        self._last_request_time: float | None = None

    def _load_config(self, config_path: str) -> dict:
        """Load scraping configuration."""
        try:
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config not found at {config_path}, using defaults")
            return {
                "global_settings": {
                    "request_timeout": 30,
                    "retry_attempts": 3,
                    "retry_backoff": 2,
                    "respect_robots_txt": True,
                },
                "user_agents": {
                    "default": "BrandClave-Aggregator/1.0",
                },
                "delays": {
                    "between_requests": {"min": 2, "max": 5},
                },
            }

    def _create_client(self) -> httpx.Client:
        """Create HTTP client with configured settings."""
        timeout = self.config.get("global_settings", {}).get("request_timeout", 30)
        user_agent = self.config.get("user_agents", {}).get("default", "BrandClave-Aggregator/1.0")

        return httpx.Client(
            timeout=timeout,
            headers={"User-Agent": user_agent},
            follow_redirects=True,
        )

    def _get_delay(self) -> float:
        """Get random delay between requests."""
        delays = self.config.get("delays", {}).get("between_requests", {"min": 2, "max": 5})
        return random.uniform(delays.get("min", 2), delays.get("max", 5))

    def _wait_between_requests(self) -> None:
        """Wait appropriate time between requests."""
        if self._last_request_time is not None:
            elapsed = time.time() - self._last_request_time
            delay = self._get_delay()
            if elapsed < delay:
                time.sleep(delay - elapsed)
        self._last_request_time = time.time()

    def _check_robots_txt(self, url: str) -> bool:
        """Check if URL is allowed by robots.txt.

        Args:
            url: URL to check

        Returns:
            True if allowed, False if disallowed
        """
        if not self.config.get("global_settings", {}).get("respect_robots_txt", True):
            return True

        parsed = urlparse(url)
        base_url = f"{parsed.scheme}://{parsed.netloc}"

        if base_url not in self._robots_cache:
            robots_url = f"{base_url}/robots.txt"
            rp = RobotFileParser()
            try:
                # Fetch robots.txt with timeout using our httpx client
                # instead of RobotFileParser.read() which has no timeout
                timeout = self.config.get("global_settings", {}).get("request_timeout", 30)
                response = self.client.get(robots_url, timeout=min(timeout, 10))

                if response.status_code == 200:
                    # Parse the robots.txt content manually
                    rp.parse(response.text.splitlines())
                else:
                    # No robots.txt or error - allow access
                    logger.debug(f"No robots.txt at {base_url} (status {response.status_code})")
                    self._robots_cache[base_url] = None
                    return True

            except Exception as e:
                logger.warning(f"Could not fetch robots.txt from {base_url}: {e}")
                # Allow if robots.txt is not available or times out
                self._robots_cache[base_url] = None
                return True
            self._robots_cache[base_url] = rp

        cached = self._robots_cache[base_url]
        if cached is None:
            return True

        user_agent = self.config.get("user_agents", {}).get("default", "*")
        return cached.can_fetch(user_agent, url)

    def fetch(self, url: str, **kwargs) -> httpx.Response | None:
        """Fetch URL with rate limiting and retry logic.

        Args:
            url: URL to fetch
            **kwargs: Additional arguments for httpx

        Returns:
            Response object or None if failed
        """
        if not self._check_robots_txt(url):
            logger.warning(f"URL disallowed by robots.txt: {url}")
            return None

        self._wait_between_requests()

        retry_attempts = self.config.get("global_settings", {}).get("retry_attempts", 3)
        retry_backoff = self.config.get("global_settings", {}).get("retry_backoff", 2)

        for attempt in range(retry_attempts):
            try:
                response = self.client.get(url, **kwargs)
                response.raise_for_status()
                self._request_count += 1
                return response

            except httpx.HTTPStatusError as e:
                logger.warning(f"HTTP error {e.response.status_code} for {url}")
                if e.response.status_code == 429:  # Rate limited
                    wait_time = (retry_backoff ** attempt) * 10
                    logger.info(f"Rate limited, waiting {wait_time}s")
                    time.sleep(wait_time)
                elif e.response.status_code >= 500:
                    wait_time = retry_backoff ** attempt
                    time.sleep(wait_time)
                else:
                    return None

            except httpx.RequestError as e:
                logger.error(f"Request error for {url}: {e}")
                wait_time = retry_backoff ** attempt
                time.sleep(wait_time)

        logger.error(f"Failed to fetch {url} after {retry_attempts} attempts")
        return None

    @abstractmethod
    def scrape(self) -> list[RawContentCreate]:
        """Scrape content from the source.

        Returns:
            List of RawContentCreate objects
        """
        pass

    def save_content(self, items: list[RawContentCreate]) -> int:
        """Save scraped content to database, deduplicating by URL.

        Args:
            items: List of RawContentCreate objects

        Returns:
            Number of new items saved
        """
        db = SessionLocal()
        saved_count = 0

        try:
            for item in items:
                # Check for existing URL
                existing = (
                    db.query(RawContentModel).filter(RawContentModel.url == item.url).first()
                )
                if existing:
                    logger.debug(f"Skipping duplicate URL: {item.url}")
                    continue

                # Create new record
                db_item = RawContentModel(
                    source=item.source,
                    source_type=item.source_type.value if hasattr(item.source_type, "value") else item.source_type,
                    url=item.url,
                    title=item.title,
                    content=item.content,
                    author=item.author,
                    published_at=item.published_at,
                    scraped_at=datetime.utcnow(),
                    metadata_json=item.metadata,
                )
                db.add(db_item)
                saved_count += 1

            db.commit()
            logger.info(f"Saved {saved_count} new items from {self.source_name}")

        except Exception as e:
            db.rollback()
            logger.error(f"Error saving content: {e}")
            raise

        finally:
            db.close()

        return saved_count

    def run(self) -> dict:
        """Run the scraper and return summary.

        Returns:
            Summary dict with scrape results
        """
        db = SessionLocal()
        job = ProcessingJobModel(
            job_type="scrape",
            source=self.source_name,
            status="running",
            started_at=datetime.utcnow(),
        )
        db.add(job)
        db.commit()
        job_id = job.id

        try:
            items = self.scrape()
            saved = self.save_content(items)

            # Update job
            job = db.query(ProcessingJobModel).filter(ProcessingJobModel.id == job_id).first()
            job.status = "completed"
            job.completed_at = datetime.utcnow()
            job.items_processed = saved
            db.commit()

            return {
                "source": self.source_name,
                "items_scraped": len(items),
                "items_saved": saved,
                "status": "completed",
            }

        except Exception as e:
            job = db.query(ProcessingJobModel).filter(ProcessingJobModel.id == job_id).first()
            job.status = "failed"
            job.completed_at = datetime.utcnow()
            job.error_message = str(e)
            db.commit()

            logger.error(f"Scraper {self.source_name} failed: {e}")
            return {
                "source": self.source_name,
                "status": "failed",
                "error": str(e),
            }

        finally:
            db.close()

    def close(self) -> None:
        """Close the HTTP client."""
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
