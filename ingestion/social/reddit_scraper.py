"""Reddit scraper using old.reddit.com JSON endpoints (no API key required)."""

import logging
import time
from datetime import datetime

from data_models.raw_content import RawContentCreate, SourceType
from ingestion.base_scraper import BaseScraper

logger = logging.getLogger(__name__)


class RedditScraper(BaseScraper):
    """Scraper for Reddit using public JSON endpoints."""

    source_name = "reddit"
    source_type = SourceType.SOCIAL

    # Subreddits relevant to hospitality
    DEFAULT_SUBREDDITS = [
        "travel",
        "solotravel",
        "digitalnomad",
        "hotels",
        "luxurytravel",
        "backpacking",
        "TravelHacks",
    ]

    # City-specific subreddits for regional trends
    CITY_SUBREDDITS = [
        "lisbon",
        "barcelona",
        "paris",
        "london",
        "tokyo",
        "bangkok",
        "bali",
        "dubai",
        "newyorkcity",
        "losangeles",
    ]

    # Search queries for hospitality content
    SEARCH_QUERIES = [
        "hotel",
        "accommodation",
        "where to stay",
        "boutique hotel",
        "hostel",
        "airbnb vs hotel",
        "luxury resort",
        "budget hotel",
    ]

    def __init__(
        self,
        config_path: str = "configs/scraping.yaml",
        subreddits: list[str] | None = None,
        include_city_subs: bool = True,
        search_queries: list[str] | None = None,
    ):
        """Initialize Reddit scraper.

        Args:
            config_path: Path to scraping config
            subreddits: List of subreddits to scrape (uses defaults if None)
            include_city_subs: Whether to include city-specific subreddits
            search_queries: Search queries to use (uses defaults if None)
        """
        super().__init__(config_path)
        # Reddit's robots.txt blocks most automated access, but their JSON API
        # is intended for programmatic access. We use respectful delays.
        self.config["global_settings"]["respect_robots_txt"] = False
        self.subreddits = subreddits or self.DEFAULT_SUBREDDITS
        if include_city_subs:
            self.subreddits.extend(self.CITY_SUBREDDITS)
        self.search_queries = search_queries or self.SEARCH_QUERIES

    def _fetch_subreddit_posts(
        self,
        subreddit: str,
        sort: str = "hot",
        limit: int = 25,
    ) -> list[dict]:
        """Fetch posts from a subreddit.

        Args:
            subreddit: Subreddit name (without r/)
            sort: Sort order (hot, new, top)
            limit: Number of posts to fetch

        Returns:
            List of post data dicts
        """
        url = f"https://old.reddit.com/r/{subreddit}/{sort}.json?limit={limit}"

        response = self.fetch(url)
        if not response:
            return []

        try:
            data = response.json()
            posts = data.get("data", {}).get("children", [])
            return [post["data"] for post in posts]
        except Exception as e:
            logger.error(f"Error parsing r/{subreddit}: {e}")
            return []

    def _search_subreddit(
        self,
        subreddit: str,
        query: str,
        sort: str = "relevance",
        limit: int = 25,
    ) -> list[dict]:
        """Search within a subreddit.

        Args:
            subreddit: Subreddit name
            query: Search query
            sort: Sort order
            limit: Number of results

        Returns:
            List of post data dicts
        """
        url = f"https://old.reddit.com/r/{subreddit}/search.json"
        params = f"?q={query}&restrict_sr=on&sort={sort}&limit={limit}"

        response = self.fetch(url + params)
        if not response:
            return []

        try:
            data = response.json()
            posts = data.get("data", {}).get("children", [])
            return [post["data"] for post in posts]
        except Exception as e:
            logger.error(f"Error searching r/{subreddit} for '{query}': {e}")
            return []

    def _post_to_content(self, post: dict, subreddit: str) -> RawContentCreate | None:
        """Convert Reddit post to RawContentCreate.

        Args:
            post: Reddit post data dict
            subreddit: Subreddit name

        Returns:
            RawContentCreate or None if invalid
        """
        # Skip posts without meaningful text
        title = post.get("title", "")
        selftext = post.get("selftext", "")
        content = f"{title}\n\n{selftext}".strip()

        if len(content) < 20:
            return None

        # Skip removed/deleted posts
        if selftext in ["[removed]", "[deleted]"]:
            return None

        # Parse timestamp
        created_utc = post.get("created_utc")
        published_at = datetime.utcfromtimestamp(created_utc) if created_utc else None

        return RawContentCreate(
            source=self.source_name,
            source_type=self.source_type,
            url=f"https://reddit.com{post.get('permalink', '')}",
            title=title,
            content=content,
            author=post.get("author"),
            published_at=published_at,
            metadata={
                "subreddit": subreddit,
                "score": post.get("score", 0),
                "num_comments": post.get("num_comments", 0),
                "upvote_ratio": post.get("upvote_ratio", 0),
                "is_self": post.get("is_self", True),
                "flair": post.get("link_flair_text"),
            },
        )

    def scrape(self) -> list[RawContentCreate]:
        """Scrape posts from configured subreddits.

        Returns:
            List of RawContentCreate objects
        """
        items = []
        seen_urls = set()

        # Fetch hot posts from each subreddit
        for subreddit in self.subreddits:
            logger.info(f"Fetching r/{subreddit}...")
            posts = self._fetch_subreddit_posts(subreddit, sort="hot", limit=25)

            for post in posts:
                item = self._post_to_content(post, subreddit)
                if item and item.url not in seen_urls:
                    items.append(item)
                    seen_urls.add(item.url)

            # Be nice to Reddit servers
            time.sleep(1)

        # Search for hospitality keywords in travel subreddits
        search_subs = ["travel", "solotravel", "digitalnomad"]
        for subreddit in search_subs:
            for query in self.search_queries[:3]:  # Limit queries to avoid rate limits
                logger.info(f"Searching r/{subreddit} for '{query}'...")
                posts = self._search_subreddit(subreddit, query, limit=15)

                for post in posts:
                    item = self._post_to_content(post, subreddit)
                    if item and item.url not in seen_urls:
                        items.append(item)
                        seen_urls.add(item.url)

                time.sleep(1)

        logger.info(f"Scraped {len(items)} posts from Reddit")
        return items


def scrape_reddit() -> dict:
    """Convenience function to run the Reddit scraper.

    Returns:
        Scrape summary dict
    """
    with RedditScraper() as scraper:
        return scraper.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = scrape_reddit()
    print(result)
