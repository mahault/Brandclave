"""Bluesky scraper — consumer travel voice via the AT Protocol public search API.

Unauthenticated, 3,000 requests / 5 min / IP, no commercial restriction in the
docs — the cleanest legal position among the social sources (plan §4.2).
Queries come from configs/sources.yaml.
"""

import logging
from datetime import datetime

from data_models.raw_content import RawContentCreate, SourceType
from ingestion.base_scraper import BaseScraper
from ingestion.registry import get_source_config

logger = logging.getLogger(__name__)

# Note: public.api.bsky.app's searchPosts sits behind a WAF that 403s scripted
# clients (verified 2026-07-27); api.bsky.app serves the same lexicon openly.
SEARCH_URL = "https://api.bsky.app/xrpc/app.bsky.feed.searchPosts"
USER_AGENT = "BrandClaveDemandBot/1.0 (https://github.com/mahault/Brandclave; mahault.albarracin@gmail.com)"

DEFAULT_QUERIES = ["hotel stay", "boutique hotel", "where to stay"]


class BlueskyScraper(BaseScraper):
    """Search Bluesky for hospitality demand conversations."""

    source_name = "bluesky"
    source_type = SourceType.SOCIAL

    def scrape(self) -> list[RawContentCreate]:
        cfg = get_source_config(self.source_name)
        queries = cfg.get("queries", DEFAULT_QUERIES)
        max_per_query = min(int(cfg.get("max_per_query", 50)), 100)
        lang = cfg.get("lang")

        items: list[RawContentCreate] = []
        for query in queries:
            params = {"q": query, "limit": max_per_query}
            if lang:
                params["lang"] = lang

            response = self.fetch(SEARCH_URL, params=params, headers={"User-Agent": USER_AGENT})
            if response is None:
                logger.warning(f"Bluesky search failed for query '{query}', continuing")
                continue

            try:
                posts = response.json().get("posts", [])
            except ValueError:
                logger.warning(f"Bluesky returned non-JSON for query '{query}'")
                continue

            for post in posts:
                item = self._post_to_item(post, query)
                if item is not None:
                    items.append(item)

        logger.info(f"Bluesky: collected {len(items)} posts across {len(queries)} queries")
        return items

    def _post_to_item(self, post: dict, query: str) -> RawContentCreate | None:
        """Convert one Bluesky post to RawContentCreate. Returns None for unusable posts."""
        try:
            record = post.get("record") or {}
            text = (record.get("text") or "").strip()
            if not text:
                return None

            author = post.get("author") or {}
            handle = author.get("handle", "unknown")

            # at://did:plc:xxx/app.bsky.feed.post/rkey -> https://bsky.app/profile/handle/post/rkey
            uri = post.get("uri", "")
            rkey = uri.rsplit("/", 1)[-1] if uri else None
            if not rkey:
                return None
            url = f"https://bsky.app/profile/{handle}/post/{rkey}"

            return RawContentCreate(
                source=self.source_name,
                source_type=self.source_type,
                url=url,
                title=None,
                content=text,
                author=author.get("displayName") or handle,
                published_at=self._parse_date(record.get("createdAt")),
                metadata={
                    "query": query,
                    "handle": handle,
                    "like_count": post.get("likeCount", 0),
                    "repost_count": post.get("repostCount", 0),
                    "reply_count": post.get("replyCount", 0),
                    "langs": record.get("langs", []),
                },
            )
        except Exception as exc:
            logger.debug(f"Skipping malformed Bluesky post: {exc}")
            return None

    @staticmethod
    def _parse_date(value: str | None) -> datetime | None:
        if not value:
            return None
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)
        except ValueError:
            return None


def scrape_bluesky() -> dict:
    """Run the Bluesky scraper standalone."""
    with BlueskyScraper() as scraper:
        return scraper.run()
