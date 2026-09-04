"""Mastodon — consumer travel voice from public hashtag timelines.

`/api/v1/timelines/tag/:hashtag` is public on most instances (no token),
rate-limited at 300 requests per 5 minutes per IP. Instances and hashtags come
from configs/sources.yaml; the scraper walks each instance × hashtag once per
run and keeps the budget far below the limit.
"""

import logging
import re
from datetime import datetime

from data_models.raw_content import RawContentCreate, SourceType
from ingestion.base_scraper import BaseScraper
from ingestion.registry import get_source_config

logger = logging.getLogger(__name__)

USER_AGENT = "BrandClaveDemandBot/1.0 (https://github.com/mahault/Brandclave; mahault.albarracin@gmail.com)"
TAG_RE = re.compile(r"<[^>]+>")


def _looks_like_a_post(text: str, acct: str | None, seen_prefixes: dict[str, int]) -> bool:
    """Reject the two spam shapes hashtag timelines carry: emoji walls with
    little prose, and the same message re-posted under many tags."""
    letters = sum(ch.isalpha() for ch in text)
    if letters < 0.5 * max(len(text), 1):
        return False
    opening = text[:60]
    symbols = sum(1 for ch in opening if not (ch.isalnum() or ch.isspace() or ch in ".,;:!?'\"()-#@/&"))
    if symbols > 6:
        return False
    key = (acct or "") + "|" + re.sub(r"\W+", "", text.lower())[:60]
    seen_prefixes[key] = seen_prefixes.get(key, 0) + 1
    return seen_prefixes[key] <= 1


def _strip_html(html: str) -> str:
    text = TAG_RE.sub(" ", html or "")
    return re.sub(r"\s+", " ", text).replace("&amp;", "&").replace("&#39;", "'").replace("&quot;", '"').strip()


class MastodonScraper(BaseScraper):
    """Public hashtag timelines across configured instances."""

    source_name = "mastodon"
    source_type = SourceType.SOCIAL

    def scrape(self) -> list[RawContentCreate]:
        cfg = get_source_config(self.source_name)
        instances = cfg.get("instances", ["mastodon.social"])
        hashtags = cfg.get("hashtags", ["hotel", "boutiquehotel", "travel", "hospitality"])
        limit = min(int(cfg.get("limit_per_tag", 40)), 40)
        langs = set(cfg.get("langs", ["en"]))

        items: list[RawContentCreate] = []
        seen: set[str] = set()
        seen_prefixes: dict[str, int] = {}
        for instance in instances:
            for tag in hashtags:
                url = f"https://{instance}/api/v1/timelines/tag/{tag}"
                response = self.fetch(url, params={"limit": limit}, headers={"User-Agent": USER_AGENT})
                if response is None:
                    logger.warning(f"Mastodon {instance} #{tag} failed, continuing")
                    continue
                try:
                    statuses = response.json()
                except ValueError:
                    continue
                for status in statuses:
                    item = self._status_to_item(status, instance, tag, langs)
                    if item is None or item.url in seen:
                        continue
                    if not _looks_like_a_post(item.content, item.metadata.get("acct"), seen_prefixes):
                        continue
                    seen.add(item.url)
                    items.append(item)
        logger.info(f"Mastodon: collected {len(items)} statuses across {len(instances)} instances")
        return items

    def _status_to_item(self, status: dict, instance: str, tag: str, langs: set[str]) -> RawContentCreate | None:
        try:
            if status.get("language") and status["language"] not in langs:
                return None
            text = _strip_html(status.get("content", ""))
            if len(text) < 20:
                return None
            account = status.get("account") or {}
            created = status.get("created_at")
            published = datetime.fromisoformat(created.replace("Z", "+00:00")).replace(tzinfo=None) if created else None
            return RawContentCreate(
                source=self.source_name,
                source_type=self.source_type,
                url=status.get("url") or status.get("uri"),
                title=None,
                content=text,
                author=account.get("display_name") or account.get("acct"),
                published_at=published,
                metadata={
                    "instance": instance,
                    "hashtag": tag,
                    "acct": account.get("acct"),
                    "favourites": status.get("favourites_count", 0),
                    "reblogs": status.get("reblogs_count", 0),
                    "replies": status.get("replies_count", 0),
                },
            )
        except Exception as exc:
            logger.debug(f"Skipping malformed Mastodon status: {exc}")
            return None


def scrape_mastodon() -> dict:
    with MastodonScraper() as scraper:
        return scraper.run()
