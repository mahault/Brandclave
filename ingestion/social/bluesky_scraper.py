"""Bluesky scraper — consumer travel voice via the AT Protocol search API.

Queries come from configs/sources.yaml.

Access model (revised 2026-09-04): the unauthenticated public AppViews are gone.
`public.api.bsky.app` began 403ing scripted clients in July 2026 and
`api.bsky.app` — the workaround adopted at the time — now does the same behind
the same openresty WAF. `bsky.social` answers the identical lexicon with 401
rather than 403, so the endpoint is open to an authenticated session. The
scraper therefore creates an AT Protocol session from a handle plus an app
password and calls searchPosts with the returned access JWT.

Credentials are optional: without them the scraper logs a warning and returns
nothing rather than failing the run, so the rest of the pipeline is unaffected.
"""

import logging
import threading
from datetime import datetime

import httpx

from config.settings import get_settings
from data_models.raw_content import RawContentCreate, SourceType
from ingestion.base_scraper import BaseScraper
from ingestion.registry import get_source_config

logger = logging.getLogger(__name__)

PDS_URL = "https://bsky.social"
SESSION_URL = f"{PDS_URL}/xrpc/com.atproto.server.createSession"
SEARCH_URL = f"{PDS_URL}/xrpc/app.bsky.feed.searchPosts"
USER_AGENT = "BrandClaveDemandBot/1.0 (https://github.com/mahault/Brandclave; mahault.albarracin@gmail.com)"

DEFAULT_QUERIES = ["hotel stay", "boutique hotel", "where to stay"]


class BlueskyAuthError(RuntimeError):
    """Raised when an AT Protocol session cannot be established."""


class _SessionCache:
    """Process-wide access token, refreshed on demand.

    Sessions are reusable across scrape runs; creating one per query would burn
    the account's rate limit for no benefit.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._token: str | None = None
        self._identifier: str | None = None

    def token(self, identifier: str, password: str, *, refresh: bool = False) -> str:
        with self._lock:
            if self._token and not refresh and self._identifier == identifier:
                return self._token
            self._token = _create_session(identifier, password)
            self._identifier = identifier
            return self._token

    def clear(self) -> None:
        with self._lock:
            self._token = None
            self._identifier = None


_SESSIONS = _SessionCache()


def _create_session(identifier: str, password: str) -> str:
    """Exchange a handle plus app password for an access JWT."""
    try:
        response = httpx.post(
            SESSION_URL,
            json={"identifier": identifier, "password": password},
            headers={"User-Agent": USER_AGENT},
            timeout=20,
        )
    except httpx.HTTPError as exc:
        raise BlueskyAuthError(f"Could not reach {SESSION_URL}: {exc}") from exc

    if response.status_code == 401:
        raise BlueskyAuthError(
            "Bluesky rejected the credentials. BLUESKY_HANDLE must be the full "
            "handle (e.g. brandclave.bsky.social) and BLUESKY_APP_PASSWORD an app "
            "password from Settings -> Privacy and security -> App passwords, "
            "not the account password."
        )
    if response.status_code != 200:
        raise BlueskyAuthError(
            f"createSession failed with HTTP {response.status_code}: {response.text[:200]}"
        )

    token = response.json().get("accessJwt")
    if not token:
        raise BlueskyAuthError("createSession returned no accessJwt")
    return token


class BlueskyScraper(BaseScraper):
    """Search Bluesky for hospitality demand conversations."""

    source_name = "bluesky"
    source_type = SourceType.SOCIAL

    def scrape(self) -> list[RawContentCreate]:
        settings = get_settings()
        handle = settings.bluesky_handle
        password = settings.bluesky_app_password

        if not handle or not password:
            logger.warning(
                "Bluesky skipped: BLUESKY_HANDLE / BLUESKY_APP_PASSWORD are not set. "
                "Public search is WAF-blocked, so an authenticated session is required."
            )
            return []

        try:
            token = _SESSIONS.token(handle, password)
        except BlueskyAuthError as exc:
            logger.error(f"Bluesky authentication failed, skipping source: {exc}")
            return []

        cfg = get_source_config(self.source_name)
        queries = cfg.get("queries", DEFAULT_QUERIES)
        max_per_query = min(int(cfg.get("max_per_query", 50)), 100)
        lang = cfg.get("lang")

        items: list[RawContentCreate] = []
        for query in queries:
            params = {"q": query, "limit": max_per_query}
            if lang:
                params["lang"] = lang

            posts, token = self._search(query, params, token, handle, password)
            for post in posts:
                item = self._post_to_item(post, query)
                if item is not None:
                    items.append(item)

        logger.info(f"Bluesky: collected {len(items)} posts across {len(queries)} queries")
        return items

    def _search(
        self, query: str, params: dict, token: str, handle: str, password: str
    ) -> tuple[list[dict], str]:
        """Run one search, refreshing the session once if the token has expired.

        Returns the posts and the token to use next: a refresh made mid-run would
        otherwise be discarded, and every later query would repeat the same 401.
        """
        for attempt in (1, 2):
            headers = {"User-Agent": USER_AGENT, "Authorization": f"Bearer {token}"}
            response = self.fetch(SEARCH_URL, params=params, headers=headers)

            if response is None:
                # Access tokens are short-lived and fetch() swallows the 401, so
                # retry once with a fresh session before giving up on the query.
                if attempt == 1:
                    try:
                        token = _SESSIONS.token(handle, password, refresh=True)
                        continue
                    except BlueskyAuthError as exc:
                        logger.error(f"Bluesky re-authentication failed: {exc}")
                        return [], token
                logger.warning(f"Bluesky search failed for query '{query}', continuing")
                return [], token

            try:
                return response.json().get("posts", []), token
            except ValueError:
                logger.warning(f"Bluesky returned non-JSON for query '{query}'")
                return [], token

        return [], token

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
