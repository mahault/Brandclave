"""YouTube scraper for hospitality video content (no API key required)."""

import json
import logging
import re
import time
from datetime import datetime

from bs4 import BeautifulSoup

from data_models.raw_content import RawContentCreate, SourceType
from ingestion.base_scraper import BaseScraper

logger = logging.getLogger(__name__)


class YouTubeScraper(BaseScraper):
    """Scraper for YouTube search results."""

    source_name = "youtube"
    source_type = SourceType.SOCIAL

    # Search queries for hospitality content
    DEFAULT_QUERIES = [
        "where to stay in lisbon",
        "where to stay in barcelona",
        "where to stay in paris",
        "where to stay in tokyo",
        "where to stay in bali",
        "best boutique hotels",
        "luxury hotel room tour",
        "hotel review 2024",
        "best hostels europe",
        "digital nomad accommodation",
        "hotel vs airbnb",
    ]

    def __init__(
        self,
        config_path: str = "configs/scraping.yaml",
        search_queries: list[str] | None = None,
    ):
        """Initialize YouTube scraper.

        Args:
            config_path: Path to scraping config
            search_queries: Search queries to use (uses defaults if None)
        """
        super().__init__(config_path)
        self.search_queries = search_queries or self.DEFAULT_QUERIES

        # Update headers for YouTube
        self.client.headers.update({
            "Accept-Language": "en-US,en;q=0.9",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        })

    def _search_youtube(self, query: str, max_results: int = 10) -> list[dict]:
        """Search YouTube and extract video info.

        Args:
            query: Search query
            max_results: Maximum results to return

        Returns:
            List of video data dicts
        """
        search_url = f"https://www.youtube.com/results?search_query={query.replace(' ', '+')}"

        response = self.fetch(search_url)
        if not response:
            return []

        try:
            html = response.text

            # YouTube embeds video data in a script tag as JSON
            # Look for ytInitialData
            pattern = r'var ytInitialData = ({.*?});'
            match = re.search(pattern, html)

            if not match:
                # Try alternate pattern
                pattern = r'ytInitialData\s*=\s*({.*?});'
                match = re.search(pattern, html)

            if not match:
                logger.warning(f"Could not find ytInitialData for query: {query}")
                return []

            data = json.loads(match.group(1))

            # Navigate to video results
            contents = (
                data.get("contents", {})
                .get("twoColumnSearchResultsRenderer", {})
                .get("primaryContents", {})
                .get("sectionListRenderer", {})
                .get("contents", [])
            )

            videos = []
            for section in contents:
                items = (
                    section.get("itemSectionRenderer", {})
                    .get("contents", [])
                )

                for item in items:
                    video_renderer = item.get("videoRenderer", {})
                    if not video_renderer:
                        continue

                    video_id = video_renderer.get("videoId")
                    if not video_id:
                        continue

                    # Extract title
                    title_runs = video_renderer.get("title", {}).get("runs", [])
                    title = title_runs[0].get("text", "") if title_runs else ""

                    # Extract description snippet
                    desc_runs = video_renderer.get("detailedMetadataSnippets", [])
                    description = ""
                    if desc_runs:
                        snippet_runs = desc_runs[0].get("snippetText", {}).get("runs", [])
                        description = "".join(r.get("text", "") for r in snippet_runs)

                    # Extract view count
                    view_text = video_renderer.get("viewCountText", {}).get("simpleText", "")
                    views = self._parse_view_count(view_text)

                    # Extract channel
                    channel_runs = video_renderer.get("ownerText", {}).get("runs", [])
                    channel = channel_runs[0].get("text", "") if channel_runs else ""

                    # Extract publish time
                    publish_text = video_renderer.get("publishedTimeText", {}).get("simpleText", "")

                    videos.append({
                        "video_id": video_id,
                        "title": title,
                        "description": description,
                        "views": views,
                        "channel": channel,
                        "publish_text": publish_text,
                        "url": f"https://www.youtube.com/watch?v={video_id}",
                    })

                    if len(videos) >= max_results:
                        break

                if len(videos) >= max_results:
                    break

            return videos

        except Exception as e:
            logger.error(f"Error parsing YouTube results for '{query}': {e}")
            return []

    def _parse_view_count(self, view_text: str) -> int:
        """Parse view count from text like '1.2M views'.

        Args:
            view_text: View count text

        Returns:
            Integer view count
        """
        if not view_text:
            return 0

        view_text = view_text.lower().replace(",", "").replace(" views", "").strip()

        try:
            if "k" in view_text:
                return int(float(view_text.replace("k", "")) * 1000)
            elif "m" in view_text:
                return int(float(view_text.replace("m", "")) * 1000000)
            elif "b" in view_text:
                return int(float(view_text.replace("b", "")) * 1000000000)
            else:
                return int(view_text)
        except (ValueError, TypeError):
            return 0

    def _video_to_content(self, video: dict, query: str) -> RawContentCreate | None:
        """Convert YouTube video to RawContentCreate.

        Args:
            video: Video data dict
            query: Search query used

        Returns:
            RawContentCreate or None if invalid
        """
        title = video.get("title", "")
        description = video.get("description", "")
        content = f"{title}\n\n{description}".strip()

        if len(content) < 10:
            return None

        return RawContentCreate(
            source=self.source_name,
            source_type=self.source_type,
            url=video.get("url", ""),
            title=title,
            content=content,
            author=video.get("channel"),
            published_at=None,  # YouTube doesn't give exact dates in search
            metadata={
                "video_id": video.get("video_id"),
                "views": video.get("views", 0),
                "channel": video.get("channel"),
                "publish_text": video.get("publish_text"),
                "search_query": query,
            },
        )

    def scrape(self) -> list[RawContentCreate]:
        """Scrape YouTube search results for hospitality content.

        Returns:
            List of RawContentCreate objects
        """
        items = []
        seen_urls = set()

        for query in self.search_queries:
            logger.info(f"Searching YouTube for '{query}'...")
            videos = self._search_youtube(query, max_results=10)

            for video in videos:
                item = self._video_to_content(video, query)
                if item and item.url not in seen_urls:
                    items.append(item)
                    seen_urls.add(item.url)

            # Be nice to YouTube servers
            time.sleep(2)

        logger.info(f"Scraped {len(items)} videos from YouTube")
        return items


def scrape_youtube() -> dict:
    """Convenience function to run the YouTube scraper.

    Returns:
        Scrape summary dict
    """
    with YouTubeScraper() as scraper:
        return scraper.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = scrape_youtube()
    print(result)
