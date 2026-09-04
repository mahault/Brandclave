"""Culture, design, food and research feeds — the signals a hotel concept is
actually built from, not just hotel-industry news about hotels.

Every source here is a public RSS/Atom feed the publisher exposes on purpose.
All of them ride RSSNewsScraper; only the URL and a per-feed relevance filter
differ. Feeds that publish on everything (Pew, Dezeen) are filtered to
hospitality-adjacent items by keyword so the corpus stays on topic.
"""

import logging
import re

from data_models.raw_content import RawContentCreate
from ingestion.news.hospitality_news import RSSNewsScraper
from ingestion.registry import get_source_config

logger = logging.getLogger(__name__)

TAG_RE = re.compile(r"<[^>]+>")

# Broad-topic feeds keep only entries that touch travel, hospitality or the
# built environment of leisure. Matching is case-insensitive over title+summary.
DEFAULT_KEYWORDS = [
    "hotel", "hospitality", "resort", "travel", "tourism", "tourist", "guest",
    "hostel", "airbnb", "vacation", "holiday", "retreat", "spa", "wellness",
    "restaurant", "bar ", "cafe", "nightlife", "club", "festival", "destination",
    "city break", "leisure", "lodging", "boutique", "villa", "coworking", "nomad",
]


class FilteredRSSScraper(RSSNewsScraper):
    """RSSNewsScraper plus a keyword gate driven by sources.yaml."""

    filter_by_default = False

    def scrape(self) -> list[RawContentCreate]:
        items = super().scrape()
        cfg = get_source_config(self.source_name)
        if not cfg.get("filter_keywords", self.filter_by_default):
            return items
        keywords = [k.strip().lower() for k in cfg.get("keywords", DEFAULT_KEYWORDS)]
        # Word-boundary match on the title and the opening of the body only:
        # feeds that ship whole pages (Dezeen) carry every keyword in their
        # footers and tag clouds, which is not a signal about the article.
        pattern = re.compile(r"\b(" + "|".join(re.escape(k) for k in keywords) + r")\b")
        kept = []
        for item in items:
            body = TAG_RE.sub(" ", item.content or "")
            haystack = f"{item.title or ''} {body[:600]}".lower()
            if pattern.search(haystack):
                kept.append(item)
        logger.info(f"{self.source_name}: kept {len(kept)} of {len(items)} entries after keyword filter")
        return kept


class DezeenScraper(FilteredRSSScraper):
    """Architecture and design direction; hotels, restaurants and hospitality interiors."""

    source_name = "dezeen"
    RSS_URL = "https://www.dezeen.com/feed/"
    filter_by_default = True


class ArchDailyScraper(FilteredRSSScraper):
    """Built projects worldwide; filtered to hospitality typologies."""

    source_name = "archdaily"
    RSS_URL = "https://www.archdaily.com/rss/"
    filter_by_default = True


class EaterScraper(FilteredRSSScraper):
    """Food culture by city — F&B is half of a hotel concept."""

    source_name = "eater"
    RSS_URL = "https://www.eater.com/rss/index.xml"
    filter_by_default = False


class PewResearchScraper(FilteredRSSScraper):
    """Consumer and social attitude research; only travel/leisure-relevant pieces."""

    source_name = "pew_research"
    RSS_URL = "https://www.pewresearch.org/feed/"
    filter_by_default = True


class GlobalWellnessInstituteScraper(FilteredRSSScraper):
    """Wellness economy research; drives the wellness-hospitality trend line."""

    source_name = "global_wellness_institute"
    RSS_URL = "https://globalwellnessinstitute.org/feed/"
    filter_by_default = False


class HospitalityNetScraper(FilteredRSSScraper):
    """Hospitality Net industry news; RSS is served even when the site's pages
    sit behind Cloudflare, so it comes back off the blocked list."""

    source_name = "hospitalitynet"
    RSS_URL = "https://www.hospitalitynet.org/rss/news.xml"
    filter_by_default = False
