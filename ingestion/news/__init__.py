"""News scrapers for BrandClave Aggregator."""

from ingestion.news.hospitalitynet_rss import HospitalityNetScraper, scrape_hospitalitynet
from ingestion.news.skift_rss import SkiftScraper, scrape_skift

__all__ = [
    "HospitalityNetScraper",
    "scrape_hospitalitynet",
    "SkiftScraper",
    "scrape_skift",
]
