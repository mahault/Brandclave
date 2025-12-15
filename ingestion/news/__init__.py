"""News scrapers for BrandClave Aggregator."""

from ingestion.news.hospitalitynet_rss import HospitalityNetScraper, scrape_hospitalitynet
from ingestion.news.skift_rss import SkiftScraper, scrape_skift
from ingestion.news.hospitality_news import (
    HotelDiveScraper,
    HotelManagementScraper,
    PhocusWireScraper,
    TravelWeeklyScraper,
    HotelNewsResourceScraper,
    TravelDailyNewsScraper,
    BusinessTravelNewsScraper,
    BoutiqueHotelierScraper,
    HotelOnlineScraper,
    HotelTechReportScraper,
    TopHotelNewsScraper,
)

__all__ = [
    "HospitalityNetScraper",
    "scrape_hospitalitynet",
    "SkiftScraper",
    "scrape_skift",
    "HotelDiveScraper",
    "HotelManagementScraper",
    "PhocusWireScraper",
    "TravelWeeklyScraper",
    "HotelNewsResourceScraper",
    "TravelDailyNewsScraper",
    "BusinessTravelNewsScraper",
    "BoutiqueHotelierScraper",
    "HotelOnlineScraper",
    "HotelTechReportScraper",
    "TopHotelNewsScraper",
]
