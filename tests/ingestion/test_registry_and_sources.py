"""Tests for the source registry and the Bluesky / Wikimedia scrapers."""

from datetime import datetime
from unittest.mock import patch

import httpx
import pytest

from ingestion.registry import active_sources, get_registry, get_scraper_class, get_source
from ingestion.social.bluesky_scraper import BlueskyScraper
from ingestion.metrics.wikimedia_pageviews import WikimediaPageviewsScraper


# ── Registry ─────────────────────────────────────────────────────────────────

def test_registry_loads_and_active_sources_have_classes():
    registry = get_registry()
    assert len(registry) > 30
    for name in active_sources():
        assert registry[name].class_path, f"active source {name} missing class"


def test_blocked_sources_excluded_from_active():
    active = set(active_sources())
    assert "reddit" not in active
    assert "tripadvisor" not in active
    assert "booking" not in active
    assert "bluesky" in active
    assert "wikimedia_pageviews" in active


def test_planned_source_has_no_class_and_raises():
    spec = get_source("eurostat_tourism")
    assert spec.status == "planned"
    with pytest.raises(ValueError, match="planned"):
        get_scraper_class("eurostat_tourism")


def test_scraper_classes_import_for_all_implemented_sources():
    for name, spec in get_registry().items():
        if spec.class_path:
            cls = get_scraper_class(name)
            assert cls.source_name == name, f"{name}: class source_name mismatch ({cls.source_name})"


def test_active_metric_and_content_kinds():
    assert "wikimedia_pageviews" in active_sources(kind="metric")
    assert "wikimedia_pageviews" not in active_sources(kind="content")
    assert "bluesky" in active_sources(kind="content")


# ── Bluesky ──────────────────────────────────────────────────────────────────

BSKY_POST = {
    "uri": "at://did:plc:abc123/app.bsky.feed.post/3kxyz",
    "author": {"handle": "traveler.bsky.social", "displayName": "Ana Traveler"},
    "record": {
        "text": "The boutique hotel in Lisbon had a rooftop onsen. I never wanted to leave.",
        "createdAt": "2026-07-25T14:30:00.000Z",
        "langs": ["en"],
    },
    "likeCount": 42,
    "repostCount": 7,
    "replyCount": 3,
}


def _bluesky_scraper_with_transport(handler) -> BlueskyScraper:
    scraper = BlueskyScraper()
    scraper.client = httpx.Client(transport=httpx.MockTransport(handler), timeout=5.0)
    # Skip robots + inter-request delays in tests
    scraper._check_robots_txt = lambda url: True
    scraper._wait_between_requests = lambda: None
    return scraper


def test_bluesky_parses_posts():
    def handler(request):
        assert "searchPosts" in str(request.url)
        return httpx.Response(200, json={"posts": [BSKY_POST]})

    with patch("ingestion.social.bluesky_scraper.get_source_config") as cfg:
        cfg.return_value = {"queries": ["boutique hotel"], "max_per_query": 10, "lang": "en"}
        scraper = _bluesky_scraper_with_transport(handler)
        items = scraper.scrape()

    assert len(items) == 1
    item = items[0]
    assert item.source == "bluesky"
    assert item.url == "https://bsky.app/profile/traveler.bsky.social/post/3kxyz"
    assert "rooftop onsen" in item.content
    assert item.author == "Ana Traveler"
    assert item.published_at == datetime(2026, 7, 25, 14, 30)
    assert item.metadata["like_count"] == 42


def test_bluesky_dead_api_returns_empty_not_raises():
    def handler(request):
        return httpx.Response(500)

    with patch("ingestion.social.bluesky_scraper.get_source_config") as cfg:
        cfg.return_value = {"queries": ["hotel stay"], "max_per_query": 10}
        with patch("ingestion.http_client._sleep"):
            scraper = _bluesky_scraper_with_transport(handler)
            assert scraper.scrape() == []


def test_bluesky_skips_empty_and_malformed_posts():
    posts = [
        BSKY_POST,
        {"uri": "at://x/app.bsky.feed.post/1", "author": {"handle": "h"}, "record": {"text": "   "}},
        {"author": {"handle": "h"}, "record": {"text": "no uri"}},
    ]

    def handler(request):
        return httpx.Response(200, json={"posts": posts})

    with patch("ingestion.social.bluesky_scraper.get_source_config") as cfg:
        cfg.return_value = {"queries": ["q"], "max_per_query": 10}
        scraper = _bluesky_scraper_with_transport(handler)
        assert len(scraper.scrape()) == 1


# ── Wikimedia pageviews ──────────────────────────────────────────────────────

def _wikimedia_scraper_with_transport(handler) -> WikimediaPageviewsScraper:
    scraper = WikimediaPageviewsScraper()
    scraper.client = httpx.Client(transport=httpx.MockTransport(handler), timeout=5.0)
    scraper._check_robots_txt = lambda url: True
    scraper._wait_between_requests = lambda: None
    return scraper


def test_wikimedia_parses_daily_points():
    def handler(request):
        assert "per-article/en.wikipedia" in str(request.url)
        assert "BrandClaveDemandBot" in request.headers["user-agent"]
        return httpx.Response(200, json={"items": [
            {"article": "Lisbon", "timestamp": "2026072000", "views": 15432},
            {"article": "Lisbon", "timestamp": "2026072100", "views": 16001},
        ]})

    with patch("ingestion.metrics.wikimedia_pageviews.get_source_config") as cfg:
        cfg.return_value = {
            "days_back": 7,
            "metric": "wikipedia_pageviews",
            "cities": [{"city": "Lisbon", "country": "Portugal", "article": "Lisbon"}],
        }
        scraper = _wikimedia_scraper_with_transport(handler)
        points = scraper.scrape()

    assert len(points) == 2
    point = points[0]
    assert point.city == "Lisbon"
    assert point.metric == "wikipedia_pageviews"
    assert point.date == datetime(2026, 7, 20)
    assert point.value == 15432.0


def test_wikimedia_dead_city_continues():
    calls = []

    def handler(request):
        calls.append(str(request.url))
        if "Lisbon" in str(request.url):
            return httpx.Response(404)
        return httpx.Response(200, json={"items": [
            {"article": "Porto", "timestamp": "2026072000", "views": 5000},
        ]})

    with patch("ingestion.metrics.wikimedia_pageviews.get_source_config") as cfg:
        cfg.return_value = {
            "days_back": 7,
            "cities": [
                {"city": "Lisbon", "country": "Portugal", "article": "Lisbon"},
                {"city": "Porto", "country": "Portugal", "article": "Porto"},
            ],
        }
        scraper = _wikimedia_scraper_with_transport(handler)
        points = scraper.scrape()

    assert len(calls) == 2
    assert len(points) == 1
    assert points[0].city == "Porto"
