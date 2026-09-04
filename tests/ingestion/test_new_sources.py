"""Tests for the September 2026 source expansion: culture feeds, Eurostat,
Overpass, Inside Airbnb, Mastodon, GDELT and SEC EDGAR."""

from datetime import datetime
from unittest.mock import patch

import httpx

from ingestion.metrics.eurostat_tourism import EurostatTourismScraper
from ingestion.metrics.inside_airbnb import InsideAirbnbScraper
from ingestion.metrics.osm_overpass import OSMOverpassScraper
from ingestion.news.culture_feeds import DezeenScraper
from ingestion.news.gdelt import GDELTScraper
from ingestion.news.sec_edgar import SECEdgarScraper
from ingestion.social.mastodon_scraper import MastodonScraper


def _with_transport(scraper, handler):
    scraper.client = httpx.Client(transport=httpx.MockTransport(handler), timeout=5.0)
    scraper._check_robots_txt = lambda url: True
    scraper._wait_between_requests = lambda: None
    return scraper


# ── Culture feeds ────────────────────────────────────────────────────────────

RSS = """<?xml version="1.0"?><rss version="2.0"><channel><title>Dezeen</title>
<item><title>Floral tiles by Casalgrande</title><link>https://x/tiles</link><description>New tile range.</description><pubDate>Mon, 01 Sep 2026 10:00:00 GMT</pubDate></item>
<item><title>Rooftop bar tops boutique hotel in Lisbon</title><link>https://x/hotel</link><description>A hotel conversion with a rooftop bar.</description><pubDate>Mon, 01 Sep 2026 11:00:00 GMT</pubDate></item>
</channel></rss>"""


def test_dezeen_keyword_filter_keeps_hospitality_items_only():
    def handler(request):
        return httpx.Response(200, text=RSS)

    with patch("ingestion.news.culture_feeds.get_source_config", return_value={"filter_keywords": True}):
        items = _with_transport(DezeenScraper(), handler).scrape()
    assert [i.title for i in items] == ["Rooftop bar tops boutique hotel in Lisbon"]
    assert items[0].source == "dezeen"


# ── Eurostat ─────────────────────────────────────────────────────────────────

JSONSTAT = {
    "id": ["freq", "c_resid", "unit", "nace_r2", "geo", "time"],
    "size": [1, 1, 1, 1, 2, 2],
    "dimension": {
        "freq": {"category": {"index": {"M": 0}}},
        "c_resid": {"category": {"index": {"TOTAL": 0}}},
        "unit": {"category": {"index": {"NR": 0}}},
        "nace_r2": {"category": {"index": {"I551-I553": 0}}},
        "geo": {"category": {"index": {"ES": 0, "PT": 1}, "label": {"ES": "Spain", "PT": "Portugal"}}},
        "time": {"category": {"index": {"2026-06": 0, "2026-07": 1}}},
    },
    "value": {"0": 100, "1": 110, "2": 50, "3": 55},
}


def test_eurostat_decodes_jsonstat_cube():
    def handler(request):
        assert "tour_occ_nim" in str(request.url)
        return httpx.Response(200, json=JSONSTAT)

    with patch("ingestion.metrics.eurostat_tourism.get_source_config", return_value={"geo": ["ES", "PT"]}):
        points = _with_transport(EurostatTourismScraper(), handler).scrape()
    by_key = {(p.city, p.date): p.value for p in points}
    assert by_key[("Spain", datetime(2026, 6, 1))] == 100
    assert by_key[("Spain", datetime(2026, 7, 1))] == 110
    assert by_key[("Portugal", datetime(2026, 7, 1))] == 55
    assert all(p.metric == "eurostat_nights_spent" for p in points)


# ── Overpass ─────────────────────────────────────────────────────────────────

def test_overpass_maps_counts_to_categories_in_order():
    def handler(request):
        assert "Lisbon" in str(request.url) and "admin_level" in str(request.url)
        return httpx.Response(200, json={"elements": [
            {"type": "count", "tags": {"total": "371"}},
            {"type": "count", "tags": {"total": "2100"}},
            {"type": "count", "tags": {"total": "640"}},
            {"type": "count", "tags": {"total": "95"}},
        ]})

    cfg = {"cities": [{"city": "Lisbon", "country": "Portugal", "osm_admin_level": 7}]}
    with patch("ingestion.metrics.osm_overpass.get_source_config", return_value=cfg), patch(
        "ingestion.metrics.osm_overpass.time.sleep"
    ):
        points = _with_transport(OSMOverpassScraper(), handler).scrape()
    values = {p.metric: p.value for p in points}
    assert values == {"osm_hotels": 371, "osm_restaurants": 2100, "osm_nightlife": 640, "osm_attractions": 95}


# ── Inside Airbnb ────────────────────────────────────────────────────────────

INDEX = """<a href="https://data.insideairbnb.com/portugal/lisbon/lisbon/2026-03-20/visualisations/listings.csv">x</a>
<a href="https://data.insideairbnb.com/portugal/lisbon/lisbon/2026-06-23/visualisations/listings.csv">y</a>"""
CSV = """id,name,room_type,price,reviews_per_month
1,A,Entire home/apt,120,2.0
2,B,Private room,40,1.0
3,C,Entire home/apt,200,
"""


def test_inside_airbnb_uses_latest_snapshot_and_summarises():
    def handler(request):
        url = str(request.url)
        if "get-the-data" in url:
            return httpx.Response(200, text=INDEX)
        assert "2026-06-23" in url
        return httpx.Response(200, text=CSV)

    cfg = {"cities": [{"city": "Lisbon", "country": "Portugal", "slug": "lisbon"}]}
    with patch("ingestion.metrics.inside_airbnb.get_source_config", return_value=cfg):
        points = _with_transport(InsideAirbnbScraper(), handler).scrape()
    values = {p.metric: p.value for p in points}
    assert values["airbnb_listings"] == 3
    assert abs(values["airbnb_entire_home_share"] - 2 / 3) < 1e-3
    assert values["airbnb_median_price"] == 120
    assert values["airbnb_reviews_per_month"] == 1.5
    assert points[0].date == datetime(2026, 6, 23)


# ── Mastodon ─────────────────────────────────────────────────────────────────

def test_mastodon_strips_html_and_filters_language():
    statuses = [
        {"id": "1", "url": "https://m/1", "content": "<p>Loved this <b>boutique hotel</b> in Porto, rooftop was unreal</p>",
         "language": "en", "created_at": "2026-09-01T10:00:00.000Z", "account": {"acct": "ana@m", "display_name": "Ana"}},
        {"id": "2", "url": "https://m/2", "content": "<p>Hotel muy bonito en Sevilla, volveremos</p>", "language": "es",
         "created_at": "2026-09-01T11:00:00.000Z", "account": {"acct": "b@m"}},
    ]

    def handler(request):
        return httpx.Response(200, json=statuses)

    cfg = {"instances": ["mastodon.social"], "hashtags": ["hotel"], "langs": ["en"]}
    with patch("ingestion.social.mastodon_scraper.get_source_config", return_value=cfg):
        items = _with_transport(MastodonScraper(), handler).scrape()
    assert len(items) == 1
    assert items[0].content == "Loved this boutique hotel in Porto, rooftop was unreal"
    assert items[0].author == "Ana"
    assert items[0].published_at == datetime(2026, 9, 1, 10, 0)


# ── GDELT ────────────────────────────────────────────────────────────────────

def test_gdelt_headline_items_and_body_fetch():
    def handler(request):
        url = str(request.url)
        if "gdeltproject" in url:
            return httpx.Response(200, json={"articles": [
                {"url": "https://news/a", "title": "Hotel group opens Lisbon flagship", "seendate": "20260901T100000Z", "domain": "news", "language": "English"},
                {"url": "https://news/b", "title": "Resort acquired", "seendate": "20260901T110000Z", "domain": "news", "language": "English"},
            ]})
        return httpx.Response(200, headers={"content-type": "text/html"},
                              text="<html><body><p>The group announced a new flagship property in Lisbon with sixty rooms and a rooftop.</p><p>x</p></body></html>")

    cfg = {"queries": ["hotel"], "fetch_bodies": 1}
    with patch("ingestion.news.gdelt.get_source_config", return_value=cfg), patch("ingestion.news.gdelt.time.sleep"):
        items = _with_transport(GDELTScraper(), handler).scrape()
    assert len(items) == 2
    assert items[0].metadata["body_fetched"] is True and "sixty rooms" in items[0].content
    assert items[1].metadata["body_fetched"] is False and items[1].content == "Resort acquired"
    assert items[0].published_at == datetime(2026, 9, 1, 10, 0)


# ── SEC EDGAR ────────────────────────────────────────────────────────────────

ATOM = """<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"><title>Marriott</title>
<entry><title>8-K  - Current report</title><link href="https://www.sec.gov/Archives/edgar/data/1048286/000119312526349395/0001193125-26-349395-index.htm"/>
<updated>2026-08-13T16:39:39-04:00</updated><summary type="html">&lt;b&gt;Filed:&lt;/b&gt; 2026-08-13</summary></entry></feed>"""
INDEX_HTML = '<a href="/Archives/edgar/data/1048286/000119312526349395/d123.htm">d123.htm</a>'
DOC_HTML = "<html><body><p>Item 2.01 Completion of Acquisition. Marriott acquired a 200-key resort in Tulum.</p></body></html>"


def test_edgar_follows_index_to_primary_document():
    def handler(request):
        url = str(request.url)
        assert request.headers["user-agent"].startswith("BrandClave ")
        if "browse-edgar" in url:
            return httpx.Response(200, text=ATOM)
        if url.endswith("-index.htm"):
            return httpx.Response(200, text=INDEX_HTML)
        return httpx.Response(200, text=DOC_HTML)

    cfg = {"companies": [{"cik": "0001048286", "name": "Marriott International"}], "forms": ["8-K"]}
    with patch("ingestion.news.sec_edgar.get_source_config", return_value=cfg):
        items = _with_transport(SECEdgarScraper(), handler).scrape()
    assert len(items) == 1
    assert items[0].title.startswith("Marriott International: 8-K")
    assert "200-key resort in Tulum" in items[0].content
    assert items[0].metadata["form"] == "8-K" and items[0].metadata["body_fetched"] is True
    assert items[0].published_at == datetime(2026, 8, 13, 16, 39, 39)
