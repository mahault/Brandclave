#!/usr/bin/env python
"""Quick test script for API endpoints."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

def test_database():
    """Test database connection."""
    print("Testing database connection...", end=" ")
    try:
        from db.database import engine
        from sqlalchemy import text
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("OK")
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        return False


def test_metrics():
    """Test metrics collector."""
    print("Testing MetricsCollector...", end=" ")
    try:
        from monitoring.metrics import MetricsCollector
        with MetricsCollector() as collector:
            metrics = collector.get_system_metrics()
        print(f"OK (total_content={metrics.total_content})")
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        return False


def test_social_pulse_service():
    """Test SocialPulseService."""
    print("Testing SocialPulseService.get_trends()...", end=" ")
    try:
        from services.social_pulse import SocialPulseService
        service = SocialPulseService()
        trends = service.get_trends(limit=5)
        print(f"OK ({len(trends)} trends)")
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        return False


def test_hotelier_bets_service():
    """Test HotelierBetsService."""
    print("Testing HotelierBetsService.get_moves()...", end=" ")
    try:
        from services.hotelier_bets import HotelierBetsService
        service = HotelierBetsService()
        moves = service.get_moves(limit=5)
        print(f"OK ({len(moves)} moves)")
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        return False


def test_demand_scan_service():
    """Test DemandScanService."""
    print("Testing DemandScanService.get_properties()...", end=" ")
    try:
        from services.demand_scan import DemandScanService
        service = DemandScanService()
        properties = service.get_properties(limit=5)
        print(f"OK ({len(properties)} properties)")
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        return False


def test_raw_content():
    """Test raw content query."""
    print("Testing raw content query...", end=" ")
    try:
        from db.database import SessionLocal
        from db.models import RawContentModel
        db = SessionLocal()
        try:
            items = db.query(RawContentModel).order_by(
                RawContentModel.scraped_at.desc()
            ).limit(5).all()
            print(f"OK ({len(items)} items)")
            return True
        finally:
            db.close()
    except Exception as e:
        print(f"FAILED: {e}")
        return False


def test_scraper_metrics():
    """Test scraper metrics."""
    print("Testing scraper metrics...", end=" ")
    try:
        from monitoring.metrics import MetricsCollector
        with MetricsCollector() as collector:
            scrapers = collector.get_all_scraper_metrics()
        print(f"OK ({len(scrapers)} scrapers)")
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        return False


def main():
    print()
    print("=" * 50)
    print("  BrandClave API Endpoint Tests")
    print("=" * 50)
    print()

    results = []

    # Run all tests
    results.append(("Database", test_database()))
    results.append(("Metrics", test_metrics()))
    results.append(("Social Pulse", test_social_pulse_service()))
    results.append(("Hotelier Bets", test_hotelier_bets_service()))
    results.append(("Demand Scan", test_demand_scan_service()))
    results.append(("Raw Content", test_raw_content()))
    results.append(("Scraper Metrics", test_scraper_metrics()))

    print()
    print("=" * 50)
    print("  Summary")
    print("=" * 50)

    passed = sum(1 for _, ok in results if ok)
    failed = sum(1 for _, ok in results if not ok)

    print(f"Passed: {passed}/{len(results)}")
    print(f"Failed: {failed}/{len(results)}")

    if failed > 0:
        print()
        print("Failed tests:")
        for name, ok in results:
            if not ok:
                print(f"  - {name}")

    print()

    if failed == 0:
        print("All tests passed! The API endpoints should work correctly.")
        print("If the dashboard still doesn't load, check the browser console for JavaScript errors.")
    else:
        print("Some tests failed. Fix the issues above before testing the dashboard.")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
