#!/usr/bin/env python
"""CLI for running BrandClave scrapers and NLP pipeline."""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

load_dotenv()

# Available scrapers
SCRAPERS = {
    # News
    "hospitalitynet": "ingestion.news.hospitalitynet_rss.HospitalityNetScraper",
    "skift": "ingestion.news.skift_rss.SkiftScraper",
    # Social
    "reddit": "ingestion.social.reddit_scraper.RedditScraper",
    "youtube": "ingestion.social.youtube_scraper.YouTubeScraper",
}


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def get_scraper_class(source: str):
    """Dynamically import and return scraper class."""
    if source not in SCRAPERS:
        raise ValueError(f"Unknown source: {source}. Available: {list(SCRAPERS.keys())}")

    module_path, class_name = SCRAPERS[source].rsplit(".", 1)
    module = __import__(module_path, fromlist=[class_name])
    return getattr(module, class_name)


def run_scraper(source: str) -> dict:
    """Run a specific scraper.

    Args:
        source: Source name (e.g., 'hospitalitynet')

    Returns:
        Scrape result summary
    """
    scraper_class = get_scraper_class(source)
    with scraper_class() as scraper:
        return scraper.run()


def run_all_scrapers() -> list[dict]:
    """Run all available scrapers.

    Returns:
        List of scrape result summaries
    """
    results = []
    for source in SCRAPERS:
        logging.info(f"Running scraper: {source}")
        try:
            result = run_scraper(source)
            results.append(result)
        except Exception as e:
            logging.error(f"Scraper {source} failed: {e}")
            results.append({"source": source, "status": "failed", "error": str(e)})
    return results


def run_nlp_pipeline(limit: int = 100) -> dict:
    """Run NLP pipeline on unprocessed content.

    Args:
        limit: Maximum items to process

    Returns:
        Processing summary
    """
    from processing.nlp_pipeline import run_pipeline

    return run_pipeline(limit=limit)


def generate_trends(days_back: int = 30, save: bool = True) -> dict:
    """Generate Social Pulse trends from content.

    Args:
        days_back: Days of content to analyze
        save: Whether to save to database

    Returns:
        Generation summary
    """
    from services.social_pulse import generate_social_pulse

    trends = generate_social_pulse(days_back=days_back, save=save)
    return {
        "trends_generated": len(trends),
        "saved": save,
    }


def main():
    parser = argparse.ArgumentParser(
        description="BrandClave Aggregator CLI - Run scrapers and NLP pipeline"
    )
    parser.add_argument(
        "--source",
        "-s",
        type=str,
        help=f"Source to scrape. Options: {list(SCRAPERS.keys())}",
    )
    parser.add_argument(
        "--all",
        "-a",
        action="store_true",
        help="Run all available scrapers",
    )
    parser.add_argument(
        "--process",
        "-p",
        action="store_true",
        help="Run NLP pipeline on unprocessed content",
    )
    parser.add_argument(
        "--limit",
        "-l",
        type=int,
        default=100,
        help="Limit for NLP processing (default: 100)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available scrapers",
    )
    parser.add_argument(
        "--trends",
        "-t",
        action="store_true",
        help="Generate Social Pulse trends from content",
    )
    parser.add_argument(
        "--days",
        "-d",
        type=int,
        default=30,
        help="Days of content for trend generation (default: 30)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    setup_logging(args.verbose)

    if args.list:
        print("Available scrapers:")
        for name in SCRAPERS:
            print(f"  - {name}")
        return

    results = []

    # Run scrapers
    if args.source:
        logging.info(f"Running scraper: {args.source}")
        result = run_scraper(args.source)
        results.append(result)
        print(f"\nScrape result: {result}")

    elif args.all:
        logging.info("Running all scrapers")
        results = run_all_scrapers()
        print("\nScrape results:")
        for result in results:
            print(f"  {result}")

    # Run NLP pipeline
    if args.process:
        logging.info(f"Running NLP pipeline (limit: {args.limit})")
        pipeline_result = run_nlp_pipeline(limit=args.limit)
        print(f"\nPipeline result: {pipeline_result}")

    # Generate trends
    if args.trends:
        logging.info(f"Generating trends (days_back: {args.days})")
        trend_result = generate_trends(days_back=args.days)
        print(f"\nTrend generation result: {trend_result}")

    if not (args.source or args.all or args.process or args.list or args.trends):
        parser.print_help()


if __name__ == "__main__":
    main()
