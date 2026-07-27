#!/usr/bin/env python
"""Standalone scheduler worker.

Runs the scraping/processing scheduler outside the web process (plan §5.6),
so web restarts and scale events never interrupt data collection. The web
service should run with SCHEDULER_ENABLED=false when this worker is deployed.
"""

import logging
import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

load_dotenv()

from monitoring.logging_config import setup_logging

setup_logging()

logger = logging.getLogger("worker")


def main() -> None:
    from scheduler.scheduler import init_scheduler

    scheduler = init_scheduler(auto_register=True)
    if not scheduler.is_available:
        logger.error("Scheduler unavailable (APScheduler not importable?); exiting")
        sys.exit(1)

    scheduler.start()
    logger.info("Scheduler worker running (jobs: %d)", len(scheduler.get_jobs()))

    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        logger.info("Worker shutting down")
        scheduler.shutdown(wait=False)


if __name__ == "__main__":
    main()
