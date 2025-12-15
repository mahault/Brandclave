"""APScheduler integration for automated scraping.

Provides background scheduling of scraper jobs with persistence.
"""

import logging
import os
from datetime import datetime
from typing import Callable

import yaml
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Try to import APScheduler
try:
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
    from apscheduler.executors.pool import ThreadPoolExecutor
    from apscheduler.triggers.cron import CronTrigger
    from apscheduler.triggers.interval import IntervalTrigger
    SCHEDULER_AVAILABLE = True
except ImportError:
    SCHEDULER_AVAILABLE = False
    logger.warning("APScheduler not installed. Scheduling disabled.")


class ScraperScheduler:
    """Manages scheduled scraping jobs.

    Uses APScheduler with SQLite persistence for job state.
    Gracefully handles APScheduler unavailability.
    """

    def __init__(
        self,
        db_url: str | None = None,
        config_path: str = "configs/scheduler.yaml",
    ):
        """Initialize scheduler.

        Args:
            db_url: SQLite URL for job persistence
            config_path: Path to scheduler configuration
        """
        self.config = self._load_config(config_path)
        self._scraper_registry: dict[str, type] = {}
        self._job_functions: dict[str, Callable] = {}
        self.scheduler = None
        self._running = False

        if not SCHEDULER_AVAILABLE:
            logger.warning("APScheduler not available, scheduling disabled")
            return

        if not os.getenv("SCHEDULER_ENABLED", "true").lower() == "true":
            logger.info("Scheduler disabled via SCHEDULER_ENABLED env var")
            return

        # Use separate SQLite database for jobs
        db_url = db_url or os.getenv(
            "DATABASE_URL",
            "sqlite:///./data/brandclave.db"
        )
        # Use a different table for APScheduler
        jobs_db = db_url.replace("brandclave.db", "scheduler_jobs.db")

        try:
            jobstores = {
                "default": SQLAlchemyJobStore(url=jobs_db)
            }

            executors = {
                "default": ThreadPoolExecutor(
                    max_workers=self.config.get("scheduler", {}).get("max_workers", 3)
                ),
            }

            job_defaults = {
                "coalesce": True,  # Combine missed runs
                "max_instances": 1,  # Only one instance per job
                "misfire_grace_time": 3600,  # 1 hour grace period
            }

            self.scheduler = BackgroundScheduler(
                jobstores=jobstores,
                executors=executors,
                job_defaults=job_defaults,
                timezone=self.config.get("scheduler", {}).get("timezone", "UTC"),
            )

            logger.info("Scheduler initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize scheduler: {e}")
            self.scheduler = None

    def _load_config(self, config_path: str) -> dict:
        """Load scheduler configuration."""
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                return yaml.safe_load(f) or {}
        return {}

    @property
    def is_available(self) -> bool:
        """Check if scheduler is available."""
        return self.scheduler is not None

    @property
    def is_running(self) -> bool:
        """Check if scheduler is running."""
        return self._running and self.scheduler is not None

    def register_scraper(self, source_name: str, scraper_class: type) -> None:
        """Register a scraper class for scheduling.

        Args:
            source_name: Name of the source
            scraper_class: Scraper class to instantiate
        """
        self._scraper_registry[source_name] = scraper_class
        logger.debug(f"Registered scraper: {source_name}")

    def register_job_function(self, name: str, func: Callable) -> None:
        """Register a generic job function.

        Args:
            name: Job name
            func: Function to call
        """
        self._job_functions[name] = func
        logger.debug(f"Registered job function: {name}")

    def _run_scraper_job(self, source_name: str) -> dict:
        """Execute a scraper job.

        Args:
            source_name: Name of the scraper source

        Returns:
            Job result dict
        """
        logger.info(f"Running scheduled scrape: {source_name}")
        try:
            scraper_class = self._scraper_registry.get(source_name)
            if not scraper_class:
                raise ValueError(f"Unknown source: {source_name}")

            with scraper_class() as scraper:
                return scraper.run()
        except Exception as e:
            logger.error(f"Scheduled scrape failed for {source_name}: {e}")
            return {"source": source_name, "status": "failed", "error": str(e)}

    def _run_job_function(self, name: str, **kwargs) -> dict:
        """Execute a registered job function.

        Args:
            name: Job name
            **kwargs: Arguments to pass to function

        Returns:
            Job result
        """
        logger.info(f"Running scheduled job: {name}")
        try:
            func = self._job_functions.get(name)
            if not func:
                raise ValueError(f"Unknown job: {name}")
            return func(**kwargs)
        except Exception as e:
            logger.error(f"Scheduled job failed for {name}: {e}")
            return {"job": name, "status": "failed", "error": str(e)}

    def add_scraper_job(
        self,
        source_name: str,
        interval_minutes: int | None = None,
        cron_expression: str | None = None,
        enabled: bool = True,
    ) -> str | None:
        """Add a scheduled scraper job.

        Args:
            source_name: Name of the scraper source
            interval_minutes: Run interval (used if no cron)
            cron_expression: Cron expression for scheduling
            enabled: Whether job is enabled

        Returns:
            Job ID or None if failed
        """
        if not self.is_available or not enabled:
            return None

        job_id = f"scraper_{source_name}"

        try:
            if cron_expression:
                trigger = CronTrigger.from_crontab(cron_expression)
            else:
                trigger = IntervalTrigger(minutes=interval_minutes or 60)

            self.scheduler.add_job(
                self._run_scraper_job,
                trigger=trigger,
                args=[source_name],
                id=job_id,
                name=f"Scrape {source_name}",
                replace_existing=True,
            )

            logger.info(f"Added scheduled job: {job_id}")
            return job_id

        except Exception as e:
            logger.error(f"Failed to add job {job_id}: {e}")
            return None

    def add_processing_job(
        self,
        name: str,
        func: Callable,
        interval_minutes: int | None = None,
        cron_expression: str | None = None,
        enabled: bool = True,
        **kwargs,
    ) -> str | None:
        """Add a scheduled processing job.

        Args:
            name: Job name
            func: Function to call
            interval_minutes: Run interval
            cron_expression: Cron expression
            enabled: Whether job is enabled
            **kwargs: Arguments to pass to function

        Returns:
            Job ID or None
        """
        if not self.is_available or not enabled:
            return None

        job_id = f"process_{name}"
        self._job_functions[name] = func

        try:
            if cron_expression:
                trigger = CronTrigger.from_crontab(cron_expression)
            else:
                trigger = IntervalTrigger(minutes=interval_minutes or 60)

            self.scheduler.add_job(
                self._run_job_function,
                trigger=trigger,
                args=[name],
                kwargs=kwargs,
                id=job_id,
                name=f"Process {name}",
                replace_existing=True,
            )

            logger.info(f"Added processing job: {job_id}")
            return job_id

        except Exception as e:
            logger.error(f"Failed to add job {job_id}: {e}")
            return None

    def remove_job(self, job_id: str) -> bool:
        """Remove a scheduled job.

        Args:
            job_id: Job identifier

        Returns:
            True if removed
        """
        if not self.is_available:
            return False

        try:
            self.scheduler.remove_job(job_id)
            logger.info(f"Removed job: {job_id}")
            return True
        except Exception as e:
            logger.warning(f"Failed to remove job {job_id}: {e}")
            return False

    def pause_job(self, job_id: str) -> bool:
        """Pause a scheduled job.

        Args:
            job_id: Job identifier

        Returns:
            True if paused
        """
        if not self.is_available:
            return False

        try:
            self.scheduler.pause_job(job_id)
            logger.info(f"Paused job: {job_id}")
            return True
        except Exception as e:
            logger.warning(f"Failed to pause job {job_id}: {e}")
            return False

    def resume_job(self, job_id: str) -> bool:
        """Resume a paused job.

        Args:
            job_id: Job identifier

        Returns:
            True if resumed
        """
        if not self.is_available:
            return False

        try:
            self.scheduler.resume_job(job_id)
            logger.info(f"Resumed job: {job_id}")
            return True
        except Exception as e:
            logger.warning(f"Failed to resume job {job_id}: {e}")
            return False

    def run_job_now(self, job_id: str) -> bool:
        """Trigger immediate execution of a job.

        Args:
            job_id: Job identifier

        Returns:
            True if triggered
        """
        if not self.is_available:
            return False

        try:
            job = self.scheduler.get_job(job_id)
            if job:
                job.modify(next_run_time=datetime.now())
                logger.info(f"Triggered job: {job_id}")
                return True
            return False
        except Exception as e:
            logger.warning(f"Failed to trigger job {job_id}: {e}")
            return False

    def get_jobs(self) -> list[dict]:
        """Get all scheduled jobs.

        Returns:
            List of job info dicts
        """
        if not self.is_available:
            return []

        jobs = []
        for job in self.scheduler.get_jobs():
            jobs.append({
                "id": job.id,
                "name": job.name,
                "next_run": job.next_run_time.isoformat() if job.next_run_time else None,
                "trigger": str(job.trigger),
                "pending": job.pending,
            })
        return jobs

    def get_job(self, job_id: str) -> dict | None:
        """Get a specific job by ID.

        Args:
            job_id: Job identifier

        Returns:
            Job info dict or None
        """
        if not self.is_available:
            return None

        job = self.scheduler.get_job(job_id)
        if job:
            return {
                "id": job.id,
                "name": job.name,
                "next_run": job.next_run_time.isoformat() if job.next_run_time else None,
                "trigger": str(job.trigger),
                "pending": job.pending,
            }
        return None

    def start(self) -> bool:
        """Start the scheduler.

        Returns:
            True if started
        """
        if not self.is_available:
            return False

        if self._running:
            logger.warning("Scheduler already running")
            return True

        try:
            self.scheduler.start()
            self._running = True
            logger.info("Scheduler started")
            return True
        except Exception as e:
            logger.error(f"Failed to start scheduler: {e}")
            return False

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the scheduler.

        Args:
            wait: Wait for running jobs to complete
        """
        if self.scheduler and self._running:
            self.scheduler.shutdown(wait=wait)
            self._running = False
            logger.info("Scheduler stopped")


# Singleton instance
_scheduler: ScraperScheduler | None = None


def get_scheduler() -> ScraperScheduler:
    """Get the singleton scheduler instance."""
    global _scheduler
    if _scheduler is None:
        _scheduler = ScraperScheduler()
    return _scheduler


def init_scheduler(auto_register: bool = True) -> ScraperScheduler:
    """Initialize and configure the scheduler.

    Args:
        auto_register: Auto-register scrapers from config

    Returns:
        Configured scheduler instance
    """
    scheduler = get_scheduler()

    if not scheduler.is_available:
        logger.warning("Scheduler not available, skipping initialization")
        return scheduler

    if auto_register:
        _register_default_scrapers(scheduler)
        _register_default_jobs(scheduler)

    return scheduler


def _register_default_scrapers(scheduler: ScraperScheduler) -> None:
    """Register default scrapers with scheduler."""
    from scripts.run_crawlers import SCRAPERS, get_scraper_class

    for source_name in SCRAPERS:
        try:
            scraper_class = get_scraper_class(source_name)
            scheduler.register_scraper(source_name, scraper_class)
        except Exception as e:
            logger.warning(f"Could not register scraper {source_name}: {e}")


def _register_default_jobs(scheduler: ScraperScheduler) -> None:
    """Register default jobs from config."""
    config = scheduler.config.get("jobs", {})

    for source_name, job_config in config.items():
        if not job_config.get("enabled", True):
            continue

        # Check if it's a scraper job
        if source_name in scheduler._scraper_registry:
            scheduler.add_scraper_job(
                source_name=source_name,
                interval_minutes=job_config.get("interval_minutes"),
                cron_expression=job_config.get("cron"),
            )
        # Check for special processing jobs
        elif source_name == "nlp_pipeline":
            from processing.nlp_pipeline import run_pipeline
            scheduler.add_processing_job(
                name="nlp_pipeline",
                func=run_pipeline,
                interval_minutes=job_config.get("interval_minutes", 180),
                limit=job_config.get("limit", 200),
            )
        elif source_name == "generate_trends":
            from services.social_pulse import generate_social_pulse
            scheduler.add_processing_job(
                name="generate_trends",
                func=generate_social_pulse,
                cron_expression=job_config.get("cron"),
                days_back=job_config.get("days_back", 30),
            )
        elif source_name == "extract_moves":
            from services.hotelier_bets import generate_hotelier_bets
            scheduler.add_processing_job(
                name="extract_moves",
                func=generate_hotelier_bets,
                cron_expression=job_config.get("cron"),
                days_back=job_config.get("days_back", 30),
            )
