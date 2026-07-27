"""Structured logging and error-tracking setup.

:func:`setup_logging` configures the root logger to emit one JSON object per
line (timestamp, level, logger, message, exception info when present) so logs
are machine-searchable in production. Set ``LOG_FORMAT=text`` for the classic
human-readable format during local development. Level comes from ``LOG_LEVEL``
(default INFO).

:func:`init_sentry` wires up Sentry error tracking when ``SENTRY_DSN`` is set
and ``sentry-sdk`` is installed; otherwise it logs one info line and moves on.
sentry-sdk is deliberately not a hard dependency.

Stdlib only - no logging dependencies.
"""

import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Optional

_TEXT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


class JsonFormatter(logging.Formatter):
    """Format log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack_info"] = self.formatStack(record.stack_info)
        return json.dumps(payload, default=str)


def setup_logging(
    level: Optional[str] = None, log_format: Optional[str] = None
) -> None:
    """Configure root logging for the whole process.

    Call once, as early as possible, before anything logs. Safe to call again
    (it replaces the root handlers rather than stacking duplicates).

    Args:
        level: Log level name; defaults to the LOG_LEVEL env var, then INFO.
        log_format: "json" (default) or "text"; defaults to the LOG_FORMAT env var.
    """
    level = (level or os.getenv("LOG_LEVEL", "INFO")).strip().upper()
    log_format = (log_format or os.getenv("LOG_FORMAT", "json")).strip().lower()

    handler = logging.StreamHandler(sys.stdout)
    if log_format == "text":
        handler.setFormatter(logging.Formatter(_TEXT_FORMAT))
    else:
        handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)


def init_sentry(dsn: Optional[str]) -> bool:
    """Initialize Sentry error tracking if configured and installed.

    Args:
        dsn: The Sentry DSN, typically ``get_settings().sentry_dsn``.

    Returns:
        True if Sentry was initialized, False if it is disabled (no DSN, or
        sentry-sdk not installed - it is an optional dependency).
    """
    logger = logging.getLogger(__name__)

    if not dsn:
        logger.info("Sentry disabled (SENTRY_DSN not set)")
        return False

    try:
        import sentry_sdk
        from sentry_sdk.integrations.fastapi import FastApiIntegration
        from sentry_sdk.integrations.starlette import StarletteIntegration
    except ImportError:
        logger.info(
            "Sentry disabled (sentry-sdk not installed; "
            "pip install 'sentry-sdk[fastapi]' to enable)"
        )
        return False

    sentry_sdk.init(
        dsn=dsn,
        integrations=[StarletteIntegration(), FastApiIntegration()],
        traces_sample_rate=0.1,
        send_default_pii=False,
    )
    logger.info("Sentry error tracking enabled")
    return True
