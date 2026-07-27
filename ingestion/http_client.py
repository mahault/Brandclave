"""Shared resilient HTTP helpers for scrapers and services.

Every outbound HTTP call in the project should go through ``resilient_get``
(or ``resilient_request`` for non-GET methods) so that:

- every request has an explicit timeout (never httpx library defaults),
- transient failures (connect/read timeouts, 429, 5xx) are retried with
  exponential backoff + jitter, honoring ``Retry-After`` on 429,
- a source that stays dead returns ``None`` and logs a WARNING instead of
  raising, so a scrape run degrades gracefully rather than aborting.
"""

import logging
import random
import time

import httpx

logger = logging.getLogger(__name__)

# Sane defaults so nothing ever runs with httpx's library defaults.
DEFAULT_TIMEOUT = 15.0
DEFAULT_RETRIES = 3
DEFAULT_BACKOFF_BASE = 1.0
MAX_RETRY_AFTER = 60.0

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

# Module-level default client with an explicit timeout, used when the caller
# does not supply its own client.
DEFAULT_CLIENT = httpx.Client(
    timeout=DEFAULT_TIMEOUT,
    follow_redirects=True,
    headers={"User-Agent": DEFAULT_USER_AGENT},
)

# Indirection so tests can patch sleeping without touching the real clock.
_sleep = time.sleep


def _parse_retry_after(response: httpx.Response) -> float | None:
    """Parse a Retry-After header into seconds, capped at MAX_RETRY_AFTER."""
    value = response.headers.get("Retry-After")
    if not value:
        return None
    try:
        seconds = float(value)
    except ValueError:
        # HTTP-date form; not worth parsing here, fall back to backoff.
        return None
    if seconds < 0:
        return None
    return min(seconds, MAX_RETRY_AFTER)


def _backoff_delay(attempt: int, backoff_base: float) -> float:
    """Exponential backoff with jitter for the given (0-indexed) attempt."""
    delay = backoff_base * (2 ** attempt)
    jitter = random.uniform(0, backoff_base * 0.5)
    return min(delay + jitter, MAX_RETRY_AFTER)


def resilient_request(
    method: str,
    url: str,
    *,
    headers: dict | None = None,
    params: dict | None = None,
    timeout: float | None = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
    backoff_base: float = DEFAULT_BACKOFF_BASE,
    client: httpx.Client | None = None,
    **kwargs,
) -> httpx.Response | None:
    """Make an HTTP request with timeouts, retries, and graceful failure.

    Retries on connect/read timeouts, transport errors, 429, and 5xx with
    exponential backoff + jitter. Honors Retry-After on 429 (capped at
    MAX_RETRY_AFTER). Never raises for these expected failure modes.

    Args:
        method: HTTP method (e.g. "GET").
        url: URL to request.
        headers: Optional per-request headers.
        params: Optional query parameters.
        timeout: Per-request timeout in seconds. Pass None to use the
            client's configured timeout instead.
        retries: Total number of attempts (not additional retries).
        backoff_base: Base delay in seconds for exponential backoff.
        client: Optional httpx.Client to use; defaults to the module client.
        **kwargs: Extra arguments passed through to httpx (e.g. json=, data=).

    Returns:
        The response (any status < 500 that is not 429 is passed through,
        including 4xx — callers should check status_code), or None after
        the final failed attempt.
    """
    http_client = client if client is not None else DEFAULT_CLIENT
    request_kwargs = dict(kwargs)
    if headers is not None:
        request_kwargs["headers"] = headers
    if params is not None:
        request_kwargs["params"] = params
    if timeout is not None:
        request_kwargs["timeout"] = timeout

    attempts = max(1, retries)
    last_reason = "unknown error"

    for attempt in range(attempts):
        try:
            response = http_client.request(method, url, **request_kwargs)
        except httpx.TimeoutException as e:
            last_reason = f"timeout ({e.__class__.__name__})"
            if attempt < attempts - 1:
                _sleep(_backoff_delay(attempt, backoff_base))
            continue
        except httpx.HTTPError as e:
            last_reason = f"{e.__class__.__name__}: {e}"
            if attempt < attempts - 1:
                _sleep(_backoff_delay(attempt, backoff_base))
            continue

        status = response.status_code
        if status == 429:
            last_reason = "HTTP 429 (rate limited)"
            if attempt < attempts - 1:
                wait = _parse_retry_after(response)
                if wait is None:
                    wait = _backoff_delay(attempt, backoff_base)
                logger.info(f"Rate limited by {url}, waiting {wait:.1f}s")
                _sleep(wait)
            continue
        if status >= 500:
            last_reason = f"HTTP {status}"
            if attempt < attempts - 1:
                _sleep(_backoff_delay(attempt, backoff_base))
            continue

        # Success or non-retryable status (e.g. 404): pass through to caller.
        return response

    logger.warning(
        f"Giving up on {method} {url} after {attempts} attempt(s): {last_reason}"
    )
    return None


def resilient_get(
    url: str,
    *,
    headers: dict | None = None,
    params: dict | None = None,
    timeout: float | None = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
    backoff_base: float = DEFAULT_BACKOFF_BASE,
    client: httpx.Client | None = None,
    **kwargs,
) -> httpx.Response | None:
    """GET a URL with timeouts, retries, and graceful failure.

    See resilient_request for full semantics.
    """
    return resilient_request(
        "GET",
        url,
        headers=headers,
        params=params,
        timeout=timeout,
        retries=retries,
        backoff_base=backoff_base,
        client=client,
        **kwargs,
    )
