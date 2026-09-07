"""Tiny in-process TTL cache for read-only endpoints.

The dashboard fires half a dozen reads on first paint. On a single small
instance every visitor repeating those queries adds up, and a minute of
staleness is invisible to a reader. Writes go through other routes, so the
cache is keyed on the call arguments only.
"""

import functools
import threading
import time
from typing import Any, Callable

_LOCK = threading.Lock()
_STORE: dict[tuple, tuple[float, Any]] = {}


def ttl_cache(seconds: float = 60.0) -> Callable:
    def decorate(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            key = (fn.__module__, fn.__qualname__, args, tuple(sorted(kwargs.items())))
            now = time.time()
            with _LOCK:
                hit = _STORE.get(key)
            if hit and now - hit[0] < seconds:
                return hit[1]
            value = fn(*args, **kwargs)
            with _LOCK:
                _STORE[key] = (now, value)
            return value

        return wrapper

    return decorate


def clear() -> None:
    with _LOCK:
        _STORE.clear()
