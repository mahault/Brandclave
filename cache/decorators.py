"""Caching decorators for API endpoints."""

import functools
import logging
from typing import Callable

from cache.redis_cache import get_cache

logger = logging.getLogger(__name__)


def cached_response(ttl: int = 300, category: str = "default"):
    """Decorator for caching API responses.

    Args:
        ttl: Cache TTL in seconds
        category: Response category for TTL lookup
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            cache = get_cache()

            # Build cache key from function name and arguments
            endpoint = func.__name__

            # Filter out non-serializable params
            params = {}
            for k, v in kwargs.items():
                if k not in ("request", "db") and not callable(v):
                    try:
                        # Test if serializable
                        import json
                        json.dumps(v)
                        params[k] = v
                    except (TypeError, ValueError):
                        pass

            # Check cache
            cached = cache.get_api_response(endpoint, params)
            if cached:
                logger.debug(f"Cache hit for {endpoint}")
                return cached

            # Call function
            result = await func(*args, **kwargs)

            # Cache result (convert Pydantic models to dict)
            try:
                if hasattr(result, "model_dump"):
                    cache_data = result.model_dump()
                elif hasattr(result, "dict"):
                    cache_data = result.dict()
                elif isinstance(result, dict):
                    cache_data = result
                else:
                    # Can't cache, return as-is
                    return result

                cache.cache_api_response(endpoint, params, cache_data, category, ttl)
            except Exception as e:
                logger.warning(f"Failed to cache response: {e}")

            return result

        return wrapper
    return decorator


def invalidate_cache(pattern: str):
    """Decorator to invalidate cache after operation.

    Args:
        pattern: Cache key pattern to invalidate
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            result = await func(*args, **kwargs)

            # Invalidate cache after successful operation
            try:
                cache = get_cache()
                cache.invalidate_pattern(pattern)
            except Exception as e:
                logger.warning(f"Failed to invalidate cache: {e}")

            return result

        return wrapper
    return decorator
