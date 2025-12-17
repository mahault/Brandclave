"""Redis caching layer for BrandClave Aggregator.

Provides caching for HTTP responses, embeddings, and API responses.
Gracefully degrades when Redis is unavailable.
"""

import hashlib
import json
import logging
import os
from typing import Any

import yaml
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Try to import redis, but don't fail if not installed
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("Redis package not installed. Caching disabled.")


class RedisCache:
    """Redis-based caching with TTL support.

    Gracefully handles Redis unavailability - all operations
    return appropriate defaults when Redis is not connected.
    """

    def __init__(self, config_path: str = "configs/cache.yaml"):
        """Initialize Redis connection.

        Args:
            config_path: Path to cache configuration
        """
        self.config = self._load_config(config_path)
        self.client = None
        self._connected = False

        if not REDIS_AVAILABLE:
            logger.info("Redis not available, caching disabled")
            return

        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")

        try:
            self.client = redis.from_url(
                redis_url,
                decode_responses=True,
                socket_connect_timeout=1,  # Fast timeout - don't block if Redis unavailable
                socket_timeout=1,
            )
            # Test connection
            self.client.ping()
            self._connected = True
            logger.info("Redis connection established")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Caching disabled.")
            self.client = None
            self._connected = False

    def _load_config(self, config_path: str) -> dict:
        """Load cache configuration."""
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        return {
            "ttl": {
                "http_response": {"default": 3600},
                "embedding": 604800,
                "api_response": {"default": 300},
            },
            "prefixes": {
                "http": "brandclave:http:",
                "embedding": "brandclave:emb:",
                "api": "brandclave:api:",
            },
        }

    @property
    def is_connected(self) -> bool:
        """Check if Redis is connected."""
        return self._connected and self.client is not None

    def _make_key(self, prefix: str, identifier: str) -> str:
        """Generate cache key with prefix."""
        return f"{prefix}{identifier}"

    def _hash_string(self, s: str) -> str:
        """Create hash of string for cache key."""
        return hashlib.sha256(s.encode()).hexdigest()[:16]

    def _get_ttl(self, category: str, subcategory: str | None = None) -> int:
        """Get TTL from config."""
        ttl_config = self.config.get("ttl", {}).get(category, {})
        if isinstance(ttl_config, dict):
            return ttl_config.get(subcategory, ttl_config.get("default", 3600))
        return ttl_config if isinstance(ttl_config, int) else 3600

    # HTTP Response Caching
    def cache_http_response(
        self,
        url: str,
        response_data: dict,
        source_type: str = "default",
        ttl: int | None = None,
    ) -> bool:
        """Cache an HTTP response.

        Args:
            url: Request URL
            response_data: Response data to cache
            source_type: Type of source (news, social, review, property)
            ttl: Time-to-live in seconds (uses config default if None)

        Returns:
            True if cached successfully
        """
        if not self.is_connected:
            return False

        try:
            key = self._make_key(
                self.config["prefixes"]["http"],
                self._hash_string(url),
            )
            ttl = ttl or self._get_ttl("http_response", source_type)

            self.client.setex(key, ttl, json.dumps(response_data))
            logger.debug(f"Cached HTTP response for: {url[:50]}...")
            return True
        except Exception as e:
            logger.error(f"Cache write error: {e}")
            return False

    def get_http_response(self, url: str) -> dict | None:
        """Get cached HTTP response.

        Args:
            url: Request URL

        Returns:
            Cached response dict or None
        """
        if not self.is_connected:
            return None

        try:
            key = self._make_key(
                self.config["prefixes"]["http"],
                self._hash_string(url),
            )
            data = self.client.get(key)
            if data:
                logger.debug(f"Cache hit for: {url[:50]}...")
                return json.loads(data)
            return None
        except Exception as e:
            logger.error(f"Cache read error: {e}")
            return None

    # Embedding Caching
    def cache_embedding(
        self,
        content_id: str,
        embedding: list[float],
        ttl: int | None = None,
    ) -> bool:
        """Cache an embedding vector.

        Args:
            content_id: Content identifier
            embedding: Embedding vector
            ttl: Time-to-live in seconds

        Returns:
            True if cached successfully
        """
        if not self.is_connected:
            return False

        try:
            key = self._make_key(
                self.config["prefixes"]["embedding"],
                content_id,
            )
            ttl = ttl or self._get_ttl("embedding")

            self.client.setex(key, ttl, json.dumps(embedding))
            return True
        except Exception as e:
            logger.error(f"Embedding cache write error: {e}")
            return False

    def get_embedding(self, content_id: str) -> list[float] | None:
        """Get cached embedding.

        Args:
            content_id: Content identifier

        Returns:
            Embedding vector or None
        """
        if not self.is_connected:
            return None

        try:
            key = self._make_key(
                self.config["prefixes"]["embedding"],
                content_id,
            )
            data = self.client.get(key)
            return json.loads(data) if data else None
        except Exception as e:
            logger.error(f"Embedding cache read error: {e}")
            return None

    # API Response Caching
    def cache_api_response(
        self,
        endpoint: str,
        params: dict,
        response: dict,
        category: str = "default",
        ttl: int | None = None,
    ) -> bool:
        """Cache an API response.

        Args:
            endpoint: API endpoint name
            params: Request parameters
            response: Response data
            category: Response category (trends, moves, search, stats)
            ttl: Time-to-live in seconds

        Returns:
            True if cached successfully
        """
        if not self.is_connected:
            return False

        try:
            param_str = json.dumps(params, sort_keys=True)
            identifier = self._hash_string(f"{endpoint}:{param_str}")
            key = self._make_key(self.config["prefixes"]["api"], identifier)
            ttl = ttl or self._get_ttl("api_response", category)

            self.client.setex(key, ttl, json.dumps(response))
            return True
        except Exception as e:
            logger.error(f"API cache write error: {e}")
            return False

    def get_api_response(self, endpoint: str, params: dict) -> dict | None:
        """Get cached API response.

        Args:
            endpoint: API endpoint name
            params: Request parameters

        Returns:
            Cached response or None
        """
        if not self.is_connected:
            return None

        try:
            param_str = json.dumps(params, sort_keys=True)
            identifier = self._hash_string(f"{endpoint}:{param_str}")
            key = self._make_key(self.config["prefixes"]["api"], identifier)

            data = self.client.get(key)
            return json.loads(data) if data else None
        except Exception as e:
            logger.error(f"API cache read error: {e}")
            return None

    # Cache Management
    def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate all keys matching pattern.

        Args:
            pattern: Key pattern (e.g., "brandclave:http:*")

        Returns:
            Number of keys deleted
        """
        if not self.is_connected:
            return 0

        try:
            keys = list(self.client.scan_iter(match=pattern))
            if keys:
                return self.client.delete(*keys)
            return 0
        except Exception as e:
            logger.error(f"Cache invalidation error: {e}")
            return 0

    def clear_all(self) -> bool:
        """Clear all BrandClave cache entries.

        Returns:
            True if successful
        """
        if not self.is_connected:
            return False

        try:
            for prefix in self.config["prefixes"].values():
                self.invalidate_pattern(f"{prefix}*")
            return True
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
            return False

    def get_stats(self) -> dict:
        """Get cache statistics.

        Returns:
            Dict with cache stats
        """
        if not self.is_connected:
            return {"status": "disconnected", "message": "Redis not available"}

        try:
            info = self.client.info("memory")
            db_size = self.client.dbsize()

            # Count keys by prefix
            prefix_counts = {}
            for name, prefix in self.config.get("prefixes", {}).items():
                count = len(list(self.client.scan_iter(match=f"{prefix}*", count=1000)))
                prefix_counts[name] = count

            return {
                "status": "connected",
                "used_memory": info.get("used_memory_human", "unknown"),
                "total_keys": db_size,
                "keys_by_type": prefix_counts,
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}


# Singleton instance
_cache: RedisCache | None = None


def get_cache() -> RedisCache:
    """Get the singleton cache instance."""
    global _cache
    if _cache is None:
        _cache = RedisCache()
    return _cache
