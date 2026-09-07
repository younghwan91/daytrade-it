"""
Redis Cache - Key-value caching with TTL support.

Provides JSON-serialized caching with key prefixing and TTL management.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

import redis.asyncio as redis

if TYPE_CHECKING:
    from krx_signal_engine.infrastructure.messaging.redis_client import RedisClient

logger = logging.getLogger(__name__)

DEFAULT_PREFIX = "gptquant:cache"
DEFAULT_TTL = 3600  # 1 hour


class CacheError(Exception):
    """Raised when cache operations fail."""


class Cache:
    """
    Redis-backed cache with JSON serialization.

    Provides get/set/delete operations with automatic key prefixing,
    JSON serialization, and TTL support.

    Usage:
        >>> cache = Cache(redis_client)
        >>> await cache.set("user:123", {"name": "John"}, ttl=3600)
        >>> data = await cache.get("user:123")
        >>> await cache.delete("user:123")
    """

    def __init__(
        self,
        redis_client: RedisClient,
        prefix: str = DEFAULT_PREFIX,
        default_ttl: int | None = DEFAULT_TTL,
    ) -> None:
        """
        Initialize cache.

        Args:
            redis_client: Connected Redis client instance.
            prefix: Key prefix for all cache entries.
            default_ttl: Default TTL in seconds (None = no expiry).
        """
        self._redis_client = redis_client
        self._prefix = prefix
        self._default_ttl = default_ttl

    def _make_key(self, key: str) -> str:
        """
        Create prefixed cache key.

        Args:
            key: User-provided cache key.

        Returns:
            Prefixed key.
        """
        return f"{self._prefix}:{key}"

    def _serialize(self, value: Any) -> str:
        """
        Serialize value to JSON string.

        Args:
            value: Value to serialize.

        Returns:
            JSON string.
        """
        return json.dumps(value, default=str)

    def _deserialize(self, data: str) -> Any:
        """
        Deserialize JSON string to value.

        Args:
            data: JSON string.

        Returns:
            Deserialized value.
        """
        return json.loads(data)

    async def get(self, key: str) -> Any | None:
        """
        Get a value from cache.

        Args:
            key: Cache key.

        Returns:
            Cached value or None if not found.

        Raises:
            CacheError: If get fails.
        """
        prefixed_key = self._make_key(key)

        try:
            client = self._redis_client.client
            data = await client.get(prefixed_key)

            if data is None:
                logger.debug("Cache miss: %s", key)
                return None

            logger.debug("Cache hit: %s", key)
            return self._deserialize(data)

        except json.JSONDecodeError:
            logger.warning("Failed to deserialize cached value: %s", key)
            return None
        except redis.RedisError as e:
            logger.exception("Cache get failed: %s", key)
            raise CacheError(f"Failed to get from cache: {e}") from e

    async def set(
        self,
        key: str,
        value: Any,
        ttl: int | None = None,
    ) -> bool:
        """
        Set a value in cache.

        Args:
            key: Cache key.
            value: Value to cache (must be JSON serializable).
            ttl: TTL in seconds (None = use default, 0 = no expiry).

        Returns:
            True if successful.

        Raises:
            CacheError: If set fails.
        """
        prefixed_key = self._make_key(key)
        effective_ttl = ttl if ttl is not None else self._default_ttl

        try:
            client = self._redis_client.client
            data = self._serialize(value)

            if effective_ttl and effective_ttl > 0:
                await client.setex(prefixed_key, effective_ttl, data)
            else:
                await client.set(prefixed_key, data)

            logger.debug("Cache set: %s (ttl: %s)", key, effective_ttl)
            return True

        except (TypeError, ValueError) as e:
            logger.exception("Failed to serialize value for cache: %s", key)
            raise CacheError(f"Failed to serialize value: {e}") from e
        except redis.RedisError as e:
            logger.exception("Cache set failed: %s", key)
            raise CacheError(f"Failed to set cache: {e}") from e

    async def delete(self, key: str) -> bool:
        """
        Delete a value from cache.

        Args:
            key: Cache key.

        Returns:
            True if key was deleted, False if not found.

        Raises:
            CacheError: If delete fails.
        """
        prefixed_key = self._make_key(key)

        try:
            client = self._redis_client.client
            result = await client.delete(prefixed_key)
            logger.debug("Cache delete: %s (found: %s)", key, bool(result))
            return bool(result)

        except redis.RedisError as e:
            logger.exception("Cache delete failed: %s", key)
            raise CacheError(f"Failed to delete from cache: {e}") from e

    async def exists(self, key: str) -> bool:
        """
        Check if a key exists in cache.

        Args:
            key: Cache key.

        Returns:
            True if key exists.

        Raises:
            CacheError: If check fails.
        """
        prefixed_key = self._make_key(key)

        try:
            client = self._redis_client.client
            return bool(await client.exists(prefixed_key))

        except redis.RedisError as e:
            logger.exception("Cache exists check failed: %s", key)
            raise CacheError(f"Failed to check cache existence: {e}") from e

    async def ttl(self, key: str) -> int | None:
        """
        Get remaining TTL for a key.

        Args:
            key: Cache key.

        Returns:
            TTL in seconds, -1 if no expiry, -2 if not found.

        Raises:
            CacheError: If TTL check fails.
        """
        prefixed_key = self._make_key(key)

        try:
            client = self._redis_client.client
            result = await client.ttl(prefixed_key)

            if result == -2:
                return None  # Key doesn't exist
            return result

        except redis.RedisError as e:
            logger.exception("Cache TTL check failed: %s", key)
            raise CacheError(f"Failed to get cache TTL: {e}") from e

    async def expire(self, key: str, ttl: int) -> bool:
        """
        Set or update TTL for a key.

        Args:
            key: Cache key.
            ttl: New TTL in seconds.

        Returns:
            True if successful, False if key doesn't exist.

        Raises:
            CacheError: If expire fails.
        """
        prefixed_key = self._make_key(key)

        try:
            client = self._redis_client.client
            result = await client.expire(prefixed_key, ttl)
            logger.debug("Cache expire: %s (ttl: %d)", key, ttl)
            return bool(result)

        except redis.RedisError as e:
            logger.exception("Cache expire failed: %s", key)
            raise CacheError(f"Failed to set cache expiry: {e}") from e

    async def get_many(self, keys: list[str]) -> dict[str, Any]:
        """
        Get multiple values from cache.

        Args:
            keys: List of cache keys.

        Returns:
            Dict mapping keys to values (missing keys excluded).

        Raises:
            CacheError: If get fails.
        """
        if not keys:
            return {}

        prefixed_keys = [self._make_key(k) for k in keys]

        try:
            client = self._redis_client.client
            values = await client.mget(prefixed_keys)

            result = {}
            for key, value in zip(keys, values, strict=True):
                if value is not None:
                    try:
                        result[key] = self._deserialize(value)
                    except json.JSONDecodeError:
                        logger.warning("Failed to deserialize: %s", key)

            logger.debug("Cache get_many: %d/%d hits", len(result), len(keys))
            return result

        except redis.RedisError as e:
            logger.exception("Cache get_many failed")
            raise CacheError(f"Failed to get multiple from cache: {e}") from e

    async def set_many(
        self,
        mapping: dict[str, Any],
        ttl: int | None = None,
    ) -> bool:
        """
        Set multiple values in cache.

        Args:
            mapping: Dict of key-value pairs to cache.
            ttl: TTL in seconds (applies to all keys).

        Returns:
            True if successful.

        Raises:
            CacheError: If set fails.
        """
        if not mapping:
            return True

        effective_ttl = ttl if ttl is not None else self._default_ttl

        try:
            client = self._redis_client.client

            # Serialize all values
            prefixed_mapping = {self._make_key(k): self._serialize(v) for k, v in mapping.items()}

            async with client.pipeline(transaction=True) as pipe:
                await pipe.mset(prefixed_mapping)

                if effective_ttl and effective_ttl > 0:
                    for key in prefixed_mapping:
                        pipe.expire(key, effective_ttl)

                await pipe.execute()

            logger.debug("Cache set_many: %d keys", len(mapping))
            return True

        except (TypeError, ValueError) as e:
            logger.exception("Failed to serialize values for cache")
            raise CacheError(f"Failed to serialize values: {e}") from e
        except redis.RedisError as e:
            logger.exception("Cache set_many failed")
            raise CacheError(f"Failed to set multiple in cache: {e}") from e

    async def delete_many(self, keys: list[str]) -> int:
        """
        Delete multiple values from cache.

        Args:
            keys: List of cache keys.

        Returns:
            Number of keys deleted.

        Raises:
            CacheError: If delete fails.
        """
        if not keys:
            return 0

        prefixed_keys = [self._make_key(k) for k in keys]

        try:
            client = self._redis_client.client
            result = await client.delete(*prefixed_keys)
            logger.debug("Cache delete_many: %d keys deleted", result)
            return result

        except redis.RedisError as e:
            logger.exception("Cache delete_many failed")
            raise CacheError(f"Failed to delete multiple from cache: {e}") from e

    async def clear_prefix(self, pattern: str = "*") -> int:
        """
        Clear all keys matching a pattern under the cache prefix.

        Args:
            pattern: Glob pattern to match (e.g., "user:*").

        Returns:
            Number of keys deleted.

        Raises:
            CacheError: If clear fails.
        """
        full_pattern = self._make_key(pattern)

        try:
            client = self._redis_client.client
            cursor = None
            deleted = 0

            while cursor != 0:
                cursor, keys = await client.scan(
                    cursor=cursor or 0,
                    match=full_pattern,
                    count=100,
                )
                if keys:
                    deleted += await client.delete(*keys)

            logger.info("Cache clear_prefix '%s': %d keys deleted", pattern, deleted)
            return deleted

        except redis.RedisError as e:
            logger.exception("Cache clear_prefix failed")
            raise CacheError(f"Failed to clear cache prefix: {e}") from e

    async def increment(self, key: str, amount: int = 1) -> int:
        """
        Increment a numeric value in cache.

        Args:
            key: Cache key.
            amount: Amount to increment by.

        Returns:
            New value after increment.

        Raises:
            CacheError: If increment fails.
        """
        prefixed_key = self._make_key(key)

        try:
            client = self._redis_client.client
            result = await client.incrby(prefixed_key, amount)
            logger.debug("Cache increment: %s by %d = %d", key, amount, result)
            return result

        except redis.RedisError as e:
            logger.exception("Cache increment failed: %s", key)
            raise CacheError(f"Failed to increment cache: {e}") from e

    async def decrement(self, key: str, amount: int = 1) -> int:
        """
        Decrement a numeric value in cache.

        Args:
            key: Cache key.
            amount: Amount to decrement by.

        Returns:
            New value after decrement.

        Raises:
            CacheError: If decrement fails.
        """
        prefixed_key = self._make_key(key)

        try:
            client = self._redis_client.client
            result = await client.decrby(prefixed_key, amount)
            logger.debug("Cache decrement: %s by %d = %d", key, amount, result)
            return result

        except redis.RedisError as e:
            logger.exception("Cache decrement failed: %s", key)
            raise CacheError(f"Failed to decrement cache: {e}") from e
