"""
Redis Client - Async connection pool and health check.

Provides a centralized Redis client with connection pooling for use
across the messaging infrastructure.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

import redis.asyncio as redis
from redis.asyncio.connection import ConnectionPool

if TYPE_CHECKING:
    from gpt_quant_v2.shared.config.settings import RedisSettings

logger = logging.getLogger(__name__)


class RedisConnectionError(Exception):
    """Raised when Redis connection fails."""


class RedisClient:
    """
    Async Redis client with connection pooling.

    Provides a shared connection pool for Redis operations,
    with health checking and graceful shutdown support.

    Usage:
        >>> client = RedisClient(settings.redis)
        >>> await client.connect()
        >>> is_healthy = await client.health_check()
        >>> await client.disconnect()
    """

    def __init__(self, settings: RedisSettings) -> None:
        """
        Initialize Redis client.

        Args:
            settings: Redis configuration settings.
        """
        self._settings = settings
        self._pool: ConnectionPool | None = None
        self._client: redis.Redis | None = None
        self._connected = False

    @property
    def client(self) -> redis.Redis:
        """Get the Redis client instance."""
        if self._client is None:
            raise RedisConnectionError("Redis client not connected. Call connect() first.")
        return self._client

    @property
    def is_connected(self) -> bool:
        """Check if client is connected."""
        return self._connected

    async def connect(self) -> None:
        """
        Establish connection to Redis with connection pooling.

        Raises:
            RedisConnectionError: If connection fails.
        """
        if self._connected:
            logger.debug("Redis client already connected")
            return

        try:
            url = str(self._settings.url)
            self._pool = ConnectionPool.from_url(
                url,
                max_connections=self._settings.max_connections,
                socket_timeout=self._settings.socket_timeout,
                socket_connect_timeout=self._settings.socket_connect_timeout,
                decode_responses=True,
            )
            self._client = redis.Redis(connection_pool=self._pool)

            # Test connection
            await self._client.ping()
            self._connected = True
            logger.info("Redis client connected to %s", url.split("@")[-1])  # Log without auth

        except redis.RedisError as e:
            self._connected = False
            logger.exception("Failed to connect to Redis")
            raise RedisConnectionError(f"Failed to connect to Redis: {e}") from e

    async def disconnect(self) -> None:
        """
        Close Redis connection and cleanup resources.
        """
        if not self._connected:
            return

        try:
            if self._client:
                await self._client.aclose()
            if self._pool:
                await self._pool.disconnect()

            self._client = None
            self._pool = None
            self._connected = False
            logger.info("Redis client disconnected")

        except redis.RedisError as e:
            logger.warning("Error during Redis disconnect: %s", e)

    async def health_check(self) -> bool:
        """
        Check if Redis connection is healthy.

        Returns:
            True if healthy, False otherwise.
        """
        if not self._connected or not self._client:
            return False

        try:
            await asyncio.wait_for(self._client.ping(), timeout=2.0)
            return True
        except (redis.RedisError, TimeoutError):
            logger.warning("Redis health check failed")
            return False

    async def __aenter__(self) -> RedisClient:
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.disconnect()
