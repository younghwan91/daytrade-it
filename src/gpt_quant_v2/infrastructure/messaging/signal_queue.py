"""
Signal Queue - Redis-backed queue for trading signals.

Provides FIFO and priority queue implementations for TradingSignal entities.
"""

from __future__ import annotations

import json
import logging
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import redis.asyncio as redis

if TYPE_CHECKING:
    from gpt_quant_v2.domain.entities.signal import TradingSignal
    from gpt_quant_v2.infrastructure.messaging.redis_client import RedisClient

logger = logging.getLogger(__name__)

QUEUE_KEY = "gptquant:signals:queue"
PRIORITY_QUEUE_KEY = "gptquant:signals:priority"


class SignalQueueError(Exception):
    """Raised when signal queue operations fail."""


class SignalQueue:
    """
    Redis-backed queue for trading signals.

    Supports both FIFO queue (using Redis lists) and priority queue
    (using Redis sorted sets with confidence as score).

    Usage:
        >>> queue = SignalQueue(redis_client)
        >>> await queue.push(signal)
        >>> signal = await queue.pop()
        >>> length = await queue.length()
    """

    def __init__(
        self,
        redis_client: RedisClient,
        queue_key: str = QUEUE_KEY,
        priority_key: str = PRIORITY_QUEUE_KEY,
    ) -> None:
        """
        Initialize signal queue.

        Args:
            redis_client: Connected Redis client instance.
            queue_key: Redis key for the FIFO queue.
            priority_key: Redis key for the priority queue.
        """
        self._redis_client = redis_client
        self._queue_key = queue_key
        self._priority_key = priority_key

    def _serialize_signal(self, signal: TradingSignal) -> str:
        """
        Serialize a trading signal to JSON.

        Args:
            signal: Trading signal to serialize.

        Returns:
            JSON string representation.
        """
        data = signal.model_dump(mode="json")
        return json.dumps(data, default=str)

    def _deserialize_signal(self, data: str) -> dict[str, Any]:
        """
        Deserialize signal data from JSON.

        Args:
            data: JSON string representation.

        Returns:
            Signal data as dictionary.
        """
        return json.loads(data)

    async def push(self, signal: TradingSignal) -> int:
        """
        Push a signal to the end of the queue (FIFO).

        Args:
            signal: Trading signal to enqueue.

        Returns:
            New length of the queue.

        Raises:
            SignalQueueError: If push fails.
        """
        try:
            client = self._redis_client.client
            message = self._serialize_signal(signal)
            length = await client.rpush(self._queue_key, message)
            logger.debug("Pushed signal %s to queue (length: %d)", signal.id, length)
            return length

        except redis.RedisError as e:
            logger.exception("Failed to push signal to queue")
            raise SignalQueueError(f"Failed to push signal: {e}") from e

    async def pop(self) -> dict[str, Any] | None:
        """
        Pop and return the first signal from the queue (FIFO).

        Returns:
            Signal data as dict, or None if queue is empty.

        Raises:
            SignalQueueError: If pop fails.
        """
        try:
            client = self._redis_client.client
            data = await client.lpop(self._queue_key)

            if data is None:
                return None

            signal_data = self._deserialize_signal(data)
            logger.debug("Popped signal from queue")
            return signal_data

        except redis.RedisError as e:
            logger.exception("Failed to pop signal from queue")
            raise SignalQueueError(f"Failed to pop signal: {e}") from e

    async def pop_blocking(self, timeout: float = 0) -> dict[str, Any] | None:
        """
        Pop signal with blocking wait if queue is empty.

        Args:
            timeout: Seconds to wait (0 = block indefinitely).

        Returns:
            Signal data as dict, or None on timeout.

        Raises:
            SignalQueueError: If pop fails.
        """
        try:
            client = self._redis_client.client
            result = await client.blpop(self._queue_key, timeout=timeout)

            if result is None:
                return None

            _, data = result
            signal_data = self._deserialize_signal(data)
            logger.debug("Blocking pop returned signal")
            return signal_data

        except redis.RedisError as e:
            logger.exception("Failed to blocking pop from queue")
            raise SignalQueueError(f"Failed to blocking pop: {e}") from e

    async def peek(self) -> dict[str, Any] | None:
        """
        Peek at the first signal without removing it.

        Returns:
            Signal data as dict, or None if queue is empty.

        Raises:
            SignalQueueError: If peek fails.
        """
        try:
            client = self._redis_client.client
            result = await client.lrange(self._queue_key, 0, 0)

            if not result:
                return None

            return self._deserialize_signal(result[0])

        except redis.RedisError as e:
            logger.exception("Failed to peek signal from queue")
            raise SignalQueueError(f"Failed to peek signal: {e}") from e

    async def length(self) -> int:
        """
        Get the number of signals in the queue.

        Returns:
            Queue length.

        Raises:
            SignalQueueError: If length check fails.
        """
        try:
            client = self._redis_client.client
            return await client.llen(self._queue_key)

        except redis.RedisError as e:
            logger.exception("Failed to get queue length")
            raise SignalQueueError(f"Failed to get queue length: {e}") from e

    async def clear(self) -> int:
        """
        Clear all signals from the queue.

        Returns:
            Number of signals removed.

        Raises:
            SignalQueueError: If clear fails.
        """
        try:
            client = self._redis_client.client
            length = await client.llen(self._queue_key)
            await client.delete(self._queue_key)
            logger.info("Cleared %d signals from queue", length)
            return length

        except redis.RedisError as e:
            logger.exception("Failed to clear queue")
            raise SignalQueueError(f"Failed to clear queue: {e}") from e

    # Priority Queue Methods (using sorted sets)

    async def push_priority(self, signal: TradingSignal) -> int:
        """
        Push a signal to the priority queue.

        Priority is based on confidence score (higher = higher priority).

        Args:
            signal: Trading signal to enqueue.

        Returns:
            Number of elements added (1 if new, 0 if updated).

        Raises:
            SignalQueueError: If push fails.
        """
        try:
            client = self._redis_client.client
            message = self._serialize_signal(signal)
            # Use confidence as score (higher confidence = higher priority)
            score = float(signal.confidence.value)
            result = await client.zadd(self._priority_key, {message: score})
            logger.debug(
                "Pushed signal %s to priority queue (confidence: %s)",
                signal.id,
                signal.confidence,
            )
            return result

        except redis.RedisError as e:
            logger.exception("Failed to push to priority queue")
            raise SignalQueueError(f"Failed to push to priority queue: {e}") from e

    async def pop_priority(self) -> dict[str, Any] | None:
        """
        Pop the highest priority signal.

        Returns:
            Signal data as dict, or None if queue is empty.

        Raises:
            SignalQueueError: If pop fails.
        """
        try:
            client = self._redis_client.client
            # Pop highest score (highest confidence)
            result = await client.zpopmax(self._priority_key)

            if not result:
                return None

            data, score = result[0]
            signal_data = self._deserialize_signal(data)
            logger.debug("Popped priority signal (score: %s)", score)
            return signal_data

        except redis.RedisError as e:
            logger.exception("Failed to pop from priority queue")
            raise SignalQueueError(f"Failed to pop from priority queue: {e}") from e

    async def pop_priority_blocking(
        self,
        timeout: float = 0,
    ) -> dict[str, Any] | None:
        """
        Pop highest priority signal with blocking wait.

        Args:
            timeout: Seconds to wait (0 = block indefinitely).

        Returns:
            Signal data as dict, or None on timeout.

        Raises:
            SignalQueueError: If pop fails.
        """
        try:
            client = self._redis_client.client
            result = await client.bzpopmax(self._priority_key, timeout=timeout)

            if result is None:
                return None

            _key, data, score = result
            signal_data = self._deserialize_signal(data)
            logger.debug("Blocking pop returned priority signal (score: %s)", score)
            return signal_data

        except redis.RedisError as e:
            logger.exception("Failed to blocking pop from priority queue")
            raise SignalQueueError(f"Failed to blocking pop: {e}") from e

    async def peek_priority(self, count: int = 1) -> list[dict[str, Any]]:
        """
        Peek at the highest priority signals without removing them.

        Args:
            count: Number of signals to peek.

        Returns:
            List of signal data dicts (highest priority first).

        Raises:
            SignalQueueError: If peek fails.
        """
        try:
            client = self._redis_client.client
            # Get top N by score (descending)
            result = await client.zrevrange(self._priority_key, 0, count - 1)

            return [self._deserialize_signal(data) for data in result]

        except redis.RedisError as e:
            logger.exception("Failed to peek priority queue")
            raise SignalQueueError(f"Failed to peek priority queue: {e}") from e

    async def priority_length(self) -> int:
        """
        Get the number of signals in the priority queue.

        Returns:
            Priority queue length.

        Raises:
            SignalQueueError: If length check fails.
        """
        try:
            client = self._redis_client.client
            return await client.zcard(self._priority_key)

        except redis.RedisError as e:
            logger.exception("Failed to get priority queue length")
            raise SignalQueueError(f"Failed to get priority queue length: {e}") from e

    async def clear_priority(self) -> int:
        """
        Clear all signals from the priority queue.

        Returns:
            Number of signals removed.

        Raises:
            SignalQueueError: If clear fails.
        """
        try:
            client = self._redis_client.client
            length = await client.zcard(self._priority_key)
            await client.delete(self._priority_key)
            logger.info("Cleared %d signals from priority queue", length)
            return length

        except redis.RedisError as e:
            logger.exception("Failed to clear priority queue")
            raise SignalQueueError(f"Failed to clear priority queue: {e}") from e

    async def get_signals_by_confidence(
        self,
        min_confidence: Decimal,
        max_confidence: Decimal | None = None,
    ) -> list[dict[str, Any]]:
        """
        Get signals within a confidence range from priority queue.

        Args:
            min_confidence: Minimum confidence score.
            max_confidence: Maximum confidence score (default: 1.0).

        Returns:
            List of signal data dicts.

        Raises:
            SignalQueueError: If query fails.
        """
        try:
            client = self._redis_client.client
            min_score = float(min_confidence)
            max_score = float(max_confidence) if max_confidence else 1.0

            result = await client.zrangebyscore(
                self._priority_key,
                min_score,
                max_score,
            )

            return [self._deserialize_signal(data) for data in result]

        except redis.RedisError as e:
            logger.exception("Failed to query signals by confidence")
            raise SignalQueueError(f"Failed to query by confidence: {e}") from e
