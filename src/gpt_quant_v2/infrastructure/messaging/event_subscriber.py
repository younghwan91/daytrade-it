"""
Event Subscriber - Redis pub/sub event subscription.

Subscribes to domain events from Redis channels with async handling.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import Callable, Coroutine
from typing import TYPE_CHECKING, Any

import redis.asyncio as redis

if TYPE_CHECKING:
    from gpt_quant_v2.infrastructure.messaging.redis_client import RedisClient

logger = logging.getLogger(__name__)

CHANNEL_PREFIX = "gptquant.events"

EventHandler = Callable[[dict[str, Any]], Coroutine[Any, Any, None]]


class EventSubscriptionError(Exception):
    """Raised when event subscription fails."""


class EventSubscriber:
    """
    Subscribes to domain events from Redis pub/sub channels.

    Supports subscribing to multiple event types with pattern matching
    and graceful shutdown.

    Usage:
        >>> subscriber = EventSubscriber(redis_client)
        >>> await subscriber.subscribe(["SignalCreated", "TradeExecuted"], handler)
        >>> await subscriber.start()  # Blocks, listening for events
        >>> await subscriber.stop()   # Graceful shutdown
    """

    def __init__(self, redis_client: RedisClient) -> None:
        """
        Initialize event subscriber.

        Args:
            redis_client: Connected Redis client instance.
        """
        self._redis_client = redis_client
        self._pubsub: redis.client.PubSub | None = None
        self._handlers: dict[str, list[EventHandler]] = {}
        self._running = False
        self._listen_task: asyncio.Task | None = None

    def _get_channel_name(self, event_type: str) -> str:
        """
        Generate channel name for an event type.

        Args:
            event_type: The event type/class name.

        Returns:
            Channel name in format "gptquant.events.{event_type}".
        """
        import re

        snake_case = re.sub(r"(?<!^)(?=[A-Z])", "_", event_type).lower()
        return f"{CHANNEL_PREFIX}.{snake_case}"

    def _deserialize_event(self, data: str) -> dict[str, Any]:
        """
        Deserialize event from JSON string.

        Args:
            data: JSON string representation.

        Returns:
            Event data as dictionary.
        """
        return json.loads(data)

    async def subscribe(
        self,
        event_types: list[str],
        handler: EventHandler,
    ) -> None:
        """
        Subscribe to event types with a handler.

        Args:
            event_types: List of event type names to subscribe to.
            handler: Async function to handle events.

        Raises:
            EventSubscriptionError: If subscription fails.
        """
        try:
            if self._pubsub is None:
                self._pubsub = self._redis_client.client.pubsub()

            channels = [self._get_channel_name(et) for et in event_types]

            for channel in channels:
                if channel not in self._handlers:
                    self._handlers[channel] = []
                self._handlers[channel].append(handler)

            await self._pubsub.subscribe(*channels)
            logger.info("Subscribed to channels: %s", channels)

        except redis.RedisError as e:
            logger.exception("Failed to subscribe to events")
            raise EventSubscriptionError(f"Failed to subscribe: {e}") from e

    async def subscribe_pattern(
        self,
        pattern: str,
        handler: EventHandler,
    ) -> None:
        """
        Subscribe to channels matching a pattern.

        Args:
            pattern: Redis pattern (e.g., "gptquant.events.*").
            handler: Async function to handle events.

        Raises:
            EventSubscriptionError: If subscription fails.
        """
        try:
            if self._pubsub is None:
                self._pubsub = self._redis_client.client.pubsub()

            if pattern not in self._handlers:
                self._handlers[pattern] = []
            self._handlers[pattern].append(handler)

            await self._pubsub.psubscribe(pattern)
            logger.info("Subscribed to pattern: %s", pattern)

        except redis.RedisError as e:
            logger.exception("Failed to subscribe to pattern")
            raise EventSubscriptionError(f"Failed to subscribe to pattern: {e}") from e

    async def unsubscribe(self, event_types: list[str]) -> None:
        """
        Unsubscribe from event types.

        Args:
            event_types: List of event type names to unsubscribe from.
        """
        if self._pubsub is None:
            return

        channels = [self._get_channel_name(et) for et in event_types]
        await self._pubsub.unsubscribe(*channels)

        for channel in channels:
            self._handlers.pop(channel, None)

        logger.info("Unsubscribed from channels: %s", channels)

    async def start(self) -> None:
        """
        Start listening for events.

        This method blocks and processes events until stop() is called.
        Events are dispatched to registered handlers.
        """
        if self._pubsub is None:
            raise EventSubscriptionError("No subscriptions. Call subscribe() first.")

        self._running = True
        logger.info("Event subscriber started")

        try:
            async for message in self._pubsub.listen():
                if not self._running:
                    break

                if message["type"] in ("message", "pmessage"):
                    await self._handle_message(message)

        except asyncio.CancelledError:
            logger.info("Event subscriber cancelled")
        except redis.RedisError as e:
            logger.exception("Event subscriber error")
            raise EventSubscriptionError(f"Subscriber error: {e}") from e
        finally:
            self._running = False

    async def start_background(self) -> asyncio.Task:
        """
        Start listening for events in a background task.

        Returns:
            The asyncio Task running the listener.
        """
        self._listen_task = asyncio.create_task(self.start())
        return self._listen_task

    async def _handle_message(self, message: dict[str, Any]) -> None:
        """
        Process a received message and dispatch to handlers.

        Args:
            message: Redis pub/sub message.
        """
        try:
            # Determine channel (pattern vs direct subscription)
            channel = message.get("channel", message.get("pattern", ""))
            if isinstance(channel, bytes):
                channel = channel.decode("utf-8")

            data = message.get("data", "")
            if isinstance(data, bytes):
                data = data.decode("utf-8")

            event_data = self._deserialize_event(data)

            # Find matching handlers
            handlers = self._handlers.get(channel, [])

            # Also check pattern handlers
            for pattern, pattern_handlers in self._handlers.items():
                if "*" in pattern or "?" in pattern:
                    handlers.extend(pattern_handlers)

            if not handlers:
                logger.warning("No handlers for channel: %s", channel)
                return

            # Dispatch to all handlers
            for handler in handlers:
                try:
                    await handler(event_data)
                except Exception:
                    logger.exception("Handler error for event on %s", channel)

        except json.JSONDecodeError:
            logger.exception("Failed to deserialize event")
        except Exception:
            logger.exception("Unexpected error handling message")

    async def stop(self) -> None:
        """
        Stop listening and cleanup resources.
        """
        self._running = False

        if self._listen_task and not self._listen_task.done():
            self._listen_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._listen_task

        if self._pubsub:
            await self._pubsub.unsubscribe()
            await self._pubsub.punsubscribe()
            await self._pubsub.aclose()
            self._pubsub = None

        self._handlers.clear()
        logger.info("Event subscriber stopped")

    @property
    def is_running(self) -> bool:
        """Check if subscriber is currently running."""
        return self._running

    async def __aenter__(self) -> EventSubscriber:
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.stop()
