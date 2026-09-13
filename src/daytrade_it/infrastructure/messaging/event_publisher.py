"""
Event Publisher - Redis pub/sub event publishing.

Publishes domain events to Redis channels for distributed event handling.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

import redis.asyncio as redis

if TYPE_CHECKING:
    from daytrade_it.domain.events.base import DomainEvent
    from daytrade_it.infrastructure.messaging.redis_client import RedisClient

logger = logging.getLogger(__name__)

CHANNEL_PREFIX = "gptquant.events"


class EventPublishError(Exception):
    """Raised when event publishing fails."""


class EventPublisher:
    """
    Publishes domain events to Redis pub/sub channels.

    Channel naming convention: "gptquant.events.{event_type}"
    Events are JSON serialized before publishing.

    Usage:
        >>> publisher = EventPublisher(redis_client)
        >>> await publisher.publish(signal_created_event)
        >>> await publisher.publish_batch([event1, event2])
    """

    def __init__(self, redis_client: RedisClient) -> None:
        """
        Initialize event publisher.

        Args:
            redis_client: Connected Redis client instance.
        """
        self._redis_client = redis_client

    def _get_channel_name(self, event_type: str) -> str:
        """
        Generate channel name for an event type.

        Args:
            event_type: The event type/class name.

        Returns:
            Channel name in format "gptquant.events.{event_type}".
        """
        # Convert CamelCase to snake_case for channel names
        import re

        snake_case = re.sub(r"(?<!^)(?=[A-Z])", "_", event_type).lower()
        return f"{CHANNEL_PREFIX}.{snake_case}"

    def _serialize_event(self, event: DomainEvent) -> str:
        """
        Serialize event to JSON string.

        Args:
            event: Domain event to serialize.

        Returns:
            JSON string representation.
        """
        return json.dumps(event.to_dict(), default=str)

    async def publish(self, event: DomainEvent) -> int:
        """
        Publish a domain event to Redis.

        Args:
            event: Domain event to publish.

        Returns:
            Number of subscribers that received the event.

        Raises:
            EventPublishError: If publishing fails.
        """
        channel = self._get_channel_name(event.event_type)
        message = self._serialize_event(event)

        try:
            client = self._redis_client.client
            subscriber_count = await client.publish(channel, message)
            logger.debug(
                "Published event %s to channel %s (%d subscribers)",
                event.event_id,
                channel,
                subscriber_count,
            )
            return subscriber_count

        except redis.RedisError as e:
            logger.exception("Failed to publish event %s", event.event_id)
            raise EventPublishError(f"Failed to publish event: {e}") from e

    async def publish_batch(self, events: list[DomainEvent]) -> dict[str, int]:
        """
        Publish multiple events in batch.

        Uses Redis pipeline for efficient batch publishing.

        Args:
            events: List of domain events to publish.

        Returns:
            Dict mapping event IDs to subscriber counts.

        Raises:
            EventPublishError: If any publish fails.
        """
        if not events:
            return {}

        results: dict[str, int] = {}

        try:
            client = self._redis_client.client
            async with client.pipeline(transaction=False) as pipe:
                for event in events:
                    channel = self._get_channel_name(event.event_type)
                    message = self._serialize_event(event)
                    pipe.publish(channel, message)

                responses = await pipe.execute()

            for event, subscriber_count in zip(events, responses, strict=True):
                results[str(event.event_id)] = subscriber_count
                logger.debug(
                    "Published event %s (%d subscribers)",
                    event.event_id,
                    subscriber_count,
                )

            logger.info("Published batch of %d events", len(events))
            return results

        except redis.RedisError as e:
            logger.exception("Failed to publish event batch")
            raise EventPublishError(f"Failed to publish event batch: {e}") from e

    async def publish_to_channel(
        self,
        channel: str,
        event: DomainEvent,
    ) -> int:
        """
        Publish event to a specific channel (bypass naming convention).

        Args:
            channel: Target channel name.
            event: Domain event to publish.

        Returns:
            Number of subscribers that received the event.

        Raises:
            EventPublishError: If publishing fails.
        """
        message = self._serialize_event(event)

        try:
            client = self._redis_client.client
            subscriber_count = await client.publish(channel, message)
            logger.debug(
                "Published event %s to custom channel %s (%d subscribers)",
                event.event_id,
                channel,
                subscriber_count,
            )
            return subscriber_count

        except redis.RedisError as e:
            logger.exception("Failed to publish to channel %s", channel)
            raise EventPublishError(f"Failed to publish to channel: {e}") from e
