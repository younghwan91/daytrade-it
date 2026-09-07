"""
Event Bridge - Redis to WebSocket.

Subscribes to Redis domain events and broadcasts them to WebSocket clients.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from typing import TYPE_CHECKING, Any

import redis.asyncio as redis

if TYPE_CHECKING:
    from krx_signal_engine.interfaces.websocket.manager import ConnectionManager

logger = logging.getLogger(__name__)

# Redis channel patterns for domain events
REDIS_EVENT_PATTERNS = [
    "gptquant.events.signal_*",  # All signal events
    "gptquant.events.trade_*",  # All trade events
    "gptquant.events.price_*",  # All price events
]

# Mapping from Redis event types to WebSocket channels
EVENT_TO_CHANNEL_MAP = {
    "signal_created": "signals",
    "signal_updated": "signals",
    "signal_executed": "signals",
    "trade_opened": "trades",
    "trade_closed": "trades",
    "trade_updated": "trades",
    "price_updated": "prices",
    "price_alert": "prices",
}


class EventBridge:
    """
    Bridges Redis pub/sub events to WebSocket clients.

    Subscribes to Redis domain events and converts them to WebSocket
    messages for subscribed clients.

    Usage:
        >>> bridge = EventBridge(connection_manager, redis_client)
        >>> await bridge.start()  # Start listening for events
        >>> await bridge.stop()   # Graceful shutdown
    """

    def __init__(
        self,
        manager: ConnectionManager,
        redis_client: redis.Redis,
    ) -> None:
        """
        Initialize event bridge.

        Args:
            manager: WebSocket connection manager.
            redis_client: Redis client instance.
        """
        self._manager = manager
        self._redis = redis_client
        self._pubsub: redis.client.PubSub | None = None
        self._running = False
        self._listen_task: asyncio.Task | None = None

    @property
    def is_running(self) -> bool:
        """Check if the event bridge is running."""
        return self._running

    async def start(self) -> None:
        """
        Start subscribing to Redis events and forwarding to WebSocket.

        This method starts a background task that listens for Redis
        pub/sub events and forwards them to subscribed WebSocket clients.
        """
        if self._running:
            logger.warning("Event bridge already running")
            return

        try:
            self._pubsub = self._redis.pubsub()

            # Subscribe to event patterns
            for pattern in REDIS_EVENT_PATTERNS:
                await self._pubsub.psubscribe(pattern)

            self._running = True
            self._listen_task = asyncio.create_task(self._listen_loop())

            logger.info(
                "Event bridge started, subscribed to patterns: %s",
                REDIS_EVENT_PATTERNS,
            )

        except redis.RedisError:
            logger.exception("Failed to start event bridge")
            self._running = False
            raise

    async def stop(self) -> None:
        """Stop the event bridge and cleanup resources."""
        self._running = False

        if self._listen_task and not self._listen_task.done():
            self._listen_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._listen_task

        if self._pubsub:
            await self._pubsub.punsubscribe()
            await self._pubsub.aclose()
            self._pubsub = None

        logger.info("Event bridge stopped")

    async def _listen_loop(self) -> None:
        """Main event listening loop."""
        try:
            async for message in self._pubsub.listen():
                if not self._running:
                    break

                if message["type"] == "pmessage":
                    await self._handle_event(message)

        except asyncio.CancelledError:
            logger.debug("Event bridge listen loop cancelled")
        except redis.RedisError:
            logger.exception("Redis error in event bridge")
            self._running = False
        except Exception:
            logger.exception("Unexpected error in event bridge")
            self._running = False

    async def _handle_event(self, message: dict[str, Any]) -> None:
        """
        Handle a Redis event and forward to WebSocket clients.

        Args:
            message: Redis pub/sub message.
        """
        try:
            # Parse channel and data
            channel = message.get("channel", "")
            if isinstance(channel, bytes):
                channel = channel.decode("utf-8")

            data = message.get("data", "")
            if isinstance(data, bytes):
                data = data.decode("utf-8")

            # Parse event data
            try:
                event_data = json.loads(data)
            except json.JSONDecodeError:
                logger.warning("Failed to parse event data: %s", data[:100])
                return

            # Determine WebSocket channels to broadcast to
            ws_channels = self._map_event_to_channels(channel, event_data)

            if not ws_channels:
                logger.debug("No WebSocket channels for event: %s", channel)
                return

            # Create WebSocket message
            ws_message = self._create_ws_message(channel, event_data)

            # Broadcast to all mapped channels
            for ws_channel in ws_channels:
                sent = await self._manager.broadcast(ws_channel, ws_message)
                logger.debug(
                    "Broadcast event to channel %s: sent to %d clients",
                    ws_channel,
                    sent,
                )

        except Exception:
            logger.exception("Error handling Redis event")

    def _map_event_to_channels(
        self,
        redis_channel: str,
        event_data: dict[str, Any],
    ) -> list[str]:
        """
        Map a Redis event to WebSocket channels.

        Args:
            redis_channel: The Redis channel the event came from.
            event_data: The event data.

        Returns:
            List of WebSocket channels to broadcast to.
        """
        channels = []

        # Extract event type from channel name
        # e.g., "gptquant.events.signal_created" -> "signal_created"
        event_type = redis_channel.split(".")[-1] if "." in redis_channel else ""

        # Get base channel type
        base_channel = EVENT_TO_CHANNEL_MAP.get(event_type)
        if not base_channel:
            # Try prefix matching
            for prefix in ("signal", "trade", "price"):
                if event_type.startswith(prefix):
                    base_channel = f"{prefix}s"  # signals, trades, prices
                    break

        if not base_channel:
            return channels

        # Always add to the "all" channel
        channels.append(f"{base_channel}:all")

        # Add ticker-specific channel if ticker is present
        ticker = event_data.get("ticker") or event_data.get("symbol")
        if ticker:
            channels.append(f"{base_channel}:{ticker}")

        return channels

    def _create_ws_message(
        self,
        redis_channel: str,
        event_data: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Create a WebSocket message from Redis event data.

        Args:
            redis_channel: The Redis channel.
            event_data: The event data.

        Returns:
            Formatted WebSocket message.
        """
        # Extract event type
        event_type = redis_channel.split(".")[-1] if "." in redis_channel else "unknown"

        return {
            "event_type": event_type,
            "event_id": event_data.get("event_id"),
            "occurred_at": event_data.get("occurred_at"),
            "payload": event_data,
        }


async def create_event_bridge(
    manager: ConnectionManager,
    redis_client: redis.Redis,
) -> EventBridge:
    """
    Create and start an event bridge.

    Args:
        manager: WebSocket connection manager.
        redis_client: Redis client instance.

    Returns:
        Started EventBridge instance.
    """
    bridge = EventBridge(manager, redis_client)
    await bridge.start()
    return bridge
