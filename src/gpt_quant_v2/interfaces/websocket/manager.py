"""
WebSocket Connection Manager.

Manages WebSocket connections, subscriptions, and message broadcasting.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
import contextlib
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from fastapi import WebSocket, WebSocketDisconnect

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class ConnectionManager:
    """
    Manages WebSocket connections and channel subscriptions.

    Provides connection lifecycle management, channel-based pub/sub,
    and both broadcast and personal messaging capabilities.

    Usage:
        >>> manager = ConnectionManager()
        >>> await manager.connect(websocket, "client-123")
        >>> await manager.subscribe("client-123", ["signals:AAPL", "trades:all"])
        >>> await manager.broadcast("signals:AAPL", {"type": "signal", "data": {...}})
        >>> await manager.disconnect("client-123")
    """

    def __init__(self) -> None:
        """Initialize connection manager with empty state."""
        # client_id -> WebSocket
        self._connections: dict[str, WebSocket] = {}
        # client_id -> set of subscribed channels
        self._client_subscriptions: dict[str, set[str]] = defaultdict(set)
        # channel -> set of client_ids
        self._channel_subscribers: dict[str, set[str]] = defaultdict(set)
        # Lock for thread-safe operations
        self._lock = asyncio.Lock()

    @property
    def active_connections(self) -> int:
        """Get the number of active connections."""
        return len(self._connections)

    @property
    def channels(self) -> list[str]:
        """Get list of active channels with subscribers."""
        return [ch for ch, subs in self._channel_subscribers.items() if subs]

    async def connect(self, websocket: WebSocket, client_id: str) -> None:
        """
        Accept and register a WebSocket connection.

        Args:
            websocket: The WebSocket connection to register.
            client_id: Unique identifier for the client.
        """
        await websocket.accept()

        async with self._lock:
            # Disconnect existing connection for this client if any
            if client_id in self._connections:
                old_ws = self._connections[client_id]
                with contextlib.suppress(Exception):
                    await old_ws.close(code=1008, reason="Replaced by new connection")

            self._connections[client_id] = websocket
            self._client_subscriptions[client_id] = set()

        logger.info(
            "WebSocket connected: client_id=%s, total_connections=%d",
            client_id,
            self.active_connections,
        )

    async def disconnect(self, client_id: str) -> None:
        """
        Disconnect and cleanup a WebSocket connection.

        Args:
            client_id: The client to disconnect.
        """
        async with self._lock:
            if client_id not in self._connections:
                return

            # Remove from all subscribed channels
            subscribed_channels = self._client_subscriptions.pop(client_id, set())
            for channel in subscribed_channels:
                self._channel_subscribers[channel].discard(client_id)
                # Clean up empty channels
                if not self._channel_subscribers[channel]:
                    del self._channel_subscribers[channel]

            # Remove connection
            websocket = self._connections.pop(client_id, None)

            if websocket:
                with contextlib.suppress(Exception):
                    await websocket.close()

        logger.info(
            "WebSocket disconnected: client_id=%s, total_connections=%d",
            client_id,
            self.active_connections,
        )

    async def subscribe(self, client_id: str, channels: list[str]) -> list[str]:
        """
        Subscribe a client to one or more channels.

        Args:
            client_id: The client to subscribe.
            channels: List of channel names to subscribe to.

        Returns:
            List of channels successfully subscribed to.

        Raises:
            ValueError: If client is not connected.
        """
        async with self._lock:
            if client_id not in self._connections:
                raise ValueError(f"Client {client_id} is not connected")

            subscribed = []
            for channel in channels:
                channel = channel.strip()
                if not channel:
                    continue

                self._client_subscriptions[client_id].add(channel)
                self._channel_subscribers[channel].add(client_id)
                subscribed.append(channel)

            logger.debug(
                "Client %s subscribed to channels: %s",
                client_id,
                subscribed,
            )
            return subscribed

    async def unsubscribe(self, client_id: str, channels: list[str]) -> list[str]:
        """
        Unsubscribe a client from one or more channels.

        Args:
            client_id: The client to unsubscribe.
            channels: List of channel names to unsubscribe from.

        Returns:
            List of channels successfully unsubscribed from.
        """
        async with self._lock:
            if client_id not in self._connections:
                return []

            unsubscribed = []
            for channel in channels:
                channel = channel.strip()
                if channel in self._client_subscriptions[client_id]:
                    self._client_subscriptions[client_id].discard(channel)
                    self._channel_subscribers[channel].discard(client_id)
                    unsubscribed.append(channel)

                    # Clean up empty channels
                    if not self._channel_subscribers[channel]:
                        del self._channel_subscribers[channel]

            logger.debug(
                "Client %s unsubscribed from channels: %s",
                client_id,
                unsubscribed,
            )
            return unsubscribed

    async def broadcast(self, channel: str, message: dict) -> int:
        """
        Broadcast a message to all subscribers of a channel.

        Args:
            channel: The channel to broadcast to.
            message: The message to send (will be JSON serialized).

        Returns:
            Number of clients the message was sent to.
        """
        async with self._lock:
            subscribers = list(self._channel_subscribers.get(channel, set()))

        if not subscribers:
            return 0

        # Add metadata to message
        envelope = {
            "channel": channel,
            "timestamp": datetime.now(UTC).isoformat(),
            "data": message,
        }

        sent_count = 0
        failed_clients = []

        for client_id in subscribers:
            try:
                websocket = self._connections.get(client_id)
                if websocket:
                    await websocket.send_json(envelope)
                    sent_count += 1
            except WebSocketDisconnect:
                failed_clients.append(client_id)
            except Exception as e:
                logger.warning(
                    "Failed to send to client %s: %s",
                    client_id,
                    e,
                )
                failed_clients.append(client_id)

        # Clean up failed connections
        for client_id in failed_clients:
            await self.disconnect(client_id)

        logger.debug(
            "Broadcast to channel %s: sent=%d, failed=%d",
            channel,
            sent_count,
            len(failed_clients),
        )
        return sent_count

    async def send_personal(self, client_id: str, message: dict) -> bool:
        """
        Send a message to a specific client.

        Args:
            client_id: The client to send to.
            message: The message to send (will be JSON serialized).

        Returns:
            True if message was sent, False otherwise.
        """
        websocket = self._connections.get(client_id)
        if not websocket:
            logger.warning("Cannot send to disconnected client: %s", client_id)
            return False

        # Add metadata to message
        envelope = {
            "channel": "_personal",
            "timestamp": datetime.now(UTC).isoformat(),
            "data": message,
        }

        try:
            await websocket.send_json(envelope)
            return True
        except WebSocketDisconnect:
            await self.disconnect(client_id)
            return False
        except Exception as e:
            logger.warning(
                "Failed to send personal message to client %s: %s",
                client_id,
                e,
            )
            await self.disconnect(client_id)
            return False

    def get_client_subscriptions(self, client_id: str) -> set[str]:
        """
        Get the channels a client is subscribed to.

        Args:
            client_id: The client to check.

        Returns:
            Set of subscribed channel names.
        """
        return self._client_subscriptions.get(client_id, set()).copy()

    def get_channel_subscribers(self, channel: str) -> set[str]:
        """
        Get the clients subscribed to a channel.

        Args:
            channel: The channel to check.

        Returns:
            Set of subscribed client IDs.
        """
        return self._channel_subscribers.get(channel, set()).copy()

    def is_connected(self, client_id: str) -> bool:
        """
        Check if a client is connected.

        Args:
            client_id: The client to check.

        Returns:
            True if connected, False otherwise.
        """
        return client_id in self._connections

    async def close_all(self) -> None:
        """Close all active connections (for shutdown)."""
        async with self._lock:
            client_ids = list(self._connections.keys())

        for client_id in client_ids:
            await self.disconnect(client_id)

        logger.info("All WebSocket connections closed")
