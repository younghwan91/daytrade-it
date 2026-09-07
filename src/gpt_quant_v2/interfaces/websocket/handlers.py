"""
WebSocket Message Handlers.

Handles incoming WebSocket messages including subscriptions,
unsubscriptions, and keepalive pings.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ValidationError

if TYPE_CHECKING:
    from gpt_quant_v2.interfaces.websocket.manager import ConnectionManager

logger = logging.getLogger(__name__)


class MessageType(StrEnum):
    """Supported WebSocket message types."""

    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    PING = "ping"
    PONG = "pong"
    AUTH = "auth"
    ERROR = "error"
    SUBSCRIBED = "subscribed"
    UNSUBSCRIBED = "unsubscribed"


class SubscribeMessage(BaseModel):
    """Message for subscribing to channels."""

    type: str = MessageType.SUBSCRIBE
    channels: list[str]


class UnsubscribeMessage(BaseModel):
    """Message for unsubscribing from channels."""

    type: str = MessageType.UNSUBSCRIBE
    channels: list[str]


class PingMessage(BaseModel):
    """Ping message for keepalive."""

    type: str = MessageType.PING
    timestamp: str | None = None


class AuthMessage(BaseModel):
    """Authentication message."""

    type: str = MessageType.AUTH
    token: str


class WebSocketMessageHandler:
    """
    Handles WebSocket message routing and processing.

    Supports subscription management, ping/pong keepalive,
    and authentication message handling.

    Usage:
        >>> handler = WebSocketMessageHandler(connection_manager)
        >>> response = await handler.handle_message(client_id, {"type": "subscribe", "channels": ["signals:005930"]})
    """

    # Valid channel patterns
    VALID_CHANNEL_PREFIXES = (
        "signals:",
        "trades:",
        "prices:",
    )
    VALID_WILDCARD_CHANNELS = (
        "signals:all",
        "trades:all",
        "prices:all",
    )

    def __init__(self, manager: ConnectionManager) -> None:
        """
        Initialize message handler.

        Args:
            manager: The connection manager instance.
        """
        self._manager = manager

    def _validate_channel(self, channel: str) -> bool:
        """
        Validate a channel name.

        Args:
            channel: The channel name to validate.

        Returns:
            True if valid, False otherwise.
        """
        # Check wildcard channels
        if channel in self.VALID_WILDCARD_CHANNELS:
            return True

        # Check prefix-based channels (e.g., signals:005930)
        for prefix in self.VALID_CHANNEL_PREFIXES:
            if channel.startswith(prefix) and len(channel) > len(prefix):
                return True

        return False

    async def handle_message(
        self,
        client_id: str,
        message: dict[str, Any],
    ) -> dict[str, Any] | None:
        """
        Route and handle an incoming WebSocket message.

        Args:
            client_id: The client sending the message.
            message: The parsed JSON message.

        Returns:
            Response message to send back, or None.
        """
        msg_type = message.get("type", "").lower()

        try:
            if msg_type == MessageType.SUBSCRIBE:
                return await self._handle_subscribe(client_id, message)
            elif msg_type == MessageType.UNSUBSCRIBE:
                return await self._handle_unsubscribe(client_id, message)
            elif msg_type == MessageType.PING:
                return await self._handle_ping(client_id, message)
            elif msg_type == MessageType.AUTH:
                return await self._handle_auth(client_id, message)
            else:
                logger.warning(
                    "Unknown message type from client %s: %s",
                    client_id,
                    msg_type,
                )
                return self._error_response(f"Unknown message type: {msg_type}")

        except ValidationError as e:
            logger.warning(
                "Invalid message from client %s: %s",
                client_id,
                e,
            )
            return self._error_response(f"Invalid message format: {e}")
        except Exception as e:
            logger.exception(
                "Error handling message from client %s",
                client_id,
            )
            return self._error_response(f"Internal error: {e!s}")

    async def _handle_subscribe(
        self,
        client_id: str,
        message: dict[str, Any],
    ) -> dict[str, Any]:
        """Handle subscription request."""
        msg = SubscribeMessage(**message)

        # Validate channels
        valid_channels = []
        invalid_channels = []
        for channel in msg.channels:
            if self._validate_channel(channel):
                valid_channels.append(channel)
            else:
                invalid_channels.append(channel)

        if invalid_channels:
            logger.warning(
                "Client %s requested invalid channels: %s",
                client_id,
                invalid_channels,
            )

        # Subscribe to valid channels
        subscribed = []
        if valid_channels:
            try:
                subscribed = await self._manager.subscribe(client_id, valid_channels)
            except ValueError as e:
                return self._error_response(str(e))

        return {
            "type": MessageType.SUBSCRIBED,
            "channels": subscribed,
            "invalid_channels": invalid_channels if invalid_channels else None,
            "timestamp": datetime.now(UTC).isoformat(),
        }

    async def _handle_unsubscribe(
        self,
        client_id: str,
        message: dict[str, Any],
    ) -> dict[str, Any]:
        """Handle unsubscription request."""
        msg = UnsubscribeMessage(**message)

        unsubscribed = await self._manager.unsubscribe(client_id, msg.channels)

        return {
            "type": MessageType.UNSUBSCRIBED,
            "channels": unsubscribed,
            "timestamp": datetime.now(UTC).isoformat(),
        }

    async def _handle_ping(
        self,
        client_id: str,  # noqa: ARG002
        message: dict[str, Any],
    ) -> dict[str, Any]:
        """Handle ping message for keepalive."""
        PingMessage(**message)  # Validate format

        return {
            "type": MessageType.PONG,
            "timestamp": datetime.now(UTC).isoformat(),
        }

    async def _handle_auth(
        self,
        client_id: str,
        message: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Handle authentication message.

        NOTE: This is a placeholder. Implement actual token validation.
        """
        msg = AuthMessage(**message)

        # TODO: Implement actual token validation
        # For now, accept any non-empty token
        if msg.token:
            logger.info("Client %s authenticated via message", client_id)
            return {
                "type": "auth_success",
                "client_id": client_id,
                "timestamp": datetime.now(UTC).isoformat(),
            }
        else:
            return self._error_response("Invalid authentication token")

    def _error_response(self, message: str) -> dict[str, Any]:
        """Create an error response message."""
        return {
            "type": MessageType.ERROR,
            "error": message,
            "timestamp": datetime.now(UTC).isoformat(),
        }


async def handle_subscription(
    manager: ConnectionManager,
    client_id: str,
    channels: list[str],
) -> dict[str, Any]:
    """
    Convenience function to handle subscription requests.

    Args:
        manager: The connection manager.
        client_id: The client requesting subscription.
        channels: List of channels to subscribe to.

    Returns:
        Response dict with subscription result.
    """
    handler = WebSocketMessageHandler(manager)
    return await handler.handle_message(
        client_id,
        {"type": "subscribe", "channels": channels},
    )


async def handle_unsubscription(
    manager: ConnectionManager,
    client_id: str,
    channels: list[str],
) -> dict[str, Any]:
    """
    Convenience function to handle unsubscription requests.

    Args:
        manager: The connection manager.
        client_id: The client requesting unsubscription.
        channels: List of channels to unsubscribe from.

    Returns:
        Response dict with unsubscription result.
    """
    handler = WebSocketMessageHandler(manager)
    return await handler.handle_message(
        client_id,
        {"type": "unsubscribe", "channels": channels},
    )
