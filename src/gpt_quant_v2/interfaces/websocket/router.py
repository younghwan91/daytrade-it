"""
WebSocket Router - Main WebSocket endpoint.

Provides the /ws endpoint for real-time communication with clients.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from gpt_quant_v2.interfaces.websocket.handlers import WebSocketMessageHandler
from gpt_quant_v2.interfaces.websocket.manager import ConnectionManager

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

router = APIRouter(tags=["websocket"])

# Global connection manager instance
# This is initialized by the application lifecycle
_manager: ConnectionManager | None = None
_handler: WebSocketMessageHandler | None = None


def get_manager() -> ConnectionManager:
    """Get the global connection manager."""
    global _manager
    if _manager is None:
        _manager = ConnectionManager()
    return _manager


def get_handler() -> WebSocketMessageHandler:
    """Get the message handler."""
    global _handler
    if _handler is None:
        _handler = WebSocketMessageHandler(get_manager())
    return _handler


def set_manager(manager: ConnectionManager) -> None:
    """Set the global connection manager (for app initialization)."""
    global _manager, _handler
    _manager = manager
    _handler = WebSocketMessageHandler(manager)


async def authenticate_client(
    websocket: WebSocket,
    token: str | None,
) -> str | None:
    """
    Authenticate a WebSocket client.

    Args:
        websocket: The WebSocket connection.
        token: Authentication token from query param.

    Returns:
        Client ID if authenticated, None otherwise.

    NOTE: This is a placeholder. Implement actual authentication.
    """
    # TODO: Implement actual JWT/OAuth token validation
    # For now, generate a client ID

    if token:
        # Validate token and extract user ID
        # For now, use token as a simple identifier
        logger.debug("Client authenticated with token")
        return f"user_{token[:8]}" if len(token) >= 8 else f"user_{token}"

    # Allow anonymous connections with generated ID
    return f"anon_{uuid.uuid4().hex[:12]}"


async def wait_for_auth_message(
    websocket: WebSocket,
    timeout: float = 10.0,
) -> str | None:
    """
    Wait for an authentication message from the client.

    Args:
        websocket: The WebSocket connection.
        timeout: Timeout in seconds.

    Returns:
        Client ID if authenticated, None otherwise.
    """
    try:
        message = await asyncio.wait_for(
            websocket.receive_json(),
            timeout=timeout,
        )

        if message.get("type") == "auth" and message.get("token"):
            token = message["token"]
            # TODO: Validate token
            logger.debug("Client authenticated via message")
            return f"user_{token[:8]}" if len(token) >= 8 else f"user_{token}"

    except asyncio.TimeoutError:
        logger.debug("Auth message timeout")
    except Exception as e:
        logger.warning("Error receiving auth message: %s", e)

    return None


@router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    token: str | None = Query(default=None, description="Authentication token"),
    client_id: str | None = Query(default=None, description="Optional client ID"),
) -> None:
    """
    Main WebSocket endpoint for real-time updates.

    Authentication:
        - Via query parameter: /ws?token=<jwt_token>
        - Via first message: {"type": "auth", "token": "<jwt_token>"}

    Message Types (client -> server):
        - subscribe: {"type": "subscribe", "channels": ["signals:AAPL", "trades:all"]}
        - unsubscribe: {"type": "unsubscribe", "channels": ["signals:AAPL"]}
        - ping: {"type": "ping"}

    Message Types (server -> client):
        - subscribed: {"type": "subscribed", "channels": [...]}
        - unsubscribed: {"type": "unsubscribed", "channels": [...]}
        - pong: {"type": "pong", "timestamp": "..."}
        - error: {"type": "error", "error": "..."}
        - data: {"channel": "...", "timestamp": "...", "data": {...}}

    Supported Channels:
        - signals:{ticker} - New signals for a specific ticker
        - signals:all - All new signals
        - trades:{ticker} - Trade updates for a ticker
        - trades:all - All trade updates
        - prices:{ticker} - Price updates for a ticker
        - prices:all - All price updates
    """
    manager = get_manager()
    handler = get_handler()

    # Authenticate client
    if token:
        # Auth via query param
        resolved_client_id = await authenticate_client(websocket, token)
    elif client_id:
        # Use provided client ID (for reconnection)
        resolved_client_id = client_id
    else:
        # Generate anonymous client ID
        resolved_client_id = f"anon_{uuid.uuid4().hex[:12]}"

    if not resolved_client_id:
        await websocket.close(code=4001, reason="Authentication failed")
        return

    try:
        # Connect client
        await manager.connect(websocket, resolved_client_id)

        # Send welcome message
        await websocket.send_json({
            "type": "connected",
            "client_id": resolved_client_id,
            "message": "Connected to GPT Quant WebSocket",
        })

        # If no token was provided via query, wait for auth message
        if not token and not client_id:
            auth_client_id = await wait_for_auth_message(websocket, timeout=10.0)
            if auth_client_id:
                # Re-register with authenticated ID
                await manager.disconnect(resolved_client_id)
                resolved_client_id = auth_client_id
                await manager.connect(websocket, resolved_client_id)
                await websocket.send_json({
                    "type": "auth_success",
                    "client_id": resolved_client_id,
                })

        # Main message loop
        while True:
            try:
                message = await websocket.receive_json()

                # Handle message and get response
                response = await handler.handle_message(resolved_client_id, message)

                if response:
                    await websocket.send_json(response)

            except ValueError as e:
                # Invalid JSON
                await websocket.send_json({
                    "type": "error",
                    "error": f"Invalid message format: {e}",
                })

    except WebSocketDisconnect:
        logger.info("Client %s disconnected", resolved_client_id)
    except Exception as e:
        logger.exception("WebSocket error for client %s", resolved_client_id)
    finally:
        await manager.disconnect(resolved_client_id)


@router.get("/ws/stats")
async def websocket_stats() -> dict[str, Any]:
    """
    Get WebSocket connection statistics.

    Returns:
        Dict with connection stats.
    """
    manager = get_manager()

    return {
        "active_connections": manager.active_connections,
        "active_channels": manager.channels,
        "channel_count": len(manager.channels),
    }
