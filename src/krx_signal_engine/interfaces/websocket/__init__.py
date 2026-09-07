"""
WebSocket Real-time Updates.

This module provides WebSocket functionality for real-time updates
including signal notifications, trade updates, and price streaming.

Components:
    - ConnectionManager: Manages WebSocket connections and subscriptions
    - WebSocketMessageHandler: Handles incoming WebSocket messages
    - EventBridge: Bridges Redis events to WebSocket clients
    - router: FastAPI router with /ws endpoint

Supported Channels:
    - signals:{ticker} - New signals for a specific ticker
    - signals:all - All new signals
    - trades:{ticker} - Trade updates for a ticker
    - trades:all - All trade updates
    - prices:{ticker} - Price updates for a ticker
    - prices:all - All price updates
"""

from krx_signal_engine.interfaces.websocket.events import EventBridge, create_event_bridge
from krx_signal_engine.interfaces.websocket.handlers import (
    WebSocketMessageHandler,
    handle_subscription,
    handle_unsubscription,
)
from krx_signal_engine.interfaces.websocket.manager import ConnectionManager
from krx_signal_engine.interfaces.websocket.router import (
    get_manager,
    router,
    set_manager,
)

__all__ = [
    "ConnectionManager",
    "EventBridge",
    "WebSocketMessageHandler",
    "create_event_bridge",
    "get_manager",
    "handle_subscription",
    "handle_unsubscription",
    "router",
    "set_manager",
]
