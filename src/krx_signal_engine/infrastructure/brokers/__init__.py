"""Trading Broker Integrations.

This module provides broker interfaces and implementations for trading operations.
"""

from krx_signal_engine.infrastructure.brokers.base import (
    Account,
    BrokerInterface,
    Order,
    OrderResult,
    OrderSide,
    OrderStatus,
    OrderStatusEnum,
    OrderType,
    Position,
    Quote,
)
from krx_signal_engine.infrastructure.brokers.factory import (
    BrokerFactoryError,
    create_broker,
)
from krx_signal_engine.infrastructure.brokers.paper_broker import (
    PaperBroker,
    PaperBrokerConfig,
)

__all__ = [
    # Base interface and data classes
    "Account",
    "BrokerInterface",
    "Order",
    "OrderResult",
    "OrderSide",
    "OrderStatus",
    "OrderStatusEnum",
    "OrderType",
    "Position",
    "Quote",
    # Factory
    "BrokerFactoryError",
    "create_broker",
    # Implementations
    "PaperBroker",
    "PaperBrokerConfig",
]
