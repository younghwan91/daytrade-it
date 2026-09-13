"""Trading Broker Integrations.

This module provides broker interfaces and implementations for trading operations.
"""

from daytrade_it.infrastructure.brokers.base import (
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
from daytrade_it.infrastructure.brokers.factory import (
    BrokerFactoryError,
    create_broker,
)
from daytrade_it.infrastructure.brokers.paper_broker import (
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
