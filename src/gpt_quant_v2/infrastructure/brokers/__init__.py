"""Trading Broker Integrations.

This module provides broker interfaces and implementations for trading operations.
"""

from gpt_quant_v2.infrastructure.brokers.base import (
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
from gpt_quant_v2.infrastructure.brokers.factory import (
    BrokerFactoryError,
    create_broker,
)
from gpt_quant_v2.infrastructure.brokers.paper_broker import (
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
