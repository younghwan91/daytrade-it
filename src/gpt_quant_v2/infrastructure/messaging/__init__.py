"""
Messaging - Redis pub/sub and caching infrastructure.

This module provides Redis-based messaging capabilities for the trading system:
- Event publishing and subscribing via Redis pub/sub
- Signal queue for trading signal processing
- Caching with TTL and key prefixing
"""

from gpt_quant_v2.infrastructure.messaging.cache import (
    Cache,
    CacheError,
)
from gpt_quant_v2.infrastructure.messaging.event_publisher import (
    EventPublisher,
    EventPublishError,
)
from gpt_quant_v2.infrastructure.messaging.event_subscriber import (
    EventSubscriber,
    EventSubscriptionError,
)
from gpt_quant_v2.infrastructure.messaging.redis_client import (
    RedisClient,
    RedisConnectionError,
)
from gpt_quant_v2.infrastructure.messaging.signal_queue import (
    SignalQueue,
    SignalQueueError,
)

__all__ = [
    # Redis client
    "RedisClient",
    "RedisConnectionError",
    # Event publishing
    "EventPublisher",
    "EventPublishError",
    # Event subscribing
    "EventSubscriber",
    "EventSubscriptionError",
    # Signal queue
    "SignalQueue",
    "SignalQueueError",
    # Cache
    "Cache",
    "CacheError",
]
