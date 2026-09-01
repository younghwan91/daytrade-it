# ADR-005: Abstract Broker Interface

## Status

Accepted

## Context

The original system was tightly coupled to the KIS (Korean Investment Securities) broker API. We need to:
- Support US markets (different brokers)
- Enable paper trading for testing
- Allow switching brokers without code changes

## Decision

We will define an abstract broker interface that all broker implementations must follow:

### Interface Definition

```python
class BrokerInterface(ABC):
    """Abstract interface for trading brokers."""

    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to broker."""
        ...

    @abstractmethod
    async def get_account(self) -> Account:
        """Get account information."""
        ...

    @abstractmethod
    async def get_positions(self) -> list[Position]:
        """Get current positions."""
        ...

    @abstractmethod
    async def place_order(self, order: Order) -> OrderResult:
        """Place a new order."""
        ...

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing order."""
        ...

    @abstractmethod
    async def get_order_status(self, order_id: str) -> OrderStatus:
        """Get status of an order."""
        ...
```

### Implementations

1. **PaperBroker**: Simulated trading for testing
2. **AlpacaBroker**: Alpaca Markets API (future)
3. **IBKRBroker**: Interactive Brokers API (future)

### Configuration

```python
# settings.py
broker_type: Literal["paper", "alpaca", "ibkr"] = "paper"
```

## Consequences

### Positive

- Easy to test with paper trading
- Can switch brokers via configuration
- Clear contract for broker implementations
- Enables multi-broker support

### Negative

- Additional abstraction layer
- May not expose all broker-specific features
- Need to handle broker-specific edge cases

### Mitigations

- Allow broker-specific extensions via metadata
- Document limitations of each implementation
