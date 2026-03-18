# ADR-003: Event-Driven Architecture

## Status

Accepted

## Context

Trading systems have multiple components that need to react to events:
- News analysis triggers signal generation
- Signals trigger trade execution
- Trades update positions
- Risk breaches trigger alerts

Tight coupling between these components leads to:
- Difficult testing
- Hard to add new behaviors
- Complex error handling

## Decision

We will implement an event-driven architecture using:

1. **Domain Events**: Immutable records of what happened
2. **Redis Pub/Sub**: Event distribution mechanism
3. **Event Handlers**: React to events and perform actions

### Event Flow

```
[News Analyzed] → Redis → [Signal Generator Handler]
                              ↓
[Signal Generated] → Redis → [Trade Executor Handler]
                              ↓
[Trade Executed] → Redis → [Position Manager Handler]
                              ↓
[Position Updated] → Redis → [Risk Monitor Handler]
```

### Event Structure

```python
class DomainEvent:
    event_id: UUID
    event_type: str
    occurred_at: datetime
    correlation_id: Optional[UUID]
    causation_id: Optional[UUID]
    payload: dict
```

## Consequences

### Positive

- Loose coupling between components
- Easy to add new event handlers
- Natural audit log (events are immutable)
- Supports async processing
- Enables replay for debugging

### Negative

- Eventual consistency (not immediate)
- More complex debugging (follow event chain)
- Need idempotent handlers

### Mitigations

- Correlation IDs for tracing
- Event store for replay
- Clear naming conventions
