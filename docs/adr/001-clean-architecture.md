# ADR-001: Clean Architecture with Domain-Driven Design

## Status

Accepted

## Context

We are building a trading system that needs to be:
- Maintainable over time
- Testable at all levels
- Flexible to changes in external dependencies
- Easy for AI agents to understand and work with

The original gpt_quant codebase had tight coupling between business logic and infrastructure, making it difficult to test and modify.

## Decision

We will structure the application using Clean Architecture principles combined with Domain-Driven Design tactical patterns:

### Layer Structure

```
┌─────────────────────────────────────────────────────────────┐
│                     Interfaces Layer                         │
│              (API, CLI, MCP Tools, WebSocket)                │
├─────────────────────────────────────────────────────────────┤
│                    Application Layer                         │
│              (Commands, Queries, Handlers)                   │
├─────────────────────────────────────────────────────────────┤
│                      Domain Layer                            │
│         (Entities, Value Objects, Services, Events)          │
├─────────────────────────────────────────────────────────────┤
│                  Infrastructure Layer                        │
│        (Persistence, External APIs, Messaging, Brokers)      │
└─────────────────────────────────────────────────────────────┘
```

### Dependency Rule

Dependencies only point inward. The domain layer has no dependencies on other layers.

### Key Patterns

1. **Entities**: Objects with identity (Stock, Trade, Signal)
2. **Value Objects**: Immutable objects defined by attributes (Price, Ticker)
3. **Domain Events**: Record of something that happened
4. **Repositories**: Abstract data access (interface in domain, implementation in infrastructure)
5. **CQRS**: Separate read and write models

## Consequences

### Positive

- Clear separation of concerns
- Domain logic is isolated and testable
- Easy to swap infrastructure (databases, APIs)
- AI agents can understand the structure
- Parallel development across layers

### Negative

- More initial boilerplate
- Learning curve for developers unfamiliar with DDD
- May be over-engineered for very simple features

### Risks

- Team must maintain discipline to follow the architecture
- Need good documentation for onboarding

## Alternatives Considered

1. **Simple layered architecture**: Rejected - too much coupling
2. **Hexagonal architecture**: Similar, but Clean Architecture is better documented
3. **Microservices**: Rejected - premature complexity for this scale
