# ADR-002: Use TimescaleDB for Time-Series Data

## Status

Accepted

## Context

The original system used two databases:
- PostgreSQL for news and trades
- QuestDB for time-series price data

Managing two databases increases operational complexity and requires different query patterns.

## Decision

We will use **TimescaleDB** as our single database solution:

- TimescaleDB is a PostgreSQL extension optimized for time-series data
- We get the full power of PostgreSQL (ACID, JSON, full SQL)
- Plus time-series optimizations (hypertables, continuous aggregates, compression)

### Schema Design

```sql
-- Regular tables for transactional data
CREATE TABLE news_articles (...);
CREATE TABLE trading_signals (...);
CREATE TABLE trades (...);
CREATE TABLE positions (...);

-- Hypertable for time-series data
CREATE TABLE price_bars (
    ticker TEXT,
    timestamp TIMESTAMPTZ,
    open NUMERIC,
    high NUMERIC,
    low NUMERIC,
    close NUMERIC,
    volume BIGINT,
    PRIMARY KEY (ticker, timestamp)
);

SELECT create_hypertable('price_bars', 'timestamp');
```

## Consequences

### Positive

- Single database to manage
- Unified query language (SQL)
- Built-in time-series optimizations
- Automatic partitioning and compression
- Strong PostgreSQL ecosystem (SQLAlchemy, asyncpg)

### Negative

- Slightly more complex than pure PostgreSQL
- Need to understand hypertables and chunks
- Less specialized than QuestDB for pure time-series workloads

## Alternatives Considered

1. **Keep QuestDB + PostgreSQL**: Rejected - operational complexity
2. **InfluxDB**: Rejected - different query language, less SQL support
3. **Pure PostgreSQL with partitioning**: Possible but more manual work
