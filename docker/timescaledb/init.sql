-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- Create schemas
CREATE SCHEMA IF NOT EXISTS trading;
CREATE SCHEMA IF NOT EXISTS analytics;

-- News articles table
CREATE TABLE IF NOT EXISTS trading.news_articles (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    title TEXT NOT NULL,
    content TEXT,
    source VARCHAR(100),
    url TEXT,
    published_at TIMESTAMPTZ NOT NULL,
    fetched_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    sentiment_score FLOAT,
    relevance_score FLOAT,
    metadata JSONB DEFAULT '{}'::jsonb,
    UNIQUE (symbol, url, published_at)
);

CREATE INDEX idx_news_articles_symbol ON trading.news_articles (symbol);
CREATE INDEX idx_news_articles_published_at ON trading.news_articles (published_at DESC);
CREATE INDEX idx_news_articles_symbol_published ON trading.news_articles (symbol, published_at DESC);
CREATE INDEX idx_news_articles_source ON trading.news_articles (source);

-- Trading signals table
CREATE TABLE IF NOT EXISTS trading.trading_signals (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    signal_type VARCHAR(50) NOT NULL,
    direction VARCHAR(10) NOT NULL CHECK (direction IN ('long', 'short', 'neutral')),
    strength FLOAT NOT NULL CHECK (strength >= 0 AND strength <= 1),
    price_at_signal DECIMAL(20, 8) NOT NULL,
    target_price DECIMAL(20, 8),
    stop_loss DECIMAL(20, 8),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ,
    metadata JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX idx_trading_signals_symbol ON trading.trading_signals (symbol);
CREATE INDEX idx_trading_signals_created_at ON trading.trading_signals (created_at DESC);
CREATE INDEX idx_trading_signals_symbol_type ON trading.trading_signals (symbol, signal_type);
CREATE INDEX idx_trading_signals_active ON trading.trading_signals (symbol, created_at DESC)
    WHERE expires_at IS NULL OR expires_at > NOW();

-- Trades table
CREATE TABLE IF NOT EXISTS trading.trades (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL CHECK (side IN ('buy', 'sell')),
    quantity DECIMAL(20, 8) NOT NULL,
    price DECIMAL(20, 8) NOT NULL,
    commission DECIMAL(20, 8) DEFAULT 0,
    commission_asset VARCHAR(20),
    order_id VARCHAR(100),
    exchange VARCHAR(50),
    strategy VARCHAR(100),
    executed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX idx_trades_symbol ON trading.trades (symbol);
CREATE INDEX idx_trades_executed_at ON trading.trades (executed_at DESC);
CREATE INDEX idx_trades_symbol_executed ON trading.trades (symbol, executed_at DESC);
CREATE INDEX idx_trades_strategy ON trading.trades (strategy);
CREATE INDEX idx_trades_order_id ON trading.trades (order_id);

-- Positions table
CREATE TABLE IF NOT EXISTS trading.positions (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL UNIQUE,
    quantity DECIMAL(20, 8) NOT NULL DEFAULT 0,
    avg_entry_price DECIMAL(20, 8),
    unrealized_pnl DECIMAL(20, 8),
    realized_pnl DECIMAL(20, 8) DEFAULT 0,
    opened_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX idx_positions_symbol ON trading.positions (symbol);
CREATE INDEX idx_positions_updated_at ON trading.positions (updated_at DESC);

-- Price bars table (time-series data)
CREATE TABLE IF NOT EXISTS trading.price_bars (
    time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    open DECIMAL(20, 8) NOT NULL,
    high DECIMAL(20, 8) NOT NULL,
    low DECIMAL(20, 8) NOT NULL,
    close DECIMAL(20, 8) NOT NULL,
    volume DECIMAL(30, 8) NOT NULL DEFAULT 0,
    quote_volume DECIMAL(30, 8),
    trades_count INTEGER,
    metadata JSONB DEFAULT '{}'::jsonb,
    PRIMARY KEY (time, symbol, timeframe)
);

-- Convert price_bars to hypertable for time-series optimization
SELECT create_hypertable(
    'trading.price_bars',
    'time',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- Create indexes for price_bars
CREATE INDEX idx_price_bars_symbol_time ON trading.price_bars (symbol, time DESC);
CREATE INDEX idx_price_bars_symbol_timeframe ON trading.price_bars (symbol, timeframe, time DESC);

-- Analytics schema tables
CREATE TABLE IF NOT EXISTS analytics.performance_metrics (
    id BIGSERIAL PRIMARY KEY,
    strategy VARCHAR(100) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(20, 8) NOT NULL,
    period_start TIMESTAMPTZ NOT NULL,
    period_end TIMESTAMPTZ NOT NULL,
    calculated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX idx_performance_metrics_strategy ON analytics.performance_metrics (strategy);
CREATE INDEX idx_performance_metrics_period ON analytics.performance_metrics (period_start, period_end);

-- Grant permissions (for the default user)
GRANT ALL PRIVILEGES ON SCHEMA trading TO gpt_quant;
GRANT ALL PRIVILEGES ON SCHEMA analytics TO gpt_quant;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA trading TO gpt_quant;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA analytics TO gpt_quant;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA trading TO gpt_quant;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA analytics TO gpt_quant;
