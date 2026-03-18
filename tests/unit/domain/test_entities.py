"""
Unit Tests for Domain Entities
"""

import pytest
from datetime import datetime, timezone
from decimal import Decimal
from uuid import uuid4

from gpt_quant_v2.domain.entities import (
    Stock,
    NewsArticle,
    TradingSignal,
    Trade,
    Position,
)
from gpt_quant_v2.domain.value_objects import Ticker, Price, Sentiment, Confidence


class TestStock:
    """Tests for Stock entity."""
    
    def test_create_stock(self):
        stock = Stock(
            ticker=Ticker(value="AAPL"),
            name="Apple Inc.",
            exchange="NASDAQ",
        )
        assert stock.ticker.value == "AAPL"
        assert stock.name == "Apple Inc."
        assert stock.id is not None
    
    def test_stock_equality_by_id(self):
        id1 = uuid4()
        stock1 = Stock(id=id1, ticker=Ticker(value="AAPL"), name="Apple", exchange="NASDAQ")
        stock2 = Stock(id=id1, ticker=Ticker(value="AAPL"), name="Apple", exchange="NASDAQ")
        assert stock1 == stock2
    
    def test_different_stocks_not_equal(self):
        stock1 = Stock(ticker=Ticker(value="AAPL"), name="Apple", exchange="NASDAQ")
        stock2 = Stock(ticker=Ticker(value="AAPL"), name="Apple", exchange="NASDAQ")
        assert stock1 != stock2  # Different IDs


class TestNewsArticle:
    """Tests for NewsArticle entity."""
    
    def test_create_news_article(self):
        article = NewsArticle(
            title="Apple Reports Record Revenue",
            content="Apple Inc. announced record quarterly revenue...",
            source="Reuters",
            url="https://example.com/article",
            published_at=datetime.now(timezone.utc),
            tickers=[Ticker(value="AAPL")],
        )
        assert article.title == "Apple Reports Record Revenue"
        assert len(article.tickers) == 1


class TestTradingSignal:
    """Tests for TradingSignal entity."""
    
    def test_create_buy_signal(self):
        signal = TradingSignal(
            ticker=Ticker(value="AAPL"),
            direction="BUY",
            confidence=Confidence(value=0.85),
            source="NEWS",
            predicted_return=5.0,
            predicted_drawdown=2.0,
            expires_at=datetime.now(timezone.utc),
        )
        assert signal.direction == "BUY"
        assert signal.risk_reward_ratio == 2.5
    
    def test_signal_is_expired(self):
        from datetime import timedelta
        past = datetime.now(timezone.utc) - timedelta(hours=1)
        signal = TradingSignal(
            ticker=Ticker(value="AAPL"),
            direction="BUY",
            confidence=Confidence(value=0.8),
            source="NEWS",
            predicted_return=5.0,
            predicted_drawdown=2.0,
            expires_at=past,
        )
        assert signal.is_expired


class TestTrade:
    """Tests for Trade entity."""
    
    def test_create_trade(self):
        trade = Trade(
            signal_id=uuid4(),
            ticker=Ticker(value="AAPL"),
            direction="BUY",
            quantity=100,
            entry_price=Price(value=Decimal("185.50")),
        )
        assert trade.status == "PENDING"
        assert trade.quantity == 100
    
    def test_trade_fill(self):
        trade = Trade(
            signal_id=uuid4(),
            ticker=Ticker(value="AAPL"),
            direction="BUY",
            quantity=100,
            entry_price=Price(value=Decimal("185.50")),
        )
        trade = trade.fill(Price(value=Decimal("185.55")))
        assert trade.status == "FILLED"


class TestPosition:
    """Tests for Position entity."""
    
    def test_create_position(self):
        position = Position(
            ticker=Ticker(value="AAPL"),
            quantity=100,
            average_price=Price(value=Decimal("185.50")),
            current_price=Price(value=Decimal("190.00")),
        )
        assert position.quantity == 100
        assert position.unrealized_pnl > 0
    
    def test_position_pnl_calculation(self):
        position = Position(
            ticker=Ticker(value="AAPL"),
            quantity=100,
            average_price=Price(value=Decimal("100.00")),
            current_price=Price(value=Decimal("110.00")),
        )
        # 100 shares * $10 gain = $1000
        assert position.unrealized_pnl == Decimal("1000.00")
