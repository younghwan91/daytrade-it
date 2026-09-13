"""
Unit Tests for Domain Entities
"""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from uuid import uuid4

import pytest

from daytrade_it.domain.entities import (
    Exchange,
    NewsArticle,
    Position,
    Stock,
    Trade,
    TradingSignal,
)
from daytrade_it.domain.value_objects import Confidence, Ticker


class TestExchange:
    """Tests for the Exchange enum (KRX markets)."""

    def test_krx_markets_present(self):
        assert Exchange.KOSPI == "KOSPI"
        assert Exchange.KOSDAQ == "KOSDAQ"
        assert Exchange.KONEX == "KONEX"

    def test_old_us_markets_removed(self):
        assert not hasattr(Exchange, "NASDAQ")
        assert not hasattr(Exchange, "NYSE")


class TestStock:
    """Tests for Stock entity."""

    def test_create_stock(self):
        stock = Stock(
            ticker=Ticker(value="005930"),
            name="삼성전자",
            exchange="KOSPI",
        )
        assert stock.ticker.value == "005930"
        assert stock.name == "삼성전자"
        assert stock.id is not None

    def test_stock_equality_by_id(self):
        id1 = uuid4()
        stock1 = Stock(id=id1, ticker=Ticker(value="005930"), name="삼성전자", exchange="KOSPI")
        stock2 = Stock(id=id1, ticker=Ticker(value="005930"), name="삼성전자", exchange="KOSPI")
        assert stock1 == stock2

    def test_different_stocks_not_equal(self):
        stock1 = Stock(ticker=Ticker(value="005930"), name="삼성전자", exchange="KOSPI")
        stock2 = Stock(ticker=Ticker(value="005930"), name="삼성전자", exchange="KOSPI")
        assert stock1 != stock2  # Different IDs

    def test_market_cap_buckets_not_implemented(self):
        """USD market-cap thresholds are disabled, not silently wrong.

        Real KRW thresholds are out of scope for this task (see plan
        Phase 2); until they exist these properties must fail loudly
        rather than misclassify a KRW market cap.
        """
        stock = Stock(
            ticker=Ticker(value="005930"),
            name="삼성전자",
            exchange="KOSPI",
            market_cap=Decimal("500_000_000_000_000"),
        )
        with pytest.raises(NotImplementedError):
            _ = stock.is_large_cap
        with pytest.raises(NotImplementedError):
            _ = stock.is_mid_cap
        with pytest.raises(NotImplementedError):
            _ = stock.is_small_cap


class TestNewsArticle:
    """Tests for NewsArticle entity."""

    def test_create_news_article(self):
        article = NewsArticle(
            title="Apple Reports Record Revenue",
            content="Apple Inc. announced record quarterly revenue...",
            source="Reuters",
            url="https://example.com/article",
            published_at=datetime.now(UTC),
            tickers=[Ticker(value="005930")],
        )
        assert article.title == "Apple Reports Record Revenue"
        assert len(article.tickers) == 1


class TestTradingSignal:
    """Tests for TradingSignal entity."""

    def test_create_buy_signal(self):
        signal = TradingSignal(
            ticker=Ticker(value="005930"),
            direction="BUY",
            confidence=Confidence(value=0.85),
            source="NEWS",
            predicted_return=Decimal("5.0"),
            predicted_drawdown=Decimal("-2.0"),
            risk_reward_ratio=Decimal("2.5"),
            expires_at=datetime.now(UTC) + timedelta(hours=1),
        )
        assert signal.direction == "BUY"
        assert signal.risk_reward_ratio == Decimal("2.5")

    def test_signal_is_expired(self):
        signal = TradingSignal(
            ticker=Ticker(value="005930"),
            direction="BUY",
            confidence=Confidence(value=0.8),
            source="NEWS",
            predicted_return=Decimal("5.0"),
            predicted_drawdown=Decimal("-2.0"),
            expires_at=datetime.now(UTC) + timedelta(hours=1),
        )
        assert not signal.is_expired

        # 생성 시점엔 만료가 미래여야 하므로(validate_expiration) 만료된 신호를
        # 곧바로 만들 수도, 할당으로 되돌릴 수도 없다. 시간이 흐른 뒤의 같은
        # 신호를 model_copy 로 재현한다.
        expired = signal.model_copy(update={"expires_at": datetime.now(UTC) - timedelta(hours=1)})
        assert expired.is_expired


class TestTrade:
    """Tests for Trade entity."""

    def test_create_trade(self):
        trade = Trade(
            signal_id=uuid4(),
            ticker=Ticker(value="005930"),
            direction="BUY",
            quantity=100,
            entry_price=Decimal("185.50"),
        )
        assert trade.status == "PENDING"
        assert trade.quantity == 100

    def test_trade_fill(self):
        trade = Trade(
            signal_id=uuid4(),
            ticker=Ticker(value="005930"),
            direction="BUY",
            quantity=100,
            entry_price=Decimal("185.50"),
        )
        trade.fill(Decimal("185.55"))
        assert trade.status == "FILLED"


class TestPosition:
    """Tests for Position entity."""

    def test_create_position(self):
        position = Position(
            ticker=Ticker(value="005930"),
            quantity=Decimal("100"),
            average_price=Decimal("185.50"),
            current_price=Decimal("190.00"),
            opened_at=datetime.now(UTC),
        )
        assert position.quantity == 100
        assert position.unrealized_pnl > 0

    def test_position_pnl_calculation(self):
        position = Position(
            ticker=Ticker(value="005930"),
            quantity=Decimal("100"),
            average_price=Decimal("100.00"),
            current_price=Decimal("110.00"),
            opened_at=datetime.now(UTC),
        )
        # 100 shares * $10 gain = $1000
        assert position.unrealized_pnl == Decimal("1000.00")
