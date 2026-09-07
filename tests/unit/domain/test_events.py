"""
Unit Tests for Domain Events
"""

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from krx_signal_engine.domain.events import (
    DomainEvent,
    NewsAnalyzed,
    PositionUpdated,
    SignalGenerated,
    TradeExecuted,
)


class TestDomainEvent:
    """Tests for base DomainEvent."""

    def test_event_has_id(self):
        event = DomainEvent()
        assert event.event_id is not None

    def test_event_has_timestamp(self):
        event = DomainEvent()
        assert event.occurred_at is not None

    def test_event_type_set_to_class_name(self):
        event = DomainEvent()
        assert event.event_type == "DomainEvent"

    def test_event_is_immutable(self):
        event = DomainEvent()
        with pytest.raises(ValidationError):
            event.event_type = "Modified"

    def test_event_to_dict(self):
        event = DomainEvent()
        data = event.to_dict()
        assert "event_id" in data
        assert "event_type" in data
        assert "occurred_at" in data

    def test_event_with_correlation_id(self):
        correlation_id = uuid4()
        event = DomainEvent()
        new_event = event.with_correlation(correlation_id)
        assert new_event.correlation_id == correlation_id


class TestNewsAnalyzed:
    """Tests for NewsAnalyzed event."""

    def test_news_analyzed_event(self):
        event = NewsAnalyzed(
            article_id=uuid4(),
            ticker="AAPL",
            sentiment_score=0.75,
            sentiment_label="BULLISH",
            confidence=0.85,
            keywords=["revenue", "growth", "record"],
        )
        assert event.event_type == "NewsAnalyzed"
        assert event.sentiment_label == "BULLISH"
        assert len(event.keywords) == 3


class TestSignalGenerated:
    """Tests for SignalGenerated event."""

    def test_signal_generated_event(self):
        event = SignalGenerated(
            signal_id=uuid4(),
            ticker="AAPL",
            direction="BUY",
            confidence=0.85,
            source="NEWS",
            predicted_return=5.0,
            predicted_drawdown=2.0,
            risk_reward_ratio=2.5,
            expires_at=datetime.now(UTC),
        )
        assert event.event_type == "SignalGenerated"
        assert event.direction == "BUY"


class TestTradeExecuted:
    """Tests for TradeExecuted event."""

    def test_trade_executed_event(self):
        from decimal import Decimal

        event = TradeExecuted(
            trade_id=uuid4(),
            signal_id=uuid4(),
            ticker="AAPL",
            direction="BUY",
            quantity=100,
            price=Decimal("185.50"),
            total_value=Decimal("18550.00"),
        )
        assert event.event_type == "TradeExecuted"
        assert event.quantity == 100


class TestPositionUpdated:
    """Tests for PositionUpdated event."""

    def test_position_updated_event(self):
        from decimal import Decimal

        event = PositionUpdated(
            position_id=uuid4(),
            ticker="AAPL",
            quantity=100,
            average_price=Decimal("185.50"),
            current_price=Decimal("190.00"),
            unrealized_pnl=Decimal("450.00"),
            unrealized_pnl_percent=2.43,
        )
        assert event.event_type == "PositionUpdated"
        assert event.unrealized_pnl == Decimal("450.00")
