"""
Unit Tests for Domain Value Objects
"""

from decimal import Decimal

import pytest

from krx_signal_engine.domain.value_objects import (
    Confidence,
    Percentage,
    Price,
    Sentiment,
    Ticker,
    Volume,
)


class TestTicker:
    """Tests for Ticker value object.

    Tickers are 6-digit numeric KRX (Korea Exchange) stock codes,
    e.g. "005930" (Samsung Electronics / 삼성전자).
    """

    def test_valid_ticker_samsung(self):
        ticker = Ticker(value="005930")
        assert ticker.value == "005930"
        assert str(ticker) == "005930"

    def test_valid_ticker_sk_hynix(self):
        ticker = Ticker(value="000660")
        assert ticker.value == "000660"

    def test_valid_ticker_kakao(self):
        ticker = Ticker(value="035720")
        assert ticker.value == "035720"

    def test_valid_ticker_preferred_share(self):
        """Preferred-share codes are still 6 digits (e.g. 005935 삼성전자우)."""
        ticker = Ticker(value="005935")
        assert ticker.value == "005935"

    def test_ticker_equality(self):
        t1 = Ticker(value="005930")
        t2 = Ticker(value="005930")
        assert t1 == t2

    def test_ticker_hash(self):
        t1 = Ticker(value="005930")
        t2 = Ticker(value="005930")
        assert hash(t1) == hash(t2)
        assert len({t1, t2}) == 1

    def test_invalid_ticker_old_us_format(self):
        """Old US-style alphabetic tickers must now be rejected."""
        with pytest.raises(ValueError):
            Ticker(value="AAPL")

    def test_invalid_ticker_too_short(self):
        with pytest.raises(ValueError):
            Ticker(value="12345")

    def test_invalid_ticker_too_long(self):
        with pytest.raises(ValueError):
            Ticker(value="1234567")

    def test_invalid_ticker_empty(self):
        with pytest.raises(ValueError):
            Ticker(value="")


class TestPrice:
    """Tests for Price value object."""

    def test_valid_price(self):
        price = Price(value=Decimal("185.50"))
        assert price.value == Decimal("185.50")

    def test_price_equality(self):
        p1 = Price(value=Decimal("100.00"))
        p2 = Price(value=Decimal("100.00"))
        assert p1 == p2

    def test_negative_price_invalid(self):
        with pytest.raises(ValueError):
            Price(value=Decimal("-10.00"))


class TestVolume:
    """Tests for Volume value object."""

    def test_valid_volume(self):
        volume = Volume(value=1000000)
        assert volume.value == 1000000

    def test_volume_addition(self):
        v1 = Volume(value=1000)
        v2 = Volume(value=500)
        result = v1 + v2
        assert result.value == 1500

    def test_negative_volume_invalid(self):
        with pytest.raises(ValueError):
            Volume(value=-100)


class TestSentiment:
    """Tests for Sentiment value object."""

    def test_bullish_sentiment(self):
        sentiment = Sentiment.bullish(confidence=0.9)
        assert sentiment.score > 0
        assert sentiment.label.value == "BULLISH"

    def test_bearish_sentiment(self):
        sentiment = Sentiment.bearish(confidence=0.8)
        assert sentiment.score < 0
        assert sentiment.label.value == "BEARISH"

    def test_neutral_sentiment(self):
        sentiment = Sentiment.neutral()
        assert sentiment.score == 0.0
        assert sentiment.label.value == "NEUTRAL"

    def test_sentiment_score_bounds(self):
        with pytest.raises(ValueError):
            Sentiment(score=1.5, label="BULLISH", confidence=0.5)


class TestConfidence:
    """Tests for Confidence value object."""

    def test_high_confidence(self):
        conf = Confidence(value=0.85)
        assert conf.is_high
        assert not conf.is_low

    def test_low_confidence(self):
        conf = Confidence(value=0.2)
        assert conf.is_low
        assert not conf.is_high

    def test_confidence_bounds(self):
        with pytest.raises(ValueError):
            Confidence(value=1.5)


class TestPercentage:
    """Tests for Percentage value object."""

    def test_percentage_as_decimal(self):
        pct = Percentage(value=15.5)
        assert pct.as_decimal == 0.155

    def test_percentage_as_basis_points(self):
        pct = Percentage(value=1.5)
        assert pct.as_basis_points == 150

    def test_percentage_of(self):
        pct = Percentage(value=10.0)
        result = pct.of(Decimal("1000"))
        assert result == Decimal("100")
