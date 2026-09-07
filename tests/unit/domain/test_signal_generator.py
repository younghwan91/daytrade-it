"""Unit tests for SignalGenerator, focused on Task 11's sentiment-only discipline.

Verifies that a NEWS-sourced signal (from ``generate_from_news``) can no
longer become an actionable/standalone signal via ``combine_signals`` --
only combined with an ML/technical/fundamental signal does it produce a
composite result. This is the concrete code consequence of deep-research
finding #1 (sentiment alone doesn't reliably beat a benchmark).
"""

from datetime import UTC, datetime

import pytest
from pydantic import HttpUrl

from krx_signal_engine.domain.entities import (
    NewsArticle,
    SignalDirection,
    SignalSource,
    TradingSignal,
)
from krx_signal_engine.domain.services.signal_generator import SignalConfig, SignalGenerator
from krx_signal_engine.domain.value_objects import Confidence, Sentiment, Ticker

TICKER = Ticker(value="005930")


def _article(**overrides: object) -> NewsArticle:
    defaults: dict = {
        "title": "Samsung Electronics announces record earnings",
        "content": "Full article body about record earnings.",
        "source": "toss",
        "url": HttpUrl("https://news.toss.im/article/1"),
        "published_at": datetime.now(UTC),
        "tickers": [TICKER],
    }
    defaults.update(overrides)
    return NewsArticle(**defaults)


def _ml_signal(direction: SignalDirection = SignalDirection.BUY, confidence: float = 0.8) -> TradingSignal:
    return TradingSignal(
        ticker=TICKER,
        direction=direction,
        confidence=Confidence(value=confidence),
        source=SignalSource.ML,
    )


def _technical_signal(
    direction: SignalDirection = SignalDirection.BUY, confidence: float = 0.7
) -> TradingSignal:
    return TradingSignal(
        ticker=TICKER,
        direction=direction,
        confidence=Confidence(value=confidence),
        source=SignalSource.TECHNICAL,
    )


class TestGenerateFromNewsIsComponentOnly:
    """generate_from_news still exists but is documented/tagged as non-standalone."""

    def test_produces_news_sourced_signal(self) -> None:
        generator = SignalGenerator()
        sentiment = Sentiment.bullish(score=0.9, confidence=0.9)
        signal = generator.generate_from_news(_article(), sentiment)

        assert signal is not None
        assert signal.source == SignalSource.NEWS

    def test_marks_signal_as_not_standalone_actionable(self) -> None:
        generator = SignalGenerator()
        sentiment = Sentiment.bullish(score=0.9, confidence=0.9)
        signal = generator.generate_from_news(_article(), sentiment)

        assert signal is not None
        assert signal.metadata["standalone_actionable"] is False


class TestCombineSignalsRefusesSentimentOnly:
    """combine_signals refuses to produce an actionable signal from sentiment alone."""

    def test_single_news_signal_no_longer_passes_through(self) -> None:
        generator = SignalGenerator()
        sentiment = Sentiment.bullish(score=0.9, confidence=0.9)
        news_signal = generator.generate_from_news(_article(), sentiment)
        assert news_signal is not None

        combined = generator.combine_signals([news_signal])

        assert combined is None

    def test_multiple_news_only_signals_still_refused(self) -> None:
        generator = SignalGenerator()
        sentiment = Sentiment.bullish(score=0.9, confidence=0.9)
        news_signal_1 = generator.generate_from_news(_article(), sentiment)
        news_signal_2 = generator.generate_from_news(_article(title="Another bullish article"), sentiment)
        assert news_signal_1 is not None
        assert news_signal_2 is not None

        combined = generator.combine_signals([news_signal_1, news_signal_2])

        assert combined is None

    def test_news_plus_ml_signal_produces_composite(self) -> None:
        generator = SignalGenerator()
        sentiment = Sentiment.bullish(score=0.9, confidence=0.9)
        news_signal = generator.generate_from_news(_article(), sentiment)
        assert news_signal is not None
        ml_signal = _ml_signal()

        combined = generator.combine_signals([news_signal, ml_signal])

        assert combined is not None
        assert combined.source == SignalSource.COMPOSITE
        assert combined.direction == SignalDirection.BUY

    def test_news_plus_technical_signal_produces_composite(self) -> None:
        generator = SignalGenerator()
        sentiment = Sentiment.bearish(score=-0.9, confidence=0.9)
        news_signal = generator.generate_from_news(_article(), sentiment)
        assert news_signal is not None
        technical_signal = _technical_signal(direction=SignalDirection.SELL)

        combined = generator.combine_signals([news_signal, technical_signal])

        assert combined is not None
        assert combined.direction == SignalDirection.SELL

    def test_lone_ml_signal_still_passes_through_unchanged(self) -> None:
        # Non-NEWS single-source lists are unaffected by this discipline --
        # only sentiment-only combinations are refused.
        generator = SignalGenerator()
        ml_signal = _ml_signal()

        combined = generator.combine_signals([ml_signal])

        assert combined is ml_signal

    def test_empty_list_returns_none(self) -> None:
        generator = SignalGenerator()
        assert generator.combine_signals([]) is None


class TestSignalConfigWeightsStillExist:
    """Step 1: confirm the sentiment/ml/technical weight blend still exists post Phases 1-3."""

    def test_default_weights(self) -> None:
        config = SignalConfig()
        assert config.sentiment_weight == pytest.approx(0.4)
        assert config.ml_weight == pytest.approx(0.4)
        assert config.technical_weight == pytest.approx(0.2)
