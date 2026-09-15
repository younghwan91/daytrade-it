"""news_entry: the poller's pair filter and signal build, shared with replay."""

import datetime as dt
from decimal import Decimal
from uuid import uuid4

import time_machine

from daytrade_it.application.signals.news_entry import (
    build_entry_candidate,
    tradable_tickers,
)
from daytrade_it.domain.entities.news import NewsArticle
from daytrade_it.domain.services.signal_generator import SignalConfig, SignalGenerator
from daytrade_it.domain.value_objects import Ticker
from daytrade_it.infrastructure.brokers.base import KST, Quote
from daytrade_it.infrastructure.external.anthropic_client import ArticleFeatures
from daytrade_it.infrastructure.external.quant_airflow_db_client import StockInfo

DAY = dt.date(2026, 9, 15)
NOW = dt.datetime(2026, 9, 15, 9, 30, tzinfo=KST)
EXPIRES = dt.datetime(2026, 9, 15, 15, 20, tzinfo=KST)
STALE = dt.datetime(2026, 9, 14, 15, 20, tzinfo=KST)


def _article(tickers: list[str], published: dt.datetime = NOW) -> NewsArticle:
    return NewsArticle(
        id=uuid4(),
        title="t",
        content="c",
        source="toss",
        url="https://tossinvest.com/news",
        published_at=published,
        tickers=[Ticker(value=t) for t in tickers],
        category="stock",
    )


BULL = ArticleFeatures(
    subject="main",
    reports_price_move=False,
    material_type="수주",
    persistence="multi_quarter",
    specificity=3,
    freshness="new",
    surprise=2,
    sentiment_direction=1,
    sentiment_strength=3,
    scale_vs_size="large",
)


class _Broker:
    def __init__(self, last: str) -> None:
        self._last = Decimal(last)

    def get_quote(self, ticker: str) -> Quote:
        return Quote(ticker=ticker, bid=self._last, ask=self._last, last_price=self._last)


class _Bar:
    def __init__(self, close: float) -> None:
        self.close = close
        self.date = DAY - dt.timedelta(days=1)


class _DB:
    def get_adjusted_daily_bars(self, ticker, *, start=None, end=None):  # noqa: ANN001, ANN201
        return [_Bar(10000.0)]


def test_tradable_tickers_keeps_known_fresh_pairs_only() -> None:
    stocks = {"005930": StockInfo(code="005930", name="삼성전자")}
    art = _article(["005930", "000660"])
    assert [t.value for t, _ in tradable_tickers(art, stocks, STALE, EXPIRES, NOW)] == ["005930"]
    stale = _article(["005930"], published=STALE - dt.timedelta(minutes=1))
    assert tradable_tickers(stale, stocks, STALE, EXPIRES, NOW) == []
    assert tradable_tickers(art, stocks, STALE, EXPIRES, EXPIRES) == []


def test_build_entry_candidate_bullish_not_extended() -> None:
    gen = SignalGenerator(SignalConfig(min_confidence=0.0))
    with time_machine.travel(NOW, tick=False):
        c = build_entry_candidate(
            _article(["005930"]),
            Ticker(value="005930"),
            BULL,
            _Broker("10200"),
            _DB(),
            gen,
            EXPIRES,
        )
    assert c is not None
    assert (c.surprise, round(c.score, 3)) == (2, round(BULL.score(), 3))
    assert c.signal.metadata["pre_entry_move"] == "0.02"
    assert c.signal.expires_at == EXPIRES


def test_build_entry_candidate_refuses_extended_and_neutral() -> None:
    gen = SignalGenerator(SignalConfig(min_confidence=0.0))
    with time_machine.travel(NOW, tick=False):
        assert (
            build_entry_candidate(
                _article(["005930"]),
                Ticker(value="005930"),
                BULL,
                _Broker("10400"),
                _DB(),
                gen,
                EXPIRES,
            )
            is None
        )
        neutral = ArticleFeatures(**{**BULL.__dict__, "subject": "mention"})
        assert (
            build_entry_candidate(
                _article(["005930"]),
                Ticker(value="005930"),
                neutral,
                _Broker("10000"),
                _DB(),
                gen,
                EXPIRES,
            )
            is None
        )
