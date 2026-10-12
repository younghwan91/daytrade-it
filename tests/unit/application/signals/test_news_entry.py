"""news_entry: the poller's candidate build, shared with replay."""

import datetime as dt
from decimal import Decimal
from uuid import uuid4

import time_machine

from daytrade_it.application.signals.news_entry import build_entry_candidate, tradable_tickers
from daytrade_it.domain.entities.news import NewsArticle
from daytrade_it.domain.entities.signal import SignalDirection, SignalSource
from daytrade_it.domain.services.signal_generator import SignalConfig, SignalGenerator
from daytrade_it.domain.value_objects import Ticker
from daytrade_it.infrastructure.brokers.base import KST, Quote
from daytrade_it.infrastructure.external.anthropic_client import ArticleFeatures
from daytrade_it.infrastructure.external.quant_airflow_db_client import (
    RegularSessionClose,
    StockInfo,
)

DAY = dt.date(2026, 9, 15)
EXPIRES = dt.datetime(2026, 9, 15, 15, 20, tzinfo=KST)
STALE = dt.datetime(2026, 9, 14, 15, 20, tzinfo=KST)


def _features(**kw: object) -> ArticleFeatures:
    base = {
        "subject": "main",
        "reports_price_move": False,
        "material_type": "수주",
        "persistence": "multi_quarter",
        "specificity": 3,
        "freshness": "new",
        "surprise": 2,
        "sentiment_direction": 1,
        "sentiment_strength": 3,
        "scale_vs_size": "large",
    }
    return ArticleFeatures(**{**base, **kw})  # type: ignore[arg-type]


def _article(published: dt.datetime, tickers: list[str]) -> NewsArticle:
    return NewsArticle(
        id=uuid4(),
        title="삼성전자, 5조원 규모 공급계약 체결",
        content="본문",
        source="toss",
        url="https://example.com/a",
        published_at=published,
        tickers=[Ticker(value=t) for t in tickers],
        category="stock",
    )


class _Broker:
    def __init__(self, price: int) -> None:
        self._p = Decimal(price)

    def get_quote(self, ticker: str) -> Quote:
        return Quote(ticker=ticker, bid=self._p - 10, ask=self._p, last_price=self._p)


class _Bar:
    close, date = 10000.0, DAY - dt.timedelta(days=1)


class _DB:
    def get_adjusted_daily_bars(self, ticker, *, start=None, end=None):  # noqa: ANN001, ANN201
        return [_Bar()]

    def get_regular_session_close(self, ticker, *, before, lookback_days):  # noqa: ANN001, ANN201
        return RegularSessionClose(code=ticker.value, date=_Bar().date, close=_Bar().close)


GEN = SignalGenerator(SignalConfig(min_confidence=0.0))
#: Signals validate expires_at > now and the filter reads today's date: run on the day.
ON_THE_DAY = time_machine.travel(dt.datetime(2026, 9, 15, 9, 1, tzinfo=KST), tick=False)


def test_tradable_tickers_drop_stale_articles_unknown_stocks_and_closed_sessions() -> None:
    stocks = {"005930": StockInfo(code="005930", name="삼성전자", market="KOSPI")}
    now = dt.datetime(2026, 9, 15, 9, 1, tzinfo=KST)
    fresh = _article(dt.datetime(2026, 9, 15, 8, 40, tzinfo=KST), ["005930", "000660"])
    assert [t.value for t, _ in tradable_tickers(fresh, stocks, STALE, EXPIRES, now)] == ["005930"]
    stale = _article(dt.datetime(2026, 9, 14, 15, 0, tzinfo=KST), ["005930"])
    assert tradable_tickers(stale, stocks, STALE, EXPIRES, now) == []
    assert tradable_tickers(fresh, stocks, STALE, EXPIRES, EXPIRES) == []


@ON_THE_DAY
def test_bullish_pair_under_the_cap_becomes_a_composite_buy_with_poller_metadata() -> None:
    article = _article(dt.datetime(2026, 9, 15, 8, 40, tzinfo=KST), ["005930"])
    c = build_entry_candidate(
        article, Ticker(value="005930"), _features(), _Broker(10100), _DB(), GEN, EXPIRES
    )
    assert c is not None
    assert (c.signal.direction, c.signal.source) == (SignalDirection.BUY, SignalSource.COMPOSITE)
    assert c.signal.news_article_id == article.id and c.signal.expires_at == EXPIRES
    assert c.signal.metadata["pre_entry_move"] == "0.01"
    assert c.signal.metadata["surprise"] == 2 and c.signal.metadata["subject"] == "main"
    assert (c.surprise, c.score) == (2, c.signal.metadata["news_score"])


@ON_THE_DAY
def test_non_bullish_and_extended_pairs_yield_no_candidate() -> None:
    article = _article(dt.datetime(2026, 9, 15, 8, 40, tzinfo=KST), ["005930"])
    t = Ticker(value="005930")
    mention = _features(subject="mention")  # score 0 -> neutral
    assert build_entry_candidate(article, t, mention, _Broker(10100), _DB(), GEN, EXPIRES) is None
    bearish = _features(sentiment_direction=-1)
    assert build_entry_candidate(article, t, bearish, _Broker(10100), _DB(), GEN, EXPIRES) is None
    extended = _Broker(10500)  # +5% over the 10,000 prior close, cap 3%
    assert build_entry_candidate(article, t, _features(), extended, _DB(), GEN, EXPIRES) is None
