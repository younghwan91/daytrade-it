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


# -- SelectionConfig (Task 4) --

from daytrade_it.application.signals.news_entry import SelectionConfig  # noqa: E402
from daytrade_it.application.trading.auto_trader import rank_batch  # noqa: E402
from daytrade_it.domain.entities.signal import (  # noqa: E402
    SignalDirection,
    SignalSource,
    TradingSignal,
)
from daytrade_it.domain.value_objects import Confidence  # noqa: E402


def _build(features: ArticleFeatures, last: str, selection=None, published=NOW):  # noqa: ANN001, ANN202
    gen = SignalGenerator(SignalConfig(min_confidence=0.0))
    kwargs = {} if selection is None else {"selection": selection}
    with time_machine.travel(NOW, tick=False):
        return build_entry_candidate(
            _article(["005930"], published=published),
            Ticker(value="005930"),
            features,
            _Broker(last),
            _DB(),
            gen,
            EXPIRES,
            **kwargs,
        )


MID = ArticleFeatures(
    **{
        **BULL.__dict__,
        "persistence": "multi_quarter",
        "scale_vs_size": "medium",
        "specificity": 2,
        "sentiment_strength": 2,
    }
)


def test_selection_defaults_equal_no_selection() -> None:
    a = _build(BULL, "10200")
    b = _build(BULL, "10200", SelectionConfig())
    assert a is not None and b is not None
    assert (a.surprise, a.score) == (b.surprise, b.score)
    assert b.signal.metadata["persistence"] == "multi_quarter"
    assert _build(BULL, "10400", SelectionConfig()) is None


def test_selection_min_score_rejects_below() -> None:
    assert round(MID.score(), 2) == 0.53
    assert _build(MID, "10000") is not None
    assert _build(MID, "10000", SelectionConfig(min_score=0.6)) is None


def test_selection_min_score_default_is_strict_threshold() -> None:
    from daytrade_it.infrastructure.external.anthropic_client import SIGNAL_THRESHOLD

    assert SelectionConfig().min_score == SIGNAL_THRESHOLD


def test_selection_min_pre_move_rejects_falling_stock() -> None:
    assert _build(BULL, "9700") is not None
    assert _build(BULL, "9700", SelectionConfig(min_pre_move=Decimal("-0.02"))) is None
    assert _build(BULL, "9900", SelectionConfig(min_pre_move=Decimal("-0.02"))) is not None


def test_selection_max_pre_move_overrides_module_cap() -> None:
    assert _build(BULL, "10200", SelectionConfig(max_pre_move=Decimal("0.01"))) is None
    assert _build(BULL, "10400", SelectionConfig(max_pre_move=Decimal("0.05"))) is not None


def test_selection_sessions_premarket_only_rejects_intraday() -> None:
    sel = SelectionConfig(sessions=("premarket",))
    assert _build(BULL, "10000", sel) is None  # published 09:30
    early = dt.datetime(2026, 9, 15, 8, 30, tzinfo=KST)
    assert _build(BULL, "10000", sel, published=early) is not None


def test_selection_subject_and_exclusions() -> None:
    partial = ArticleFeatures(**{**BULL.__dict__, "subject": "partial"})
    assert _build(partial, "10000", SelectionConfig(subjects=("main",))) is None
    assert _build(BULL, "10000", SelectionConfig(exclude_persistence=("multi_quarter",))) is None
    assert _build(BULL, "10000", SelectionConfig(exclude_material=("수주",))) is None


def _sig(code: str, persistence: str | None = None) -> TradingSignal:
    return TradingSignal(
        ticker=Ticker(value=code),
        direction=SignalDirection.BUY,
        confidence=Confidence(value=0.8),
        source=SignalSource.COMPOSITE,
        expires_at=dt.datetime.now(dt.UTC) + dt.timedelta(hours=1),
        metadata={} if persistence is None else {"persistence": persistence},
    )


def test_rank_batch_keys() -> None:
    a, b, c = (
        _sig("000001", "structural"),
        _sig("000002", "one_off"),
        _sig("000003", "multi_quarter"),
    )
    items = [(a, 0, 0.9), (b, 2, 0.3), (c, 1, 0.5)]
    assert [s.ticker.value for s in rank_batch(items)] == ["000002", "000003", "000001"]
    assert rank_batch(items) == rank_batch(items, key="surprise")
    assert [s.ticker.value for s in rank_batch(items, key="score")] == [
        "000001",
        "000003",
        "000002",
    ]
    assert [s.ticker.value for s in rank_batch(items, key="persistence")] == [
        "000001",
        "000003",
        "000002",
    ]
