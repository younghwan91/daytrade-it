# 리플레이 엔진 + 동일성 게이트 (베이즈 최적화 0단계) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 운영 코드(`AutoTrader`, `execute_trade`, `SignalGenerator`, `generate_no_chase_confirmation`)를 과거 하루에 그대로 돌려, 09-14·09-15 실거래 원장과 같은 주문이 나오는지 확인한다. 통과하면 1~4단계 최적화의 평가기로 쓴다.

**Architecture:** 시계는 `time-machine`으로 과거 시각에 고정하고, 브로커만 분봉 기반 `SimBroker`로 바꾼다. 신호 생성 로직은 폴러에서 `application/signals/news_entry.py`로 옮겨 폴러와 리플레이가 같은 함수를 부른다. 리플레이 입력은 "판정된 (기사, 종목) 이벤트" JSONL이고, 운영 판정 아카이브 또는 kr_quant 기사에서 만든다.

**Tech Stack:** Python 3.13, uv, pytest(asyncio), time-machine(신규 dev 의존성), psycopg2(kr_quant), structlog `capture_logs`.

**Spec:** `docs/superpowers/specs/2026-09-15-bayes-pipeline-optimization-design.md`

## Global Constraints

- 백테스트 = 실매매: 운영 모듈을 복사하지 않고 import해서 부른다. 리플레이 전용 분기(`if replay:`)를 운영 코드에 넣지 않는다.
- 매매 시점에 볼 수 없는 정보 금지: `SimBroker`는 현재 분 이후 분봉을 어떤 경로로도 노출하지 않는다. DART 스냅샷은 `rcept_dt ≤ 리플레이 날짜`만.
- 데이터 범위 B안: 09-14~ 는 `data/news_judgments` 아카이브만, 09-07~11은 kr_quant `news_articles`(운영 필터 통과분).
- 평가 스크립트 진입점은 `require_backtest_host()` (simnode 전용). 단위 테스트는 네트워크·DB 없이 돈다.
- 비용: 매도세 0.20% + 수수료 편도 0.015%.
- 동일성 게이트 통과 기준: 진입·추가매수·청산 종목과 수량 완전 일치, 체결가 차이 종목당 ±0.5% 이내.
- 커밋 메시지 끝: `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`

## File Structure

| 파일 | 책임 |
|---|---|
| `src/daytrade_it/application/signals/news_entry.py` (신규) | 기사 → 거래 가능 종목 필터, 판정 → 진입 후보 신호. 폴러와 리플레이 공용 |
| `scripts/news_signal_poller.py` (수정) | 위 함수를 부르도록 변경. 동작 불변 |
| `src/daytrade_it/infrastructure/brokers/sim_broker.py` (신규) | 분봉 기반 `BrokerInterface` 구현. 호가·체결·보유·현금 |
| `src/daytrade_it/application/replay/__init__.py`, `day_replay.py` (신규) | 이벤트 + 설정 → 하루 리플레이, 원장·스킵 사유 반환 |
| `src/daytrade_it/application/replay/events.py` (신규) | `ReplayEvent` JSONL 직렬화, 아카이브·kr_quant 행 → 이벤트 변환(순수 함수) |
| `scripts/eval/build_replay_events.py` (신규) | 이벤트 파일 생성(아카이브·데몬 로그·kr_quant·판정 캐시) |
| `scripts/eval/replay_parity.py` (신규) | 09-14·15 리플레이 vs 운영 원장 비교 |
| `scripts/eval/replay_baseline.py` (신규) | 전 거래일 운영 정책 리플레이, 거래별 손익 JSONL |
| `tests/unit/application/signals/test_news_entry.py`, `tests/unit/infrastructure/test_sim_broker.py`, `tests/unit/application/replay/test_day_replay.py`, `tests/unit/application/replay/test_events.py` (신규) | 단위 테스트 |
| `docs/research/2026-09-16-bayes-pipeline-eval.md` (신규) | 게이트 결과, 기준선 |

---

### Task 1: 신호 생성 로직을 공용 모듈로 추출

**Files:**
- Create: `src/daytrade_it/application/signals/news_entry.py`
- Modify: `scripts/news_signal_poller.py` (`process_pair` 228~265행, `process_article` 317~326행)
- Test: `tests/unit/application/signals/test_news_entry.py`

**Interfaces:**
- Produces:
  - `tradable_tickers(article: NewsArticle, stocks: dict[str, StockInfo], stale_before: dt.datetime, expires_at: dt.datetime, now: dt.datetime) -> list[tuple[Ticker, StockInfo]]`
  - `@dataclass(frozen=True) class EntryCandidate: signal: TradingSignal; surprise: int; score: float`
  - `build_entry_candidate(article: NewsArticle, ticker: Ticker, features: ArticleFeatures, broker: BrokerInterface, db_client: QuantAirflowDBClient, gen: SignalGenerator, expires_at: dt.datetime) -> EntryCandidate | None`

- [ ] **Step 1: dev 의존성 추가**

Run: `uv add --dev time-machine`
Expected: `pyproject.toml` dev 그룹에 `time-machine` 추가, `uv.lock` 갱신.

- [ ] **Step 2: 실패하는 테스트 작성**

```python
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
```

- [ ] **Step 3: 실패 확인**

Run: `uv run pytest tests/unit/application/signals/test_news_entry.py -v`
Expected: FAIL — `ModuleNotFoundError: daytrade_it.application.signals.news_entry`

- [ ] **Step 4: 구현**

`src/daytrade_it/application/signals/news_entry.py`:

```python
"""News pair → entry candidate: the live poller's rules, shared with replay.

`scripts/news_signal_poller.py` and `application/replay/day_replay.py` both
call these, so a backtest runs exactly the filter and signal build the live
daemon runs (spec 2026-09-15-bayes-pipeline-optimization-design.md).
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from daytrade_it.application.signals.entry_filter import generate_no_chase_confirmation
from daytrade_it.domain.entities.news import NewsArticle
from daytrade_it.domain.entities.signal import TradingSignal
from daytrade_it.domain.services.signal_generator import SignalGenerator
from daytrade_it.domain.value_objects import Ticker
from daytrade_it.infrastructure.brokers.base import BrokerInterface
from daytrade_it.infrastructure.external.anthropic_client import PROMPT_VERSION, ArticleFeatures
from daytrade_it.infrastructure.external.quant_airflow_db_client import (
    QuantAirflowDBClient,
    StockInfo,
)


@dataclass(frozen=True)
class EntryCandidate:
    signal: TradingSignal
    surprise: int
    score: float


def tradable_tickers(
    article: NewsArticle,
    stocks: dict[str, StockInfo],
    stale_before: dt.datetime,
    expires_at: dt.datetime,
    now: dt.datetime,
) -> list[tuple[Ticker, StockInfo]]:
    """Tagged tickers the daemon would judge: known stock, fresh article, session open."""
    # Published before the previous session closed: that session already
    # had the chance to trade it, and the eval never measured such articles.
    if article.published_at < stale_before or now >= expires_at:
        return []
    return [(t, stocks[t.value]) for t in article.tickers if t.value in stocks]


def build_entry_candidate(
    article: NewsArticle,
    ticker: Ticker,
    features: ArticleFeatures,
    broker: BrokerInterface,
    db_client: QuantAirflowDBClient,
    gen: SignalGenerator,
    expires_at: dt.datetime,
) -> EntryCandidate | None:
    """Composite BUY for a bullish, not-extended pair, else None."""
    sentiment = features.to_sentiment()
    # Entries are BUY-only: KRX retail can't short.
    if not sentiment.is_bullish:
        return None
    news_signal = gen.generate_from_news(
        article.model_copy(update={"tickers": [ticker]}), sentiment
    )
    if news_signal is None:
        return None
    entry_signal = generate_no_chase_confirmation(ticker, broker, db_client, expires_at)
    if entry_signal is None:
        return None
    composite = gen.combine_signals([news_signal, entry_signal])
    if composite is None:
        return None
    composite = composite.model_copy(
        update={
            "news_article_id": article.id,
            "expires_at": expires_at,
            "metadata": {
                **composite.metadata,
                "prompt_version": PROMPT_VERSION,
                "article_title": article.title[:200],
                "subject": features.subject,
                "material_type": features.material_type,
                "surprise": features.surprise,
                "news_score": sentiment.score,
                "pre_entry_move": entry_signal.metadata["pre_entry_move"],
            },
        }
    )
    return EntryCandidate(composite, int(features.surprise), float(sentiment.score))
```

`scripts/news_signal_poller.py` 변경:

1. import 추가: `from daytrade_it.application.signals.news_entry import build_entry_candidate, tradable_tickers`. 사용하지 않게 된 `generate_no_chase_confirmation` import는 삭제.
2. `process_pair`의 228~265행(`# Entries are BUY-only` 주석부터 `composite = composite.model_copy(...)` 끝까지)을 아래로 교체:

```python
try:
    candidate = build_entry_candidate(article, ticker, features, broker, db_client, gen, expires_at)
except Exception:
    logger.exception("news_signal_poller: entry filter failed", ticker=ticker.value)
    return
if candidate is None:
    return
composite = candidate.signal
```

3. 그 아래 `ACTIONABLE SIGNAL PERSISTED` 로그의 `pre_entry_move=entry_signal.metadata["pre_entry_move"]`를 `pre_entry_move=composite.metadata["pre_entry_move"]`로, `news_score=round(sentiment.score, 3)`는 그대로 둔다. `pending.append((saved, int(features.surprise), float(sentiment.score)))`를 `pending.append((saved, candidate.surprise, candidate.score))`로.
4. `process_article`의 317~326행(stale 검사, expires 검사, `known = ...`, `if not known: return`)을 아래로 교체:

```python
    known = tradable_tickers(article, stocks, stale_before, expires_at, dt.datetime.now(KST))
    if not known:
        return
```

- [ ] **Step 5: 테스트 통과 확인 (신규 + 기존 폴러 테스트)**

Run: `uv run pytest tests/unit/application/signals/test_news_entry.py tests/unit/scripts/test_news_signal_poller.py -v`
Expected: 전부 PASS

- [ ] **Step 6: 린트·타입**

Run: `uv run ruff check src scripts tests && uv run ruff format --check src scripts tests && uv run mypy src/daytrade_it/application/signals/news_entry.py`
Expected: 오류 없음

- [ ] **Step 7: 커밋**

```bash
git add pyproject.toml uv.lock src/daytrade_it/application/signals/news_entry.py scripts/news_signal_poller.py tests/unit/application/signals/test_news_entry.py
git commit -m "refactor(signals): move the poller's pair filter and signal build to news_entry for replay

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: SimBroker (분봉 기반 브로커)

**Files:**
- Create: `src/daytrade_it/infrastructure/brokers/sim_broker.py`
- Test: `tests/unit/infrastructure/test_sim_broker.py`

**Interfaces:**
- Consumes: `BrokerInterface`, `MinuteBar`, `Order`, `OrderResult`, `OrderStatus`, `OrderStatusEnum`, `Quote`, `Position`, `Account`, `KST` (`infrastructure/brokers/base.py`), `BrokerError` (`shared/errors/exceptions.py`), `tick_size` (`krx_quant_core.market.ticks`)
- Produces: `SimBroker(bars: dict[str, list[MinuteBar]], prev_close: dict[str, Decimal], clock: Callable[[], dt.datetime], cash: Decimal = Decimal("1100000"))`, 속성 `fills: list[dict]` (각 `{"ts", "ticker", "side", "qty", "price", "order_id"}`)

체결 규칙(스펙 표 그대로):
- 호가: 현재 분 m의 분봉이 있으면 `ask = open`, `bid = open − 1틱`, `last = open`. 없으면 m 이전 마지막 분봉 종가로 같은 계산. 분봉이 전혀 없으면 bid = ask = 0.
- 상한가 잠김: 현재 분봉 `high == low` 이고 `open ≥ prev_close × 1.295` → `ask = 0`. 하한가 잠김: `high == low` 이고 `open ≤ prev_close × 0.705` → `bid = 0`.
- 주문 즉시 체결: 매수 지정가 ≥ ask(> 0) → ask에 체결. 매도 지정가 ≤ bid(> 0) → bid에 체결.
- 미체결 주문: 다음 분부터 매 분봉 확인. 매수는 `low ≤ 지정가` 이고 잠김이 아니면 `min(지정가, open)`, 매도는 `high ≥ 지정가` 이고 잠김이 아니면 `max(지정가, open)`.
- 미래 차단: 공개 메서드는 모두 먼저 `_sync()`로 현재 분 **직전까지** 끝난 분봉으로만 미체결 주문을 처리한다. `get_minute_bars`는 `ts < 현재 분`만 준다.
- `get_order_status`: 미체결이면 `OrderStatus(SUBMITTED, 0, 수량)`, 아니면 `BrokerError("Order not found")` (키움과 같음). `cancel_order`: 미체결이면 제거 후 True, 아니면 False.

- [ ] **Step 1: 실패하는 테스트 작성**

```python
"""SimBroker: minute-bar fills with no look-ahead."""

import datetime as dt
from decimal import Decimal

import pytest

from daytrade_it.infrastructure.brokers.base import KST, MinuteBar, Order, OrderSide, OrderType
from daytrade_it.infrastructure.brokers.sim_broker import SimBroker
from daytrade_it.shared.errors.exceptions import BrokerError


def _bar(hm: str, o: int, h: int, l: int, c: int, v: int = 100) -> MinuteBar:
    hh, mm = map(int, hm.split(":"))
    return MinuteBar(
        ts=dt.datetime(2026, 9, 15, hh, mm),
        open=Decimal(o),
        high=Decimal(h),
        low=Decimal(l),
        close=Decimal(c),
        volume=v,
    )


class _Clock:
    def __init__(self) -> None:
        self.now = dt.datetime(2026, 9, 15, 9, 1, 10, tzinfo=KST)

    def __call__(self) -> dt.datetime:
        return self.now


BARS = {
    "000001": [
        _bar("09:00", 1000, 1010, 990, 1000),
        _bar("09:01", 1005, 1020, 1000, 1010),
        _bar("09:02", 1010, 1015, 980, 990),
        _bar("09:03", 995, 1000, 970, 975),
    ]
}


def _broker(clock: _Clock) -> SimBroker:
    return SimBroker(BARS, {"000001": Decimal(1000)}, clock, cash=Decimal(1_000_000))


def _buy(limit: int) -> Order:
    return Order(
        ticker="000001",
        quantity=Decimal(1),
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        limit_price=Decimal(limit),
    )


def _sell(limit: int) -> Order:
    return Order(
        ticker="000001",
        quantity=Decimal(1),
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        limit_price=Decimal(limit),
    )


def test_quote_uses_current_minute_open_one_tick_spread() -> None:
    q = _broker(_Clock()).get_quote("000001")
    assert (q.ask, q.bid, q.last_price) == (Decimal(1005), Decimal(1004), Decimal(1005))


def test_marketable_buy_fills_at_ask_and_updates_positions_and_cash() -> None:
    c = _Clock()
    b = _broker(c)
    r = b.place_order(_buy(1005))
    assert r.filled_quantity == 1 and r.filled_avg_price == Decimal(1005)
    assert [(p.ticker, p.quantity) for p in b.get_positions()] == [("000001", Decimal(1))]
    assert b.get_account().cash == Decimal(1_000_000 - 1005)
    with pytest.raises(BrokerError):
        b.get_order_status(r.order_id)


def test_resting_buy_fills_only_on_a_later_completed_bar() -> None:
    c = _Clock()
    b = _broker(c)
    r = b.place_order(_buy(985))
    assert b.get_order_status(r.order_id).remaining_quantity == 1
    c.now = dt.datetime(2026, 9, 15, 9, 2, 30, tzinfo=KST)  # 09:02 bar not finished
    assert b.get_order_status(r.order_id).remaining_quantity == 1
    c.now = dt.datetime(2026, 9, 15, 9, 3, 0, tzinfo=KST)  # 09:02 bar (low 980) done
    with pytest.raises(BrokerError):
        b.get_order_status(r.order_id)
    assert b.fills[-1]["price"] == Decimal(985)


def test_cancel_open_order_true_then_false() -> None:
    b = _broker(_Clock())
    r = b.place_order(_buy(900))
    assert b.cancel_order(r.order_id) is True
    assert b.cancel_order(r.order_id) is False


def test_marketable_sell_fills_at_bid() -> None:
    c = _Clock()
    b = _broker(c)
    b.place_order(_buy(1005))
    r = b.place_order(_sell(990))
    assert r.filled_avg_price == Decimal(1004)
    assert b.get_positions() == []


def test_minute_bars_hide_current_and_future_minutes() -> None:
    c = _Clock()
    assert [x.ts.minute for x in _broker(c).get_minute_bars("000001")] == [0]


def test_limit_up_lock_has_no_ask_and_never_fills_buys() -> None:
    bars = {
        "000002": [_bar("09:00", 1300, 1300, 1300, 1300), _bar("09:01", 1300, 1300, 1300, 1300)]
    }
    c = _Clock()
    b = SimBroker(bars, {"000002": Decimal(1000)}, c)
    assert b.get_quote("000002").ask == 0
    r = b.place_order(
        Order(
            ticker="000002",
            quantity=Decimal(1),
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            limit_price=Decimal(1300),
        )
    )
    c.now = dt.datetime(2026, 9, 15, 9, 5, tzinfo=KST)
    assert b.get_order_status(r.order_id).remaining_quantity == 1
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest tests/unit/infrastructure/test_sim_broker.py -v`
Expected: FAIL — `ModuleNotFoundError: ...sim_broker`

- [ ] **Step 3: 구현**

`src/daytrade_it/infrastructure/brokers/sim_broker.py`:

```python
"""Minute-bar broker for replaying a past session through the live trader.

Fill rules are the spec's (docs/superpowers/specs/2026-09-15-bayes-pipeline-
optimization-design.md): the quote at minute m is that bar's open (ask) with a
one-tick spread; marketable limits fill at once at the quote; resting limits
fill on a later bar that trades through them. Every public call first settles
resting orders against bars that have *finished* before the current minute,
and `get_minute_bars` hides the current minute -- nothing after "now" leaks.
"""

from __future__ import annotations

import datetime as dt
import itertools
from collections.abc import Callable
from decimal import Decimal

from krx_quant_core.market.ticks import tick_size

from daytrade_it.infrastructure.brokers.base import (
    Account,
    BrokerInterface,
    MinuteBar,
    Order,
    OrderResult,
    OrderSide,
    OrderStatus,
    OrderStatusEnum,
    Position,
    Quote,
)
from daytrade_it.shared.errors.exceptions import BrokerError

_UP_LOCK = Decimal("1.295")
_DOWN_LOCK = Decimal("0.705")


class SimBroker(BrokerInterface):
    def __init__(
        self,
        bars: dict[str, list[MinuteBar]],
        prev_close: dict[str, Decimal],
        clock: Callable[[], dt.datetime],
        cash: Decimal = Decimal("1100000"),
    ) -> None:
        self._bars = {t: sorted(b, key=lambda x: x.ts) for t, b in bars.items()}
        self._prev_close = prev_close
        self._clock = clock
        self._cash = cash
        self._held: dict[str, int] = {}
        self._avg: dict[str, Decimal] = {}
        self._open: dict[str, dict] = {}
        self._ids = itertools.count(1)
        self.fills: list[dict] = []

    # -- time --------------------------------------------------------------

    def _minute(self) -> dt.datetime:
        return self._clock().replace(tzinfo=None, second=0, microsecond=0)

    def _bar_at(self, ticker: str, minute: dt.datetime) -> MinuteBar | None:
        for b in self._bars.get(ticker, []):
            if b.ts == minute:
                return b
        return None

    def _locked(self, ticker: str, bar: MinuteBar) -> tuple[bool, bool]:
        prev = self._prev_close.get(ticker)
        if not prev or bar.high != bar.low:
            return False, False
        return bar.open >= prev * _UP_LOCK, bar.open <= prev * _DOWN_LOCK

    # -- settlement --------------------------------------------------------

    def _fill(self, oid: str, price: Decimal) -> None:
        o = self._open.pop(oid)
        t, qty = o["ticker"], o["qty"]
        if o["side"] == OrderSide.BUY:
            held = self._held.get(t, 0)
            self._avg[t] = (self._avg.get(t, Decimal(0)) * held + price * qty) / (held + qty)
            self._held[t] = held + qty
            self._cash -= price * qty
        else:
            self._held[t] = self._held.get(t, 0) - qty
            self._cash += price * qty
        self.fills.append(
            {
                "ts": self._clock(),
                "ticker": t,
                "side": o["side"].value,
                "qty": qty,
                "price": price,
                "order_id": oid,
            }
        )

    def _sync(self) -> None:
        minute = self._minute()
        for oid, o in list(self._open.items()):
            for b in self._bars.get(o["ticker"], []):
                if b.ts <= o["checked"] or b.ts >= minute:
                    continue
                o["checked"] = b.ts
                up, down = self._locked(o["ticker"], b)
                if o["side"] == OrderSide.BUY and not up and b.low <= o["limit"]:
                    self._fill(oid, min(o["limit"], b.open))
                    break
                if o["side"] == OrderSide.SELL and not down and b.high >= o["limit"]:
                    self._fill(oid, max(o["limit"], b.open))
                    break

    # -- BrokerInterface ---------------------------------------------------

    def connect(self) -> bool:
        return True

    def disconnect(self) -> None:
        return None

    def is_connected(self) -> bool:
        return True

    def get_account(self) -> Account:
        self._sync()
        return Account(
            account_id="sim", cash=self._cash, buying_power=self._cash, equity=self._cash
        )

    def get_positions(self) -> list[Position]:
        self._sync()
        return [
            Position(ticker=t, quantity=Decimal(q), average_price=self._avg[t])
            for t, q in self._held.items()
            if q > 0
        ]

    def get_quote(self, ticker: str) -> Quote:
        self._sync()
        minute = self._minute()
        bar = self._bar_at(ticker, minute)
        if bar is not None:
            price = bar.open
            up, down = self._locked(ticker, bar)
        else:
            done = [b for b in self._bars.get(ticker, []) if b.ts < minute]
            if not done:
                return Quote(ticker=ticker, bid=Decimal(0), ask=Decimal(0))
            price, up, down = done[-1].close, False, False
        bid = price - tick_size(price)
        return Quote(
            ticker=ticker,
            bid=Decimal(0) if down else bid,
            ask=Decimal(0) if up else price,
            last_price=price,
        )

    def place_order(self, order: Order) -> OrderResult:
        self._sync()
        if order.limit_price is None:
            raise BrokerError("SimBroker takes LIMIT orders only", broker_name="sim")
        quote = self.get_quote(order.ticker)
        oid = f"S{next(self._ids):07d}"
        qty = int(order.quantity)
        if order.side == OrderSide.SELL and self._held.get(order.ticker, 0) < qty:
            raise BrokerError(f"no holdings to sell: {order.ticker}", broker_name="sim")
        self._open[oid] = {
            "ticker": order.ticker,
            "side": order.side,
            "qty": qty,
            "limit": order.limit_price,
            "checked": self._minute(),
        }
        filled_price: Decimal | None = None
        if order.side == OrderSide.BUY and quote.ask > 0 and order.limit_price >= quote.ask:
            filled_price = quote.ask
        if order.side == OrderSide.SELL and quote.bid > 0 and order.limit_price <= quote.bid:
            filled_price = quote.bid
        if filled_price is not None:
            self._fill(oid, filled_price)
        return OrderResult(
            order_id=oid,
            client_order_id=None,
            ticker=order.ticker,
            quantity=order.quantity,
            filled_quantity=Decimal(qty if filled_price is not None else 0),
            side=order.side,
            order_type=order.order_type,
            status=OrderStatusEnum.FILLED
            if filled_price is not None
            else OrderStatusEnum.SUBMITTED,
            limit_price=order.limit_price,
            filled_avg_price=filled_price,
        )

    def cancel_order(self, order_id: str) -> bool:
        self._sync()
        return self._open.pop(order_id, None) is not None

    def get_order_status(self, order_id: str) -> OrderStatus:
        self._sync()
        o = self._open.get(order_id)
        if o is None:
            raise BrokerError(f"Order not found: {order_id}", broker_name="sim")
        return OrderStatus(
            order_id=order_id,
            status=OrderStatusEnum.SUBMITTED,
            filled_quantity=Decimal(0),
            remaining_quantity=Decimal(o["qty"]),
        )

    def get_minute_bars(self, ticker: str, day: dt.date | None = None) -> list[MinuteBar]:
        self._sync()
        minute = self._minute()
        return [b for b in self._bars.get(ticker, []) if b.ts < minute]
```

주의: `test_limit_up_lock...`에서 매수 지정가 1300은 즉시 체결 조건(ask = 0)에 걸리지 않고, 이후 분봉은 잠김이라 체결되지 않는다. `test_resting_buy...`에서 09:01 분봉(low 1000)은 제출 분이라 확인하지 않고(`checked` = 09:01), 09:02 분봉(low 980)에서 `min(985, 1010) = 985`로 체결된다.

`tick_size(price: Decimal | int | float) -> Decimal` (krx-quant-core, 2026-09-15 확인 — 시장 인자 없음).

- [ ] **Step 4: 테스트 통과 확인**

Run: `uv run pytest tests/unit/infrastructure/test_sim_broker.py -v`
Expected: 7개 PASS

- [ ] **Step 5: 커밋**

```bash
git add src/daytrade_it/infrastructure/brokers/sim_broker.py tests/unit/infrastructure/test_sim_broker.py
git commit -m "feat(replay): minute-bar SimBroker with no look-ahead fills

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: 리플레이 이벤트 형식과 변환

**Files:**
- Create: `src/daytrade_it/application/replay/__init__.py` (빈 파일), `src/daytrade_it/application/replay/events.py`
- Test: `tests/unit/application/replay/__init__.py` (빈 파일), `tests/unit/application/replay/test_events.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) class ReplayEvent: key: str; article_id: str; ticker: str; name: str; market: str | None; sector: str | None; title: str; category: str | None; published_at: dt.datetime; decide_at: dt.datetime; batch_at: dt.datetime; features: dict; source: str` (`source` ∈ `"archive"`, `"kr_quant"`)
  - `to_article(e: ReplayEvent) -> NewsArticle`
  - `features_of(e: ReplayEvent) -> ArticleFeatures`
  - `dump_events(path: Path, events: list[ReplayEvent]) -> None`, `load_events(path: Path) -> list[ReplayEvent]`
  - `events_from_archive(rows: list[dict], poll_times: list[dt.datetime], stocks: dict[str, StockInfo]) -> list[ReplayEvent]` — 같은 `key` 첫 행만, `decide_at = judged_at`, 폴 k(= `poll_times[k] ≤ judged_at < poll_times[k+1]`)의 `batch_at = 그 폴 마지막 judged_at + 1초`
  - `median_detect_delay(rows: list[dict]) -> dt.timedelta` — `published_at ≥ 09:00:30`인 행의 `judged_at − published_at` 중앙값
  - `schedule_detected(items: list[tuple[str, str, dt.datetime]], day: dt.date, delay: dt.timedelta, poll_every: dt.timedelta, judge_gap: dt.timedelta) -> dict[tuple[str, str], tuple[dt.datetime, dt.datetime]]` — `(article_id, ticker, published_at)` → `(decide_at, batch_at)`. 발견 = `max(published + delay, 09:00:30)`를 폴 격자(09:00:30 + k × poll_every)의 다음 칸으로 올림, 같은 칸 안에서 발행순 i번째는 `slot + i × judge_gap`, `batch_at` = 칸 마지막 `decide_at + 1초`

- [ ] **Step 1: 실패하는 테스트 작성**

```python
"""Replay events: archive rows and kr_quant rows → timed, judged pairs."""

import datetime as dt
from pathlib import Path

from daytrade_it.application.replay.events import (
    dump_events,
    events_from_archive,
    features_of,
    load_events,
    median_detect_delay,
    schedule_detected,
    to_article,
)
from daytrade_it.infrastructure.brokers.base import KST
from daytrade_it.infrastructure.external.quant_airflow_db_client import StockInfo

V2 = {
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
    "score": 0.8,
}


def _row(aid: str, ticker: str, pub: str, judged: str) -> dict:
    return {
        "key": f"{aid}|{ticker}",
        "article_id": aid,
        "ticker": ticker,
        "name": "n",
        "title": "t",
        "category": "stock",
        "published_at": pub,
        "judged_at": judged,
        "v2": V2,
    }


def _t(hms: str) -> dt.datetime:
    h, m, s = map(int, hms.split(":"))
    return dt.datetime(2026, 9, 15, h, m, s, tzinfo=KST)


A = "0a0a0a0a-0000-5000-8000-000000000001"
B = "0a0a0a0a-0000-5000-8000-000000000002"
STOCKS = {"005930": StockInfo(code="005930", name="삼성전자", market="KOSPI", sector="전기/전자")}


def test_archive_dedups_and_batches_by_poll() -> None:
    rows = [
        _row(A, "005930", "2026-09-15T08:40:00+09:00", "2026-09-15T09:00:35+09:00"),
        _row(A, "005930", "2026-09-15T08:40:00+09:00", "2026-09-15T09:01:40+09:00"),  # dup daemon
        _row(B, "005930", "2026-09-15T09:02:00+09:00", "2026-09-15T09:04:10+09:00"),
    ]
    polls = [_t("09:00:31"), _t("09:04:02")]
    ev = events_from_archive(rows, polls, STOCKS)
    assert [(e.article_id, e.decide_at, e.batch_at) for e in ev] == [
        (A, _t("09:00:35"), _t("09:00:36")),
        (B, _t("09:04:10"), _t("09:04:11")),
    ]
    assert ev[0].sector == "전기/전자" and ev[0].source == "archive"
    assert features_of(ev[0]).surprise == 2
    assert to_article(ev[0]).tickers[0].value == "005930"


def test_median_detect_delay_ignores_premarket() -> None:
    rows = [
        _row(A, "1", "2026-09-15T08:00:00+09:00", "2026-09-15T09:00:35+09:00"),
        _row(A, "2", "2026-09-15T10:00:00+09:00", "2026-09-15T10:10:00+09:00"),
        _row(A, "3", "2026-09-15T11:00:00+09:00", "2026-09-15T11:30:00+09:00"),
    ]
    assert median_detect_delay(rows) == dt.timedelta(minutes=20)


def test_schedule_detected_snaps_to_poll_grid() -> None:
    items = [(A, "005930", _t("08:10:00")), (B, "005930", _t("09:10:00"))]
    out = schedule_detected(
        items,
        dt.date(2026, 9, 15),
        dt.timedelta(minutes=10),
        dt.timedelta(minutes=3),
        dt.timedelta(seconds=2),
    )
    assert out[(A, "005930")] == (_t("09:00:30"), _t("09:00:31"))
    # 09:20:00 → next grid slot 09:21:30
    assert out[(B, "005930")] == (_t("09:21:30"), _t("09:21:31"))


def test_dump_load_roundtrip(tmp_path: Path) -> None:
    rows = [_row(A, "005930", "2026-09-15T08:40:00+09:00", "2026-09-15T09:00:35+09:00")]
    ev = events_from_archive(rows, [_t("09:00:31")], STOCKS)
    p = tmp_path / "e.jsonl"
    dump_events(p, ev)
    assert load_events(p) == ev
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest tests/unit/application/replay/test_events.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: 구현**

`src/daytrade_it/application/replay/events.py`:

```python
"""Replay input: one judged (article, ticker) pair with the time the daemon acted on it.

Live days come from the daemon's judgment archive (`data/news_judgments`):
`decide_at` is the real `judged_at`, and a poll's signals are entered right
after its last judgment (`batch_at`). Days before the daemon ran come from
kr_quant articles, detected at publish + the archive's median delay on the
daemon's poll grid. Features are the stored v2 judgment -- replay never
re-asks the model for a pair the daemon already judged.
"""

from __future__ import annotations

import bisect
import datetime as dt
import json
import statistics
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from uuid import UUID

from daytrade_it.domain.entities.news import NewsArticle
from daytrade_it.domain.value_objects import Ticker
from daytrade_it.infrastructure.external.anthropic_client import ArticleFeatures
from daytrade_it.infrastructure.external.quant_airflow_db_client import StockInfo

FIRST_POLL = dt.time(9, 0, 30)
_FEATURE_FIELDS = {f.name for f in fields(ArticleFeatures)}


@dataclass(frozen=True)
class ReplayEvent:
    key: str
    article_id: str
    ticker: str
    name: str
    market: str | None
    sector: str | None
    title: str
    category: str | None
    published_at: dt.datetime
    decide_at: dt.datetime
    batch_at: dt.datetime
    features: dict
    source: str


def to_article(e: ReplayEvent) -> NewsArticle:
    return NewsArticle(
        id=UUID(e.article_id),
        title=e.title,
        content=e.title,
        source="toss",
        url="https://tossinvest.com/news",
        published_at=e.published_at,
        tickers=[Ticker(value=e.ticker)],
        category=e.category,
    )


def features_of(e: ReplayEvent) -> ArticleFeatures:
    return ArticleFeatures(**{k: v for k, v in e.features.items() if k in _FEATURE_FIELDS})


def dump_events(path: Path, events: list[ReplayEvent]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for e in events:
            row = asdict(e)
            for k in ("published_at", "decide_at", "batch_at"):
                row[k] = row[k].isoformat()
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_events(path: Path) -> list[ReplayEvent]:
    out = []
    for line in path.open():
        row = json.loads(line)
        for k in ("published_at", "decide_at", "batch_at"):
            row[k] = dt.datetime.fromisoformat(row[k])
        out.append(ReplayEvent(**row))
    return out


def events_from_archive(
    rows: list[dict], poll_times: list[dt.datetime], stocks: dict[str, StockInfo]
) -> list[ReplayEvent]:
    first: dict[str, dict] = {}
    for r in rows:
        if r.get("v2") and r["key"] not in first:
            first[r["key"]] = r
    kept = sorted(first.values(), key=lambda r: r["judged_at"])
    polls = sorted(poll_times)

    def poll_of(ts: dt.datetime) -> int:
        return bisect.bisect_right(polls, ts) - 1

    last_in_poll: dict[int, dt.datetime] = {}
    for r in kept:
        ts = dt.datetime.fromisoformat(r["judged_at"])
        last_in_poll[poll_of(ts)] = ts
    events = []
    for r in kept:
        judged = dt.datetime.fromisoformat(r["judged_at"])
        stock = stocks.get(r["ticker"])
        events.append(
            ReplayEvent(
                key=r["key"],
                article_id=r["article_id"],
                ticker=r["ticker"],
                name=r["name"],
                market=stock.market if stock else None,
                sector=stock.sector if stock else None,
                title=r["title"],
                category=r.get("category"),
                published_at=dt.datetime.fromisoformat(r["published_at"]),
                decide_at=judged,
                batch_at=last_in_poll[poll_of(judged)] + dt.timedelta(seconds=1),
                features=r["v2"],
                source="archive",
            )
        )
    return events


def median_detect_delay(rows: list[dict]) -> dt.timedelta:
    delays = []
    for r in rows:
        pub = dt.datetime.fromisoformat(r["published_at"])
        if pub.timetz().replace(tzinfo=None) >= FIRST_POLL:
            delays.append((dt.datetime.fromisoformat(r["judged_at"]) - pub).total_seconds())
    return dt.timedelta(seconds=statistics.median(delays))


def schedule_detected(
    items: list[tuple[str, str, dt.datetime]],
    day: dt.date,
    delay: dt.timedelta,
    poll_every: dt.timedelta,
    judge_gap: dt.timedelta,
) -> dict[tuple[str, str], tuple[dt.datetime, dt.datetime]]:
    tz = items[0][2].tzinfo if items else None
    start = dt.datetime.combine(day, FIRST_POLL, tzinfo=tz)
    slots: dict[dt.datetime, list[tuple[str, str, dt.datetime]]] = {}
    for aid, ticker, pub in sorted(items, key=lambda x: x[2]):
        seen = max(pub + delay, start)
        k = -(-(seen - start) // poll_every)  # ceil
        slots.setdefault(start + k * poll_every, []).append((aid, ticker, pub))
    out = {}
    for slot, group in slots.items():
        batch = slot + (len(group) - 1) * judge_gap + dt.timedelta(seconds=1)
        for i, (aid, ticker, _) in enumerate(group):
            out[(aid, ticker)] = (slot + i * judge_gap, batch)
    return out
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `uv run pytest tests/unit/application/replay/test_events.py -v`
Expected: 4개 PASS

- [ ] **Step 5: 커밋**

```bash
git add src/daytrade_it/application/replay tests/unit/application/replay
git commit -m "feat(replay): judged-pair event format from the daemon archive and kr_quant

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: 하루 리플레이 엔진

**Files:**
- Create: `src/daytrade_it/application/replay/day_replay.py`
- Test: `tests/unit/application/replay/test_day_replay.py`

**Interfaces:**
- Consumes: `build_entry_candidate`, `tradable_tickers`, `EntryCandidate` (Task 1); `SimBroker` (Task 2); `ReplayEvent`, `to_article`, `features_of` (Task 3); `AutoTrader`, `TradingLimits`, `rank_batch` (`application/trading/auto_trader.py`); `entry_filter.MAX_PRE_ENTRY_MOVE`; `interfaces.mcp.tools._DART_DB_PATH`, `_KILL_SWITCH_PATH`
- Produces:
  - `@dataclass(frozen=True) class DayConfig: limits: TradingLimits; max_pre_entry_move: Decimal; manage_every: dt.timedelta = dt.timedelta(seconds=30); exits_until: dt.time = dt.time(15, 30)`
  - `@dataclass class DayResult: ledger: list[dict]; skips: list[dict]; fills: list[dict]`
  - `snapshot_dart(src: Path, day: dt.date, dst: Path) -> Path` — `rcept_dt ≤ day` 행만 복사한 sqlite
  - `replay_day(day: dt.date, events: list[ReplayEvent], broker: SimBroker, db_client, stocks: dict[str, StockInfo], stale_before: dt.datetime, config: DayConfig, work_dir: Path, dart_db: Path) -> DayResult`

동작(운영 `run()` 루프 순서 그대로):
1. 타임라인 = 09:00:30부터 `exits_until`까지 `manage_every` 간격 틱 ∪ 모든 `decide_at` ∪ 모든 `batch_at`, 시각순. 같은 시각이면 manage → decide → batch.
2. manage 틱: `trader.manage()` (예외는 로그만).
3. decide: `tradable_tickers(to_article(e), stocks, stale_before, expires_at, now)`에 이벤트 종목이 있으면 `build_entry_candidate(...)` → 후보를 그 이벤트의 `batch_at` 대기열에 넣는다.
4. batch: 대기열을 `rank_batch([(c.signal, c.surprise, c.score) ...])` 순서로 `await trader.enter(signal)`.
5. 시계: 전체를 `time_machine.travel(first_tick, tick=False)`로 감싸고 각 시점마다 `traveller.move_to(t)`. `AutoTrader(clock=lambda: dt.datetime.now(KST))`, 원장은 `work_dir/ledger`.
6. 운영 전역 치환(끝나면 원복): `entry_filter.MAX_PRE_ENTRY_MOVE = config.max_pre_entry_move`, `tools._DART_DB_PATH = snapshot`, `tools._KILL_SWITCH_PATH = work_dir / "no_kill_switch"`.
7. 스킵 사유: `structlog.testing.capture_logs()`로 `event == "auto_trader: entry skipped"`와 `"entry_filter: extended, no entry"`를 모은다.
8. 끝에 `trader.report_leftovers()` 후 원장 JSONL을 읽어 `DayResult`로 반환.

- [ ] **Step 1: 실패하는 테스트 작성**

```python
"""replay_day drives the live AutoTrader through a synthetic session."""

import datetime as dt
import sqlite3
from decimal import Decimal
from pathlib import Path

from daytrade_it.application.replay.day_replay import DayConfig, replay_day, snapshot_dart
from daytrade_it.application.replay.events import ReplayEvent
from daytrade_it.application.trading.auto_trader import TradingLimits
from daytrade_it.domain.value_objects import Ticker
from daytrade_it.infrastructure.brokers.base import KST, MinuteBar
from daytrade_it.infrastructure.brokers.sim_broker import SimBroker
from daytrade_it.infrastructure.external.quant_airflow_db_client import StockInfo

DAY = dt.date(2026, 9, 15)
V2 = {
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


def _bars(ticker_open: int) -> list[MinuteBar]:
    out, t = [], dt.datetime(2026, 9, 15, 9, 0)
    while t.time() <= dt.time(15, 30):
        p = Decimal(ticker_open)
        out.append(MinuteBar(ts=t, open=p, high=p + 10, low=p - 10, close=p, volume=1000))
        t += dt.timedelta(minutes=1)
    return out


def _event(aid_suffix: str, ticker: str, hms: str) -> ReplayEvent:
    h, m, s = map(int, hms.split(":"))
    at = dt.datetime(2026, 9, 15, h, m, s, tzinfo=KST)
    return ReplayEvent(
        key=f"k{aid_suffix}",
        article_id=f"0a0a0a0a-0000-5000-8000-00000000000{aid_suffix}",
        ticker=ticker,
        name="n",
        market="KOSPI",
        sector="s",
        title="t",
        category="stock",
        published_at=at - dt.timedelta(minutes=5),
        decide_at=at,
        batch_at=at + dt.timedelta(seconds=1),
        features=V2,
        source="archive",
    )


class _Bar:
    def __init__(self) -> None:
        self.close, self.date = 10000.0, DAY - dt.timedelta(days=1)


class _DB:
    def get_adjusted_daily_bars(self, ticker, *, start=None, end=None):  # noqa: ANN001, ANN201
        return [_Bar()]


def _dart(tmp: Path) -> Path:
    p = tmp / "dart.db"
    con = sqlite3.connect(p)
    con.execute(
        "CREATE TABLE disclosures (rcept_no TEXT PRIMARY KEY, rcept_dt TEXT NOT NULL, "
        "code TEXT NOT NULL, corp_name TEXT, corp_cls TEXT, report_nm TEXT, "
        "material_type TEXT, is_correction INTEGER)"
    )
    con.execute("INSERT INTO disclosures VALUES ('1','20260916','000002','x','Y','상장폐지','x',0)")
    con.commit()
    con.close()
    return p


def test_snapshot_dart_drops_future_filings(tmp_path: Path) -> None:
    snap = snapshot_dart(_dart(tmp_path), DAY, tmp_path / "snap.db")
    assert sqlite3.connect(snap).execute("SELECT count(*) FROM disclosures").fetchone() == (0,)


def test_replay_enters_ranked_batch_respects_cap_and_flattens(tmp_path: Path) -> None:
    clock_holder: dict = {}
    bars = {"000001": _bars(10100), "000002": _bars(10100), "000003": _bars(10100)}
    broker = SimBroker(bars, {t: Decimal(10000) for t in bars}, lambda: clock_holder["now"]())
    stocks = {t: StockInfo(code=t, name=t, market="KOSPI", sector="s") for t in bars}
    events = [
        _event("1", "000001", "09:00:35"),
        _event("2", "000002", "09:00:35"),
        _event("3", "000003", "09:00:35"),
    ]
    cfg = DayConfig(
        limits=TradingLimits(max_open_positions=2, vwap_exit_gap=None, add_on_min_gain=None),
        max_pre_entry_move=Decimal("0.03"),
    )
    res = replay_day(
        DAY,
        events,
        broker,
        _DB(),
        stocks,
        dt.datetime(2026, 9, 14, 15, 20, tzinfo=KST),
        cfg,
        tmp_path,
        _dart(tmp_path),
        clock_holder=clock_holder,
    )
    entries = [r["ticker"] for r in res.ledger if r["event"] == "entry_submitted"]
    exits = [r["ticker"] for r in res.ledger if r["event"] == "exit_submitted"]
    assert len(entries) == 2 and sorted(exits) == sorted(entries)
    assert any(s["reason"] == "concurrent position cap reached" for s in res.skips)
    assert all(r["ts"].startswith("2026-09-15") for r in res.ledger)
    assert [f["side"] for f in res.fills].count("sell") == 2
```

`replay_day`는 테스트 편의를 위해 선택 인자 `clock_holder: dict | None = None`을 받아, 주어지면 `clock_holder["now"] = lambda: dt.datetime.now(KST)`를 넣는다(브로커가 리플레이 시계를 공유하도록). 스크립트에서는 `SimBroker(..., clock=lambda: dt.datetime.now(KST))`로 직접 만든다.

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest tests/unit/application/replay/test_day_replay.py -v`
Expected: FAIL — `ModuleNotFoundError: ...day_replay`

- [ ] **Step 3: 구현**

`src/daytrade_it/application/replay/day_replay.py`:

```python
"""Replay one past session through the live trader.

The daemon's loop (`scripts/news_signal_poller.py::run`) is reproduced as a
timeline: manage ticks every 30 s, each pair judged at its `decide_at`, each
poll's candidates entered at `batch_at` in `rank_batch` order. The clock is
frozen with time-machine at each instant, so every `datetime.now()` inside
the live code -- signal expiry, the no-chase filter's "today", DART lookback --
sees the replayed time. Only the broker (`SimBroker`) and three module paths
(no-chase cap, DART snapshot, kill-switch file) differ from live.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import sqlite3
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any

import time_machine
from structlog.testing import capture_logs

from daytrade_it.application.signals import entry_filter
from daytrade_it.application.signals.news_entry import (
    EntryCandidate,
    build_entry_candidate,
    tradable_tickers,
)
from daytrade_it.application.replay.events import ReplayEvent, features_of, to_article
from daytrade_it.application.trading.auto_trader import AutoTrader, TradingLimits, rank_batch
from daytrade_it.domain.services.signal_generator import SignalConfig, SignalGenerator
from daytrade_it.domain.value_objects import Ticker
from daytrade_it.infrastructure.brokers.base import KST
from daytrade_it.infrastructure.brokers.sim_broker import SimBroker
from daytrade_it.infrastructure.external.quant_airflow_db_client import StockInfo
from daytrade_it.interfaces.mcp import tools
from daytrade_it.shared.logging import get_logger

logger = get_logger(__name__)

FIRST_POLL = dt.time(9, 0, 30)
SESSION_CLOSE = dt.time(15, 20)
_SKIP_EVENTS = {"auto_trader: entry skipped", "entry_filter: extended, no entry"}


@dataclass(frozen=True)
class DayConfig:
    limits: TradingLimits
    max_pre_entry_move: Decimal
    manage_every: dt.timedelta = dt.timedelta(seconds=30)
    exits_until: dt.time = dt.time(15, 30)


@dataclass
class DayResult:
    ledger: list[dict] = field(default_factory=list)
    skips: list[dict] = field(default_factory=list)
    fills: list[dict] = field(default_factory=list)


def snapshot_dart(src: Path, day: dt.date, dst: Path) -> Path:
    """Copy of dart.db without filings received after `day` (no look-ahead)."""
    dst.unlink(missing_ok=True)
    con = sqlite3.connect(dst)
    con.execute("ATTACH DATABASE ? AS src", (f"file:{src}?mode=ro",))
    con.execute(
        "CREATE TABLE disclosures AS SELECT * FROM src.disclosures WHERE rcept_dt <= ?",
        (day.strftime("%Y%m%d"),),
    )
    con.commit()
    con.execute("DETACH DATABASE src")
    con.close()
    return dst


def _timeline(
    day: dt.date, events: list[ReplayEvent], cfg: DayConfig
) -> list[tuple[dt.datetime, int, Any]]:
    items: list[tuple[dt.datetime, int, Any]] = []
    t = dt.datetime.combine(day, FIRST_POLL, tzinfo=KST)
    end = dt.datetime.combine(day, cfg.exits_until, tzinfo=KST)
    while t <= end:
        items.append((t, 0, None))
        t += cfg.manage_every
    for e in events:
        items.append((e.decide_at, 1, e))
        items.append((e.batch_at, 2, e.batch_at))
    items.sort(key=lambda x: (x[0], x[1]))
    return items


def replay_day(
    day: dt.date,
    events: list[ReplayEvent],
    broker: SimBroker,
    db_client: Any,
    stocks: dict[str, StockInfo],
    stale_before: dt.datetime,
    config: DayConfig,
    work_dir: Path,
    dart_db: Path,
    clock_holder: dict | None = None,
) -> DayResult:
    work_dir.mkdir(parents=True, exist_ok=True)
    ledger_dir = work_dir / "ledger"
    expires_at = dt.datetime.combine(day, SESSION_CLOSE, tzinfo=KST)
    saved = (entry_filter.MAX_PRE_ENTRY_MOVE, tools._DART_DB_PATH, tools._KILL_SWITCH_PATH)
    entry_filter.MAX_PRE_ENTRY_MOVE = config.max_pre_entry_move
    tools._DART_DB_PATH = snapshot_dart(dart_db, day, work_dir / "dart.db")
    tools._KILL_SWITCH_PATH = work_dir / "no_kill_switch"
    if clock_holder is not None:
        clock_holder["now"] = lambda: dt.datetime.now(KST)
    timeline = _timeline(day, events, config)
    try:
        with time_machine.travel(timeline[0][0], tick=False) as traveller, capture_logs() as logs:
            asyncio.run(
                _run(
                    timeline,
                    traveller,
                    broker,
                    db_client,
                    stocks,
                    stale_before,
                    expires_at,
                    config,
                    ledger_dir,
                )
            )
    finally:
        entry_filter.MAX_PRE_ENTRY_MOVE, tools._DART_DB_PATH, tools._KILL_SWITCH_PATH = saved
    path = ledger_dir / f"{day.isoformat()}.jsonl"
    ledger = [json.loads(line) for line in path.open()] if path.exists() else []
    skips = [
        {"ticker": r.get("ticker"), "reason": r.get("reason", "extended")}
        for r in logs
        if r.get("event") in _SKIP_EVENTS
    ]
    return DayResult(ledger=ledger, skips=skips, fills=list(broker.fills))


async def _run(
    timeline: list[tuple[dt.datetime, int, Any]],
    traveller: Any,
    broker: SimBroker,
    db_client: Any,
    stocks: dict[str, StockInfo],
    stale_before: dt.datetime,
    expires_at: dt.datetime,
    config: DayConfig,
    ledger_dir: Path,
) -> None:
    trader = AutoTrader(
        broker, ledger_dir, live=True, limits=config.limits, clock=lambda: dt.datetime.now(KST)
    )
    gen = SignalGenerator(SignalConfig(min_confidence=0.0))
    pending: dict[dt.datetime, list[EntryCandidate]] = {}
    for at, kind, payload in timeline:
        traveller.move_to(at)
        if kind == 0:
            try:
                trader.manage()
            except Exception:
                logger.exception("replay: manage failed")
        elif kind == 1:
            e: ReplayEvent = payload
            article = to_article(e)
            if not any(
                t.value == e.ticker
                for t, _ in tradable_tickers(article, stocks, stale_before, expires_at, at)
            ):
                continue
            try:
                c = build_entry_candidate(
                    article,
                    Ticker(value=e.ticker),
                    features_of(e),
                    broker,
                    db_client,
                    gen,
                    expires_at,
                )
            except Exception:
                logger.exception("replay: entry filter failed", ticker=e.ticker)
                continue
            if c is not None:
                pending.setdefault(e.batch_at, []).append(c)
        else:
            batch = pending.pop(payload, [])
            if not batch:
                continue
            for signal in rank_batch([(c.signal, c.surprise, c.score) for c in batch]):
                try:
                    await trader.enter(signal)
                except Exception:
                    logger.exception("replay: enter failed", ticker=signal.ticker.value)
    trader.report_leftovers()
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `uv run pytest tests/unit/application/replay -v`
Expected: 전부 PASS. `execute_trade`의 `RiskManager.validate_trade`가 테스트 계좌에서 거부하면 실패 메시지의 `reason`을 보고, 운영 계좌 조건(예수금 110만원, 보유 없음)과 같은지 확인한다. **리스크 설정을 테스트 통과용으로 바꾸지 않는다.**

- [ ] **Step 5: 전체 단위 테스트**

Run: `uv run pytest tests/unit -q`
Expected: 기존 테스트 포함 전부 PASS

- [ ] **Step 6: 커밋**

```bash
git add src/daytrade_it/application/replay/day_replay.py tests/unit/application/replay/test_day_replay.py
git commit -m "feat(replay): replay a session through the live AutoTrader under a frozen clock

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: 이벤트 파일 생성 스크립트

**Files:**
- Create: `scripts/eval/build_replay_events.py`

**Interfaces:**
- Consumes: `events_from_archive`, `median_detect_delay`, `schedule_detected`, `dump_events`, `ReplayEvent` (Task 3); `AnthropicClient.score_article`; `TOSS_NEWS_NAMESPACE`; `QuantAirflowDBClient.get_stock_master`
- Produces: `data/eval/replay_events/YYYY-MM-DD.jsonl` (거래일별), `data/eval/replay_events/meta.json` (`{"median_delay_s", "judge_gap_s", "poll_every_s", "archive_days", "kr_quant_days"}`)

실행 위치: **trader** (아카이브·데몬 로그·Anthropic 키가 여기 있다). 결과 파일을 simnode로 복사한다. 이 스크립트는 주문·백테스트를 하지 않으므로 `require_backtest_host()`를 부르지 않는다.

- [ ] **Step 1: 스크립트 작성**

```python
"""Build replay events per trading day (spec 2026-09-15, data option B).

    uv run python scripts/eval/build_replay_events.py --kr-quant-days 2026-09-07,2026-09-08,2026-09-09,2026-09-10,2026-09-11

- Days with a daemon archive (`data/news_judgments/DAY.jsonl`): archive only.
  Poll times come from `data/news_signal_daemon.log` ("Toss news poll complete").
- `--kr-quant-days`: kr_quant `news_articles` pairs, keyed like the daemon
  (uuid5(TOSS_NEWS_NAMESPACE, id)), judged with the live `score_article`
  (cached in `data/eval/cache/replay_v2.jsonl`), timed with the archive's
  median detect delay on the daemon's poll grid.
Run on trader, then copy `data/eval/replay_events/` to simnode.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import os
import statistics
from dataclasses import asdict
from pathlib import Path
from uuid import uuid5

import psycopg2

from daytrade_it.application.replay.events import (
    ReplayEvent,
    dump_events,
    events_from_archive,
    median_detect_delay,
    schedule_detected,
)
from daytrade_it.domain.value_objects import Ticker
from daytrade_it.infrastructure.brokers.base import KST
from daytrade_it.infrastructure.external.anthropic_client import AnthropicClient
from daytrade_it.infrastructure.external.quant_airflow_db_client import QuantAirflowDBClient
from daytrade_it.infrastructure.external.toss_news_client import TOSS_NEWS_NAMESPACE
from daytrade_it.shared.config.settings import settings

ARCHIVE = Path("data/news_judgments")
DAEMON_LOG = Path("data/news_signal_daemon.log")
OUT = Path("data/eval/replay_events")
CACHE = Path("data/eval/cache/replay_v2.jsonl")


def poll_times(day: dt.date) -> list[dt.datetime]:
    out = []
    for line in DAEMON_LOG.open():
        if "Toss news poll complete" not in line:
            continue
        try:
            ts = dt.datetime.fromisoformat(json.loads(line)["timestamp"].replace("Z", "+00:00"))
        except (ValueError, KeyError):
            continue
        ts = ts.astimezone(KST)
        if ts.date() == day:
            out.append(ts)
    return out


def judge_gap(rows: list[dict]) -> dt.timedelta:
    ts = sorted(dt.datetime.fromisoformat(r["judged_at"]) for r in rows)
    gaps = [(b - a).total_seconds() for a, b in zip(ts, ts[1:]) if 0 < (b - a).total_seconds() < 60]
    return dt.timedelta(seconds=statistics.median(gaps))


def kr_quant_pairs(day: dt.date, prev_close_day: dt.date) -> list[dict]:
    """Pairs the daemon's filters would keep: published after the prior close, before 15:20."""
    with psycopg2.connect(os.environ["KR_QUANT_DB"], connect_timeout=5) as c, c.cursor() as cur:
        cur.execute(
            """SELECT a.id, t.ticker, a.title, a.summary, a.content, a.category, a.published_at
               FROM news_articles a JOIN news_article_tickers t ON t.article_id = a.id
               WHERE a.source = 'toss' AND a.published_at >= %s AND a.published_at < %s""",
            (
                dt.datetime.combine(prev_close_day, dt.time(15, 20), tzinfo=KST),
                dt.datetime.combine(day, dt.time(15, 20), tzinfo=KST),
            ),
        )
        cols = ["id", "ticker", "title", "summary", "content", "category", "published_at"]
        return [dict(zip(cols, r)) for r in cur.fetchall()]


async def judge(pairs: list[dict], stocks: dict) -> dict[str, dict]:
    cache: dict[str, dict] = {}
    if CACHE.exists():
        cache = {json.loads(l)["key"]: json.loads(l)["v2"] for l in CACHE.open()}
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    async with AnthropicClient() as claude:
        for p in pairs:
            key = f"{p['live_id']}|{p['ticker']}"
            if key in cache:
                continue
            s = stocks[p["ticker"]]
            f = await claude.score_article(
                code=p["ticker"],
                name=s.name,
                market=s.market,
                sector=s.sector,
                title=p["title"],
                body=p["summary"] or p["content"],
                category=p["category"],
            )
            cache[key] = {**asdict(f), "score": f.score()}
            with CACHE.open("a") as out:
                out.write(json.dumps({"key": key, "v2": cache[key]}, ensure_ascii=False) + "\n")
    return cache


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kr-quant-days", default="")
    args = ap.parse_args()
    stocks = QuantAirflowDBClient().get_stock_master()
    archive_days = sorted(dt.date.fromisoformat(p.stem) for p in ARCHIVE.glob("*.jsonl"))
    all_rows: list[dict] = []
    for day in archive_days:
        rows = [json.loads(l) for l in (ARCHIVE / f"{day}.jsonl").open()]
        all_rows += rows
        events = events_from_archive(rows, poll_times(day), stocks)
        dump_events(OUT / f"{day}.jsonl", events)
        print(day, "archive events", len(events))
    delay, gap = median_detect_delay(all_rows), judge_gap(all_rows)
    poll_every = dt.timedelta(seconds=settings.toss_poll_interval_seconds)
    kq_days = [dt.date.fromisoformat(d) for d in args.kr_quant_days.split(",") if d]
    db = QuantAirflowDBClient()
    for day in kq_days:
        prior = db.get_adjusted_daily_bars(
            Ticker(value="005930"),
            start=day - dt.timedelta(days=14),
            end=day - dt.timedelta(days=1),
        )
        pairs = [p for p in kr_quant_pairs(day, prior[-1].date) if p["ticker"] in stocks]
        for p in pairs:
            p["live_id"] = str(uuid5(TOSS_NEWS_NAMESPACE, p["id"]))
        judged = asyncio.run(judge(pairs, stocks))
        sched = schedule_detected(
            [(p["live_id"], p["ticker"], p["published_at"].astimezone(KST)) for p in pairs],
            day,
            delay,
            poll_every,
            gap,
        )
        events = []
        seen: set[str] = set()
        for p in sorted(pairs, key=lambda x: x["published_at"]):
            key = f"{p['live_id']}|{p['ticker']}"
            if key in seen:
                continue
            seen.add(key)
            decide, batch = sched[(p["live_id"], p["ticker"])]
            s = stocks[p["ticker"]]
            events.append(
                ReplayEvent(
                    key=key,
                    article_id=p["live_id"],
                    ticker=p["ticker"],
                    name=s.name,
                    market=s.market,
                    sector=s.sector,
                    title=p["title"],
                    category=p["category"],
                    published_at=p["published_at"].astimezone(KST),
                    decide_at=decide,
                    batch_at=batch,
                    features=judged[key],
                    source="kr_quant",
                )
            )
        dump_events(OUT / f"{day}.jsonl", events)
        print(day, "kr_quant events", len(events))
    (OUT / "meta.json").write_text(
        json.dumps(
            {
                "median_delay_s": delay.total_seconds(),
                "judge_gap_s": gap.total_seconds(),
                "poll_every_s": poll_every.total_seconds(),
                "archive_days": [str(d) for d in archive_days],
                "kr_quant_days": [str(d) for d in kq_days],
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: trader에서 실행**

Run:
```bash
export $(grep -E '^KR_QUANT_DB=' ../quant-airflow/.env | xargs)
export $(grep -E '^ANTHROPIC_API_KEY=' .env | xargs)
uv run python scripts/eval/build_replay_events.py --kr-quant-days 2026-09-07,2026-09-08,2026-09-09,2026-09-10,2026-09-11
```
Expected: 날짜별 이벤트 수 출력. 09-14 약 147, 09-15 약 127, kr_quant 날짜 각 40~70. `meta.json`의 `median_delay_s`가 수백~1,500초 범위. 판정 호출 약 280쌍(Haiku, $1 미만).

- [ ] **Step 3: 운영 판정 재현성 점검 (kr_quant 판정이 운영과 같은 조건인지)**

아카이브 날짜에서 kr_quant에도 있는 기사(09-14·15 약 65쌍)를 같은 `judge()`로 판정해 아카이브 v2와 비교하는 한 번짜리 확인:

```bash
uv run python - <<'EOF'
import json, collections
arch = {}
for d in ("2026-09-14", "2026-09-15"):
    for l in open(f"data/news_judgments/{d}.jsonl"):
        r = json.loads(l); arch.setdefault(r["key"], r["v2"])
cache = {json.loads(l)["key"]: json.loads(l)["v2"] for l in open("data/eval/cache/replay_v2.jsonl")}
both = [k for k in cache if k in arch]
same_buy = sum((arch[k]["score"] > 0.15) == (cache[k]["score"] > 0.15) for k in both)
print("shared", len(both), "same BUY decision", same_buy)
EOF
```
먼저 `--kr-quant-days 2026-09-14,2026-09-15`로 한 번 돌려 캐시를 채운 뒤(이벤트 파일은 아카이브 날짜로 다시 덮어쓰도록 그다음 원래 명령을 재실행) 위 확인을 한다.
Expected: 겹치는 쌍 60개 안팎, BUY 판단 일치율을 결과 문서에 기록. 90% 미만이면 입력 차이(summary vs content)를 조사하고 멈춘다.

- [ ] **Step 4: simnode로 복사, 커밋**

```bash
rsync -a data/eval/replay_events/ young@192.168.45.9:~/git/daytrade-it/data/eval/replay_events/
rsync -a data/news_judgments/ young@192.168.45.9:~/git/daytrade-it/data/news_judgments/
rsync -a data/auto_trader/ young@192.168.45.9:~/git/daytrade-it/data/auto_trader/
git add scripts/eval/build_replay_events.py
git commit -m "feat(eval): build replay events from the daemon archive and kr_quant articles

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: 동일성 게이트 (09-14, 09-15)

**Files:**
- Create: `scripts/eval/replay_parity.py`
- Create: `docs/research/2026-09-16-bayes-pipeline-eval.md` (게이트 절)

**Interfaces:**
- Consumes: `replay_day`, `DayConfig`, `DayResult` (Task 4); `SimBroker` (Task 2); `load_events` (Task 3)
- Produces: `load_session_bars(codes: list[str], day: dt.date) -> dict[str, list[MinuteBar]]` (kr_quant `minute_bars`, `ts` 날짜 = day, tz-naive KST), `prev_closes(codes, day) -> dict[str, Decimal]` (`daily_bars_adjusted` day 이전 마지막 종가), `LIVE_CONFIGS: dict[dt.date, DayConfig]`

그날 운영 설정(원장·커밋 시각으로 확인한 값):

| 날짜 | max_open_positions | no-chase 상한 | VWAP 청산 | 추가매수 | 비고 |
|---|---|---|---|---|---|
| 09-14 | 2 | 0.02 | 없음(None) | 없음(None) | `4ebd39d` 이전 코드 |
| 09-15 | 5 | 0.03 | gap 0.01 / +0.02 | +0.01 | `48d6cb1` 코드 |

- [ ] **Step 1: 스크립트 작성**

```python
"""Parity gate: replay live days and diff against the live ledger.

    uv run python scripts/eval/replay_parity.py --days 2026-09-14,2026-09-15   # simnode

Pass = same tickers and quantities for entries, add-ons and exits; fill price
within ±0.5% per ticker (live fills from Kiwoom ka10076 via the ledger limits).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from decimal import Decimal
from pathlib import Path

import psycopg2

from daytrade_it.application.replay.day_replay import DayConfig, replay_day
from daytrade_it.application.replay.events import load_events
from daytrade_it.application.trading.auto_trader import TradingLimits
from daytrade_it.domain.value_objects import Ticker
from daytrade_it.infrastructure.brokers.base import KST, MinuteBar
from daytrade_it.infrastructure.brokers.sim_broker import SimBroker
from daytrade_it.infrastructure.external.quant_airflow_db_client import QuantAirflowDBClient
from daytrade_it.shared.runtime_host import require_backtest_host

EVENTS = Path("data/eval/replay_events")
LIVE_LEDGER = Path("data/auto_trader")
DART = Path("../scalp-it/data/dart.db")
WORK = Path("data/eval/replay_work")

LIVE_CONFIGS = {
    dt.date(2026, 9, 14): DayConfig(
        limits=TradingLimits(max_open_positions=2, vwap_exit_gap=None, add_on_min_gain=None),
        max_pre_entry_move=Decimal("0.02"),
    ),
    dt.date(2026, 9, 15): DayConfig(limits=TradingLimits(), max_pre_entry_move=Decimal("0.03")),
}


def _conn():  # noqa: ANN202
    return psycopg2.connect(os.environ["KR_QUANT_DB"], connect_timeout=5)


def load_session_bars(codes: list[str], day: dt.date) -> dict[str, list[MinuteBar]]:
    out: dict[str, list[MinuteBar]] = {c: [] for c in codes}
    with _conn() as c, c.cursor() as cur:
        cur.execute("SET max_parallel_workers_per_gather = 0")
        cur.execute(
            "SELECT code, ts, open, high, low, close, volume FROM minute_bars "
            "WHERE code = ANY(%s) AND ts >= %s AND ts < %s ORDER BY code, ts",
            (codes, day, day + dt.timedelta(days=1)),
        )
        for code, ts, o, h, l, cl, v in cur.fetchall():
            out[code].append(
                MinuteBar(
                    ts=ts.replace(tzinfo=None),
                    open=Decimal(str(o)),
                    high=Decimal(str(h)),
                    low=Decimal(str(l)),
                    close=Decimal(str(cl)),
                    volume=int(v),
                )
            )
    return out


def prev_closes(codes: list[str], day: dt.date) -> dict[str, Decimal]:
    with _conn() as c, c.cursor() as cur:
        cur.execute(
            "SELECT DISTINCT ON (code) code, close FROM daily_bars_adjusted "
            "WHERE code = ANY(%s) AND date < %s ORDER BY code, date DESC",
            (codes, day),
        )
        return {code: Decimal(str(close)) for code, close in cur.fetchall()}


def summarize(ledger: list[dict]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {"entry": {}, "add_on": {}, "exit": {}}
    for r in ledger:
        kind = {
            "entry_submitted": "entry",
            "add_on_submitted": "add_on",
            "exit_submitted": "exit",
        }.get(r["event"])
        if kind is None:
            continue
        qty = int(r.get("quantity", 1))
        if kind == "exit" and int(r.get("attempt", 0)) > 0:
            continue  # re-priced retries of the same exit
        out[kind][r["ticker"]] = out[kind].get(r["ticker"], 0) + qty
    return out


def main() -> None:
    require_backtest_host()
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", required=True)
    args = ap.parse_args()
    stocks = QuantAirflowDBClient().get_stock_master()
    db = QuantAirflowDBClient()
    ok_all = True
    for d in args.days.split(","):
        day = dt.date.fromisoformat(d)
        events = load_events(EVENTS / f"{day}.jsonl")
        codes = sorted({e.ticker for e in events})
        bars = load_session_bars(codes, day)
        broker = SimBroker(bars, prev_closes(codes, day), lambda: dt.datetime.now(KST))
        stale_before = dt.datetime.combine(
            db.get_adjusted_daily_bars(
                Ticker(value="005930"),
                start=day - dt.timedelta(days=14),
                end=day - dt.timedelta(days=1),
            )[-1].date,
            dt.time(15, 20),
            tzinfo=KST,
        )
        res = replay_day(
            day, events, broker, db, stocks, stale_before, LIVE_CONFIGS[day], WORK / str(day), DART
        )
        live = [json.loads(l) for l in (LIVE_LEDGER / f"{day}.jsonl").open()]
        s_live, s_sim = summarize(live), summarize(res.ledger)
        same = s_live == s_sim
        ok_all &= same
        print(f"== {day} parity {'PASS' if same else 'FAIL'}")
        for kind in ("entry", "add_on", "exit"):
            print(f"  {kind:6} live={s_live[kind]} sim={s_sim[kind]}")
        buys = {f["ticker"]: f["price"] for f in res.fills if f["side"] == "buy"}
        for r in live:
            if r["event"] == "entry_submitted" and r["ticker"] in buys:
                dev = float(buys[r["ticker"]]) / float(r["limit"]) - 1
                print(
                    f"  fill {r['ticker']} live_limit={r['limit']} sim={buys[r['ticker']]} dev={dev:+.2%}"
                )
        print("  skips:", sorted({(s["ticker"], s["reason"]) for s in res.skips})[:20])
        (WORK / str(day) / "result.json").write_text(
            json.dumps(
                {"ledger": res.ledger, "skips": res.skips, "fills": res.fills},
                default=str,
                ensure_ascii=False,
                indent=1,
            )
        )
    raise SystemExit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: simnode에서 브랜치 받기와 09-15 분봉 확인**

Run:
```bash
ssh young@192.168.45.9 'cd ~/git/daytrade-it && git fetch origin && git switch feat/bayes-pipeline-opt && uv sync'
ssh young@192.168.45.9 'cd ~/git/daytrade-it && export $(grep -E "^KR_QUANT_DB=" ../quant-airflow/.env | xargs) && uv run python -c "
import os,psycopg2; c=psycopg2.connect(os.environ[\"KR_QUANT_DB\"]).cursor()
c.execute(\"select count(distinct code) from minute_bars where ts >= %s and ts < %s\", (\"2026-09-15\",\"2026-09-16\")); print(c.fetchone())"'
```
(브랜치를 먼저 trader에서 `git push -u origin feat/bayes-pipeline-opt`.)
Expected: 09-15 분봉 종목 수 > 0. 0이면 quant-airflow 적재(장 마감 후)를 기다리거나, simnode에서 `scalp-intraday --codes <09-15 이벤트 종목> --since 20260915 --no-resume --sleep 0.3`로 채운다(장 시간 금지).

- [ ] **Step 3: 게이트 실행**

Run: `ssh young@192.168.45.9 'cd ~/git/daytrade-it && export $(grep -E "^KR_QUANT_DB=" ../quant-airflow/.env | xargs) && uv run python scripts/eval/replay_parity.py --days 2026-09-14,2026-09-15'`
Expected: 두 날짜 PASS와 체결가 차이 ±0.5% 이내. 09-15 운영 원장에는 중복 데몬 탓에 기아·이마트 진입이 2번씩 기록돼 있다 — `summarize`의 entry 수량이 live 2 / sim 1로 나오면 **그 두 종목만** 사고로 표시하고 나머지가 같으면 통과로 본다(스펙: 첫 데몬 기준).

- [ ] **Step 4: FAIL 처리 규칙**

FAIL이면 최적화로 넘어가지 않는다. 차이를 한 줄씩 원인 분류해 문서에 적는다:
1. 이벤트 시각 차이(`decide_at`·`batch_at`) → Task 3 규칙 수정
2. 체결 가정 차이(호가 스프레드, TTL 내 체결) → Task 2 규칙 수정 + 테스트 추가
3. 운영 설정 차이(그날 배포 코드) → `LIVE_CONFIGS` 수정
4. 데이터 누락(분봉 없음) → 백필
수정 후 Step 3 재실행.

- [ ] **Step 5: 결과 문서 작성과 커밋**

`docs/research/2026-09-16-bayes-pipeline-eval.md`에 아래 절을 쓴다(실제 출력 값으로):

```markdown
# 뉴스 매매 파이프라인 베이즈 최적화 — 결과

스펙 `docs/superpowers/specs/2026-09-15-bayes-pipeline-optimization-design.md`, 계획 `docs/superpowers/plans/2026-09-15-replay-engine-parity.md`.

## 0단계: 동일성 게이트

| 날짜 | 진입 일치 | 추가매수 일치 | 청산 일치 | 체결가 최대 차이 | 판정 |
|---|---|---|---|---|---|
| 09-14 | (출력) | (출력) | (출력) | (출력) | PASS/FAIL |
| 09-15 | (출력) | (출력) | (출력) | (출력) | PASS/FAIL |

- kr_quant 판정 재현성: 겹치는 쌍 N, BUY 판단 일치 M (Task 5 Step 3)
- 이벤트 시각 가정: 발견 지연 중앙값 (meta.json), 판정 간격, 폴 간격
- 알려진 차이와 처리: (Step 4 분류)
```

```bash
git add scripts/eval/replay_parity.py docs/research/2026-09-16-bayes-pipeline-eval.md
git commit -m "feat(eval): replay parity gate against live ledgers; record gate result

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: 기준선 리플레이 (운영 정책, 전 거래일)

**Files:**
- Create: `scripts/eval/replay_baseline.py`
- Modify: `docs/research/2026-09-16-bayes-pipeline-eval.md` (기준선 절)

**Interfaces:**
- Consumes: `replay_day`, `DayConfig` (Task 4); `load_session_bars`, `prev_closes` (Task 6, `scripts/eval/replay_parity.py`에서 import); `load_events` (Task 3)
- Produces: `trade_pnl(fills: list[dict]) -> list[dict]` — 종목별 `{"ticker", "buy_qty", "buy_value", "sell_value", "net"}`, `net = sell_value × (1 − 0.0020 − 0.00015) − buy_value × (1 + 0.00015)`; `data/eval/replay_baseline/{day}.json`; 표준 출력에 날짜별 거래 수·순손익(원)·손실 거래 수

- [ ] **Step 1: `trade_pnl` 테스트 작성**

`tests/unit/scripts/test_replay_baseline.py`:

```python
import sys
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "eval"))

from replay_baseline import trade_pnl  # noqa: E402


def test_trade_pnl_nets_costs_per_ticker() -> None:
    fills = [
        {"ticker": "A", "side": "buy", "qty": 1, "price": Decimal(10000)},
        {"ticker": "A", "side": "buy", "qty": 1, "price": Decimal(10100)},
        {"ticker": "A", "side": "sell", "qty": 2, "price": Decimal(10200)},
    ]
    [row] = trade_pnl(fills)
    assert row["buy_qty"] == 2
    expected = Decimal(20400) * (1 - Decimal("0.00215")) - Decimal(20100) * Decimal("1.00015")
    assert row["net"] == expected
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest tests/unit/scripts/test_replay_baseline.py -v`
Expected: FAIL — `ModuleNotFoundError: replay_baseline`

- [ ] **Step 3: 구현**

```python
"""Live policy replayed over every event day -- the baseline stages 1-4 must beat.

uv run python scripts/eval/replay_baseline.py   # simnode
"""

from __future__ import annotations

import datetime as dt
import json
from decimal import Decimal
from pathlib import Path

SELL_COST = Decimal("0.00215")  # 0.20% tax + 0.015% commission
BUY_COST = Decimal("0.00015")
OUT = Path("data/eval/replay_baseline")


def trade_pnl(fills: list[dict]) -> list[dict]:
    rows: dict[str, dict] = {}
    for f in fills:
        r = rows.setdefault(
            f["ticker"],
            {
                "ticker": f["ticker"],
                "buy_qty": 0,
                "buy_value": Decimal(0),
                "sell_value": Decimal(0),
            },
        )
        value = Decimal(str(f["price"])) * int(f["qty"])
        if f["side"] == "buy":
            r["buy_qty"] += int(f["qty"])
            r["buy_value"] += value
        else:
            r["sell_value"] += value
    for r in rows.values():
        r["net"] = r["sell_value"] * (1 - SELL_COST) - r["buy_value"] * (1 + BUY_COST)
    return list(rows.values())


def main() -> None:
    from daytrade_it.application.replay.day_replay import DayConfig, replay_day  # noqa: PLC0415
    from daytrade_it.application.replay.events import load_events  # noqa: PLC0415
    from daytrade_it.application.trading.auto_trader import TradingLimits  # noqa: PLC0415
    from daytrade_it.domain.value_objects import Ticker  # noqa: PLC0415
    from daytrade_it.infrastructure.brokers.base import KST  # noqa: PLC0415
    from daytrade_it.infrastructure.brokers.sim_broker import SimBroker  # noqa: PLC0415
    from daytrade_it.infrastructure.external.quant_airflow_db_client import (  # noqa: PLC0415
        QuantAirflowDBClient,
    )
    from daytrade_it.shared.runtime_host import require_backtest_host  # noqa: PLC0415
    from replay_parity import DART, EVENTS, WORK, load_session_bars, prev_closes  # noqa: PLC0415

    require_backtest_host()
    db = QuantAirflowDBClient()
    stocks = db.get_stock_master()
    cfg = DayConfig(limits=TradingLimits(), max_pre_entry_move=Decimal("0.03"))
    OUT.mkdir(parents=True, exist_ok=True)
    for path in sorted(EVENTS.glob("2026-*.jsonl")):
        day = dt.date.fromisoformat(path.stem)
        events = load_events(path)
        codes = sorted({e.ticker for e in events})
        broker = SimBroker(
            load_session_bars(codes, day), prev_closes(codes, day), lambda: dt.datetime.now(KST)
        )
        prior = db.get_adjusted_daily_bars(
            Ticker(value="005930"),
            start=day - dt.timedelta(days=14),
            end=day - dt.timedelta(days=1),
        )[-1].date
        stale_before = dt.datetime.combine(prior, dt.time(15, 20), tzinfo=KST)
        res = replay_day(
            day, events, broker, db, stocks, stale_before, cfg, WORK / f"baseline-{day}", DART
        )
        pnl = trade_pnl(res.fills)
        total = sum((r["net"] for r in pnl), Decimal(0))
        losers = sum(r["net"] < 0 for r in pnl)
        src = events[0].source if events else "-"
        print(f"{day} [{src}] trades={len(pnl)} net={total:,.0f} losers={losers}")
        (OUT / f"{day}.json").write_text(
            json.dumps(
                {"day": str(day), "source": src, "trades": pnl, "net": total, "skips": res.skips},
                default=str,
                ensure_ascii=False,
                indent=1,
            )
        )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `uv run pytest tests/unit/scripts/test_replay_baseline.py -v`
Expected: PASS

- [ ] **Step 5: simnode 실행 (게이트 PASS 이후에만)**

Run: `ssh young@192.168.45.9 'cd ~/git/daytrade-it && git pull && export $(grep -E "^KR_QUANT_DB=" ../quant-airflow/.env | xargs) && uv run python scripts/eval/replay_baseline.py'`
Expected: 7거래일 각 한 줄. 09-15 행의 순손익이 실거래(금액 비공개)와 같은 부호·비슷한 크기(중복 데몬분 제외).

- [ ] **Step 6: 문서 기준선 절 추가, 커밋**

`docs/research/2026-09-16-bayes-pipeline-eval.md`에 추가:

```markdown
## 기준선: 운영 정책 리플레이

| 날짜 | 출처 | 거래 | 순손익(원) | 손실 거래 |
|---|---|---|---|---|
| (출력 그대로 7행) |

- 09-07~11(kr_quant)은 운영이 봤을 기사의 약 25%라 한도 경쟁이 약하다. 출처별로 나눠 읽는다.
- 1~4단계 후보는 이 표의 날짜별 순손익과 비교한다(스펙 "베이즈 방법").
```

```bash
git add scripts/eval/replay_baseline.py tests/unit/scripts/test_replay_baseline.py docs/research/2026-09-16-bayes-pipeline-eval.md
git commit -m "feat(eval): baseline replay of the live policy over all event days

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
git push
```

---

## 다음 계획 (이 계획 범위 밖)

게이트 PASS와 기준선이 나오면 `docs/superpowers/plans/2026-09-XX-bayes-stages.md`를 따로 쓴다: 1단계 프롬프트 후보 판정(캐시·예산 $20), 2~3단계 Optuna 탐색을 `replay_day`에 `DayConfig`·필터 파라미터로 연결, 4단계 계층 t 모형 판정. 게이트 결과(특히 체결 가정 수정)가 그 설계를 바꾸기 때문에 지금 쓰지 않는다.
