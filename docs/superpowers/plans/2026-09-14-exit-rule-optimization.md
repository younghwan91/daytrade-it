# 청산·진입 규칙 최적화 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 뉴스 BUY 표본의 1분봉 경로로 청산 규칙(1단계)과 매수가·추가매수가(2단계)를 비교해, 15:00 일괄 매도 기준선보다 일관되게 나은 규칙이 있는지 문서로 결론낸다.

**Architecture:** 순수 함수 시뮬레이터(`application/trading/exit_rules.py`)가 분봉 리스트와 규칙을 받아 결과를 낸다. 표본 구성(`scripts/eval/exit_samples.py`)과 평가(`scripts/eval/eval_exit_rules.py`)는 DB를 읽는 얇은 스크립트다. 분봉은 scalp-it의 기존 수집기 `scalp-intraday`(ka10080 → kr_quant `minute_bars`)로 채운다. 평가는 simnode에서만 실행한다.

**Tech Stack:** Python 3.12, uv, pytest, psycopg2, kr_quant TimescaleDB(primary 192.168.45.9:5433), gptquant Postgres.

**Spec:** `docs/superpowers/specs/2026-09-14-exit-rule-optimization-design.md`

## Global Constraints

- 평가 스크립트는 simnode에서만: 진입점 첫 줄에서 `require_backtest_host()` (`daytrade_it.shared.runtime_host`).
- 키움 호출(백필)은 **장 시간 08:30~15:40 금지**. scalp-it 실거래와 토큰 공유. 조회 TR만.
- 운영 코드(`auto_trader.py`, `execute_trade`)는 수정하지 않는다.
- 비용: 매도세 0.20%, 수수료 편도 0.015%.
- 체결: 종가 기준 발동 → 다음 분봉 시가. 한 봉에서 손절·익절 동시 → 손절. 15:00 이후 첫 봉 시가에 강제 청산.
- dev = 거래일 ≤ 2026-09-08, test = 2026-09-09~09-11. test는 규칙별 파라미터를 문서에 고정한 뒤 한 번만 본다.
- 선택 기준: 기준선 대비 쌍별 순수익률 차이의 중앙값. 채택: 운영·확장 세트 test 모두 중앙값 > 0 이고 부트스트랩 95% CI 하한 ≥ −0.5%p.
- 2단계 시작 조건: 운영 세트 dev·test 각각 분봉 있는 쌍 ≥ 40.
- 운영 BUY 판정 = v2 점수(`drop_mention=True`) ≥ 0.15 (`SIGNAL_THRESHOLD`), 운영 진입 필터 = 기사 전 상승 ≤ +2%.

## File Structure

| 파일 | 책임 |
|---|---|
| `src/daytrade_it/application/trading/exit_rules.py` (신규) | `Bar`, 규칙 클래스, `simulate_exit`, `simulate_position`, 비용 계산. DB·IO 없음 |
| `tests/unit/application/trading/test_exit_rules.py` (신규) | 시뮬레이터 단위 테스트 |
| `scripts/eval/exit_samples.py` (신규) | 평가셋 + v2 캐시 + gptquant 신호 → `data/eval/exit_samples.jsonl`, 백필 대상 목록 |
| `tests/unit/scripts/test_exit_samples.py` (신규) | 진입 시각 계산·세트 플래그 테스트 |
| `scripts/eval/eval_exit_rules.py` (신규) | 분봉 로드 → 격자 시뮬 → dev 선택 → test 1회 → 표 출력 |
| `docs/research/2026-09-XX-exit-rule-eval.md` (신규) | 결과 문서 |

---

### Task 1: 청산 시뮬레이터 코어

**Files:**
- Create: `src/daytrade_it/application/trading/exit_rules.py`
- Test: `tests/unit/application/trading/test_exit_rules.py`

**Interfaces:**
- Produces:
  - `Bar(ts: datetime, open: float, high: float, low: float, close: float, volume: int)` (frozen dataclass)
  - `ExitResult(exit_ts: datetime, exit_price: float, reason: str, net_return: float)`
  - 규칙 클래스 `Baseline()`, `Trailing(pct: float)`, `StopTake(stop_pct: float, take_pct: float | None)`, `TimeExit(minutes: int)`, `MomentumFade(min_gain_pct: float, window: int)` — 모두 `name: str` 속성
  - `simulate_exit(bars: Sequence[Bar], entry_idx: int, entry_price: float, rule: ExitRule, flatten_at: time = time(15, 0)) -> ExitResult`
  - `net_return(entry_price: float, exit_price: float) -> float`
  - 상수 `SELL_TAX = 0.002`, `FEE = 0.00015`

- [ ] **Step 1: 실패하는 테스트 작성**

```python
"""exit_rules 시뮬레이터 — 체결 가정은 스펙 '시뮬레이터' 절."""

from datetime import datetime, time

import pytest

from daytrade_it.application.trading.exit_rules import (
    Bar,
    Baseline,
    MomentumFade,
    StopTake,
    TimeExit,
    Trailing,
    net_return,
    simulate_exit,
)


def bars_from(closes: list[float], start: str = "2026-09-01T09:00", vols: list[int] | None = None) -> list[Bar]:
    t0 = datetime.fromisoformat(start)
    out = []
    for i, c in enumerate(closes):
        prev = closes[i - 1] if i else c
        out.append(
            Bar(
                ts=t0.replace(hour=t0.hour + (t0.minute + i) // 60, minute=(t0.minute + i) % 60),
                open=prev,
                high=max(prev, c),
                low=min(prev, c),
                close=c,
                volume=(vols[i] if vols else 1000),
            )
        )
    return out


def test_net_return_includes_tax_and_fees() -> None:
    assert net_return(100.0, 100.0) == pytest.approx((1 - 0.002 - 0.00015) / (1 + 0.00015) - 1)


def test_baseline_exits_at_first_bar_at_or_after_flatten() -> None:
    bars = bars_from([100.0] * 10, start="2026-09-01T14:55")
    r = simulate_exit(bars, 0, 100.0, Baseline())
    assert r.exit_ts.time() == time(15, 0)
    assert r.reason == "flatten"


def test_data_end_before_flatten_exits_at_last_close() -> None:
    bars = bars_from([100.0, 101.0, 102.0])
    r = simulate_exit(bars, 0, 100.0, Baseline())
    assert (r.exit_price, r.reason) == (102.0, "data_end")


def test_trailing_triggers_on_close_and_fills_next_open() -> None:
    # 고점 110, 종가 107 ≤ 110×0.98 → 다음 봉 시가(=107)
    bars = bars_from([100.0, 110.0, 107.0, 90.0])
    r = simulate_exit(bars, 0, 100.0, Trailing(0.02))
    assert r.reason == "trailing"
    assert r.exit_ts == bars[3].ts
    assert r.exit_price == bars[3].open


def test_stop_and_take_in_same_bar_takes_stop() -> None:
    b = Bar(datetime(2026, 9, 1, 9, 1), 100.0, 106.0, 96.0, 100.0, 1000)
    bars = [Bar(datetime(2026, 9, 1, 9, 0), 100.0, 100.0, 100.0, 100.0, 1000), b]
    r = simulate_exit(bars, 0, 100.0, StopTake(0.03, 0.05))
    assert (r.reason, r.exit_price) == ("stop", 97.0)


def test_stop_gap_down_fills_at_open() -> None:
    bars = [
        Bar(datetime(2026, 9, 1, 9, 0), 100.0, 100.0, 100.0, 100.0, 1000),
        Bar(datetime(2026, 9, 1, 9, 1), 90.0, 91.0, 89.0, 90.0, 1000),
    ]
    assert simulate_exit(bars, 0, 100.0, StopTake(0.03, None)).exit_price == 90.0


def test_take_profit_fills_at_target() -> None:
    bars = bars_from([100.0, 106.0, 106.0])
    r = simulate_exit(bars, 0, 100.0, StopTake(0.03, 0.05))
    assert (r.reason, r.exit_price) == ("take", 105.0)


def test_time_exit_after_n_minutes() -> None:
    bars = bars_from([100.0] * 40)
    r = simulate_exit(bars, 0, 100.0, TimeExit(30))
    assert (r.reason, r.exit_ts) == ("time", bars[30].ts)


def test_momentum_fade_needs_gain_stall_and_volume_drop() -> None:
    closes = [100.0, 102.0, 104.0] + [103.0] * 6
    vols = [1000, 1000, 1000, 1000, 1000, 1000, 100, 100, 100]
    bars = bars_from(closes, vols=vols)
    r = simulate_exit(bars, 0, 100.0, MomentumFade(0.03, 3))
    # 고점 bar2(104). bar7 종가 시점: 고점 뒤 5봉 신고가 없음, 최근 3봉 거래량 1200 < 0.5×3000 → bar8 시가
    assert (r.reason, r.exit_ts) == ("fade", bars[8].ts)


def test_momentum_fade_not_triggered_without_min_gain() -> None:
    closes = [100.0, 101.0] + [100.5] * 10
    vols = [1000] * 6 + [10] * 6
    r = simulate_exit(bars_from(closes, vols=vols), 0, 100.0, MomentumFade(0.03, 3))
    assert r.reason == "data_end"
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest tests/unit/application/trading/test_exit_rules.py -q -p no:cacheprovider`
Expected: FAIL — `ModuleNotFoundError: daytrade_it.application.trading.exit_rules`

- [ ] **Step 3: 구현**

```python
"""청산 규칙 시뮬레이터 — 1분봉 경로 위에서 규칙을 평가한다.

스펙: docs/superpowers/specs/2026-09-14-exit-rule-optimization-design.md.
체결 가정은 보수적이다: 종가 조건은 다음 봉 시가에 팔고, 가격 조건(손절·익절)은
그 봉 안에서 체결하되 둘 다 걸리면 손절로 본다. 호가 공백은 반영하지 못한다.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, time
from typing import Protocol

SELL_TAX = 0.002
FEE = 0.00015


@dataclass(frozen=True)
class Bar:
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int


@dataclass(frozen=True)
class ExitResult:
    exit_ts: datetime
    exit_price: float
    reason: str
    net_return: float


def net_return(entry_price: float, exit_price: float) -> float:
    return exit_price * (1 - SELL_TAX - FEE) / (entry_price * (1 + FEE)) - 1


@dataclass(frozen=True)
class PriceHit:
    price: float
    reason: str


class ExitRule(Protocol):
    name: str

    def intrabar(self, bar: Bar, entry_price: float) -> PriceHit | None: ...

    def on_close(self, bars: Sequence[Bar], entry_idx: int, i: int, entry_price: float) -> str | None: ...


class _NoIntrabar:
    def intrabar(self, bar: Bar, entry_price: float) -> PriceHit | None:  # noqa: ARG002
        return None


class _NoClose:
    def on_close(self, bars: Sequence[Bar], entry_idx: int, i: int, entry_price: float) -> str | None:  # noqa: ARG002
        return None


@dataclass(frozen=True)
class Baseline(_NoIntrabar, _NoClose):
    name: str = "B"


@dataclass(frozen=True)
class Trailing(_NoIntrabar):
    pct: float

    @property
    def name(self) -> str:
        return f"T{self.pct:g}"

    def on_close(self, bars: Sequence[Bar], entry_idx: int, i: int, entry_price: float) -> str | None:  # noqa: ARG002
        peak = max(b.high for b in bars[entry_idx : i + 1])
        return "trailing" if bars[i].close <= peak * (1 - self.pct) else None


@dataclass(frozen=True)
class StopTake(_NoClose):
    stop_pct: float
    take_pct: float | None

    @property
    def name(self) -> str:
        return f"SL{self.stop_pct:g}/TP{self.take_pct if self.take_pct is not None else '-'}"

    def intrabar(self, bar: Bar, entry_price: float) -> PriceHit | None:
        stop = entry_price * (1 - self.stop_pct)
        if bar.low <= stop:
            return PriceHit(min(bar.open, stop), "stop")
        if self.take_pct is not None:
            take = entry_price * (1 + self.take_pct)
            if bar.high >= take:
                return PriceHit(max(bar.open, take), "take")
        return None


@dataclass(frozen=True)
class TimeExit(_NoIntrabar):
    minutes: int

    @property
    def name(self) -> str:
        return f"H{self.minutes}"

    def on_close(self, bars: Sequence[Bar], entry_idx: int, i: int, entry_price: float) -> str | None:  # noqa: ARG002
        return "time" if i - entry_idx + 1 >= self.minutes else None


@dataclass(frozen=True)
class MomentumFade(_NoIntrabar):
    min_gain_pct: float
    window: int

    @property
    def name(self) -> str:
        return f"M{self.min_gain_pct:g}/{self.window}"

    def on_close(self, bars: Sequence[Bar], entry_idx: int, i: int, entry_price: float) -> str | None:
        span = bars[entry_idx : i + 1]
        if len(span) < 2 * self.window:
            return None
        peak_off = max(range(len(span)), key=lambda k: span[k].high)
        if span[peak_off].high < entry_price * (1 + self.min_gain_pct):
            return None
        if len(span) - 1 - peak_off < self.window:
            return None
        recent = sum(b.volume for b in span[-self.window :])
        prior = sum(b.volume for b in span[-2 * self.window : -self.window])
        return "fade" if recent < 0.5 * prior else None


def simulate_exit(
    bars: Sequence[Bar],
    entry_idx: int,
    entry_price: float,
    rule: ExitRule,
    flatten_at: time = time(15, 0),
) -> ExitResult:
    pending: str | None = None
    for i in range(entry_idx, len(bars)):
        bar = bars[i]
        if pending is not None:
            return ExitResult(bar.ts, bar.open, pending, net_return(entry_price, bar.open))
        if bar.ts.time() >= flatten_at:
            return ExitResult(bar.ts, bar.open, "flatten", net_return(entry_price, bar.open))
        # 진입 봉은 시가에 샀으므로 그 봉의 가격 조건부터 본다.
        hit = rule.intrabar(bar, entry_price)
        if hit is not None:
            return ExitResult(bar.ts, hit.price, hit.reason, net_return(entry_price, hit.price))
        pending = rule.on_close(bars, entry_idx, i, entry_price)
    last = bars[-1]
    return ExitResult(last.ts, last.close, "data_end", net_return(entry_price, last.close))
```

주의: 테스트 헬퍼는 `high=max(직전 종가, 종가)`라 bar3의 high도 104다. `max(range, key=...)`는 동률에서 첫 번째(index 2)를 고른다.

- [ ] **Step 4: 통과 확인**

Run: `uv run pytest tests/unit/application/trading/test_exit_rules.py -q -p no:cacheprovider`
Expected: 10 passed

- [ ] **Step 5: 커밋**

```bash
git add src/daytrade_it/application/trading/exit_rules.py tests/unit/application/trading/test_exit_rules.py
git commit -m "feat(eval): minute-bar exit rule simulator"
```

---

### Task 2: 표본 구성과 백필 대상 목록

**Files:**
- Create: `scripts/eval/exit_samples.py`
- Test: `tests/unit/scripts/test_exit_samples.py`

**Interfaces:**
- Consumes: `data/eval/news_eval_set.jsonl` (필드 `article_id, ticker, session, published_kst, trade_day, pre_news_move, market_median_oc`), `data/eval/cache/v2__claude-haiku-4-5-20251001.jsonl` (`key = "{article_id}|{ticker}"`), `scripts/eval/eval_news_prompts.py::v2_score`, gptquant `trading_signals` (`direction='BUY'`, `created_at`, `ticker`, `metadata_json.pre_entry_move`)
- Produces: `data/eval/exit_samples.jsonl` 한 줄 = `{"key", "ticker", "trade_day": "YYYY-MM-DD", "entry_ts": "YYYY-MM-DDTHH:MM", "split": "dev"|"test"|"forward", "ops": bool, "wide": bool, "market_median_oc": float|null}`; 표준출력에 `code,YYYYMMDD` 백필 대상. 함수 `entry_minute(session: str, published_kst: str, trade_day: str) -> str`, `split_of(trade_day: str) -> str`.

- [ ] **Step 1: 실패하는 테스트**

```python
import importlib.util
from pathlib import Path

_path = Path(__file__).resolve().parents[3] / "scripts" / "eval" / "exit_samples.py"
_spec = importlib.util.spec_from_file_location("exit_samples", _path)
es = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(es)


def test_pre_market_and_after_close_enter_at_open() -> None:
    assert es.entry_minute("pre_market", "2026-09-02T07:10:00", "2026-09-02") == "2026-09-02T09:00"
    assert es.entry_minute("after_close", "2026-09-01T17:00:00", "2026-09-02") == "2026-09-02T09:00"


def test_intraday_enters_minute_after_publish_plus_60s() -> None:
    assert es.entry_minute("intraday", "2026-09-01T11:30:46", "2026-09-01") == "2026-09-01T11:32"
    assert es.entry_minute("intraday", "2026-09-01T11:30:00", "2026-09-01") == "2026-09-01T11:31"


def test_split() -> None:
    assert es.split_of("2026-09-08") == "dev"
    assert es.split_of("2026-09-09") == "test"
    assert es.split_of("2026-09-14") == "forward"
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest tests/unit/scripts/test_exit_samples.py -q -p no:cacheprovider`
Expected: FAIL — `FileNotFoundError` (스크립트 없음)

- [ ] **Step 3: 구현**

```python
"""청산 규칙 평가 표본 — 평가셋(09-01~11) + 운영 신호(09-14~).

    uv run python scripts/eval/exit_samples.py > data/eval/exit_backfill_targets.csv

진입 시각은 프롬프트 평가와 같다(docs/research/2026-09-13-news-prompt-eval.md):
장전·장후 기사는 거래일 09:00, 장중은 발표 +60초가 속한 분의 다음 분.
운영 신호는 (종목, 거래일)당 첫 신호만 쓴다.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
import os
from pathlib import Path

import psycopg2

EVAL_SET = Path("data/eval/news_eval_set.jsonl")
V2_CACHE = Path("data/eval/cache/v2__claude-haiku-4-5-20251001.jsonl")
OUT = Path("data/eval/exit_samples.jsonl")
DEV_END = dt.date(2026, 9, 8)
TEST_END = dt.date(2026, 9, 11)
BUY_THRESHOLD = 0.15
MAX_PRE_MOVE = 0.02
KST = dt.timezone(dt.timedelta(hours=9))

_p = Path(__file__).with_name("eval_news_prompts.py")
_s = importlib.util.spec_from_file_location("eval_news_prompts", _p)
_enp = importlib.util.module_from_spec(_s)
_s.loader.exec_module(_enp)


def entry_minute(session: str, published_kst: str, trade_day: str) -> str:
    if session != "intraday":
        return f"{trade_day}T09:00"
    t = dt.datetime.fromisoformat(published_kst) + dt.timedelta(seconds=60)
    if t.second or t.microsecond:  # 발표+60초 이후 첫 틱이 속한 분 = 올림
        t = t.replace(second=0, microsecond=0) + dt.timedelta(minutes=1)
    return t.strftime("%Y-%m-%dT%H:%M")


def split_of(trade_day: str) -> str:
    d = dt.date.fromisoformat(trade_day)
    return "dev" if d <= DEV_END else "test" if d <= TEST_END else "forward"


def eval_rows() -> list[dict]:
    cache = {}
    for line in V2_CACHE.open():
        rec = json.loads(line)
        cache[rec["key"]] = rec["judgment"]
    out = []
    for line in EVAL_SET.open():
        r = json.loads(line)
        key = f"{r['article_id']}|{r['ticker']}"
        j = cache.get(key)
        score = _enp.v2_score(j, drop_mention=True, drop_price_move=False, use_surprise=False) if j else 0.0
        wide = r["pre_news_move"] <= MAX_PRE_MOVE
        out.append(
            {
                "key": key,
                "ticker": r["ticker"],
                "trade_day": r["trade_day"],
                "entry_ts": entry_minute(r["session"], r["published_kst"], r["trade_day"]),
                "split": split_of(r["trade_day"]),
                "ops": wide and score >= BUY_THRESHOLD,
                "wide": wide,
                "market_median_oc": r["market_median_oc"],
            }
        )
    return out


def forward_rows(dsn: str) -> list[dict]:
    with psycopg2.connect(dsn, connect_timeout=5) as c, c.cursor() as cur:
        cur.execute(
            "select id::text, ticker, created_at from trading_signals "
            "where direction = 'BUY' and created_at >= '2026-09-14' order by created_at"
        )
        rows = cur.fetchall()
    seen: set[tuple[str, str]] = set()
    out = []
    for sid, ticker, created in rows:
        local = created.astimezone(KST)
        day = local.date().isoformat()
        if (ticker, day) in seen:
            continue
        seen.add((ticker, day))
        t = local.replace(second=0, microsecond=0, tzinfo=None) + dt.timedelta(minutes=1)
        out.append(
            {
                "key": f"signal:{sid}",
                "ticker": ticker,
                "trade_day": day,
                "entry_ts": t.strftime("%Y-%m-%dT%H:%M"),
                "split": "forward",
                "ops": True,
                "wide": True,
                "market_median_oc": None,
            }
        )
    return out


def main() -> None:
    dsn = os.environ["GPTQ_DATABASE_URL"].replace("postgresql+asyncpg", "postgresql")
    rows = eval_rows() + forward_rows(dsn)
    with OUT.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for code, day in sorted({(r["ticker"], r["trade_day"]) for r in rows if r["wide"]}):
        print(f"{code},{day.replace('-', '')}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 통과 확인 + 실데이터 실행**

Run: `uv run pytest tests/unit/scripts/test_exit_samples.py -q -p no:cacheprovider`
Expected: 3 passed

Run (trader 또는 simnode, DB 읽기만):
```bash
export GPTQ_DATABASE_URL=$(grep '^GPTQ_DATABASE_URL=' .env | cut -d= -f2- | tr -d '"')
uv run python scripts/eval/exit_samples.py > data/eval/exit_backfill_targets.csv
uv run python -c "import json;r=[json.loads(l) for l in open('data/eval/exit_samples.jsonl')];import collections;print(collections.Counter((x['split'],x['ops'],x['wide']) for x in r))"
```
Expected: ops=True 이면서 split dev+test 합 ≈ 51, wide=True dev+test 합 = 123, forward ≥ 1.

- [ ] **Step 5: 커밋**

```bash
git add scripts/eval/exit_samples.py tests/unit/scripts/test_exit_samples.py
git commit -m "feat(eval): exit rule sample set and minute backfill targets"
```

---

### Task 3: simnode 준비와 분봉 백필 (운영 작업, 코드 없음)

**Files:** 없음 (simnode 셸 작업). 결과는 Task 4 문서에 기록.

**Interfaces:**
- Consumes: `data/eval/exit_backfill_targets.csv` (Task 2), simnode `~/git/scalp-it`의 `scalp-intraday` (`--codes`, `--since YYYYMMDD`, `--no-resume`, `--sleep`)
- Produces: kr_quant `minute_bars`에 대상 (종목, 거래일) 분봉; simnode `~/git/daytrade-it` clone

- [ ] **Step 1: 장 시간이 아닌지 확인** — `date +%H%M`이 0830~1540 사이면 중단하고 15:40 이후로 미룬다.

- [ ] **Step 2: simnode에 daytrade-it clone과 의존성**

```bash
ORIGIN=$(git -C ~/git/daytrade-it remote get-url origin)   # trader에서 평가
ssh simnode-local "cd ~/git && { [ -d daytrade-it ] || git clone $ORIGIN; } && cd daytrade-it && git pull --ff-only && uv sync"
ssh simnode-local 'cd ~/git/daytrade-it && uv run python -c "from daytrade_it.shared.runtime_host import require_backtest_host; require_backtest_host(); print(\"host ok\")"'
```
Expected: `host ok`. `.env`(GPTQ_DATABASE_URL)가 필요하면 trader의 값을 scp로 복사하지 말고 simnode `quant-airflow/.env`의 `KR_QUANT_DB`만 쓴다 — 평가 단계는 kr_quant만 읽는다. `data/eval/`은 git에 없으므로 `rsync -a data/eval/news_eval_set.jsonl data/eval/exit_samples.jsonl data/eval/exit_backfill_targets.csv simnode-local:~/git/daytrade-it/data/eval/` 로 복사한다.

- [ ] **Step 3: 이미 분봉이 충분한 대상은 제외**

```bash
ssh simnode-local 'cd ~/git/daytrade-it && export $(grep -E "^KR_QUANT_DB=" ../quant-airflow/.env | xargs) && uv run python - <<PY
import csv, os, psycopg2
t = [tuple(r) for r in csv.reader(open("data/eval/exit_backfill_targets.csv"))]
c = psycopg2.connect(os.environ["KR_QUANT_DB"]); cur = c.cursor()
need = set()
for code, day in t:
    cur.execute("select count(*) from minute_bars where code=%s and ts::date=to_date(%s,%s)", (code, day, "YYYYMMDD"))
    if cur.fetchone()[0] < 300:
        need.add(code)
print(",".join(sorted(need)))
PY' > data/eval/exit_backfill_codes.txt; wc -c data/eval/exit_backfill_codes.txt
```

- [ ] **Step 4: 백필 실행 (simnode, 백그라운드, 로그 남김)**

```bash
ssh simnode-local 'cd ~/git/scalp-it && export $(grep -E "^(KR_QUANT_DB|KIWOOM_APP_KEY|KIWOOM_APP_SECRET)=" ../quant-airflow/.env | xargs) && nohup uv run scalp-intraday --codes "'"$(cat data/eval/exit_backfill_codes.txt)"'" --since 20260831 --no-resume --sleep 0.3 > ~/git/daytrade-it/data/minute_backfill_exit_$(date +%Y%m%d).log 2>&1 &'
```
`scalp-intraday`가 쓰는 환경변수 이름은 실행 전에 `grep -n "environ" ~/git/scalp-it/src/scalp_it/cli_intraday.py`로 확인하고 맞춘다. sleep 0.3s는 09-13 백필과 같은 값이다.

- [ ] **Step 5: 커버리지 확인** — Step 3 쿼리를 대상 전체에 다시 돌려 `(ops, split)`별로 분봉 ≥ 300인 쌍 수를 센다. 결과 표를 Task 4 결과 문서의 "데이터" 절에 넣는다. 운영 세트 dev·test 각 쌍 수를 기록한다(2단계 시작 조건 판단용).

---

### Task 4: 1단계 평가 스크립트와 결과 문서

**Files:**
- Create: `scripts/eval/eval_exit_rules.py`
- Create: `docs/research/2026-09-XX-exit-rule-eval.md` (XX = 실행일)

**Interfaces:**
- Consumes: `exit_rules.Bar, Baseline, Trailing, StopTake, TimeExit, MomentumFade, simulate_exit` (Task 1), `data/eval/exit_samples.jsonl` (Task 2), kr_quant `minute_bars(code, ts, open, high, low, close, volume)`
- Produces: 표준출력 표 + `data/eval/exit_results.jsonl` (`{"key","rule","split","ops","wide","net","base_net","reason"}`)

- [ ] **Step 1: 스크립트 작성**

```python
"""청산 규칙 1단계 평가 — simnode 전용.

    uv run python scripts/eval/eval_exit_rules.py --phase dev
    uv run python scripts/eval/eval_exit_rules.py --phase test --frozen "T0.03,SL0.03/TP-,H120,M0.03/30"

dev에서 규칙별 파라미터를 고르고, 그 이름을 문서에 적은 뒤 test를 한 번만 본다.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import random
import statistics
from collections import defaultdict
from pathlib import Path

import psycopg2

from daytrade_it.application.trading.exit_rules import (
    Bar,
    Baseline,
    MomentumFade,
    StopTake,
    TimeExit,
    Trailing,
    simulate_exit,
)
from daytrade_it.shared.runtime_host import require_backtest_host

SAMPLES = Path("data/eval/exit_samples.jsonl")
RESULTS = Path("data/eval/exit_results.jsonl")
MIN_BARS = 300
MAX_GAP_RATIO = 0.10

GRID = {
    "T": [Trailing(x) for x in (0.01, 0.02, 0.03, 0.05)],
    "SL/TP": [StopTake(y, z) for y in (0.02, 0.03, 0.05) for z in (0.03, 0.05, 0.08, None)],
    "H": [TimeExit(n) for n in (30, 60, 120, 240)],
    "M": [MomentumFade(a, k) for a in (0.01, 0.03) for k in (15, 30, 60)],
}


def load_bars(cur, code: str, day: str) -> list[Bar]:
    cur.execute(
        "select ts, open, high, low, close, volume from minute_bars "
        "where code = %s and ts >= %s::date + time '09:00' and ts < %s::date + time '15:31' order by ts",
        (code, day, day),
    )
    return [Bar(ts, float(o), float(h), float(lo), float(c), int(v)) for ts, o, h, lo, c, v in cur.fetchall()]


def entry_index(bars: list[Bar], entry_ts: str) -> int | None:
    t = dt.datetime.fromisoformat(entry_ts)
    for i, b in enumerate(bars):
        if b.ts.replace(tzinfo=None) >= t:
            return i
    return None


def usable(bars: list[Bar], i: int) -> bool:
    if len(bars) < MIN_BARS:
        return False
    start = bars[i].ts.replace(tzinfo=None)
    end = start.replace(hour=15, minute=0)
    expected = max(int((end - start).total_seconds() // 60), 1)
    have = sum(1 for b in bars[i:] if b.ts.replace(tzinfo=None) < end)
    return have >= expected * (1 - MAX_GAP_RATIO)


def boot_ci(vals: list[float], n: int = 2000) -> tuple[float, float]:
    rng = random.Random(0)
    meds = sorted(statistics.median(rng.choices(vals, k=len(vals))) for _ in range(n))
    return meds[int(0.025 * n)], meds[int(0.975 * n)]


def winsor_mean(vals: list[float], cap: float = 0.10) -> float:
    return statistics.fmean(max(-cap, min(cap, v)) for v in vals)


def run(phase: str, frozen: list[str] | None) -> None:
    require_backtest_host()
    samples = [json.loads(line) for line in SAMPLES.open()]
    samples = [s for s in samples if s["split"] == phase and s["wide"]]
    rules = [Baseline()] + [r for rs in GRID.values() for r in rs]
    if frozen is not None:
        rules = [Baseline()] + [r for r in rules if r.name in frozen]
    rows = []
    skipped = 0
    with psycopg2.connect(os.environ["KR_QUANT_DB"], connect_timeout=5) as c, c.cursor() as cur:
        for s in samples:
            bars = load_bars(cur, s["ticker"], s["trade_day"])
            i = entry_index(bars, s["entry_ts"]) if bars else None
            if i is None or not usable(bars, i):
                skipped += 1
                continue
            base = simulate_exit(bars, i, bars[i].open, Baseline())
            for rule in rules:
                r = simulate_exit(bars, i, bars[i].open, rule)
                rows.append({**s, "rule": rule.name, "net": r.net_return, "base_net": base.net_return, "reason": r.reason})
    with RESULTS.open("a") as f:
        for r in rows:
            f.write(json.dumps({k: r[k] for k in ("key", "rule", "split", "ops", "wide", "net", "base_net", "reason")}) + "\n")
    print(f"phase={phase} samples={len(samples)} skipped(no/gappy bars)={skipped}")
    for subset in ("ops", "wide"):
        by_rule: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            if r[subset]:
                by_rule[r["rule"]].append(r)
        print(f"\n[{subset}]  rule | n | mean | median | hit | winsor | Δmedian vs B | Δ 95% CI")
        for name, rs in sorted(by_rule.items(), key=lambda kv: -statistics.median(x["net"] - x["base_net"] for x in kv[1])):
            nets = [x["net"] for x in rs]
            diffs = [x["net"] - x["base_net"] for x in rs]
            lo, hi = boot_ci(diffs)
            print(
                f"{name:>14} | {len(rs):3d} | {statistics.fmean(nets):+.2%} | {statistics.median(nets):+.2%} | "
                f"{sum(n > 0 for n in nets) / len(nets):.0%} | {winsor_mean(nets):+.2%} | "
                f"{statistics.median(diffs):+.2%} | [{lo:+.2%}, {hi:+.2%}]"
            )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["dev", "test", "forward"], required=True)
    ap.add_argument("--frozen", default=None, help="test/forward: 쉼표 구분 규칙 이름")
    a = ap.parse_args()
    frozen = a.frozen.split(",") if a.frozen else None
    if a.phase != "dev" and frozen is None:
        ap.error("test/forward는 --frozen 필수 (dev에서 고른 규칙만 본다)")
    run(a.phase, frozen)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 로컬에서 가드 동작 확인**

Run (trader): `uv run python scripts/eval/eval_exit_rules.py --phase dev`
Expected: `require_backtest_host`가 호스트 오류를 내며 종료(분봉 조회 전).

- [ ] **Step 3: 커밋 후 simnode에서 dev 실행**

```bash
git add scripts/eval/eval_exit_rules.py && git commit -m "feat(eval): exit rule grid evaluation (simnode-only)" && git push
ssh simnode-local 'cd ~/git/daytrade-it && git pull --ff-only && export $(grep -E "^KR_QUANT_DB=" ../quant-airflow/.env | xargs) && rm -f data/eval/exit_results.jsonl && uv run python scripts/eval/eval_exit_rules.py --phase dev'
```
Expected: `[ops]`, `[wide]` 표 두 개. 규칙군(T, SL/TP, H, M)마다 **`[ops]` 표에서 Δmedian이 가장 큰 파라미터 하나**를 고른다. 동률이면 `[wide]` Δmedian이 큰 쪽.

- [ ] **Step 4: 결과 문서에 dev 표와 고른 4개 이름을 먼저 기록하고 커밋** (test 실행 전)

`docs/research/2026-09-XX-exit-rule-eval.md` 구성: 결론(비워둔 채 커밋하지 말고 "test 전 고정" 절만 채운다) / 데이터(Task 3 Step 5 커버리지 표) / dev 표 전문 / **test 전 고정한 규칙 4개** / 체결 가정 요약.

```bash
git add docs/research/2026-09-*-exit-rule-eval.md && git commit -m "docs(research): exit rule dev results, freeze candidates before test"
```

- [ ] **Step 5: test 한 번 실행**

```bash
ssh simnode-local 'cd ~/git/daytrade-it && export $(grep -E "^KR_QUANT_DB=" ../quant-airflow/.env | xargs) && uv run python scripts/eval/eval_exit_rules.py --phase test --frozen "<Step 3에서 고른 4개>"'
```

- [ ] **Step 6: 채택 판정과 문서 완성** — Global Constraints의 채택 조건을 `[ops]`, `[wide]` test 표에 적용한다. 결론 절에 채택 규칙(또는 "15:00 유지")을 쓰고, forward 실행 방법(`--phase forward --frozen ...`)과 "전진 30건 전 운영 반영 금지"를 적는다. 운영 반영은 사용자 결정으로 남긴다.

```bash
git add docs/research/2026-09-*-exit-rule-eval.md && git commit -m "docs(research): exit rule test result and decision" && git push
```

---

### Task 5: 2단계 — 매수가·추가매수가 (조건부)

**시작 조건:** Task 3 Step 5에서 운영 세트 dev·test 각각 분봉 있는 쌍 ≥ 40. 아니면 이 태스크는 건너뛰고 결과 문서에 "표본 부족으로 보류, 전진 세트 누적 후 재시도"와 실제 쌍 수를 기록한다.

**Files:**
- Modify: `src/daytrade_it/application/trading/exit_rules.py` (파일 끝에 추가)
- Modify: `tests/unit/application/trading/test_exit_rules.py` (끝에 추가)
- Modify: `scripts/eval/eval_exit_rules.py` (`--stage entry` 추가)

**Interfaces:**
- Consumes: Task 1의 `Bar`, `ExitRule`, `simulate_exit`, `SELL_TAX`, `FEE`; Task 4에서 채택한 청산 규칙(없으면 `Baseline()`)
- Produces: `EntryPolicy(discount: float, ttl: int)`, `AddPolicy(drop: float | None)`, `PositionResult(filled: bool, shares: int, avg_cost: float, net_return: float, reason: str)`, `simulate_position(bars, signal_idx, ref_price, entry: EntryPolicy, add: AddPolicy, rule: ExitRule) -> PositionResult`

- [ ] **Step 1: 실패하는 테스트 추가**

```python
from daytrade_it.application.trading.exit_rules import AddPolicy, EntryPolicy, simulate_position


def test_unfilled_entry_returns_zero() -> None:
    bars = bars_from([100.0] * 20)
    r = simulate_position(bars, 0, 100.0, EntryPolicy(0.02, 5), AddPolicy(None), Baseline())
    assert (r.filled, r.net_return) == (False, 0.0)


def test_discount_entry_fills_when_low_touches_limit() -> None:
    bars = bars_from([100.0, 99.0, 97.5, 101.0, 101.0])
    r = simulate_position(bars, 0, 100.0, EntryPolicy(0.02, 5), AddPolicy(None), Baseline())
    assert r.filled and r.avg_cost == 98.0 and r.shares == 1


def test_add_on_buy_averages_cost() -> None:
    bars = bars_from([100.0, 100.0, 96.0, 99.0, 99.0])
    r = simulate_position(bars, 0, 100.0, EntryPolicy(0.0, 5), AddPolicy(0.03), Baseline())
    assert r.shares == 2 and r.avg_cost == pytest.approx((100.0 + 97.0) / 2)
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest tests/unit/application/trading/test_exit_rules.py -q -p no:cacheprovider`
Expected: FAIL — `ImportError: cannot import name 'AddPolicy'`

- [ ] **Step 3: 구현 (exit_rules.py 끝에 추가)**

```python
@dataclass(frozen=True)
class EntryPolicy:
    """신호 시점 가격 대비 `discount` 아래 지정가, `ttl`분 안에 안 닿으면 포기."""

    discount: float
    ttl: int


@dataclass(frozen=True)
class AddPolicy:
    """첫 체결가 대비 `drop` 아래에 1주 추가(1회). None이면 추가 없음."""

    drop: float | None


@dataclass(frozen=True)
class PositionResult:
    filled: bool
    shares: int
    avg_cost: float
    net_return: float
    reason: str


def simulate_position(
    bars: Sequence[Bar],
    signal_idx: int,
    ref_price: float,
    entry: EntryPolicy,
    add: AddPolicy,
    rule: ExitRule,
    flatten_at: time = time(15, 0),
) -> PositionResult:
    """체결 안 되면 수익 0(놓친 기회도 목적함수에 들어가게). 수익률은 투입 원금 대비."""
    limit = ref_price * (1 - entry.discount)
    fill_idx = None
    for i in range(signal_idx, min(signal_idx + entry.ttl, len(bars))):
        if bars[i].ts.time() >= flatten_at:
            break
        if bars[i].low <= limit:
            fill_idx = i
            break
    if fill_idx is None:
        return PositionResult(False, 0, 0.0, 0.0, "unfilled")
    first = min(bars[fill_idx].open, limit)
    costs = [first]
    exit_ = simulate_exit(bars, fill_idx, first, rule, flatten_at)
    if add.drop is not None:
        add_px = first * (1 - add.drop)
        for i in range(fill_idx + 1, len(bars)):
            if bars[i].ts >= exit_.exit_ts:
                break
            if bars[i].low <= add_px:
                costs.append(min(bars[i].open, add_px))
                # 청산 규칙은 평균단가 기준으로 다시 돌린다(손절 기준이 평균단가로 바뀜).
                exit_ = simulate_exit(bars, fill_idx, statistics.fmean(costs), rule, flatten_at)
                break
    invested = sum(c * (1 + FEE) for c in costs)
    proceeds = exit_.exit_price * len(costs) * (1 - SELL_TAX - FEE)
    return PositionResult(True, len(costs), statistics.fmean(costs), proceeds / invested - 1, exit_.reason)
```

파일 상단 import에 `import statistics`를 추가한다.

- [ ] **Step 4: 통과 확인**

Run: `uv run pytest tests/unit/application/trading/test_exit_rules.py -q -p no:cacheprovider`
Expected: 13 passed

- [ ] **Step 5: 평가 스크립트에 `--stage entry` 추가**

`eval_exit_rules.py`에 추가:

```python
from daytrade_it.application.trading.exit_rules import AddPolicy, EntryPolicy, simulate_position

ENTRY_GRID = [EntryPolicy(d, n) for d in (0.0, 0.005, 0.01, 0.02) for n in (5, 30, 60)]
ADD_GRID = [AddPolicy(e) for e in (0.02, 0.03, 0.05, None)]
```

`run()`에 `stage` 인자를 받아 `stage == "entry"`면 규칙 루프 대신 다음을 쓴다. 청산 규칙은 `--exit-rule` 이름으로 `GRID`와 `Baseline()`에서 찾는다. 기준선은 `EntryPolicy(0.0, 5)` + `AddPolicy(None)` (현재 운영).

```python
            ref = bars[i].open
            base = simulate_position(bars, i, ref, EntryPolicy(0.0, 5), AddPolicy(None), exit_rule)
            for ep in ENTRY_GRID:
                for ap in ADD_GRID:
                    name = f"E{ep.discount:g}/{ep.ttl}+A{ap.drop if ap.drop is not None else '-'}"
                    if frozen is not None and name not in frozen:
                        continue
                    r = simulate_position(bars, i, ref, ep, ap, exit_rule)
                    rows.append({**s, "rule": name, "net": r.net_return, "base_net": base.net_return, "reason": r.reason})
```

argparse에 `--stage {exit,entry}`(기본 exit), `--exit-rule`(기본 `B`) 추가. 표 출력은 그대로 재사용한다(체결 안 된 쌍은 net=0으로 들어가 있다).

- [ ] **Step 6: dev → 고정 → test 1회** — Task 4 Step 3~6과 같은 절차. 격자가 48조합이므로 dev 선택은 `[ops]` Δmedian 상위 1개만 고정한다. 추가매수 후보가 채택돼도 문서에 "`execute_trade` 1주 하드캡 변경은 사용자 결정"을 명시한다.

- [ ] **Step 7: 커밋**

```bash
git add src/daytrade_it/application/trading/exit_rules.py tests/unit/application/trading/test_exit_rules.py scripts/eval/eval_exit_rules.py docs/research/2026-09-*-exit-rule-eval.md
git commit -m "feat(eval): entry price and add-on buy grid (stage 2)" && git push
```
