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

    def on_close(
        self, bars: Sequence[Bar], entry_idx: int, i: int, entry_price: float
    ) -> str | None: ...


class _NoIntrabar:
    def intrabar(self, bar: Bar, entry_price: float) -> PriceHit | None:  # noqa: ARG002
        return None


class _NoClose:
    def on_close(
        self,
        bars: Sequence[Bar],  # noqa: ARG002
        entry_idx: int,  # noqa: ARG002
        i: int,  # noqa: ARG002
        entry_price: float,  # noqa: ARG002
    ) -> str | None:
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

    def on_close(
        self,
        bars: Sequence[Bar],  # noqa: ARG002
        entry_idx: int,  # noqa: ARG002
        i: int,  # noqa: ARG002
        entry_price: float,  # noqa: ARG002
    ) -> str | None:
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

    def on_close(
        self,
        bars: Sequence[Bar],  # noqa: ARG002
        entry_idx: int,  # noqa: ARG002
        i: int,  # noqa: ARG002
        entry_price: float,  # noqa: ARG002
    ) -> str | None:
        return "time" if i - entry_idx + 1 >= self.minutes else None


@dataclass(frozen=True)
class MomentumFade(_NoIntrabar):
    min_gain_pct: float
    window: int

    @property
    def name(self) -> str:
        return f"M{self.min_gain_pct:g}/{self.window}"

    def on_close(
        self, bars: Sequence[Bar], entry_idx: int, i: int, entry_price: float
    ) -> str | None:
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
