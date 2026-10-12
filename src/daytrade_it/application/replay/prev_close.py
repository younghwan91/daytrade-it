"""Previous closes frozen per replay day, without post-trade revisions.

`daily_bars_adjusted` rows are rewritten after the fact, so a replay reading
it today could use a close nobody could see on the trade date. The replay
freezes, per ticker, the prior session's last minute bar with ts <= 20:00
(`closes`, the daily close incl. KRX after-market trades 16:00-20:00 that
kr_quant's daily bars carry from 2026-09-14) and the KRX regular-session close
= last bar 09:00 <= ts < 15:33 (closing auction at 15:30, or 15:32 when a VI
extends it; `krx`). Frozen once to `data/eval/prev_close/YYYY-MM-DD.json`.

Served to SimBroker (`krx` is the price-limit base, R18) and, through
`FrozenPrevCloseDB`, to the no-chase filter (R17) on the basis live used that
day: before `entry_filter.REGULAR_CLOSE_FROM` live read the daily close, so the
stub serves no regular-session close and the filter falls back to `closes`;
from that day on it serves `krx` and, as the daily input, the frozen last
`daily_bars_adjusted` row (`daily`) -- the two values live's filter reads -- so
`entry_filter.choose_prev_close`, the one implementation, picks exactly as live
does. Serving `closes` as the daily input there would diverge for a ticker with
no minute bars for the prior session but some for an older one: live compares
the stale regular bar to the prior session's daily row and falls back, while
`closes` would carry the same stale date and keep the stale regular close
(2026-10-12 328380: live 3,600 vs replay 3,760).
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

from daytrade_it.application.signals.entry_filter import (
    DAILY_FALLBACK,
    LOOKBACK_DAYS,
    REGULAR_CLOSE_FROM,
    PrevClose,
    choose_prev_close,
)
from daytrade_it.domain.value_objects import Ticker
from daytrade_it.infrastructure.external.quant_airflow_db_client import (
    AdjustedDailyBar,
    RegularSessionClose,
)

SOURCE = (
    "kr_quant minute_bars before the day: close = last bar ts <= 20:00 (KRX after-market incl.); "
    "krx_close = last bar 09:00 <= ts < 15:33"
)


@dataclass(frozen=True)
class FrozenPrevCloses:
    day: dt.date
    #: code -> (close, session date of that bar)
    closes: dict[str, tuple[Decimal, dt.date]]
    #: code -> (KRX regular-session close, date): price-limit base (R18) and, from
    #: REGULAR_CLOSE_FROM, the no-chase prior close (R17).
    krx: dict[str, tuple[Decimal, dt.date]] = field(default_factory=dict)
    #: Tickers whose frozen no-chase basis differs from an independent reference by > 0.1%
    #: (replay_parity.vintage_flags).
    flags: list[dict] = field(default_factory=list)
    #: Event tickers with no bar by 20:00 in the lookback (no frozen `closes` row).
    missing: list[str] = field(default_factory=list)
    #: code -> (close, date) of the last ``daily_bars_adjusted`` row before `day`, as
    #: kr_quant held it at freeze time: the daily input of live's no-chase filter from
    #: REGULAR_CLOSE_FROM. None = frozen before this field existed (2026-10-13); the stub
    #: then serves `closes` as the daily input (pre-switch days only, see replay_parity).
    daily: dict[str, tuple[Decimal, dt.date]] | None = None

    @property
    def prices(self) -> dict[str, Decimal]:
        return {c: p for c, (p, _) in self.closes.items()}

    @property
    def krx_prices(self) -> dict[str, Decimal]:
        """R18: the KRX regular-session close (price-limit base), not the after-market-inclusive close."""
        return {c: p for c, (p, _) in self.krx.items()}

    @property
    def regular_no_chase(self) -> bool:
        """Did live's no-chase filter read the regular-session close on this day?"""
        return self.day >= REGULAR_CLOSE_FROM

    def daily_input(
        self, code: str, *, start: dt.date | None = None, end: dt.date | None = None
    ) -> tuple[Decimal, dt.date] | None:
        """The daily row live's filter would read for `code` in [start, end], or None.

        From REGULAR_CLOSE_FROM the frozen `daily_bars_adjusted` row (`daily`); before it
        (and for files frozen without `daily`) the after-market-inclusive `closes`.
        """
        src = self.daily if self.daily is not None and self.regular_no_chase else self.closes
        hit = src.get(code)
        if hit is None:
            return None
        if (start is not None and hit[1] < start) or (end is not None and hit[1] > end):
            return None
        return hit

    def regular_input(
        self, code: str, *, before: dt.date, lookback_days: int
    ) -> tuple[Decimal, dt.date] | None:
        """The regular-session close live's filter would read: none before REGULAR_CLOSE_FROM."""
        if not self.regular_no_chase:
            return None
        hit = self.krx.get(code)
        if hit is None or not (before - dt.timedelta(days=lookback_days) <= hit[1] < before):
            return None
        return hit

    @property
    def no_chase_basis(self) -> dict[str, PrevClose]:
        """Per code, the prior close live's no-chase filter measured against on this day.

        The same inputs `FrozenPrevCloseDB` serves the filter (same lookback window),
        decided by `entry_filter.choose_prev_close` -- not a second implementation.
        """
        out = {}
        for c in sorted(set(self.closes) | set(self.daily or {}) | set(self.krx)):
            d = self.daily_input(
                c,
                start=self.day - dt.timedelta(days=LOOKBACK_DAYS),
                end=self.day - dt.timedelta(days=1),
            )
            r = self.regular_input(c, before=self.day, lookback_days=LOOKBACK_DAYS)
            p = choose_prev_close(_daily_bar(c, d), _regular(c, r))
            if p is not None:
                out[c] = p
        return out

    @property
    def no_chase_prices(self) -> dict[str, Decimal]:
        """The prior close live's no-chase filter measured against on this day."""
        return {c: p.price for c, p in self.no_chase_basis.items()}

    @property
    def no_chase_fallbacks(self) -> list[str]:
        """Codes whose no-chase basis falls back to the daily close on a regular-close day."""
        if not self.regular_no_chase:
            return []
        return sorted(c for c, p in self.no_chase_basis.items() if p.source == DAILY_FALLBACK)

    @property
    def prev_session(self) -> dt.date:
        return max(d for _, d in self.closes.values())


def dump_prev_closes(path: Path, frozen: FrozenPrevCloses) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "day": frozen.day.isoformat(),
        "source": SOURCE,
        "closes": {
            c: {
                "close": str(p),
                "date": d.isoformat(),
                "krx_close": str(frozen.krx[c][0]) if c in frozen.krx else None,
                "krx_date": frozen.krx[c][1].isoformat() if c in frozen.krx else None,
            }
            for c, (p, d) in frozen.closes.items()
        },
        "flags": frozen.flags,
        "missing": frozen.missing,
    }
    if frozen.daily is not None:
        row["daily"] = {
            c: {"close": str(p), "date": d.isoformat()} for c, (p, d) in frozen.daily.items()
        }
    path.write_text(json.dumps(row, ensure_ascii=False, indent=1, sort_keys=True))


def load_prev_closes(path: Path) -> FrozenPrevCloses:
    row = json.loads(path.read_text())
    return FrozenPrevCloses(
        day=dt.date.fromisoformat(row["day"]),
        closes={
            c: (Decimal(v["close"]), dt.date.fromisoformat(v["date"]))
            for c, v in row["closes"].items()
        },
        krx={
            c: (Decimal(v["krx_close"]), dt.date.fromisoformat(v["krx_date"]))
            for c, v in row["closes"].items()
            if v.get("krx_close") is not None
        },
        flags=row.get("flags", []),
        missing=row.get("missing", []),
        daily=None
        if row.get("daily") is None
        else {
            c: (Decimal(v["close"]), dt.date.fromisoformat(v["date"]))
            for c, v in row["daily"].items()
        },
    )


class FrozenPrevCloseDB:
    """Replay-only stand-in for the two `QuantAirflowDBClient` reads the no-chase filter makes.

    `get_adjusted_daily_bars` serves `FrozenPrevCloses.daily_input` (the frozen
    `daily_bars_adjusted` row from REGULAR_CLOSE_FROM, `closes` before);
    `get_regular_session_close` serves `krx` only from REGULAR_CLOSE_FROM, so the
    filter reproduces each day's live basis.
    """

    def __init__(self, frozen: FrozenPrevCloses) -> None:
        self._frozen = frozen

    def get_adjusted_daily_bars(
        self,
        ticker: Ticker,
        *,
        start: dt.date | str | None = None,
        end: dt.date | str | None = None,
    ) -> list[AdjustedDailyBar]:
        hit = self._frozen.daily_input(
            ticker.value,
            start=None if start is None else _date(start),
            end=None if end is None else _date(end),
        )
        bar = _daily_bar(ticker.value, hit)
        return [] if bar is None else [bar]

    def get_regular_session_close(
        self, ticker: Ticker, *, before: dt.date, lookback_days: int
    ) -> RegularSessionClose | None:
        return _regular(
            ticker.value,
            self._frozen.regular_input(ticker.value, before=before, lookback_days=lookback_days),
        )


def _daily_bar(code: str, hit: tuple[Decimal, dt.date] | None) -> AdjustedDailyBar | None:
    if hit is None:
        return None
    return AdjustedDailyBar(code=code, date=hit[1], close=float(hit[0]), source="replay_frozen")


def _regular(code: str, hit: tuple[Decimal, dt.date] | None) -> RegularSessionClose | None:
    return None if hit is None else RegularSessionClose(code=code, date=hit[1], close=float(hit[0]))


def _date(v: dt.date | str) -> dt.date:
    return v if isinstance(v, dt.date) else dt.date.fromisoformat(v)
