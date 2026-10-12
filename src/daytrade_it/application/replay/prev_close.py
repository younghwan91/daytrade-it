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
from that day on it serves `krx` and the filter picks it exactly as live does.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

from daytrade_it.application.signals.entry_filter import REGULAR_CLOSE_FROM
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
    #: Event tickers with no regular bar in the lookback (no prev close).
    missing: list[str] = field(default_factory=list)

    @property
    def prices(self) -> dict[str, Decimal]:
        return {c: p for c, (p, _) in self.closes.items()}

    @property
    def krx_prices(self) -> dict[str, Decimal]:
        """R18: the KRX regular-session close (price-limit base), not the unified close."""
        return {c: p for c, (p, _) in self.krx.items()}

    @property
    def regular_no_chase(self) -> bool:
        """Did live's no-chase filter read the regular-session close on this day?"""
        return self.day >= REGULAR_CLOSE_FROM

    @property
    def no_chase_prices(self) -> dict[str, Decimal]:
        """The prior close live's no-chase filter measured against on this day.

        From REGULAR_CLOSE_FROM: the regular close, else (no regular bar, or one older
        than the after-market-inclusive close) `closes` -- `entry_filter.choose_prev_close`.
        """
        if not self.regular_no_chase:
            return self.prices
        out = {}
        for c, (p, d) in self.closes.items():
            k = self.krx.get(c)
            out[c] = k[0] if k is not None and k[1] >= d else p
        return out

    @property
    def no_chase_fallbacks(self) -> list[str]:
        """Codes whose no-chase basis falls back to `closes` on a regular-close day."""
        if not self.regular_no_chase:
            return []
        return sorted(
            c for c, (_, d) in self.closes.items() if c not in self.krx or self.krx[c][1] < d
        )

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
    )


class FrozenPrevCloseDB:
    """Replay-only stand-in for the two `QuantAirflowDBClient` reads the no-chase filter makes.

    `get_adjusted_daily_bars` serves `closes` (the daily close as kr_quant held it);
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
        hit = self._frozen.closes.get(ticker.value)
        if hit is None:
            return []
        close, date = hit
        if (start is not None and date < _date(start)) or (end is not None and date > _date(end)):
            return []
        return [
            AdjustedDailyBar(
                code=ticker.value, date=date, close=float(close), source="replay_frozen"
            )
        ]

    def get_regular_session_close(
        self, ticker: Ticker, *, before: dt.date, lookback_days: int
    ) -> RegularSessionClose | None:
        if not self._frozen.regular_no_chase:
            return None
        hit = self._frozen.krx.get(ticker.value)
        if hit is None:
            return None
        close, date = hit
        if not (before - dt.timedelta(days=lookback_days) <= date < before):
            return None
        return RegularSessionClose(code=ticker.value, date=date, close=float(close))


def _date(v: dt.date | str) -> dt.date:
    return v if isinstance(v, dt.date) else dt.date.fromisoformat(v)
