"""Previous closes frozen per replay day, without post-trade revisions.

`daily_bars_adjusted` rows are rewritten after the fact, so a replay reading
it today could use a close nobody could see on the trade date. The live
no-chase filter reads kr_quant's unified KRX+NXT daily close, which is the
NXT after-market last price (controller R17). The replay therefore freezes,
per ticker, the prior session's last minute bar with ts <= 20:00 (`closes`)
and, for later selection work, the KRX close = last regular bar
09:00 <= ts < 15:33 (closing auction at 15:30, or 15:32 when a VI extends
it; `krx`). Frozen once to `data/eval/prev_close/YYYY-MM-DD.json` and served
to SimBroker and, through `FrozenPrevCloseDB`, to the no-chase filter.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

from daytrade_it.domain.value_objects import Ticker
from daytrade_it.infrastructure.external.quant_airflow_db_client import AdjustedDailyBar

SOURCE = (
    "kr_quant minute_bars before the day: close = last bar ts <= 20:00 (unified KRX+NXT); "
    "krx_close = last bar 09:00 <= ts < 15:33"
)


@dataclass(frozen=True)
class FrozenPrevCloses:
    day: dt.date
    #: code -> (close, session date of that bar)
    closes: dict[str, tuple[Decimal, dt.date]]
    #: code -> (KRX regular-session close, date); not used by replay yet.
    krx: dict[str, tuple[Decimal, dt.date]] = field(default_factory=dict)
    #: Tickers whose frozen close differs from daily_bars_adjusted by > 0.1%.
    flags: list[dict] = field(default_factory=list)
    #: Event tickers with no regular bar in the lookback (no prev close).
    missing: list[str] = field(default_factory=list)

    @property
    def prices(self) -> dict[str, Decimal]:
        return {c: p for c, (p, _) in self.closes.items()}

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
    """Replay-only stand-in for `QuantAirflowDBClient.get_adjusted_daily_bars`."""

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


def _date(v: dt.date | str) -> dt.date:
    return v if isinstance(v, dt.date) else dt.date.fromisoformat(v)
