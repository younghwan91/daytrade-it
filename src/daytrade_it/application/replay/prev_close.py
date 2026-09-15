"""Previous closes frozen per replay day, without post-trade revisions.

`daily_bars_adjusted` rows are rewritten after the fact (split/dividend
adjustment, intraday snapshots overwritten by the final close), so a replay
reading it today could use a close nobody could see on the trade date. The
replay instead takes each ticker's prior-session close from its last regular
minute bar (09:00 <= ts < 15:31) before the day, freezes it to
`data/eval/prev_close/YYYY-MM-DD.json` once, and serves it both to SimBroker
(price limits) and to the no-chase filter through `FrozenPrevCloseDB`.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

from daytrade_it.domain.value_objects import Ticker
from daytrade_it.infrastructure.external.quant_airflow_db_client import AdjustedDailyBar

SOURCE = "kr_quant minute_bars: last bar 09:00 <= ts < 15:31 before the day"


@dataclass(frozen=True)
class FrozenPrevCloses:
    day: dt.date
    #: code -> (close, session date of that bar)
    closes: dict[str, tuple[Decimal, dt.date]]
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
            c: {"close": str(p), "date": d.isoformat()} for c, (p, d) in frozen.closes.items()
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
