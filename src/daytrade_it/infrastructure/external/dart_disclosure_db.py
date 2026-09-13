"""Recent DART disclosures for the entry risk gate, from scalp-it's dart.db.

Why this source: `kr_quant.disclosures` (quant-airflow) is collected twice a
day and ended at 2026-09-10 15:00 KST when this was written. scalp-it's cron
refreshes `data/dart.db` from OpenDART at 08:40 (7 days back) and every 10
minutes 09:00-15:50 on weekdays, on this same host. It is the only same-day
disclosure data here.

Fail-closed by design: a gate over a file nobody has refreshed would pass
every entry while looking wired. Missing, unreadable or stale data raises
`DisclosureDataUnavailable`; the caller refuses the *entry*. Exits never
consult this module.
"""

from __future__ import annotations

import datetime as dt
import sqlite3
from collections.abc import Callable
from pathlib import Path

from daytrade_it.domain.services.risk_gate import (
    DisclosureSeverity,
    classify_disclosure_severity,
)
from daytrade_it.infrastructure.brokers.base import KST, now_kst

#: Refresh cadence is 10 minutes in session; 30 leaves room for one slow or
#: skipped run. The 08:40 run keeps the first entries (09:00:30) inside it.
DEFAULT_MAX_STALENESS = dt.timedelta(minutes=30)

#: Calendar days, inclusive of today. A delisting review or embezzlement
#: filing from earlier in the week still argues against a fresh entry.
DEFAULT_LOOKBACK_DAYS = 7


class DisclosureDataUnavailable(RuntimeError):
    """The disclosure source can't vouch for the ticker right now."""


class DartDisclosureDB:
    def __init__(
        self,
        path: Path,
        *,
        max_staleness: dt.timedelta = DEFAULT_MAX_STALENESS,
        lookback_days: int = DEFAULT_LOOKBACK_DAYS,
        clock: Callable[[], dt.datetime] = now_kst,
    ) -> None:
        self._path = Path(path)
        self._max_staleness = max_staleness
        self._lookback_days = lookback_days
        self._clock = clock

    def hard_disclosure(self, ticker: str) -> str | None:
        """Most recent hard-severity report title for `ticker` in the lookback, else None."""
        now = self._clock()
        if not self._path.exists():
            raise DisclosureDataUnavailable(f"dart.db not found: {self._path}")
        refreshed = dt.datetime.fromtimestamp(self._path.stat().st_mtime, KST)
        age = now - refreshed
        if age > self._max_staleness:
            raise DisclosureDataUnavailable(
                f"dart.db stale: last refreshed {refreshed:%Y-%m-%d %H:%M} KST "
                f"({int(age.total_seconds() // 60)} min ago, limit "
                f"{int(self._max_staleness.total_seconds() // 60)} min)"
            )
        since = (now.date() - dt.timedelta(days=self._lookback_days)).strftime("%Y%m%d")
        try:
            con = sqlite3.connect(f"file:{self._path}?mode=ro", uri=True)
            try:
                rows = con.execute(
                    "SELECT report_nm FROM disclosures WHERE code = ? AND rcept_dt >= ? "
                    "ORDER BY rcept_dt DESC, rcept_no DESC",
                    (ticker, since),
                ).fetchall()
            finally:
                con.close()
        except sqlite3.Error as exc:
            raise DisclosureDataUnavailable(f"dart.db unreadable: {exc}") from exc
        for (report_nm,) in rows:
            if classify_disclosure_severity(report_nm) is DisclosureSeverity.HARD:
                return str(report_nm)
        return None
