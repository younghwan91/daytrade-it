"""Read-only client for the user's ``quant-airflow`` TimescaleDB.

``quant-airflow`` (a sibling project, not part of this repo) already runs a
daily Airflow pipeline that ingests Korean market data — prices, supply/
demand, sector indices, earnings, consensus, and news/LLM judgments — into a
TimescaleDB instance exposed read-only on the LAN. Phase 0 spike 004 of the
gpt-quant-v2 KRX redevelopment concluded that gpt-quant-v2 should **consume**
that data rather than duplicating the ingestion (kiwoom-client/DART/etc.),
so this client is the read side of that hybrid data-layer decision.

Connection pattern reused from ``kr-quant``'s ``storage.py``
(``~/Documents/git/kr-quant/src/kr_quant/storage.py``): a plain
``psycopg2.connect(...)`` over a ``postgresql://`` DSN, no ORM, no
connection pool — this is a LAN host queried occasionally for batch/EOD
data, not a high-throughput service dependency.

Read-only is enforced in two independent ways, per the task brief:

1. At the connection level: every connection this client opens calls
   ``conn.set_session(readonly=True)`` immediately after connecting, which
   makes Postgres itself reject any DML/DDL for the lifetime of the
   connection (``ERROR: cannot execute INSERT in a read-only transaction``,
   etc.) — a server-enforced guarantee, not just a code-review convention.
2. In code: this module contains no INSERT/UPDATE/DELETE statements
   anywhere, by construction.

This client must never modify ``quant-airflow`` or ``kr-quant`` — it only
reads from the shared TimescaleDB they populate.
"""

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING, Any

import psycopg2
from krx_quant_core.market.session import KST
from pydantic import BaseModel, Field

from daytrade_it.domain.value_objects import Ticker
from daytrade_it.shared.errors import error_codes
from daytrade_it.shared.errors.exceptions import DatabaseError
from daytrade_it.shared.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Sequence

    from daytrade_it.shared.config.settings import QuantAirflowDBSettings

logger = get_logger(__name__)

# quant-airflow's news_judgments.judged_at is stored TIMESTAMPTZ in UTC
# (docs/schema.md:36 in quant-airflow — a documented real gotcha they hit).
# Any date-boundary comparison against it (e.g. "today's KST trading day")
# must convert to KST first, or rows land on the wrong calendar day. `KST`
# is krx-quant-core's (imported above) and stays re-exported from here.
UTC = dt.UTC


def judged_at_to_kst(judged_at: dt.datetime) -> dt.datetime:
    """Convert a ``news_judgments.judged_at`` value (UTC) to KST.

    ``judged_at`` comes back from psycopg2 as a timezone-aware
    ``datetime`` (the column is ``TIMESTAMPTZ``), but callers must not
    assume its wall-clock hour is already KST — it is stored in UTC. This
    makes the conversion explicit and is the one thing this module tests
    without any DB connection.

    Args:
        judged_at: The raw value read from ``news_judgments.judged_at``.
            If naive (no tzinfo), it is assumed to already be UTC — this
            matches what a bare ``TIMESTAMPTZ`` column reads back as in the
            rare case the driver strips tzinfo, and keeps the function total
            rather than raising on that edge case.

    Returns:
        The same instant expressed in Asia/Seoul wall-clock time.
    """
    aware = judged_at if judged_at.tzinfo is not None else judged_at.replace(tzinfo=UTC)
    return aware.astimezone(KST)


# =============================================================================
# Row DTOs
#
# No existing domain entity (checked `domain/entities/`) matches these
# shapes: `Stock` has no OHLCV fields, and the only `Bar`-named model in the
# codebase (`infrastructure/external/polygon_client.py::Bar`) is a
# Polygon-wire-format DTO — alias-mapped to Polygon's `o/h/l/c/v` JSON keys
# and a Unix-ms timestamp, which does not fit these date-keyed KRW rows.
# Rather than force either into KRX data, these are lightweight Pydantic
# DTOs scoped to this client, following the same pattern this repo already
# uses for `tiingo_client.NewsItem` and `polygon_client.Bar`/`Quote`.
# =============================================================================


class AdjustedDailyBar(BaseModel):
    """One row of ``daily_bars_adjusted`` (split-adjusted daily OHLCV, KRW).

    From 2026-09-14 the high/low/close/volume/trade_value include KRX after-market
    trades (16:00-20:00 continuous session, ETF/ETN excluded); only the open is the
    regular-session value. The next day's price-limit base is the regular-session close
    (minute bars 09:00 <= ts < 15:33), not this close. ``trade_value`` is in millions
    of KRW.
    """

    code: str
    date: dt.date
    open: float | None = None
    high: float | None = None
    low: float | None = None
    close: float | None = None
    volume: int | None = None
    trade_value: int | None = Field(default=None, description="Traded value, millions of KRW")
    source: str = Field(description="Provenance, e.g. 'kiwoom' or 'naver' (delisted backfill)")


class StockInfo(BaseModel):
    """One row of ``stocks`` (listed-stock master)."""

    code: str
    name: str
    market: str | None = None
    sector: str | None = None


class SupplyDemandRow(BaseModel):
    """One row of ``supply_demand`` (investor net-buy flows).

    Investor columns are net-bought **quantities** (shares), not KRW. ``close`` includes
    KRX after-market trades from 2026-09-14; ``close / (1 + flu_rt / 1e4)`` is the day's
    price-limit base (the previous regular-session close).
    """

    code: str
    date: dt.date
    close: int | None = Field(
        default=None, description="Close; includes KRX after-market trades from 2026-09-14"
    )
    flu_rt: float | None = Field(
        default=None,
        description="Change vs. the day's base price in percent x 100 (21.0 = +0.21%)",
    )
    acc_trde_qty: int | None = Field(default=None, description="Accumulated trading volume")
    individual: int | None = None
    foreign_: int | None = None
    institution: int | None = None
    source: str


class SectorIndexRow(BaseModel):
    """One row of ``sector_index`` (KOSPI/KOSDAQ sector indices).

    OHLC are index points x 100 stored as integers (KOSPI 6,384.24 -> 638424).
    Rows start 2024-01-26.
    """

    code: str
    name: str | None = None
    date: dt.date
    open: int | None = None
    high: int | None = None
    low: int | None = None
    close: int | None = None
    volume: int | None = None
    trade_value: int | None = None


class EarningsRow(BaseModel):
    """One row of ``earnings`` (DART financials, versioned by ``knowledge_date``).

    ``avail_date`` is the statutory filing deadline, not the disclosure date: 17.7% of
    periods with a periodic-report receipt date in ``earnings_disclosures`` were filed
    after it (2026-10-12), so filtering on ``avail_date`` alone looks ahead. Use
    ``known_date`` (first usable the next trading day) for as-of work.
    """

    code: str
    period: str
    avail_date: dt.date | None = Field(
        default=None,
        description="Statutory filing deadline - NOT lookahead-safe; see known_date",
    )
    netinc: float | None = None
    netinc_prior: float | None = None
    revenue: float | None = None
    revenue_prior: float | None = None
    op_income: float | None = None
    op_income_prior: float | None = None
    knowledge_date: dt.date
    known_date: dt.date | None = Field(
        default=None,
        description=(
            "max(avail_date, first periodic-report receipt date); usable from the next "
            "trading day. Falls back to avail_date when no periodic receipt exists "
            "(before 2018Q3)."
        ),
    )


class ConsensusRow(BaseModel):
    """One row of ``consensus`` (analyst consensus snapshot)."""

    code: str
    date: dt.date
    target_mean: float | None = None
    recomm_mean: float | None = None
    base_date: str | None = None
    fwd_eps: float | None = None
    prev_eps: float | None = None
    est_year: str | None = None


class NewsJudgmentRow(BaseModel):
    """One row of ``news_judgments`` (LLM sentiment/event judgments).

    ``judged_at_kst`` is derived from the raw UTC-stored ``judged_at`` via
    :func:`judged_at_to_kst` at construction time, so callers never have to
    remember the UTC gotcha themselves.
    """

    source_type: str
    source_id: str
    ticker: str
    event_type: str | None = None
    sentiment_direction: int | None = None
    is_stale_repeat: bool | None = None
    price_impact_likely: bool | None = None
    confidence: int | None = None
    knowledge_date: dt.date
    judged_at_utc: dt.datetime
    judged_at_kst: dt.datetime

    @classmethod
    def from_raw(cls, *, judged_at: dt.datetime, **fields: Any) -> NewsJudgmentRow:
        """Build from a raw DB row, deriving ``judged_at_kst`` from UTC."""
        return cls(judged_at_utc=judged_at, judged_at_kst=judged_at_to_kst(judged_at), **fields)


# =============================================================================
# Client
# =============================================================================


class QuantAirflowDBClient:
    """Read-only client over quant-airflow's TimescaleDB.

    Usage:
        >>> with QuantAirflowDBClient() as client:
        ...     bars = client.get_adjusted_daily_bars(
        ...         Ticker(value="005930"), start="2026-01-01", end="2026-01-31"
        ...     )

    Every connection this client opens is placed into a Postgres read-only
    session (``set_session(readonly=True)``) — see module docstring. There
    is deliberately no `execute()`-with-arbitrary-SQL escape hatch on this
    class; only the specific read methods below.
    """

    def __init__(
        self,
        *,
        host: str | None = None,
        port: int | None = None,
        dbname: str | None = None,
        user: str | None = None,
        password: str | None = None,
        connect_timeout: float | None = None,
        settings: QuantAirflowDBSettings | None = None,
    ) -> None:
        """Initialize the client. Explicit args win; otherwise falls back to settings.

        Args:
            host: TimescaleDB host. Defaults to configured settings.
            port: TimescaleDB port. Defaults to configured settings.
            dbname: Database name. Defaults to configured settings.
            user: Query user. Defaults to configured settings.
            password: Password for `user`. Defaults to configured settings.
            connect_timeout: Connection timeout in seconds. Defaults to
                configured settings.
            settings: Explicit settings object (mainly for tests). If not
                given, lazily pulled from ``get_settings().quant_airflow_db``
                only when needed, so constructing this client with all
                connection args passed explicitly never touches process-wide
                settings.
        """
        self._explicit = {
            "host": host,
            "port": port,
            "dbname": dbname,
            "user": user,
            "password": password,
            "connect_timeout": connect_timeout,
        }
        self._settings = settings
        self._conn: Any | None = None

    def _resolve_settings(self) -> QuantAirflowDBSettings:
        if self._settings is not None:
            return self._settings
        from daytrade_it.shared.config import get_settings  # noqa: PLC0415

        self._settings = get_settings().quant_airflow_db
        return self._settings

    def _connection_kwargs(self) -> dict[str, Any]:
        settings = self._resolve_settings()
        return {
            "host": self._explicit["host"] or settings.host,
            "port": self._explicit["port"] or settings.port,
            "dbname": self._explicit["dbname"] or settings.dbname,
            "user": self._explicit["user"] or settings.user,
            "password": self._explicit["password"] or settings.password.get_secret_value(),
            "connect_timeout": int(self._explicit["connect_timeout"] or settings.connect_timeout),
        }

    def connect(self) -> Any:
        """Open (or reuse) a read-only psycopg2 connection.

        Returns:
            The open psycopg2 connection.

        Raises:
            DatabaseError: If the connection cannot be established.
        """
        if self._conn is not None and not self._conn.closed:
            return self._conn

        kwargs = self._connection_kwargs()
        try:
            conn = psycopg2.connect(**kwargs)
        except psycopg2.Error as exc:
            logger.error(
                "Failed to connect to quant-airflow TimescaleDB",
                host=kwargs["host"],
                port=kwargs["port"],
                dbname=kwargs["dbname"],
                error=str(exc),
            )
            raise DatabaseError(
                message="Failed to connect to quant-airflow TimescaleDB",
                error_code=error_codes.DATABASE_CONNECTION_ERROR,
                cause=exc,
                details={
                    "host": kwargs["host"],
                    "port": kwargs["port"],
                    "dbname": kwargs["dbname"],
                },
            ) from exc

        # Server-enforced read-only: any INSERT/UPDATE/DELETE/DDL on this
        # connection now fails at the Postgres level, independent of what
        # this client's own code does or doesn't send.
        conn.set_session(readonly=True, autocommit=True)

        logger.info(
            "Connected to quant-airflow TimescaleDB (read-only)",
            host=kwargs["host"],
            port=kwargs["port"],
            dbname=kwargs["dbname"],
        )
        self._conn = conn
        return conn

    def close(self) -> None:
        """Close the connection if open."""
        if self._conn is not None and not self._conn.closed:
            self._conn.close()
            logger.debug("Closed quant-airflow TimescaleDB connection")
        self._conn = None

    def __enter__(self) -> QuantAirflowDBClient:
        self.connect()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def ping(self) -> bool:
        """Run ``SELECT 1`` to confirm connectivity. Used for reachability checks."""
        conn = self.connect()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                return cur.fetchone() == (1,)
        except psycopg2.Error as exc:
            raise DatabaseError(
                message="quant-airflow TimescaleDB ping failed",
                error_code=error_codes.DATABASE_QUERY_ERROR,
                cause=exc,
                query="SELECT 1",
            ) from exc

    def _query(self, sql: str, params: tuple[Any, ...]) -> list[tuple[Any, ...]]:
        conn = self.connect()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                return cur.fetchall()
        except psycopg2.Error as exc:
            logger.error("Query failed", error=str(exc), query=sql[:200])
            raise DatabaseError(
                message="quant-airflow TimescaleDB query failed",
                error_code=error_codes.DATABASE_QUERY_ERROR,
                cause=exc,
                query=sql,
            ) from exc

    # -------------------------------------------------------------------
    # daily_bars_adjusted
    # -------------------------------------------------------------------

    def get_adjusted_daily_bars(
        self,
        ticker: Ticker,
        *,
        start: dt.date | str | None = None,
        end: dt.date | str | None = None,
    ) -> list[AdjustedDailyBar]:
        """Fetch split-adjusted daily bars for `ticker` over `[start, end]`.

        Args:
            ticker: KRX 6-digit code.
            start: Inclusive start date (``YYYY-MM-DD`` or ``date``). ``None``
                means no lower bound.
            end: Inclusive end date. ``None`` means no upper bound.

        Returns:
            Bars ordered by date ascending.
        """
        where = ["code = %s"]
        params: list[Any] = [ticker.value]
        if start is not None:
            where.append("date >= %s")
            params.append(start)
        if end is not None:
            where.append("date <= %s")
            params.append(end)

        sql = (
            "SELECT code, date, open, high, low, close, volume, trade_value, source "
            f"FROM daily_bars_adjusted WHERE {' AND '.join(where)} ORDER BY date ASC"
        )
        rows = self._query(sql, tuple(params))
        return [
            AdjustedDailyBar(
                code=r[0],
                date=r[1],
                open=r[2],
                high=r[3],
                low=r[4],
                close=r[5],
                volume=r[6],
                trade_value=r[7],
                source=r[8],
            )
            for r in rows
        ]

    # -------------------------------------------------------------------
    # stocks
    # -------------------------------------------------------------------

    def get_stock_master(self) -> dict[str, StockInfo]:
        """All listed stocks keyed by code (``stocks`` table, ~2.6k rows)."""
        rows = self._query("SELECT code, name, market, sector FROM stocks", ())
        return {r[0]: StockInfo(code=r[0], name=r[1], market=r[2], sector=r[3]) for r in rows}

    # -------------------------------------------------------------------
    # supply_demand
    # -------------------------------------------------------------------

    def get_latest_supply_demand(self, ticker: Ticker) -> SupplyDemandRow | None:
        """Fetch the most recent ``supply_demand`` row for `ticker`."""
        sql = (
            "SELECT code, date, close, flu_rt, acc_trde_qty, individual, "
            "foreign_, institution, source FROM supply_demand "
            "WHERE code = %s ORDER BY date DESC LIMIT 1"
        )
        rows = self._query(sql, (ticker.value,))
        if not rows:
            return None
        r = rows[0]
        return SupplyDemandRow(
            code=r[0],
            date=r[1],
            close=r[2],
            flu_rt=r[3],
            acc_trde_qty=r[4],
            individual=r[5],
            foreign_=r[6],
            institution=r[7],
            source=r[8],
        )

    # -------------------------------------------------------------------
    # sector_index
    # -------------------------------------------------------------------

    def get_latest_sector_index(self, sector_code: str) -> SectorIndexRow | None:
        """Fetch the most recent ``sector_index`` row for `sector_code`.

        Args:
            sector_code: quant-airflow sector code (e.g. ``"001"`` = KOSPI
                composite, ``"101"`` = KOSDAQ composite) — not a `Ticker`,
                sector codes are not stock codes.
        """
        sql = (
            "SELECT code, name, date, open, high, low, close, volume, trade_value "
            "FROM sector_index WHERE code = %s ORDER BY date DESC LIMIT 1"
        )
        rows = self._query(sql, (sector_code,))
        if not rows:
            return None
        r = rows[0]
        return SectorIndexRow(
            code=r[0],
            name=r[1],
            date=r[2],
            open=r[3],
            high=r[4],
            low=r[5],
            close=r[6],
            volume=r[7],
            trade_value=r[8],
        )

    # -------------------------------------------------------------------
    # earnings
    # -------------------------------------------------------------------

    def get_earnings(
        self,
        ticker: Ticker,
        *,
        asof: dt.date | str | None = None,
        known_before: dt.date | None = None,
    ) -> list[EarningsRow]:
        """Fetch earnings for `ticker`, one row per period, as-of `asof`.

        ``earnings`` is versioned by ``knowledge_date`` — a DART restatement
        lands as a new row rather than overwriting the original (mirrors
        ``kr_quant.storage.read_earnings``). This picks the newest version
        per period whose ``knowledge_date <= asof`` (or the latest known
        version if ``asof`` is ``None``) to avoid a plain ``SELECT *``
        silently duplicating rows once restatements exist.

        ``asof`` bounds the restatement version only. It does not make a row
        lookahead-safe: ``avail_date`` is the filing deadline and 17.7% of
        periodic reports came later. For point-in-time use pass
        ``known_before=<trading day>``, which keeps rows with
        ``known_date < known_before`` (disclosed on an earlier day, so usable
        from that trading day's open).

        Args:
            ticker: KRX 6-digit code.
            asof: Ceiling on ``knowledge_date``. ``None`` means latest-known.
            known_before: Keep only periods whose ``known_date`` is strictly
                before this date. ``None`` keeps all (not lookahead-safe).

        Returns:
            One row per ``period``, ordered by period ascending.
        """
        cols = (
            "e.code, e.period, e.avail_date, e.netinc, e.netinc_prior, e.revenue, "
            "e.revenue_prior, e.op_income, e.op_income_prior, e.knowledge_date, "
            "GREATEST(e.avail_date, (SELECT min(d.rcept_dt) FROM earnings_disclosures d "
            "WHERE d.code = e.code AND d.period = e.period AND d.kind = 'periodic'))"
        )
        if asof is not None:
            sql = (
                f"SELECT DISTINCT ON (e.code, e.period) {cols} FROM earnings e "
                "WHERE e.code = %s AND e.knowledge_date <= %s "
                "ORDER BY e.code, e.period, e.knowledge_date DESC"
            )
            params: tuple[Any, ...] = (ticker.value, asof)
        else:
            sql = (
                f"SELECT DISTINCT ON (e.code, e.period) {cols} FROM earnings e "
                "WHERE e.code = %s ORDER BY e.code, e.period, e.knowledge_date DESC"
            )
            params = (ticker.value,)

        rows = self._query(sql, params)
        results = [
            EarningsRow(
                code=r[0],
                period=r[1],
                avail_date=r[2],
                netinc=r[3],
                netinc_prior=r[4],
                revenue=r[5],
                revenue_prior=r[6],
                op_income=r[7],
                op_income_prior=r[8],
                knowledge_date=r[9],
                known_date=r[10],
            )
            for r in rows
        ]
        if known_before is not None:
            results = [
                e for e in results if e.known_date is not None and e.known_date < known_before
            ]
        return sorted(results, key=lambda e: e.period)

    # -------------------------------------------------------------------
    # consensus
    # -------------------------------------------------------------------

    def get_consensus(self, ticker: Ticker) -> list[ConsensusRow]:
        """Fetch all consensus snapshots for `ticker`, ordered by date ascending."""
        sql = (
            "SELECT code, date, target_mean, recomm_mean, base_date, fwd_eps, "
            "prev_eps, est_year FROM consensus WHERE code = %s ORDER BY date ASC"
        )
        rows = self._query(sql, (ticker.value,))
        return [
            ConsensusRow(
                code=r[0],
                date=r[1],
                target_mean=r[2],
                recomm_mean=r[3],
                base_date=r[4],
                fwd_eps=r[5],
                prev_eps=r[6],
                est_year=r[7],
            )
            for r in rows
        ]

    def get_latest_consensus(self, ticker: Ticker) -> ConsensusRow | None:
        """Fetch the most recent consensus snapshot for `ticker`."""
        rows = self.get_consensus(ticker)
        return rows[-1] if rows else None

    # -------------------------------------------------------------------
    # news_judgments (UTC judged_at -> KST, see module docstring)
    # -------------------------------------------------------------------

    def get_news_judgments(
        self,
        ticker: Ticker,
        *,
        judged_after_kst: dt.datetime | None = None,
        judged_before_kst: dt.datetime | None = None,
        limit: int = 200,
    ) -> list[NewsJudgmentRow]:
        """Fetch LLM news judgments for `ticker`, optionally bounded by KST time.

        ``judged_at`` is stored UTC in the DB (see module docstring). The
        `judged_after_kst`/`judged_before_kst` bounds are given in KST by
        the caller and converted to UTC here before hitting the DB, so
        callers never have to do the UTC/KST arithmetic themselves.

        Args:
            ticker: KRX 6-digit code.
            judged_after_kst: Inclusive lower bound on `judged_at`, in KST.
            judged_before_kst: Inclusive upper bound on `judged_at`, in KST.
            limit: Maximum rows to return, most recent first.

        Returns:
            Rows ordered by ``judged_at`` descending (most recent first),
            each with both `judged_at_utc` (raw) and `judged_at_kst`
            (converted) populated.
        """
        where = ["ticker = %s"]
        params: list[Any] = [ticker.value]
        if judged_after_kst is not None:
            where.append("judged_at >= %s")
            params.append(self._kst_to_utc(judged_after_kst))
        if judged_before_kst is not None:
            where.append("judged_at <= %s")
            params.append(self._kst_to_utc(judged_before_kst))

        sql = (
            "SELECT source_type, source_id, ticker, event_type, sentiment_direction, "
            "is_stale_repeat, price_impact_likely, confidence, knowledge_date, judged_at "
            f"FROM news_judgments WHERE {' AND '.join(where)} "
            "ORDER BY judged_at DESC LIMIT %s"
        )
        params.append(limit)
        rows = self._query(sql, tuple(params))
        return [
            NewsJudgmentRow.from_raw(
                source_type=r[0],
                source_id=r[1],
                ticker=r[2],
                event_type=r[3],
                sentiment_direction=r[4],
                is_stale_repeat=r[5],
                price_impact_likely=r[6],
                confidence=r[7],
                knowledge_date=r[8],
                judged_at=r[9],
            )
            for r in rows
        ]

    @staticmethod
    def _kst_to_utc(value: dt.datetime) -> dt.datetime:
        aware = value if value.tzinfo is not None else value.replace(tzinfo=KST)
        return aware.astimezone(UTC)


__all__: Sequence[str] = (
    "KST",
    "AdjustedDailyBar",
    "ConsensusRow",
    "EarningsRow",
    "NewsJudgmentRow",
    "QuantAirflowDBClient",
    "SectorIndexRow",
    "SupplyDemandRow",
    "judged_at_to_kst",
)
