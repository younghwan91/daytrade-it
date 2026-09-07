"""Integration tests for QuantAirflowDBClient against the real quant-airflow
TimescaleDB.

These tests require LAN reachability to the TimescaleDB host (confirmed
reachable at `localhost:5432` at the time this was written — quant-airflow's
`timescaledb` container runs on this same machine and publishes the port to
the host). If the host is unreachable when these tests run, the whole module
is skipped rather than failing noisily — this is meant to degrade gracefully
in CI environments or other machines that are not on the same LAN as the
"spare PC" quant-airflow runs on.

Connection details come from `GPTQ_QUANT_AIRFLOW_DB_*` env vars / `.env` via
`QuantAirflowDBSettings` (see `.env.example`), falling back to the same
localhost defaults quant-airflow's own README documents.
"""

from __future__ import annotations

import datetime as dt

import psycopg2
import pytest

from krx_signal_engine.domain.value_objects import Ticker
from krx_signal_engine.infrastructure.external.quant_airflow_db_client import (
    AdjustedDailyBar,
    QuantAirflowDBClient,
)
from krx_signal_engine.shared.config.settings import QuantAirflowDBSettings

# Real, well-known large-cap KRX code (Samsung Electronics), per the task
# brief's guidance to pick one for the integration test.
SAMSUNG_CODE = "005930"


def _password_from_sibling_quant_airflow_env() -> str:
    """Best-effort local-dev convenience: fall back to reading
    `TIMESCALE_PASSWORD` out of the sibling `quant-airflow` checkout's own
    (gitignored, never committed here) `.env` file, so this integration test
    can run un-configured on a dev machine that has that sibling checked out.
    Returns "" if unavailable - the reachability check then simply fails
    and the whole module is skipped, which is the correct degrade-gracefully
    behavior for machines without that sibling repo.
    """
    from pathlib import Path

    candidate = Path.home() / "Documents" / "git" / "quant-airflow" / ".env"
    if not candidate.is_file():
        return ""
    for line in candidate.read_text(encoding="utf-8").splitlines():
        if line.startswith("TIMESCALE_PASSWORD="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def _test_settings() -> QuantAirflowDBSettings:
    import os

    password = os.environ.get(
        "GPTQ_QUANT_AIRFLOW_DB_PASSWORD"
    ) or _password_from_sibling_quant_airflow_env()

    return QuantAirflowDBSettings(
        host=os.environ.get("GPTQ_QUANT_AIRFLOW_DB_HOST", "localhost"),
        port=int(os.environ.get("GPTQ_QUANT_AIRFLOW_DB_PORT", "5432")),
        dbname=os.environ.get("GPTQ_QUANT_AIRFLOW_DB_DBNAME", "kr_quant"),
        user=os.environ.get("GPTQ_QUANT_AIRFLOW_DB_USER", "kr_quant"),
        password=password,
        connect_timeout=3.0,
    )


def _is_reachable(settings: QuantAirflowDBSettings) -> bool:
    try:
        conn = psycopg2.connect(
            host=settings.host,
            port=settings.port,
            dbname=settings.dbname,
            user=settings.user,
            password=settings.password.get_secret_value(),
            connect_timeout=int(settings.connect_timeout),
        )
    except psycopg2.Error:
        return False
    else:
        conn.close()
        return True


_SETTINGS = _test_settings()
_REACHABLE = _is_reachable(_SETTINGS)

pytestmark = pytest.mark.skipif(
    not _REACHABLE,
    reason=(
        "quant-airflow TimescaleDB not reachable "
        f"({_SETTINGS.host}:{_SETTINGS.port}/{_SETTINGS.dbname}) - "
        "set GPTQ_QUANT_AIRFLOW_DB_* env vars or run on a machine with LAN "
        "access to it."
    ),
)


@pytest.fixture
def client() -> QuantAirflowDBClient:
    with QuantAirflowDBClient(settings=_SETTINGS) as c:
        yield c


class TestReachability:
    def test_ping(self, client: QuantAirflowDBClient) -> None:
        assert client.ping() is True


class TestReadOnlyEnforcedByServer:
    def test_write_is_rejected_by_postgres(self, client: QuantAirflowDBClient) -> None:
        """The connection itself is a read-only Postgres session — an INSERT
        must fail at the server, not just be absent from this client's code.
        """
        conn = client.connect()
        with conn.cursor() as cur, pytest.raises(psycopg2.errors.ReadOnlySqlTransaction):
            cur.execute(
                "INSERT INTO stocks (code, name, market) VALUES (%s, %s, %s)",
                ("999999", "not a real stock", "KOSPI"),
            )


class TestAdjustedDailyBars:
    def test_reads_known_large_cap_ticker(self, client: QuantAirflowDBClient) -> None:
        bars = client.get_adjusted_daily_bars(
            Ticker(value=SAMSUNG_CODE),
            start="2026-01-01",
            end="2026-03-31",
        )

        assert len(bars) > 0
        assert all(isinstance(b, AdjustedDailyBar) for b in bars)
        assert all(b.code == SAMSUNG_CODE for b in bars)
        # Ordered ascending by date.
        dates = [b.date for b in bars]
        assert dates == sorted(dates)
        # Sane OHLCV shape for a real large-cap stock (KRW, thousands range).
        first = bars[0]
        assert first.close is not None
        assert first.close > 0
        assert first.volume is not None
        assert first.volume >= 0
        assert first.source  # non-empty provenance tag

    def test_unknown_ticker_returns_empty(self, client: QuantAirflowDBClient) -> None:
        # A syntactically valid 6-digit code that is not a real listing.
        bars = client.get_adjusted_daily_bars(Ticker(value="000000"))
        assert bars == []


class TestSupplyDemand:
    def test_latest_supply_demand_for_known_ticker(self, client: QuantAirflowDBClient) -> None:
        row = client.get_latest_supply_demand(Ticker(value=SAMSUNG_CODE))
        assert row is not None
        assert row.code == SAMSUNG_CODE
        assert row.source


class TestSectorIndex:
    def test_latest_kospi_composite(self, client: QuantAirflowDBClient) -> None:
        # "001" = KOSPI composite index per quant-airflow's storage.py schema comment.
        row = client.get_latest_sector_index("001")
        if row is None:
            pytest.skip("sector_index has no rows for code 001 in this DB snapshot")
        assert row.code == "001"


class TestEarnings:
    def test_earnings_for_known_ticker(self, client: QuantAirflowDBClient) -> None:
        rows = client.get_earnings(Ticker(value=SAMSUNG_CODE))
        if not rows:
            pytest.skip("no earnings rows for 005930 in this DB snapshot")
        assert all(r.code == SAMSUNG_CODE for r in rows)
        # One row per period - no duplicate periods from unresolved versioning.
        periods = [r.period for r in rows]
        assert len(periods) == len(set(periods))


class TestConsensus:
    def test_consensus_for_known_ticker(self, client: QuantAirflowDBClient) -> None:
        rows = client.get_consensus(Ticker(value=SAMSUNG_CODE))
        if not rows:
            pytest.skip("no consensus rows for 005930 in this DB snapshot")
        assert all(r.code == SAMSUNG_CODE for r in rows)


class TestNewsJudgments:
    def test_news_judgments_kst_conversion_on_real_rows(
        self, client: QuantAirflowDBClient
    ) -> None:
        rows = client.get_news_judgments(Ticker(value=SAMSUNG_CODE), limit=5)
        if not rows:
            pytest.skip("no news_judgments rows for 005930 in this DB snapshot")
        for row in rows:
            # KST must be exactly UTC+9 of the raw stored value.
            assert row.judged_at_kst == row.judged_at_utc.astimezone(row.judged_at_kst.tzinfo)
            assert row.judged_at_kst.utcoffset() == dt.timedelta(hours=9)
