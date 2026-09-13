"""Unit tests for QuantAirflowDBClient.

These tests do not touch the network or a real database — they cover:
- The KST/UTC conversion logic for `news_judgments.judged_at` with fixed
  inputs (the documented UTC-storage gotcha).
- Row -> DTO mapping logic, using a mocked psycopg2 connection/cursor.
- Read-only enforcement: `connect()` must call `set_session(readonly=True)`.

The DB-reachable integration test lives in
`tests/integration/test_quant_airflow_db_client_integration.py`.
"""

from __future__ import annotations

import datetime as dt
from unittest.mock import MagicMock

import pytest

from daytrade_it.domain.value_objects import Ticker
from daytrade_it.infrastructure.external.quant_airflow_db_client import (
    KST,
    AdjustedDailyBar,
    NewsJudgmentRow,
    QuantAirflowDBClient,
    judged_at_to_kst,
)
from daytrade_it.shared.config.settings import QuantAirflowDBSettings

# =============================================================================
# KST/UTC conversion (fixed inputs, no DB)
# =============================================================================


class TestJudgedAtToKst:
    """`judged_at` is stored TIMESTAMPTZ **in UTC** — a documented gotcha
    (quant-airflow docs/schema.md:36). KST = UTC+9, no DST.
    """

    def test_converts_aware_utc_to_kst(self) -> None:
        judged_at = dt.datetime(2026, 3, 15, 15, 30, tzinfo=dt.UTC)
        result = judged_at_to_kst(judged_at)
        assert result == dt.datetime(2026, 3, 16, 0, 30, tzinfo=KST)

    def test_crosses_midnight_into_next_kst_day(self) -> None:
        """15:01 UTC is 00:01 KST the *next* calendar day — the exact class
        of bug this conversion exists to prevent (a naive same-day compare
        against a KST-naive column would misclassify this row).
        """
        judged_at = dt.datetime(2026, 3, 15, 15, 1, tzinfo=dt.UTC)
        result = judged_at_to_kst(judged_at)
        assert result.date() == dt.date(2026, 3, 16)
        assert judged_at.date() == dt.date(2026, 3, 15)

    def test_just_before_midnight_utc_stays_same_kst_day(self) -> None:
        judged_at = dt.datetime(2026, 3, 15, 14, 59, tzinfo=dt.UTC)
        result = judged_at_to_kst(judged_at)
        assert result.date() == dt.date(2026, 3, 15)

    def test_naive_datetime_assumed_utc(self) -> None:
        judged_at = dt.datetime(2026, 3, 15, 15, 30)  # noqa: DTZ001 - deliberately naive
        result = judged_at_to_kst(judged_at)
        assert result == dt.datetime(2026, 3, 16, 0, 30, tzinfo=KST)

    def test_result_is_kst_offset_plus_nine(self) -> None:
        judged_at = dt.datetime(2026, 6, 1, 0, 0, tzinfo=dt.UTC)
        result = judged_at_to_kst(judged_at)
        assert result.utcoffset() == dt.timedelta(hours=9)


class TestKstToUtc:
    """The inverse direction, used by `get_news_judgments`'s KST-bounded query params."""

    def test_kst_bound_converted_to_utc_for_query(self) -> None:
        kst_value = dt.datetime(2026, 3, 16, 0, 30, tzinfo=KST)
        utc_value = QuantAirflowDBClient._kst_to_utc(kst_value)
        assert utc_value == dt.datetime(2026, 3, 15, 15, 30, tzinfo=dt.UTC)

    def test_naive_kst_bound_assumed_kst(self) -> None:
        naive_kst = dt.datetime(2026, 3, 16, 0, 30)  # noqa: DTZ001 - deliberately naive
        utc_value = QuantAirflowDBClient._kst_to_utc(naive_kst)
        assert utc_value == dt.datetime(2026, 3, 15, 15, 30, tzinfo=dt.UTC)


# =============================================================================
# Row -> DTO mapping (mocked cursor, no DB)
# =============================================================================


def _client_with_mock_cursor(rows: list[tuple]) -> tuple[QuantAirflowDBClient, MagicMock]:
    """Build a client whose `.connect()` returns a mock connection wired to
    return `rows` from `fetchall()`, without ever calling psycopg2.connect.
    """
    client = QuantAirflowDBClient(settings=QuantAirflowDBSettings(host="unused", password="unused"))
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = rows
    mock_cursor.__enter__.return_value = mock_cursor
    mock_cursor.__exit__.return_value = False

    mock_conn = MagicMock()
    mock_conn.closed = False
    mock_conn.cursor.return_value = mock_cursor

    client._conn = mock_conn  # bypass real connect()
    return client, mock_cursor


class TestAdjustedDailyBarsMapping:
    def test_maps_rows_to_dtos_in_order(self) -> None:
        rows = [
            (
                "005930",
                dt.date(2026, 1, 2),
                70000.0,
                71000.0,
                69500.0,
                70500.0,
                1_000_000,
                70_000_000_000,
                "kiwoom",
            ),
            (
                "005930",
                dt.date(2026, 1, 5),
                70500.0,
                72000.0,
                70000.0,
                71800.0,
                1_200_000,
                85_000_000_000,
                "kiwoom",
            ),
        ]
        client, mock_cursor = _client_with_mock_cursor(rows)

        bars = client.get_adjusted_daily_bars(
            Ticker(value="005930"), start="2026-01-01", end="2026-01-31"
        )

        assert len(bars) == 2
        assert all(isinstance(b, AdjustedDailyBar) for b in bars)
        assert bars[0].code == "005930"
        assert bars[0].date == dt.date(2026, 1, 2)
        assert bars[0].close == 70500.0
        assert bars[0].source == "kiwoom"
        assert bars[1].date == dt.date(2026, 1, 5)

        # WHERE clause used both bounds and the ticker.
        sql, params = mock_cursor.execute.call_args[0]
        assert "code = %s" in sql
        assert "date >= %s" in sql
        assert "date <= %s" in sql
        assert params == ("005930", "2026-01-01", "2026-01-31")

    def test_empty_result_returns_empty_list(self) -> None:
        client, _ = _client_with_mock_cursor([])
        bars = client.get_adjusted_daily_bars(Ticker(value="005930"))
        assert bars == []

    def test_no_bounds_omits_date_predicates(self) -> None:
        client, mock_cursor = _client_with_mock_cursor([])
        client.get_adjusted_daily_bars(Ticker(value="005930"))
        sql, params = mock_cursor.execute.call_args[0]
        assert "date >=" not in sql
        assert "date <=" not in sql
        assert params == ("005930",)


class TestSupplyDemandMapping:
    def test_latest_row_mapped(self) -> None:
        rows = [("005930", dt.date(2026, 3, 1), 71000, 1.2, 15_000_000, 100, -200, 150, "kiwoom")]
        client, _ = _client_with_mock_cursor(rows)
        row = client.get_latest_supply_demand(Ticker(value="005930"))
        assert row is not None
        assert row.code == "005930"
        assert row.foreign_ == -200
        assert row.institution == 150

    def test_no_rows_returns_none(self) -> None:
        client, _ = _client_with_mock_cursor([])
        assert client.get_latest_supply_demand(Ticker(value="005930")) is None


class TestNewsJudgmentMapping:
    def test_from_raw_derives_kst_from_utc(self) -> None:
        judged_at_utc = dt.datetime(2026, 3, 15, 15, 1, tzinfo=dt.UTC)
        row = NewsJudgmentRow.from_raw(
            source_type="toss",
            source_id="abc123",
            ticker="005930",
            event_type="EARNINGS",
            sentiment_direction=1,
            is_stale_repeat=False,
            price_impact_likely=True,
            confidence=80,
            knowledge_date=dt.date(2026, 3, 15),
            judged_at=judged_at_utc,
        )
        assert row.judged_at_utc == judged_at_utc
        assert row.judged_at_kst.date() == dt.date(2026, 3, 16)  # crossed midnight

    def test_get_news_judgments_maps_rows_and_converts_bounds(self) -> None:
        judged_at_utc = dt.datetime(2026, 3, 15, 15, 1, tzinfo=dt.UTC)
        rows = [
            (
                "toss",
                "abc123",
                "005930",
                "EARNINGS",
                1,
                False,
                True,
                80,
                dt.date(2026, 3, 15),
                judged_at_utc,
            )
        ]
        client, mock_cursor = _client_with_mock_cursor(rows)

        kst_lower_bound = dt.datetime(2026, 3, 16, 0, 0, tzinfo=KST)
        results = client.get_news_judgments(
            Ticker(value="005930"), judged_after_kst=kst_lower_bound
        )

        assert len(results) == 1
        assert results[0].judged_at_kst.date() == dt.date(2026, 3, 16)

        _, params = mock_cursor.execute.call_args[0]
        # The KST bound passed to the query must have been converted to UTC.
        assert params[1] == dt.datetime(2026, 3, 15, 15, 0, tzinfo=dt.UTC)


# =============================================================================
# Read-only enforcement
# =============================================================================


class TestReadOnlyEnforcement:
    def test_connect_enforces_readonly_session(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_conn = MagicMock()
        mock_conn.closed = False
        mock_connect = MagicMock(return_value=mock_conn)
        monkeypatch.setattr(
            "daytrade_it.infrastructure.external.quant_airflow_db_client.psycopg2.connect",
            mock_connect,
        )

        client = QuantAirflowDBClient(
            settings=QuantAirflowDBSettings(host="db.local", password="secret")
        )
        client.connect()

        mock_conn.set_session.assert_called_once_with(readonly=True, autocommit=True)

    def test_no_write_statements_anywhere_in_module(self) -> None:
        """Static guard: this module must never contain SQL write verbs."""
        import inspect

        from daytrade_it.infrastructure.external import quant_airflow_db_client as mod

        source = inspect.getsource(mod)
        for verb in ("INSERT INTO", "UPDATE ", "DELETE FROM", "DROP ", "TRUNCATE", "ALTER "):
            assert verb not in source, f"Found forbidden write verb {verb!r} in client module"

    def test_client_exposes_no_raw_execute_escape_hatch(self) -> None:
        """Only the private `_query` helper (read-only SELECTs by construction)
        touches the cursor — there is no public `execute()`/`execute_sql()`
        that would let a caller smuggle in arbitrary SQL.
        """
        public_methods = [
            name
            for name in dir(QuantAirflowDBClient)
            if not name.startswith("_") and callable(getattr(QuantAirflowDBClient, name))
        ]
        assert "execute" not in public_methods
        assert "execute_sql" not in public_methods
        assert "cursor" not in public_methods
