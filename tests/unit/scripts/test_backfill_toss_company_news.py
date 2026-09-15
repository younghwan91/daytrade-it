"""backfill_toss_company_news: pure helpers (pause window, block guard, paging, live key)."""

import datetime as dt
import sys
from pathlib import Path
from uuid import uuid5

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "eval"))

import backfill_toss_company_news as bf  # noqa: E402
from krx_news_client.scrapers.base import make_article_id  # noqa: E402
from krx_news_client.scrapers.toss import build_article_url  # noqa: E402

from daytrade_it.infrastructure.external.toss_news_client import TOSS_NEWS_NAMESPACE  # noqa: E402

KST = dt.timezone(dt.timedelta(hours=9))


def _kst(y, mo, d, h, mi, s=0):
    return dt.datetime(y, mo, d, h, mi, s, tzinfo=KST)


# --- market-hours pause -------------------------------------------------------


def test_pause_zero_outside_window_on_weekday():
    assert bf.market_pause_seconds(_kst(2026, 9, 16, 1, 7)) == 0  # Wed 01:07
    assert bf.market_pause_seconds(_kst(2026, 9, 16, 8, 19, 59)) == 0
    assert bf.market_pause_seconds(_kst(2026, 9, 16, 15, 45)) == 0
    assert bf.market_pause_seconds(_kst(2026, 9, 16, 23, 0)) == 0


def test_pause_until_1545_inside_window_on_weekday():
    assert bf.market_pause_seconds(_kst(2026, 9, 16, 8, 20)) == 7 * 3600 + 25 * 60
    assert bf.market_pause_seconds(_kst(2026, 9, 16, 15, 44, 30)) == 30
    assert bf.market_pause_seconds(_kst(2026, 9, 18, 12, 0)) == 3 * 3600 + 45 * 60  # Fri


def test_pause_zero_on_weekend():
    assert bf.market_pause_seconds(_kst(2026, 9, 19, 10, 0)) == 0  # Sat
    assert bf.market_pause_seconds(_kst(2026, 9, 20, 10, 0)) == 0  # Sun


def test_pause_converts_other_timezones_to_kst():
    # 00:00 UTC == 09:00 KST on a Wednesday -> inside the window
    utc = dt.datetime(2026, 9, 16, 0, 0, tzinfo=dt.UTC)
    assert bf.market_pause_seconds(utc) == 6 * 3600 + 45 * 60


# --- block guard ----------------------------------------------------------------


def test_block_guard_trips_after_three_consecutive_429_or_403():
    g = bf.BlockGuard(limit=3)
    assert not g.record(429)
    assert not g.record(403)
    assert g.record(429)
    assert g.blocked


def test_block_guard_resets_on_non_block_status():
    g = bf.BlockGuard(limit=3)
    g.record(429)
    g.record(429)
    g.record(200)
    assert not g.record(429)
    g.record(503)  # a 5xx is not a block signal and breaks the streak
    assert not g.record(403)
    assert not g.blocked


def test_backoff_delay_is_exponential_and_capped():
    assert bf.backoff_delay(0, base=5, cap=60) == 5
    assert bf.backoff_delay(1, base=5, cap=60) == 10
    assert bf.backoff_delay(3, base=5, cap=60) == 40
    assert bf.backoff_delay(10, base=5, cap=60) == 60


# --- paging until --since ----------------------------------------------------------


def _art(news_id, created):
    return {
        "id": news_id,
        "title": "t",
        "summary": "s",
        "createdAt": created,
        "stockCodes": None,
        "newsType": "N",
        "ticsTitles": [],
        "source": {"code": "X"},
    }


def test_page_rows_keeps_since_and_newer_and_signals_more():
    body = [_art("a", "2026-09-15T10:00:00"), _art("b", "2026-03-16T00:00:01")]
    rows, more = bf.page_rows(body, since="2026-03-16", code="005930")
    assert [r["id"] for r in rows] == ["a", "b"]
    assert rows[0]["source"] == "X" and rows[0]["code"] == "005930"
    assert more


def test_page_rows_keeps_paging_when_order_is_loose():
    # Toss lists are only roughly newest-first (verified 2026-09-16: ~45 inversions per page of
    # 005930), so an old last row must not stop paging while any row is still >= since.
    body = [
        _art("a", "2026-03-15T10:00:00"),
        _art("b", "2026-03-17T00:00:00"),
        _art("c", "2026-03-10T00:00:00"),
    ]
    rows, more = bf.page_rows(body, since="2026-03-16", code="005930")
    assert [r["id"] for r in rows] == ["b"]
    assert more


def test_page_rows_stops_only_when_whole_page_is_older_or_empty():
    body = [_art("a", "2026-03-15T23:59:59"), _art("b", "2026-03-01T00:00:00")]
    assert bf.page_rows(body, since="2026-03-16", code="1") == ([], False)
    assert bf.page_rows([], since="2026-03-16", code="1") == ([], False)


# --- live key -------------------------------------------------------------------------


def test_live_article_id_matches_daemon_derivation():
    news_id = "1234567890abcdef"
    expected = str(uuid5(TOSS_NEWS_NAMESPACE, make_article_id("toss", build_article_url(news_id))))
    assert bf.live_article_id(news_id) == expected
    assert bf.live_article_id(news_id) != bf.live_article_id("other")


# --- progress ---------------------------------------------------------------------------


def test_progress_line_reports_counts_and_eta():
    line = bf.progress_line(
        done=50,
        total=150,
        articles=1234,
        requests=80,
        errors=2,
        elapsed_active_s=100.0,
        done_this_run=50,
    )
    assert "codes 50/150" in line and "articles 1234" in line
    assert "requests 80" in line and "errors 2" in line
    assert "eta 0h03m" in line  # 100 codes left at 2 s/code = 200 s
