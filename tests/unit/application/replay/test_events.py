"""Replay events: archive rows and kr_quant rows → timed, judged pairs."""

import datetime as dt
from pathlib import Path

import pytest

from daytrade_it.application.replay.events import (
    dump_events,
    events_from_archive,
    features_of,
    load_events,
    median_detect_delay,
    schedule_detected,
    to_article,
)
from daytrade_it.infrastructure.brokers.base import KST
from daytrade_it.infrastructure.external.quant_airflow_db_client import StockInfo

V2 = {
    "subject": "main",
    "reports_price_move": False,
    "material_type": "수주",
    "persistence": "multi_quarter",
    "specificity": 3,
    "freshness": "new",
    "surprise": 2,
    "sentiment_direction": 1,
    "sentiment_strength": 3,
    "scale_vs_size": "large",
    "score": 0.8,
}


def _row(aid: str, ticker: str, pub: str, judged: str) -> dict:
    return {
        "key": f"{aid}|{ticker}",
        "article_id": aid,
        "ticker": ticker,
        "name": "n",
        "title": "t",
        "category": "stock",
        "published_at": pub,
        "judged_at": judged,
        "v2": V2,
    }


def _t(hms: str) -> dt.datetime:
    h, m, s = map(int, hms.split(":"))
    return dt.datetime(2026, 9, 15, h, m, s, tzinfo=KST)


A = "0a0a0a0a-0000-5000-8000-000000000001"
B = "0a0a0a0a-0000-5000-8000-000000000002"
STOCKS = {"005930": StockInfo(code="005930", name="삼성전자", market="KOSPI", sector="전기/전자")}


def test_archive_dedups_and_batches_by_poll() -> None:
    rows = [
        _row(A, "005930", "2026-09-15T08:40:00+09:00", "2026-09-15T09:00:35+09:00"),
        _row(A, "005930", "2026-09-15T08:40:00+09:00", "2026-09-15T09:01:40+09:00"),  # dup daemon
        _row(B, "005930", "2026-09-15T09:02:00+09:00", "2026-09-15T09:04:10+09:00"),
    ]
    polls = [_t("09:00:31"), _t("09:04:02")]
    ev = events_from_archive(rows, polls, STOCKS)
    assert [(e.article_id, e.decide_at, e.batch_at) for e in ev] == [
        (A, _t("09:00:35"), _t("09:00:36")),
        (B, _t("09:04:10"), _t("09:04:11")),
    ]
    assert ev[0].sector == "전기/전자" and ev[0].source == "archive"
    assert features_of(ev[0]).surprise == 2
    assert to_article(ev[0]).tickers[0].value == "005930"


def test_median_detect_delay_ignores_premarket() -> None:
    rows = [
        _row(A, "1", "2026-09-15T08:00:00+09:00", "2026-09-15T09:00:35+09:00"),
        _row(A, "2", "2026-09-15T10:00:00+09:00", "2026-09-15T10:10:00+09:00"),
        _row(A, "3", "2026-09-15T11:00:00+09:00", "2026-09-15T11:30:00+09:00"),
    ]
    assert median_detect_delay(rows) == dt.timedelta(minutes=20)


def test_median_detect_delay_raises_when_all_premarket() -> None:
    rows = [
        _row(A, "1", "2026-09-15T08:00:00+09:00", "2026-09-15T08:30:00+09:00"),
        _row(A, "2", "2026-09-15T08:10:00+09:00", "2026-09-15T08:40:00+09:00"),
    ]
    with pytest.raises(ValueError, match="no post-open articles to measure detect delay"):
        median_detect_delay(rows)


def test_schedule_detected_snaps_to_poll_grid() -> None:
    items = [(A, "005930", _t("08:10:00")), (B, "005930", _t("09:10:00"))]
    out = schedule_detected(
        items,
        dt.date(2026, 9, 15),
        dt.timedelta(minutes=10),
        dt.timedelta(minutes=3),
        dt.timedelta(seconds=2),
    )
    assert out[(A, "005930")] == (_t("09:00:30"), _t("09:00:31"))
    # 09:20:00 → next grid slot 09:21:30
    assert out[(B, "005930")] == (_t("09:21:30"), _t("09:21:31"))


def test_dump_load_roundtrip(tmp_path: Path) -> None:
    rows = [_row(A, "005930", "2026-09-15T08:40:00+09:00", "2026-09-15T09:00:35+09:00")]
    ev = events_from_archive(rows, [_t("09:00:31")], STOCKS)
    p = tmp_path / "e.jsonl"
    dump_events(p, ev)
    assert load_events(p) == ev
