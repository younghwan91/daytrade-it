"""build_universe_events (plan C3): synthetic C1 files + fake daily values, no DB."""

import datetime as dt
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "eval"))

from build_universe_events import (  # noqa: E402
    bucket_articles,
    build_day_events,
    day_pairs,
    load_articles,
    measure_judge_lag,
    pair_rows,
    parse_days,
    run_day,
)

from daytrade_it.application.news.universe import live_article_id, next_visit
from daytrade_it.application.replay.events import load_events
from daytrade_it.infrastructure.brokers.base import KST
from daytrade_it.infrastructure.external.quant_airflow_db_client import StockInfo

D = dt.date(2026, 9, 15)
PREV = dt.date(2026, 9, 14)
FEATS = {
    "subject": "main",
    "reports_price_move": False,
    "material_type": "수주",
    "persistence": "multi_quarter",
    "specificity": 3,
    "freshness": "new",
    "surprise": 1,
    "sentiment_direction": 1,
    "sentiment_strength": 2,
    "scale_vs_size": "medium",
    "score": 0.5,
}
STOCKS = {
    "005930": StockInfo(code="005930", name="삼성전자", market="KOSPI", sector="전기전자"),
    "000660": StockInfo(code="000660", name="SK하이닉스", market="KOSPI", sector="전기전자"),
    "123456": StockInfo(code="123456", name="작은회사", market="KOSDAQ", sector="IT"),
}


def _write_c1(root: Path, code: str, rows: list[dict]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    with (root / f"{code}.jsonl").open("a") as f:
        for r in rows:
            f.write(json.dumps({**r, "code": code}, ensure_ascii=False) + "\n")


def _row(news_id: str, created: str, title: str = "t", summary: str = "s") -> dict:
    return {
        "id": news_id,
        "title": title,
        "summary": summary,
        "createdAt": created,
        "source": "x",
        "newsType": "NEWS",
    }


@pytest.fixture
def c1(tmp_path: Path) -> Path:
    root = tmp_path / "toss_universe"
    # n1: pre-open on d, under two codes (one is outside the universe)
    _write_c1(root, "005930", [_row("n1", "2026-09-15T07:30:00", "삼성 수주", "요약1")])
    _write_c1(root, "123456", [_row("n1", "2026-09-15T07:30:00", "삼성 수주", "요약1")])
    # n2: intraday on d under 000660; n3: prior day after 15:20 (in window);
    # n4: prior day before 15:20 (out of window); n5: d after 15:20 (out)
    _write_c1(
        root,
        "000660",
        [
            _row("n2", "2026-09-15T10:00:00.5"),
            _row("n3", "2026-09-14T16:00:00"),
            _row("n4", "2026-09-14T15:00:00"),
            _row("n5", "2026-09-15T15:30:00"),
        ],
    )
    return root


def test_load_articles_groups_codes_by_article_id(c1: Path) -> None:
    arts = load_articles(c1)
    assert arts["n1"]["codes"] == ("005930", "123456")
    assert arts["n1"]["summary"] == "요약1"
    assert arts["n2"]["published_at"] == dt.datetime(2026, 9, 15, 10, 0, 0, 500000, tzinfo=KST)


def test_day_pairs_window_universe_and_visit_times(c1: Path) -> None:
    universe = ["005930", "000660"]
    pairs = day_pairs(load_articles(c1), D, PREV, universe)
    got = {(p["news_id"], p["code"]) for p in pairs}
    assert got == {("n1", "005930"), ("n2", "000660"), ("n3", "000660")}
    by = {p["news_id"]: p for p in pairs}
    start = dt.datetime(2026, 9, 15, 8, 0, tzinfo=KST)
    # pre-open / prior-day articles are first seen on the sweep from 08:00
    assert by["n1"]["visit_at"] == start
    assert by["n3"]["visit_at"] == start + dt.timedelta(seconds=1)
    assert by["n2"]["visit_at"] == next_visit(
        "000660", universe, D, dt.datetime(2026, 9, 15, 10, 0, 0, 500000, tzinfo=KST)
    )
    assert by["n1"]["key"] == f"{live_article_id('n1')}|005930"
    assert by["n1"]["session"] == "pre_market"
    assert by["n2"]["session"] == "intraday"


def test_day_pairs_drops_articles_whose_visit_is_after_cutoff(tmp_path: Path) -> None:
    root = tmp_path / "u"
    _write_c1(root, "005930", [_row("late", "2026-09-15T15:19:59.5")])
    universe = ["000660", "005930"]  # 005930 slot is odd seconds -> 15:20:01 > cutoff
    assert day_pairs(load_articles(root), D, PREV, universe) == []


def test_build_day_events_uses_cache_and_counts_excluded(c1: Path) -> None:
    universe = ["005930", "000660"]
    pairs = day_pairs(load_articles(c1), D, PREV, universe)
    cache = {pairs[0]["key"]: FEATS}
    lag = dt.timedelta(seconds=7)
    events, excluded = build_day_events(pairs, STOCKS, cache, lag)
    assert excluded == len(pairs) - 1
    (e,) = events
    assert e.source == "universe"
    assert e.decide_at == pairs[0]["visit_at"] + lag
    assert e.batch_at == e.decide_at + dt.timedelta(seconds=1)
    assert e.features == FEATS
    assert e.article_id == live_article_id(pairs[0]["news_id"])
    assert e.name == STOCKS[e.ticker].name


def test_pair_rows_carry_c4_fields(c1: Path) -> None:
    pairs = day_pairs(load_articles(c1), D, PREV, ["005930", "000660"])
    rows = pair_rows(pairs, STOCKS)
    r = next(r for r in rows if r["code"] == "005930")
    for k in ("key", "code", "name", "market", "sector", "title", "body", "category"):
        assert k in r
    assert r["body"] == "요약1"
    assert r["name"] == "삼성전자"
    assert r["month"] == "2026-09"
    assert r["session"] == "pre_market"


def test_measure_judge_lag_is_median_first_judgment_after_poll() -> None:
    polls = [
        dt.datetime(2026, 9, 15, 9, 0, 0, tzinfo=KST),
        dt.datetime(2026, 9, 15, 9, 3, 0, tzinfo=KST),
    ]
    rows = [
        {"judged_at": "2026-09-15T09:00:04+09:00"},
        {"judged_at": "2026-09-15T09:00:09+09:00"},  # queued behind first -- not a lag sample
        {"judged_at": "2026-09-15T09:03:06+09:00"},
    ]
    assert measure_judge_lag(rows, polls) == dt.timedelta(seconds=5)


def test_parse_days_list_and_range() -> None:
    days = parse_days(
        "2026-09-11..2026-09-15",
        trading_days=[dt.date(2026, 9, 11), dt.date(2026, 9, 14), dt.date(2026, 9, 15)],
    )
    assert days == [dt.date(2026, 9, 11), dt.date(2026, 9, 14), dt.date(2026, 9, 15)]
    assert parse_days("2026-09-14,2026-09-13", trading_days=[dt.date(2026, 9, 14)]) == [
        dt.date(2026, 9, 14)
    ]


def test_run_day_writes_event_file_with_fake_daily_values(c1: Path, tmp_path: Path) -> None:
    def daily_values(day: dt.date) -> dict[str, float]:
        assert day == PREV
        return {"005930": 3e12, "000660": 2e12, "123456": 0.0}

    arts = load_articles(c1)
    pairs = day_pairs(arts, D, PREV, ["005930", "000660"])
    cache = {p["key"]: FEATS for p in pairs}
    out = tmp_path / "events"
    n, excluded = run_day(arts, D, PREV, daily_values, STOCKS, cache, dt.timedelta(seconds=5), out)
    assert (n, excluded) == (3, 0)
    events = load_events(out / f"{D}.jsonl")
    assert [e.decide_at for e in events] == sorted(e.decide_at for e in events)
    assert all(e.source == "universe" for e in events)


def test_all_pairs_of_an_article_share_the_first_related_visit(tmp_path: Path) -> None:
    root = tmp_path / "u"
    row = _row("multi", "2026-09-15T10:00:00.5")
    _write_c1(root, "005930", [row])  # slot 0: even seconds -> own visit 10:00:02
    _write_c1(root, "000660", [row])  # slot 1: odd seconds -> own visit 10:00:01
    universe = ["005930", "000660"]
    pairs = day_pairs(load_articles(root), D, PREV, universe)
    assert {p["code"] for p in pairs} == {"005930", "000660"}
    first = dt.datetime(2026, 9, 15, 10, 0, 1, tzinfo=KST)
    assert [p["visit_at"] for p in pairs] == [first, first]
    events, _ = build_day_events(
        pairs, STOCKS, {p["key"]: FEATS for p in pairs}, dt.timedelta(seconds=3)
    )
    assert {e.decide_at for e in events} == {first + dt.timedelta(seconds=3)}


def test_bucketed_day_pairs_match_full_scan(c1: Path) -> None:
    arts = load_articles(c1)
    trading = [dt.date(2026, 9, 11), PREV, D]
    buckets = bucket_articles(arts, trading)
    universe = ["005930", "000660", "123456"]
    for prev, day in [(dt.date(2026, 9, 11), PREV), (PREV, D)]:
        assert day_pairs(buckets.get(day, []), day, prev, universe) == day_pairs(
            arts, day, prev, universe
        )
    assert sum(len(v) for v in buckets.values()) <= len(arts)
