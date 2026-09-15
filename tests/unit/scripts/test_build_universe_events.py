"""build_universe_events (plan C3): synthetic C1 files + fake daily values, no DB."""

import datetime as dt
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "eval"))

from build_universe_events import (  # noqa: E402
    IncompleteDay,
    bucket_articles,
    build_day_events,
    check_day_coverage,
    day_pairs,
    load_articles,
    load_c1,
    measure_judge_lag,
    pair_rows,
    parse_days,
    prepare_day,
)

from daytrade_it.application.news.universe import live_article_id, next_visit
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


def _write_c1(
    root: Path, code: str, rows: list[dict], status: str = "done", since: str = "2026-09-01"
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    with (root / f"{code}.jsonl").open("a") as f:
        for r in rows:
            f.write(json.dumps({**r, "code": code}, ensure_ascii=False) + "\n")
    (root / f"{code}.done").write_text(json.dumps({"status": status, "rows": 0, "since": since}))


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
    cache = {pairs[0]["key"]: [{"v2": FEATS}]}
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
        pairs, STOCKS, {p["key"]: [{"v2": FEATS}] for p in pairs}, dt.timedelta(seconds=3)
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


# --- F2 (S12): C1 coverage -----------------------------------------------------


def test_load_c1_reads_only_finished_codes(tmp_path: Path) -> None:
    root = tmp_path / "u"
    _write_c1(root, "005930", [_row("a", "2026-09-15T09:00:00")])
    # still being backfilled: no .done, plus a partial file
    (root / "000660.jsonl").write_text(
        json.dumps({**_row("b", "2026-09-15T09:00:00"), "code": "000660"}) + "\n"
    )
    (root / "000660.partial.jsonl").write_text(
        json.dumps({**_row("c", "2026-09-15T09:00:00"), "code": "000660"}) + "\n"
    )
    arts, coverage = load_c1(root)
    assert set(arts) == {"a"}
    assert set(coverage) == {"005930"}
    assert load_articles(root).keys() == arts.keys()


def test_coverage_start_is_since_for_done_and_earliest_row_for_page_cap(tmp_path: Path) -> None:
    root = tmp_path / "u"
    _write_c1(root, "005930", [_row("a", "2026-09-15T09:00:00")], since="2026-09-01")
    _write_c1(
        root,
        "000660",
        [
            _row("b", "2026-09-15T09:00:00"),
            _row("c", "2026-09-14T17:00:00"),
            _row("d", "2026-09-15T08:00:00"),
        ],
        status="page_cap",
    )
    _, coverage = load_c1(root)
    assert coverage["005930"]["status"] == "done"
    assert coverage["005930"]["start"] == dt.datetime(2026, 9, 1, tzinfo=KST)
    assert coverage["000660"]["status"] == "page_cap"
    assert coverage["000660"]["start"] == dt.datetime(2026, 9, 14, 17, 0, tzinfo=KST)


def test_day_pairs_drops_truncated_code_before_its_coverage_start(tmp_path: Path) -> None:
    root = tmp_path / "u"
    row = _row("x", "2026-09-15T09:00:00")
    _write_c1(root, "005930", [row])
    # 000660 page_cap: earliest row 09-14 17:00 > D's window start (09-14 15:20) -> not covered on D
    _write_c1(root, "000660", [row, _row("y", "2026-09-14T17:00:00")], status="page_cap")
    arts, coverage = load_c1(root)
    universe = ["005930", "000660"]
    stats: dict = {}
    pairs = day_pairs(arts, D, PREV, universe, coverage=coverage, stats=stats)
    assert {(p["news_id"], p["code"]) for p in pairs} == {
        ("x", "005930")
    }  # y and (x, 000660) dropped
    assert stats["excluded_pairs"] == 2
    # the next day's window starts 09-15 15:20 >= coverage start: covered again
    nxt = dt.date(2026, 9, 16)
    late = _row("z", "2026-09-16T09:00:00")
    _write_c1(root, "000660", [late], status="page_cap")
    arts, coverage = load_c1(root)
    assert {p["code"] for p in day_pairs(arts, nxt, D, universe, coverage=coverage)} == {"000660"}


def test_first_backfill_day_is_fully_excluded(tmp_path: Path) -> None:
    root = tmp_path / "u"
    _write_c1(root, "005930", [_row("a", "2026-03-16T09:00:00")], since="2026-03-16")
    arts, coverage = load_c1(root)
    stats: dict = {}
    # 2026-03-16's window starts 2026-03-13 15:20, before since -> nothing is covered
    pairs = day_pairs(arts, dt.date(2026, 3, 16), dt.date(2026, 3, 13), ["005930"], coverage, stats)
    assert pairs == [] and stats["excluded_pairs"] == 1


def test_check_day_coverage_refuses_missing_codes(tmp_path: Path) -> None:
    root = tmp_path / "u"
    _write_c1(root, "005930", [])
    _write_c1(root, "000660", [_row("y", "2026-09-15T08:00:00")], status="page_cap")
    _, coverage = load_c1(root)
    universe = ["005930", "000660", "123456"]
    with pytest.raises(IncompleteDay, match="123456"):
        check_day_coverage(universe, coverage, D, PREV)
    report = check_day_coverage(universe, coverage, D, PREV, allow_incomplete=True)
    assert report["missing"] == ["123456"]
    assert report["truncated"] == ["000660"]
    assert report["uncovered"] == ["000660", "123456"]
    assert report["universe"] == 3


def test_prepare_day_prints_coverage_and_dashboard_overlap(tmp_path: Path) -> None:
    root = tmp_path / "u"
    _write_c1(root, "005930", [_row("a", "2026-09-15T09:00:00")])
    _write_c1(
        root,
        "000660",
        [_row("a", "2026-09-15T09:00:00"), _row("b", "2026-09-15T08:00:00")],
        status="page_cap",
    )
    arts, coverage = load_c1(root)
    archive = tmp_path / "news_judgments"
    archive.mkdir()
    (archive / f"{D}.jsonl").write_text(
        json.dumps({"key": f"{live_article_id('a')}|005930"}) + "\n"
    )
    pairs, line = prepare_day(
        arts.values(), D, PREV, ["005930", "000660"], coverage, False, archive
    )
    assert [p["code"] for p in pairs] == ["005930"]
    assert "universe 2 truncated 1 missing 0 uncovered 1 pairs 1 excluded_pairs 2" in line
    assert line.endswith("dashboard_overlap 1")
    with pytest.raises(IncompleteDay):
        prepare_day(arts.values(), D, PREV, ["005930", "999999"], coverage, False, archive)
    _, line = prepare_day(arts.values(), D, PREV, ["005930"], coverage, False, tmp_path / "none")
    assert line.endswith("dashboard_overlap n/a")


# --- F5: body fallback + input-hash cache -------------------------------------


def test_pair_rows_body_falls_back_to_title_when_summary_empty(tmp_path: Path) -> None:
    root = tmp_path / "u"
    _write_c1(root, "005930", [_row("e", "2026-09-15T09:00:00", title="제목만", summary="")])
    pairs = day_pairs(load_articles(root), D, PREV, ["005930"])
    (row,) = pair_rows(pairs, STOCKS)
    assert row["body"] == "제목만" and row["summary"] == ""


def test_events_use_only_cache_rows_matching_the_pair_input(c1: Path) -> None:
    from judge_batch import pair_input_hash

    from daytrade_it.infrastructure.external.anthropic_client import DEFAULT_MODEL

    universe = ["005930", "000660"]
    pairs = day_pairs(load_articles(c1), D, PREV, universe)
    rows = {r["key"]: r for r in pair_rows(pairs, STOCKS)}
    k0, k1, k2 = (p["key"] for p in pairs)
    cache = {
        k0: [{"v2": FEATS, "input_hash": pair_input_hash(rows[k0], DEFAULT_MODEL)}],
        k1: [{"v2": FEATS, "input_hash": "stale"}],
        k2: [{"v2": FEATS}],  # legacy row; c1 summaries are non-empty -> input unchanged
    }
    stats: dict = {}
    events, excluded = build_day_events(pairs, STOCKS, cache, dt.timedelta(0), stats=stats)
    assert {e.key for e in events} == {k0, k2}
    assert excluded == 1 and stats["stale_cache"] == 1


def test_load_cache_keeps_every_row_per_key(tmp_path: Path) -> None:
    from build_universe_events import load_cache

    path = tmp_path / "c.jsonl"
    path.write_text(
        json.dumps({"key": "a|1", "v2": {"x": 1}})
        + "\n"
        + json.dumps({"key": "a|1", "v2": {"x": 2}, "input_hash": "h"})
        + "\n"
    )
    assert [r.get("input_hash") for r in load_cache(path)["a|1"]] == [None, "h"]
