"""universe_readiness (F7): per-day minute-bar + C1 news readiness for A6 -- pure, no DB."""

import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "eval"))

from build_universe_events import load_c1, load_coverage  # noqa: E402
from universe_readiness import (  # noqa: E402
    day_readiness,
    load_event_tickers,
    readiness_report,
)

from daytrade_it.infrastructure.brokers.base import KST  # noqa: E402

D0 = dt.date(2026, 9, 11)
D1 = dt.date(2026, 9, 14)
D2 = dt.date(2026, 9, 15)


def _c1(root: Path, code: str, created: list[str], status: str = "done", since: str = "2026-09-01"):
    root.mkdir(parents=True, exist_ok=True)
    rows = [
        {"id": f"{code}-{i}", "title": "t", "createdAt": c, "code": code}
        for i, c in enumerate(created)
    ]
    (root / f"{code}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (root / f"{code}.done").write_text(json.dumps({"status": status, "since": since}))


def test_load_coverage_matches_load_c1(tmp_path: Path) -> None:
    _c1(tmp_path, "000001", ["2026-09-14T17:00:00", "2026-09-15T09:00:00"], status="page_cap")
    _c1(tmp_path, "000002", ["2026-09-10T09:00:00"])
    assert load_coverage(tmp_path) == load_c1(tmp_path)[1]
    assert load_coverage(tmp_path)["000001"]["start"] == dt.datetime(2026, 9, 14, 17, tzinfo=KST)


def test_day_readiness_ready_partial_not(tmp_path: Path) -> None:
    _c1(tmp_path, "000001", ["2026-09-10T09:00:00"])
    _c1(tmp_path, "000002", ["2026-09-14T15:00:00"], status="page_cap")  # covered from D2 only
    coverage = load_coverage(tmp_path)
    universe = ["000001", "000002"]
    full = {(d, c): 380 for d in (D1, D2) for c in [*universe, "069500"]}

    ready = day_readiness(D2, D1, universe, full, coverage)
    assert ready["status"] == "ready"

    partial = day_readiness(D1, D0, universe, full, coverage)  # 000002 news not covered on D1
    assert partial["status"] == "partial"
    assert partial["news_uncovered"] == ["000002"] and partial["news_truncated"] == ["000002"]

    few_bars = {**full, (D2, "000002"): 10}
    r = day_readiness(D2, D1, universe, few_bars, coverage)
    assert r["status"] == "partial" and r["bars_missing"] == ["000002"]

    no_etf = {k: v for k, v in full.items() if k != (D2, "069500")}
    assert day_readiness(D2, D1, universe, no_etf, coverage)["status"] == "not_ready"

    missing_done = day_readiness(D2, D1, [*universe, "000003"], full, coverage)
    assert missing_done["status"] == "not_ready" and missing_done["news_missing"] == ["000003"]


def test_readiness_report_counts_and_lists_ready_days(tmp_path: Path) -> None:
    _c1(tmp_path, "000001", ["2026-09-10T09:00:00"])
    coverage = load_coverage(tmp_path)
    universes = {D1: ["000001"], D2: ["000001"]}
    prev = {D1: D0, D2: D1}
    bars = {(D1, "000001"): 380, (D1, "069500"): 380, (D2, "000001"): 380}
    report = readiness_report(universes, prev, bars, coverage)
    assert report["ready_days"] == ["2026-09-14"]
    assert report["counts"] == {"ready": 1, "partial": 0, "not_ready": 1}
    assert [d["day"] for d in report["days"]] == ["2026-09-14", "2026-09-15"]


def test_load_event_tickers_from_pairs_file(tmp_path: Path) -> None:
    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text(
        "".join(
            json.dumps(r) + "\n"
            for r in [
                {"key": "a|000001", "code": "000001", "day": "2026-09-14"},
                {"key": "b|000001", "code": "000001", "day": "2026-09-14"},
                {"key": "b|000002", "code": "000002", "day": "2026-09-14"},
                {"key": "c|000003", "code": "000003", "day": "2026-09-15"},
            ]
        )
    )
    assert load_event_tickers(pairs) == {D1: ["000001", "000001", "000002"], D2: ["000003"]}


def test_load_event_tickers_from_events_dir(tmp_path: Path) -> None:
    events = tmp_path / "events"
    events.mkdir()
    (events / "2026-09-15.jsonl").write_text(
        json.dumps({"key": "c|000003", "ticker": "000003"})
        + "\n"
        + json.dumps({"key": "d|000004", "ticker": "000004"})
        + "\n"
    )
    (events / "meta.json").write_text("{}")  # not a day file: ignored
    assert load_event_tickers(events) == {D2: ["000003", "000004"]}


def test_day_readiness_event_coverage_rule(tmp_path: Path) -> None:
    _c1(tmp_path, "000001", ["2026-09-10T09:00:00"])
    _c1(tmp_path, "000002", ["2026-09-10T09:00:00"])
    _c1(tmp_path, "000003", ["2026-09-10T09:00:00"])
    coverage = load_coverage(tmp_path)
    universe = ["000001", "000002", "000003"]
    # 000003 has no bars, so the strict every-universe-code rule says `partial`.
    bars = {(D2, c): 380 for c in ["000001", "000002", "069500"]}

    # 20 events, 19 on tickers that have bars -> 95% of the rows, but only 2 of
    # the 3 distinct event tickers (66.7%), which is below the rule.
    rows = ["000001"] * 10 + ["000002"] * 9 + ["000003"]
    r = day_readiness(D2, D1, universe, bars, coverage, event_tickers={D2: rows})
    assert r["status"] == "partial"
    assert r["usable"] is False
    assert r["event_tickers"] == 3 and r["event_rows"] == 20
    assert r["event_bars_missing"] == ["000003"]
    assert r["event_bar_frac"] == 2 / 3
    assert r["event_row_frac"] == 19 / 20
    assert r["dropped_events"] == 1

    # Every event ticker has bars: usable even though the universe does not.
    ok = day_readiness(D2, D1, universe, bars, coverage, event_tickers={D2: ["000001", "000002"]})
    assert ok["status"] == "partial" and ok["usable"] is True
    assert ok["dropped_events"] == 0 and ok["event_bar_frac"] == 1.0

    # A lower bar makes the 2-of-3 day usable.
    loose = day_readiness(
        D2, D1, universe, bars, coverage, event_tickers={D2: rows}, event_bar_frac=0.6
    )
    assert loose["usable"] is True and loose["dropped_events"] == 1

    # 069500 without bars is never usable, however good the event coverage is.
    no_etf = {k: v for k, v in bars.items() if k != (D2, "069500")}
    bad = day_readiness(D2, D1, universe, no_etf, coverage, event_tickers={D2: ["000001"]})
    assert bad["status"] == "not_ready" and bad["usable"] is False

    # A day with no events at all is not usable.
    empty = day_readiness(D2, D1, universe, bars, coverage, event_tickers={D2: []})
    assert empty["usable"] is False and empty["event_tickers"] == 0

    # Without an event-ticker source the relaxed view is not computed.
    assert "usable" not in day_readiness(D2, D1, universe, bars, coverage)


def test_readiness_report_usable_days(tmp_path: Path) -> None:
    _c1(tmp_path, "000001", ["2026-09-10T09:00:00"])
    _c1(tmp_path, "000002", ["2026-09-10T09:00:00"])
    coverage = load_coverage(tmp_path)
    universes = {D1: ["000001", "000002"], D2: ["000001", "000002"]}
    prev = {D1: D0, D2: D1}
    bars = {
        (D1, "000001"): 380,
        (D1, "069500"): 380,
        (D2, "000001"): 380,
        (D2, "069500"): 380,
    }  # 000002 never has bars
    report = readiness_report(
        universes,
        prev,
        bars,
        coverage,
        event_tickers={D1: ["000001", "000002"], D2: ["000001"]},
    )
    assert report["counts"] == {"ready": 0, "partial": 2, "not_ready": 0, "usable": 1}
    assert report["ready_days"] == []
    assert report["usable_days"] == ["2026-09-15"]
    assert report["dropped_events"] == 1
