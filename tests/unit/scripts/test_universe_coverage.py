"""select_universe / compute_universes / coverage_rows / summarize — pure, no DB."""

import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "eval"))

from universe_coverage import (  # noqa: E402
    compute_universes,
    coverage_rows,
    missing_pairs,
    select_universe,
    summarize,
)


def test_select_universe_ranks_desc_and_tie_breaks_by_code():
    rows = [("003", 10.0), ("001", 30.0), ("002", 30.0), ("004", 5.0)]
    assert select_universe(rows, size=3) == ["001", "002", "003", "069500"]


def test_select_universe_size_cap_excludes_lower_ranked():
    rows = [(f"{i:03d}", float(1000 - i)) for i in range(10)]
    top = select_universe(rows, size=5)
    assert top[:5] == ["000", "001", "002", "003", "004"]
    assert "069500" in top
    assert len(top) == 6  # 5 ranked + the always-included ETF


def test_select_universe_always_included_not_duplicated_if_already_top():
    rows = [("069500", 999.0), ("001", 5.0)]
    out = select_universe(rows, size=2)
    assert out.count("069500") == 1


def test_select_universe_deterministic_on_ties():
    rows = [("b", 10.0), ("a", 10.0), ("c", 10.0)]
    assert select_universe(rows, size=2)[:2] == ["a", "b"]


def test_compute_universes_uses_previous_trading_day():
    ordered_days = [dt.date(2026, 3, 13), dt.date(2026, 3, 16), dt.date(2026, 3, 17)]
    rows_by_day = {
        dt.date(2026, 3, 13): [("001", 100.0), ("002", 50.0)],
        dt.date(2026, 3, 16): [("001", 10.0), ("002", 200.0)],
    }
    universes = compute_universes(
        rows_by_day,
        ordered_days,
        target_days=[dt.date(2026, 3, 16), dt.date(2026, 3, 17)],
        size=1,
    )
    # 2026-03-16's universe comes from 2026-03-13's values (001 > 002)
    assert universes[dt.date(2026, 3, 16)] == ["001", "069500"]
    # 2026-03-17's universe comes from 2026-03-16's values (002 > 001)
    assert universes[dt.date(2026, 3, 17)] == ["002", "069500"]


def test_coverage_rows_defaults_missing_pairs_to_zero_bars():
    universes = {dt.date(2026, 3, 16): ["001", "002"]}
    bar_counts = {(dt.date(2026, 3, 16), "001"): 390}
    rows = coverage_rows(universes, bar_counts)
    assert rows == [
        (dt.date(2026, 3, 16), "001", 390),
        (dt.date(2026, 3, 16), "002", 0),
    ]


def test_summarize_counts_covered_vs_missing_per_month():
    rows = [
        (dt.date(2026, 3, 16), "001", 390),  # covered
        (dt.date(2026, 3, 16), "002", 10),  # missing
        (dt.date(2026, 4, 1), "001", 5),  # missing
    ]
    summary = summarize(rows, threshold=300)
    assert summary["total_universe_pairs"] == 3
    assert summary["total_covered_pairs"] == 1
    assert summary["total_missing_pairs"] == 2
    assert summary["by_month"]["2026-03"] == {
        "universe_pairs": 2,
        "covered": 1,
        "missing": 1,
    }
    assert summary["by_month"]["2026-04"] == {
        "universe_pairs": 1,
        "covered": 0,
        "missing": 1,
    }
    assert dict(summary["top_missing_codes"])["002"] == 1
    assert dict(summary["top_missing_codes"])["001"] == 1


def test_missing_pairs_filters_by_threshold():
    rows = [
        (dt.date(2026, 3, 16), "001", 390),
        (dt.date(2026, 3, 16), "002", 10),
    ]
    assert missing_pairs(rows, threshold=300) == [(dt.date(2026, 3, 16), "002", 10)]
