import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "eval"))

from spend_plan import build_plan  # noqa: E402


def _row(day: str, cost: str, missing: str = "10") -> dict:
    return {
        "day": day,
        "pairs_total": "100",
        "pairs_kept": "80",
        "already_judged": "70",
        "missing": missing,
        "est_cost_usd": cost,
    }


def test_orders_usable_days_cheapest_first() -> None:
    rows = [_row("2026-09-01", "5.0"), _row("2026-08-01", "1.0"), _row("2026-07-01", "3.0")]
    usable = {"2026-09-01", "2026-08-01", "2026-07-01"}
    out = build_plan(usable, rows, [10])
    assert [p["day"] for p in out["plan_cheapest_first"]] == [
        "2026-08-01",
        "2026-07-01",
        "2026-09-01",
    ]


def test_excludes_days_not_in_usable_set() -> None:
    rows = [_row("2026-09-01", "1.0"), _row("2026-08-01", "1.0")]
    out = build_plan({"2026-09-01"}, rows, [10])
    assert [p["day"] for p in out["plan_cheapest_first"]] == ["2026-09-01"]
    assert out["n_usable_days_total"] == 1
    assert out["n_days_total_in_summary"] == 2


def test_threshold_stops_at_first_day_that_would_exceed_budget() -> None:
    rows = [_row("2026-08-01", "4.0"), _row("2026-07-01", "4.0"), _row("2026-06-01", "4.0")]
    usable = {"2026-08-01", "2026-07-01", "2026-06-01"}
    out = build_plan(usable, rows, [10])
    t = out["thresholds"]["10"]
    assert t["n_days"] == 2
    assert t["actual_cost"] == 8.0
    assert set(t["days"]) == {"2026-08-01", "2026-07-01"}


def test_threshold_with_no_affordable_day_is_empty() -> None:
    rows = [_row("2026-08-01", "5.0")]
    out = build_plan({"2026-08-01"}, rows, [1])
    t = out["thresholds"]["1"]
    assert t == {
        "n_days": 0,
        "actual_cost": 0.0,
        "oldest_day": None,
        "newest_day": None,
        "days": [],
    }
