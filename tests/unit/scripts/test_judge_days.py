"""judge_days: budget -> whole judged days. Pure logic + a fake judge_batch layer.

Nothing here touches the Anthropic API, the network or the DB. The real
`judge_batch` subprocess is replaced by `fake_judge`, so the budget arithmetic,
the day selection and the >=99% verification are all exercised without spend.
"""

import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "eval"))

from judge_days import (  # noqa: E402
    BudgetOverrun,
    DayRow,
    ProbeFailed,
    format_table,
    load_summary,
    pick_days,
    run_days,
)

COST = 0.0007335


def row(day: str, missing: int, *, kept: int | None = None, judged: int = 0) -> DayRow:
    kept = missing + judged if kept is None else kept
    return DayRow(
        day=day,
        pairs_total=kept + 10,
        pairs_kept=kept,
        already_judged=judged,
        missing=missing,
        est_cost_usd=round(missing * COST, 6),
    )


# --------------------------------------------------------------------------
# load_summary
# --------------------------------------------------------------------------


def write_csv(path: Path, rows: list[DayRow]) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["day", "pairs_total", "pairs_kept", "already_judged", "missing", "est_cost_usd"]
        )
        for r in rows:
            w.writerow(
                [r.day, r.pairs_total, r.pairs_kept, r.already_judged, r.missing, r.est_cost_usd]
            )


class TestLoadSummary:
    def test_reads_rows_and_orders_most_recent_first(self, tmp_path: Path) -> None:
        path = tmp_path / "s.csv"
        write_csv(path, [row("2026-09-01", 100), row("2026-09-15", 200), row("2026-09-10", 50)])
        days = [r.day for r in load_summary(path)]
        assert days == ["2026-09-15", "2026-09-10", "2026-09-01"]

    def test_keeps_the_numbers(self, tmp_path: Path) -> None:
        path = tmp_path / "s.csv"
        write_csv(path, [row("2026-09-15", 1000, judged=300)])
        (r,) = load_summary(path)
        assert (r.pairs_kept, r.already_judged, r.missing) == (1300, 300, 1000)
        assert r.est_cost_usd == pytest.approx(0.7335)

    def test_rejects_a_file_without_the_expected_columns(self, tmp_path: Path) -> None:
        path = tmp_path / "s.csv"
        path.write_text("day,cost\n2026-09-15,1.0\n")
        with pytest.raises(ValueError, match="missing column"):
            load_summary(path)


# --------------------------------------------------------------------------
# pick_days -- the hard cap
# --------------------------------------------------------------------------


class TestPickDays:
    def test_takes_the_most_recent_days_that_fit(self) -> None:
        rows = [row(f"2026-09-{d:02d}", 1000) for d in (15, 14, 13, 12)]  # $0.7335 each
        picked, cost = pick_days(rows, budget_usd=1.60)
        assert [p.day for p in picked] == ["2026-09-15", "2026-09-14"]
        assert cost == pytest.approx(1.467)

    def test_never_exceeds_the_budget(self) -> None:
        rows = [row(f"2026-09-{d:02d}", 1000) for d in (15, 14)]
        picked, cost = pick_days(rows, budget_usd=0.70)
        assert picked == []
        assert cost == 0.0

    def test_stops_at_the_first_day_that_does_not_fit_keeping_the_window_contiguous(self) -> None:
        rows = [row("2026-09-15", 1000), row("2026-09-14", 100_000), row("2026-09-13", 10)]
        picked, _ = pick_days(rows, budget_usd=1.00)
        assert [p.day for p in picked] == ["2026-09-15"]

    def test_already_complete_days_are_free_and_do_not_consume_budget(self) -> None:
        rows = [row("2026-09-15", 0, judged=1000), row("2026-09-14", 1000)]
        picked, cost = pick_days(rows, budget_usd=0.80)
        assert [p.day for p in picked] == ["2026-09-15", "2026-09-14"]
        assert cost == pytest.approx(0.7335)

    def test_zero_budget_picks_only_free_days(self) -> None:
        rows = [row("2026-09-15", 0, judged=10), row("2026-09-14", 1)]
        picked, cost = pick_days(rows, budget_usd=0.0)
        assert [p.day for p in picked] == ["2026-09-15"]
        assert cost == 0.0

    def test_negative_budget_is_refused(self) -> None:
        with pytest.raises(ValueError, match="budget"):
            pick_days([row("2026-09-15", 1)], budget_usd=-1.0)


# --------------------------------------------------------------------------
# run_days -- fake judge_batch layer
# --------------------------------------------------------------------------


class FakeJudge:
    """Stands in for a `judge_batch --days <day>` subprocess.

    Records the per-day budget it was handed and reports a spend, so a test can
    make the day cost more than estimated and check the cap still holds.
    """

    def __init__(self, spend: dict[str, float] | None = None, fail: set[str] | None = None) -> None:
        self.calls: list[tuple[str, float]] = []
        self.spend = spend or {}
        self.fail = fail or set()

    def __call__(self, day: str, budget_usd: float) -> dict:
        self.calls.append((day, budget_usd))
        if day in self.fail:
            raise RuntimeError(f"judge_batch failed on {day}")
        return {"estimate": {"usd": self.spend.get(day, budget_usd)}, "succeeded": 1, "failed": 0}


def fixed_verifier(fracs: dict[str, float]):
    def verify(day: str) -> tuple[int, int]:
        kept = 1000
        return int(round(fracs.get(day, 1.0) * kept)), kept

    return verify


def ok_probe() -> None:
    return None


class TestRunDays:
    def test_refuses_to_start_when_the_balance_probe_fails(self) -> None:
        judge = FakeJudge()

        def bad_probe() -> None:
            raise ProbeFailed("credit balance is too low")

        with pytest.raises(ProbeFailed):
            run_days(
                [row("2026-09-15", 1000)],
                budget_usd=10.0,
                judge=judge,
                verify=fixed_verifier({}),
                probe=bad_probe,
            )
        assert judge.calls == [], "no day may be judged after a failed probe"

    def test_probe_runs_before_any_judging(self) -> None:
        order: list[str] = []
        judge = FakeJudge()

        def probe() -> None:
            order.append("probe")

        def counting(day: str, budget_usd: float) -> dict:
            order.append(day)
            return judge(day, budget_usd)

        run_days(
            [row("2026-09-15", 1000)],
            budget_usd=10.0,
            judge=counting,
            verify=fixed_verifier({}),
            probe=probe,
        )
        assert order[0] == "probe"

    def test_judges_exactly_the_picked_days_most_recent_first(self) -> None:
        rows = [row(f"2026-09-{d:02d}", 1000) for d in (15, 14, 13)]
        judge = FakeJudge()
        results = run_days(
            rows, budget_usd=1.60, judge=judge, verify=fixed_verifier({}), probe=ok_probe
        )
        assert [d for d, _ in judge.calls] == ["2026-09-15", "2026-09-14"]
        assert [r["day"] for r in results] == ["2026-09-15", "2026-09-14"]
        assert all(r["complete"] for r in results)

    def test_per_day_budget_never_exceeds_what_is_left(self) -> None:
        rows = [row(f"2026-09-{d:02d}", 1000) for d in (15, 14)]
        judge = FakeJudge()
        run_days(rows, budget_usd=1.50, judge=judge, verify=fixed_verifier({}), probe=ok_probe)
        handed = [b for _, b in judge.calls]
        assert sum(handed) <= 1.50 + 1e-9
        assert all(b > 0 for b in handed)

    def test_a_day_that_overspends_stops_the_run_before_the_next_day(self) -> None:
        rows = [row(f"2026-09-{d:02d}", 1000) for d in (15, 14, 13)]
        # day 15 reports back more spend than the whole cap allows
        judge = FakeJudge(spend={"2026-09-15": 2.40})
        with pytest.raises(BudgetOverrun):
            run_days(rows, budget_usd=2.30, judge=judge, verify=fixed_verifier({}), probe=ok_probe)
        assert [d for d, _ in judge.calls] == ["2026-09-15"]

    def test_a_day_below_the_threshold_is_reported_incomplete_and_stops_the_run(self) -> None:
        rows = [row(f"2026-09-{d:02d}", 1000) for d in (15, 14)]
        judge = FakeJudge()
        results = run_days(
            rows,
            budget_usd=1.60,
            judge=judge,
            verify=fixed_verifier({"2026-09-15": 0.90}),
            probe=ok_probe,
        )
        assert results[0]["complete"] is False
        assert results[0]["judged_frac"] == pytest.approx(0.90)
        assert [d for d, _ in judge.calls] == ["2026-09-15"], "stop once a day cannot be finished"

    def test_exactly_99_percent_counts_as_complete(self) -> None:
        judge = FakeJudge()
        results = run_days(
            [row("2026-09-15", 1000)],
            budget_usd=1.0,
            judge=judge,
            verify=fixed_verifier({"2026-09-15": 0.99}),
            probe=ok_probe,
        )
        assert results[0]["complete"] is True

    def test_a_judge_failure_is_reported_not_swallowed(self) -> None:
        rows = [row("2026-09-15", 1000), row("2026-09-14", 1000)]
        judge = FakeJudge(fail={"2026-09-15"})
        results = run_days(
            rows, budget_usd=1.60, judge=judge, verify=fixed_verifier({}), probe=ok_probe
        )
        assert results[0]["complete"] is False
        assert "judge_batch failed" in results[0]["error"]
        assert [d for d, _ in judge.calls] == ["2026-09-15"]

    def test_free_days_are_verified_without_calling_the_judge(self) -> None:
        rows = [row("2026-09-15", 0, judged=1000)]
        judge = FakeJudge()
        results = run_days(
            rows, budget_usd=0.0, judge=judge, verify=fixed_verifier({}), probe=ok_probe
        )
        assert judge.calls == []
        assert results[0]["complete"] is True
        assert results[0]["spent_usd"] == 0.0

    def test_nothing_affordable_judges_nothing(self) -> None:
        judge = FakeJudge()
        results = run_days(
            [row("2026-09-15", 1000)],
            budget_usd=0.10,
            judge=judge,
            verify=fixed_verifier({}),
            probe=ok_probe,
        )
        assert results == []
        assert judge.calls == []


class TestFormatTable:
    def test_lists_completed_days_and_the_total(self) -> None:
        out = format_table(
            [
                {
                    "day": "2026-09-15",
                    "judged": 1000,
                    "kept": 1000,
                    "judged_frac": 1.0,
                    "spent_usd": 0.73,
                    "complete": True,
                    "error": None,
                },
                {
                    "day": "2026-09-14",
                    "judged": 900,
                    "kept": 1000,
                    "judged_frac": 0.9,
                    "spent_usd": 0.70,
                    "complete": False,
                    "error": None,
                },
            ]
        )
        assert "2026-09-15" in out and "2026-09-14" in out
        assert "1 complete" in out
        assert "1.43" in out  # total spend
