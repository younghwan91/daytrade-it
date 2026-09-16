import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "eval"))

from learning_curve import aggregate, plan_windows  # noqa: E402

DAYS = [dt.date(2026, 6, 1) + dt.timedelta(days=i) for i in range(43)]


def test_every_window_leaves_the_holdout_strictly_later() -> None:
    for k, o in plan_windows(DAYS, [5, 10, 20], holdout=8, origins=4):
        train = DAYS[o : o + k]
        held = DAYS[o + k : o + k + 8]
        assert len(train) == k
        assert len(held) == 8
        assert max(train) < min(held)  # never train on a day at or after the score day


def test_drops_a_size_that_cannot_fit() -> None:
    plan = plan_windows(DAYS, [5, 40, 100], holdout=8, origins=3)
    assert {k for k, _ in plan} == {5}  # 40 + 8 > 43, 100 + 8 > 43


def test_origins_are_distinct_and_spread_over_the_range() -> None:
    starts = [o for k, o in plan_windows(DAYS, [10], holdout=8, origins=5)]
    assert len(starts) == len(set(starts)) == 5
    assert starts[0] == 0
    assert starts[-1] == len(DAYS) - 10 - 8  # the last window that still fits


def test_one_origin_is_allowed_but_stays_a_single_draw() -> None:
    assert plan_windows(DAYS, [10], holdout=8, origins=1) == [(10, 0)]


def test_exact_fit_yields_one_window() -> None:
    assert plan_windows(DAYS[:18], [10], holdout=8, origins=5) == [(10, 0)]


def _result(kind: str, size: int, origin: int, cand: list[float], base: list[float]) -> dict:
    days = [str(dt.date(2026, 7, 1) + dt.timedelta(days=i)) for i in range(len(cand))]
    mk = lambda d, n: {  # noqa: E731
        "day": d,
        "net": str(n),
        "trades": 1,
        "losers": 0 if n >= 0 else 1,
        "worst_trade": str(min(0.0, n)),
        "errors": 0,
    }
    return {
        "kind": kind,
        "size": size,
        "origin": origin,
        "name": f"{kind}-k{size}-o{origin}",
        "train_days": [str(dt.date(2026, 6, 1))] * size,
        "eval_days": days,
        "candidate": [mk(d, n) for d, n in zip(days, cand, strict=True)],
        "baseline": [mk(d, n) for d, n in zip(days, base, strict=True)],
        "oos_candidate_net": sum(cand),
        "oos_baseline_net": sum(base),
    }


def test_aggregate_reports_dispersion_across_origins_not_just_a_mean() -> None:
    results = [
        _result("optimizer", 10, 0, [100.0, 100.0], [50.0, 50.0]),
        _result("optimizer", 10, 5, [-300.0, 100.0], [50.0, 50.0]),
        _result("optimizer", 10, 9, [200.0, 200.0], [50.0, 50.0]),
    ]
    curve = aggregate(results)["optimizer"]["10"]
    assert curve["candidate_per_day"]["n"] == 3
    assert curve["candidate_per_day"]["min"] == -100.0  # the unlucky origin survives the summary
    assert curve["candidate_per_day"]["max"] == 200.0
    assert curve["candidate_per_day"]["stdev"] > 0
    # per-day deltas: (100-50), (-100-50), (200-50)
    assert curve["delta_per_day"]["mean"] == (50.0 + -150.0 + 150.0) / 3


def test_aggregate_separates_the_two_learners() -> None:
    results = [
        _result("optimizer", 5, 0, [10.0], [0.0]),
        _result("bandit", 5, 0, [0.0], [0.0]),
    ]
    curve = aggregate(results)
    assert set(curve) == {"optimizer", "bandit"}
    assert curve["optimizer"]["5"]["delta_per_day"]["mean"] == 10.0
    assert curve["bandit"]["5"]["delta_per_day"]["mean"] == 0.0


def test_aggregate_attaches_a_bayes_judge_verdict_per_window() -> None:
    results = [_result("optimizer", 5, 0, [10.0, 10.0], [0.0, 0.0])]
    curve = aggregate(results)
    assert "adopt" in results[0]["judge"]
    assert curve["optimizer"]["5"]["adopt_count"] in (0, 1)
