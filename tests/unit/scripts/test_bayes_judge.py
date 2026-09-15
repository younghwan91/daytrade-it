"""bayes_judge: grid-posterior Bayesian judge for candidate vs baseline day_net results."""

from __future__ import annotations

import sys
from decimal import Decimal
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "eval"))

from bayes_judge import judge, posterior_mu  # noqa: E402


def _day(net: float, trades: int = 1, losers: int = 0) -> dict:
    return {
        "net": Decimal(str(net)),
        "trades": trades,
        "losers": losers,
        "worst_trade": Decimal(str(min(net, 0))),
        "errors": 0,
        "leftovers": {},
    }


def _days(day_prefix: str, n: int, mk) -> dict[str, dict]:
    return {f"{day_prefix}-{i:02d}": mk(i) for i in range(n)}


def test_posterior_mu_grid_sums_to_one() -> None:
    _, post = posterior_mu([100.0, -50.0, 200.0, 0.0, 300.0], prior_sd=5000.0)
    assert post.sum() == pytest.approx(1.0, abs=1e-9)


def test_posterior_mu_all_positive_strongly_favors_mu_gt_0() -> None:
    mu_grid, post = posterior_mu([3000.0] * 7, prior_sd=5000.0)
    p_gt_0 = float(post[mu_grid > 0].sum())
    assert p_gt_0 > 0.95


def test_posterior_mu_all_zero_is_near_half() -> None:
    mu_grid, post = posterior_mu([0.0] * 7, prior_sd=5000.0)
    p_gt_0 = float(post[mu_grid > 0].sum())
    assert 0.4 <= p_gt_0 <= 0.6


def test_posterior_mu_all_negative_strongly_favors_mu_lt_0() -> None:
    mu_grid, post = posterior_mu([-3000.0] * 7, prior_sd=5000.0)
    p_gt_0 = float(post[mu_grid > 0].sum())
    assert p_gt_0 < 0.05


def test_posterior_mu_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="empty"):
        posterior_mu([], prior_sd=5000.0)


def test_judge_adopts_when_candidate_clearly_better() -> None:
    candidate = _days("2026-09", 7, lambda _: _day(3000.0, trades=2, losers=0))
    baseline = _days("2026-09", 7, lambda _: _day(-500.0, trades=2, losers=1))

    verdict = judge(candidate, baseline, prior_sd=5000.0)

    assert verdict["adopt"] is True
    assert verdict["p_mu_gt_0"] >= 0.90
    assert verdict["n_days"] == 7
    assert verdict["days_ge_0"] == 7
    assert verdict["worst_day_candidate"] >= verdict["worst_day_baseline"]
    assert verdict["loser_ratio_candidate"] <= verdict["loser_ratio_baseline"]
    assert verdict["forward_pending"] is True
    assert verdict["reasons"] == []


def test_judge_rejects_when_worst_day_gets_worse() -> None:
    candidate = _days("2026-09", 7, lambda i: _day(3000.0 if i < 6 else -100000.0))
    baseline = _days("2026-09", 7, lambda _: _day(100.0))

    verdict = judge(candidate, baseline, prior_sd=5000.0)

    assert verdict["adopt"] is False
    assert verdict["worst_day_candidate"] < verdict["worst_day_baseline"]
    assert any("worst" in r for r in verdict["reasons"])


def test_judge_reports_days_ge_0_threshold() -> None:
    # 3 of 7 days d_t >= 0 -- fails the ceil(5/7*7)=5 threshold.
    candidate = _days("2026-09", 7, lambda i: _day(100.0 if i < 3 else -100.0))
    baseline = _days("2026-09", 7, lambda _: _day(0.0))

    verdict = judge(candidate, baseline, prior_sd=5000.0)

    assert verdict["days_ge_0"] == 3
    assert verdict["adopt"] is False
    assert any("days_ge_0" in r or "5" in r for r in verdict["reasons"])


def test_judge_loser_ratio_is_total_losers_over_total_trades() -> None:
    candidate = {
        "d1": _day(100.0, trades=4, losers=1),
        "d2": _day(100.0, trades=2, losers=1),
    }
    baseline = {
        "d1": _day(0.0, trades=5, losers=0),
        "d2": _day(0.0, trades=5, losers=0),
    }

    verdict = judge(candidate, baseline, prior_sd=5000.0)

    assert verdict["loser_ratio_candidate"] == pytest.approx(2 / 6)
    assert verdict["loser_ratio_baseline"] == pytest.approx(0.0)


def test_judge_handles_zero_trades_without_dividing_by_zero() -> None:
    candidate = {"d1": _day(0.0, trades=0, losers=0)}
    baseline = {"d1": _day(0.0, trades=0, losers=0)}

    verdict = judge(candidate, baseline, prior_sd=5000.0)

    assert verdict["loser_ratio_candidate"] == 0.0
    assert verdict["loser_ratio_baseline"] == 0.0


def test_judge_raises_on_mismatched_day_sets() -> None:
    candidate = {"2026-09-01": _day(100.0)}
    baseline = {"2026-09-02": _day(100.0)}

    with pytest.raises(ValueError, match="day"):
        judge(candidate, baseline, prior_sd=5000.0)


def test_judge_result_is_json_serializable() -> None:
    import json

    candidate = _days("2026-09", 7, lambda _: _day(3000.0))
    baseline = _days("2026-09", 7, lambda _: _day(-500.0))
    verdict = judge(candidate, baseline, prior_sd=5000.0)

    json.dumps(verdict)  # must not raise (no numpy scalars leaking through)
