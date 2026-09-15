"""optimize_policy: LODO x Optuna TPE over NewsPolicy search spaces (no DB, fake day_net)."""

from __future__ import annotations

import csv
import dataclasses
import datetime as dt
import json
import sys
from decimal import Decimal
from functools import partial
from pathlib import Path

import optuna
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "eval"))

import optimize_policy as op  # noqa: E402

from daytrade_it.application.trading.policy import NewsPolicy, load_policy  # noqa: E402

DAYS = [dt.date(2026, 9, 7), dt.date(2026, 9, 8), dt.date(2026, 9, 9)]


def test_stage2_live_params_rebuild_the_live_policy() -> None:
    base = NewsPolicy()
    params = op.base_params("stage2", base)
    assert params["max_pre_move"] == 0.03  # live None = the 3% no-chase cap
    assert op.SPACES["stage2"](optuna.trial.FixedTrial(params), base) == base


def test_stage3_base_params_rebuild_a_stage2_policy() -> None:
    base = op.SPACES["stage2"](
        optuna.trial.FixedTrial({**op.base_params("stage2", NewsPolicy()), "stop_time_min": 60}),
        NewsPolicy(),
    )
    assert base.limits.stop_time == dt.timedelta(minutes=60)
    rebuilt = op.SPACES["stage3"](optuna.trial.FixedTrial(op.base_params("stage3", base)), base)
    assert rebuilt == base


def test_stage_params_convert_to_policy_types() -> None:
    base = NewsPolicy()
    params = {
        **op.base_params("stage3", base),
        "take_profit_pct": 0.05,
        "flatten_at": "14:30",
    }
    p = op.SPACES["stage3"](optuna.trial.FixedTrial(params), base)
    assert p.limits.take_profit_pct == Decimal("0.05")
    assert p.limits.flatten_at == dt.time(14, 30)
    params2 = {
        **op.base_params("stage2", base),
        "entry_delay_min": 5,
        "stop_orb_min": 15,
        "subjects": "main",
        "sessions": "premarket",
        "min_pre_move": -0.02,
    }
    q = op.SPACES["stage2"](optuna.trial.FixedTrial(params2), base)
    assert q.limits.entry_delay == dt.timedelta(minutes=5)
    assert q.limits.stop_orb == dt.timedelta(minutes=15)
    assert q.selection.subjects == ("main",)
    assert q.selection.sessions == ("premarket",)
    assert q.selection.min_pre_move == Decimal("-0.02")


def test_base_params_rejects_a_value_outside_the_space() -> None:
    lim = dataclasses.replace(NewsPolicy().limits, stop_loss_pct=Decimal("0.025"))
    with pytest.raises(ValueError, match="stop_loss_pct"):
        op.base_params("stage2", NewsPolicy(limits=lim))


def test_objective_penalizes_the_worst_losing_day() -> None:
    def r(net: int, errors: int = 0) -> dict:
        return {"net": Decimal(net), "errors": errors}

    assert op.objective([r(1000), r(2000)]) == 3000
    assert op.objective([r(5000), r(-2000)]) == 3000 - 1000
    assert op.objective([r(5000), r(-2000), r(-4000)]) == -1000 - 2000
    assert op.objective([r(9000), r(10, errors=1)]) == -1e12


def _fake_env() -> op.Env:
    """Net depends on a few searched knobs so TPE has something to find."""

    def evaluate(day: dt.date, policy: NewsPolicy) -> dict:
        lim, sel = policy.limits, policy.selection
        net = 1000 + day.day * 10 - int(sel.min_score * 1000)
        if lim.stop_loss_pct is not None:
            net += 500
        if day == DAYS[1]:
            net -= 3000
        return {
            "day": day,
            "net": Decimal(net),
            "trades": 2,
            "losers": int(net < 0),
            "worst_trade": Decimal(min(net, 0)),
            "errors": 0,
            "leftovers": {},
        }

    return op.Env(evaluate=evaluate)


@pytest.mark.parametrize("jobs", [1, 2])
def test_run_writes_fold_oof_full_judge_policy_and_trial_tables(tmp_path: Path, jobs: int) -> None:
    out = tmp_path / "out"
    base = NewsPolicy()
    op.run(
        "stage2",
        DAYS[:2],
        base,
        trials=5,
        jobs=jobs,
        out=out,
        make_env=partial(_fake_env),
    )
    names = sorted(p.name for p in out.iterdir())
    for d in DAYS[:2]:
        assert f"fold-{d}.json" in names
        assert f"trials-fold-{d}.csv" in names
    for n in ("oof.json", "full.json", "judge.json", "policy.json", "trials-full.csv"):
        assert n in names

    fold = json.loads((out / f"fold-{DAYS[0]}.json").read_text())
    assert fold["held_out"] == str(DAYS[0]) and fold["train_days"] == [str(DAYS[1])]
    # Trial 0 is the base policy: its train objective is the baseline's.
    assert fold["baseline_objective"] == op.objective([_fake_env().evaluate(DAYS[1], base)])
    assert fold["best_objective"] >= fold["baseline_objective"]
    assert fold["held_out_result"]["day"] == str(DAYS[0])

    oof = json.loads((out / "oof.json").read_text())
    assert [r["day"] for r in oof["candidate"]] == [str(d) for d in DAYS[:2]]
    assert [r["day"] for r in oof["baseline"]] == [str(d) for d in DAYS[:2]]
    assert Decimal(oof["baseline"][1]["net"]) == _fake_env().evaluate(DAYS[1], base)["net"]

    judge = json.loads((out / "judge.json").read_text())
    assert "adopt" in judge and judge["n_days"] == 2
    assert isinstance(load_policy(out / "policy.json"), NewsPolicy)

    with (out / f"trials-fold-{DAYS[0]}.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 5 and rows[0]["number"] == "0"
    assert float(rows[0]["objective"]) == fold["baseline_objective"]
    assert f"net_{DAYS[1]}" in rows[0] and "param_min_score" in rows[0]


def test_run_is_deterministic_per_seed(tmp_path: Path) -> None:
    kw = {"trials": 6, "jobs": 1, "make_env": partial(_fake_env)}
    op.run("stage2", DAYS[:2], NewsPolicy(), out=tmp_path / "a", **kw)
    op.run("stage2", DAYS[:2], NewsPolicy(), out=tmp_path / "b", **kw)
    for name in ("full.json", f"fold-{DAYS[0]}.json"):
        assert (tmp_path / "a" / name).read_text() == (tmp_path / "b" / name).read_text()
