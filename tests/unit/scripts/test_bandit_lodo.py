"""bandit_lodo: inner-LODO hyperparameters, strict outer LODO replays, judge and final model."""

import datetime as dt
import json
import sys
from functools import partial
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "eval"))

import bandit_lodo as bl  # noqa: E402
import optimize_policy as op  # noqa: E402

from daytrade_it.application.trading.bandit import FEATURES, BanditModel  # noqa: E402
from daytrade_it.application.trading.policy import NewsPolicy, load_policy  # noqa: E402

DAYS = [dt.date(2026, 9, d) for d in (8, 9, 10, 11)]


def _rows(sign: float = 1.0, n: int = 40) -> list[dict]:
    """Reward follows pre_entry_move (sign=+1) or is always a loss (sign=0)."""
    rng = np.random.default_rng(7)
    rows = []
    for day in DAYS:
        for i in range(n):
            x = {f: float(rng.normal()) for f in FEATURES}
            x["pre_entry_move"] = float(rng.uniform(-0.03, 0.03))
            r = sign * x["pre_entry_move"] + float(rng.normal(scale=0.003)) - 0.004 * (sign == 0)
            rows.append({"day": str(day), "key": f"{day}-{i}", "reward": r, "features": x})
    rows.append({"day": str(DAYS[0]), "key": "skip", "reward": None, "features": None})
    return rows


def test_choose_finds_a_positive_rule_when_reward_follows_a_feature() -> None:
    hp = bl.choose(bl.trainable(_rows()))
    assert hp["objective"] > 0
    assert set(hp["inner"]) == {str(d) for d in DAYS}
    assert (hp["lam"], hp["min_prob"], hp["min_edge"]) in bl.GRID


def test_choose_trades_nothing_when_every_candidate_loses() -> None:
    hp = bl.choose(bl.trainable(_rows(sign=0.0)))
    assert hp["objective"] == pytest.approx(0.0)
    assert all(v == 0 for v in hp["inner"].values())


def test_objective_penalises_the_worst_day() -> None:
    assert bl.objective([0.03, -0.01]) == pytest.approx(0.02 - 0.005)
    assert bl.objective([0.01, 0.02]) == pytest.approx(0.03)


def _fake_env(seen: list) -> op.Env:  # noqa: ANN001
    def evaluate(day: dt.date, policy: NewsPolicy) -> dict:
        model_path = policy.selection.bandit_model
        trained = None
        if model_path is not None:
            trained = BanditModel.from_json(Path(model_path).read_text()).trained_days
            assert str(day) not in trained  # strict LODO
            assert policy.selection.rank_key == "bandit"
        seen.append((str(day), trained))
        net = 100 if model_path else 50
        return {
            "day": day,
            "net": net,
            "trades": 2,
            "losers": 0,
            "worst_trade": 0,
            "errors": 0,
            "leftovers": {},
        }

    return op.Env(evaluate=evaluate)


def test_run_writes_folds_oof_judge_final_model_and_policy(tmp_path: Path) -> None:
    seen: list = []
    out = tmp_path / "out"
    verdict = bl.run(
        _rows(), DAYS, NewsPolicy(), jobs=1, out=out, make_env=partial(_fake_env, seen)
    )
    assert len(seen) == 2 * len(DAYS)
    for d in DAYS:
        fold = json.loads((out / "folds" / f"{d}.json").read_text())
        assert fold["held_out"] == str(d)
        assert str(d) not in fold["train_days"] and len(fold["train_days"]) == 3
        assert {"lam", "min_prob", "min_edge", "objective"} <= set(fold["hyperparams"])
        assert fold["bandit"]["net"] == 100 and fold["base"]["net"] == 50
        assert fold["n_train"] == 120
    oof = json.loads((out / "oof.json").read_text())
    assert [r["net"] for r in oof["candidate"]] == [100] * 4
    assert [r["net"] for r in oof["baseline"]] == [50] * 4
    assert json.loads((out / "judge.json").read_text()) == json.loads(json.dumps(verdict))
    final = BanditModel.from_json((out / "final_model.json").read_text())
    assert final.trained_days == tuple(str(d) for d in DAYS)
    policy = load_policy(out / "policy.json")
    assert policy.selection.rank_key == "bandit"
    assert Path(policy.selection.bandit_model) == out / "final_model.json"
    assert json.loads((out / "policy.json").read_text())["selection"]["bandit_model"] == (
        "final_model.json"
    )


def test_run_is_deterministic(tmp_path: Path) -> None:
    a = bl.run(_rows(), DAYS, NewsPolicy(), 1, tmp_path / "a", partial(_fake_env, []))
    b = bl.run(_rows(), DAYS, NewsPolicy(), 1, tmp_path / "b", partial(_fake_env, []))
    assert a == b
    fa = json.loads((tmp_path / "a" / "folds" / f"{DAYS[1]}.json").read_text())
    fb = json.loads((tmp_path / "b" / "folds" / f"{DAYS[1]}.json").read_text())
    assert fa["hyperparams"] == fb["hyperparams"]
