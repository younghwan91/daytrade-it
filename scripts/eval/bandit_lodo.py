"""Leave-one-day-out training and replay evaluation of the entry bandit.

uv run python scripts/eval/bandit_lodo.py --dataset data/eval/bandit/dataset.jsonl \\
    --dart data/eval/dart_trader_20260915.db                                     # simnode

Spec: docs/superpowers/specs/2026-09-16-bandit-entry-design.md (평가와 채택).

For each held-out day d (strict: nothing from d is used to train d's model):
1. Inner LODO over the other days picks (lam, min_prob, min_edge) from `GRID`,
   maximizing `objective` over the inner held-out days' sums of rewards of the
   candidates the bandit enters (`bandit.decide`). Ties keep the earlier grid
   entry, and the grid runs from the most to the least conservative.
2. `bandit.fit` on all training rows with those values -> folds/<d>.model.json.
3. Day d is replayed with the CAPPED base policy plus that model and
   `rank_key="bandit"`, and with the base policy alone (`replay_common.day_net`).

Then `bayes_judge.judge(bandit oof, base oof)`. `final_model.json` is trained on
every day with inner-LODO hyperparameters; `policy.json` = base policy +
`bandit_model: final_model.json` + `rank_key: bandit`.

Folds replay in `multiprocessing.Pool(jobs)` (optimize_policy's per-worker
environment: cached days, `--work-root/<pid>`, run dirs removed).
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import multiprocessing
import sys
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path

import numpy as np
import optimize_policy as op
from bandit_dataset import load_rows, trainable

from daytrade_it.application.trading import bandit
from daytrade_it.application.trading.policy import NewsPolicy, policy_to_dict

LAMS = (100.0, 30.0, 10.0, 3.0, 1.0)
MIN_PROBS = (0.7, 0.6, 0.55, 0.5)
MIN_EDGES = (0.005, 0.002, 0.0)
#: Most conservative first: a tie keeps the stronger shrinkage / stricter gate.
GRID: tuple[tuple[float, float, float], ...] = tuple(
    (lam, p, e) for lam in LAMS for p in MIN_PROBS for e in MIN_EDGES
)
WORST_DAY_WEIGHT = 0.5
OUT = Path("data/eval/opt/bandit")
WORK = Path("data/eval/bandit_lodo_work")

__all__ = ["GRID", "choose", "objective", "run", "trainable"]


def objective(day_sums: list[float]) -> float:
    """Sum of held-out day reward sums - 0.5 x max(0, -worst day)."""
    return sum(day_sums) - WORST_DAY_WEIGHT * max(0.0, -min(day_sums))


def fit_rows(
    rows: list[dict], lam: float, min_prob: float = 0.5, min_edge: float = 0.0
) -> bandit.BanditModel:
    X = np.array([[r["features"][f] for f in bandit.FEATURES] for r in rows], dtype=float)
    y = np.array([r["reward"] for r in rows], dtype=float)
    return bandit.fit(
        X,
        y,
        lam=lam,
        min_prob=min_prob,
        min_edge=min_edge,
        trained_days=tuple(sorted({r["day"] for r in rows})),
    )


def entered_reward(model: bandit.BanditModel, rows: list[dict]) -> float:
    """Sum of rewards of the rows `bandit.decide` enters."""
    return float(sum(r["reward"] for r in rows if bandit.decide(model, r["features"])[0]))


def choose(rows: list[dict]) -> dict:
    """Inner-LODO (lam, min_prob, min_edge) over the days in `rows` (trainable rows)."""
    days = sorted({r["day"] for r in rows})
    if len(days) < 2:
        raise ValueError(f"inner LODO needs >= 2 training days, got {days}")
    sums: dict[tuple[float, float, float], dict[str, float]] = {g: {} for g in GRID}
    for lam in LAMS:
        for h in days:
            model = fit_rows([r for r in rows if r["day"] != h], lam)
            held = [r for r in rows if r["day"] == h]
            for p in MIN_PROBS:
                for e in MIN_EDGES:
                    gated = dataclasses.replace(model, min_prob=p, min_edge=e)
                    sums[(lam, p, e)][h] = entered_reward(gated, held)
    best = max(GRID, key=lambda g: (objective(list(sums[g].values())), -GRID.index(g)))
    lam, p, e = best
    return {
        "lam": lam,
        "min_prob": p,
        "min_edge": e,
        "objective": objective(list(sums[best].values())),
        "inner": sums[best],
    }


def bandit_policy(base: NewsPolicy, model_path: Path) -> NewsPolicy:
    return dataclasses.replace(
        base,
        selection=dataclasses.replace(
            base.selection, bandit_model=str(model_path.resolve()), rank_key="bandit"
        ),
    )


def train(rows: list[dict], model_path: Path) -> dict:
    """Choose hyperparameters, fit on `rows`, write the model; returns the hyperparameters."""
    hp = choose(rows)
    model = fit_rows(rows, hp["lam"], hp["min_prob"], hp["min_edge"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text(model.to_json())
    return hp


@dataclasses.dataclass(frozen=True)
class FoldTask:
    held_out: dt.date
    rows: tuple  # every trainable row; the fold drops the held-out day's
    base: NewsPolicy
    out: Path


def _run_fold(task: FoldTask) -> dict:
    env = op._ENV
    assert env is not None  # noqa: S101
    started = time.monotonic()
    d = str(task.held_out)
    train_rows = [r for r in task.rows if r["day"] != d]
    model_path = task.out / "folds" / f"{d}.model.json"
    hp = train(train_rows, model_path)
    return {
        "held_out": d,
        "train_days": sorted({r["day"] for r in train_rows}),
        "n_train": len(train_rows),
        "hyperparams": hp,
        "model": str(model_path),
        "bandit": op._jsonable(env.evaluate(task.held_out, bandit_policy(task.base, model_path))),
        "base": op._jsonable(env.evaluate(task.held_out, task.base)),
        "seconds": time.monotonic() - started,
    }


def run(
    rows: list[dict],
    days: list[dt.date],
    base: NewsPolicy,
    jobs: int,
    out: Path,
    make_env: Callable[[], op.Env],
) -> dict:
    """Outer LODO folds, judge, final model and policy; returns the judge verdict."""
    from bayes_judge import judge  # noqa: PLC0415

    days = sorted(days)
    train_rows = tuple(trainable(rows))
    tasks = [FoldTask(d, train_rows, base, out) for d in days]
    (out / "folds").mkdir(parents=True, exist_ok=True)
    if jobs <= 1:
        saved, op._ENV = op._ENV, make_env()
        try:
            folds = [_run_fold(t) for t in tasks]
        finally:
            op._ENV = saved
    else:
        with multiprocessing.Pool(min(jobs, len(tasks)), op._init_worker, (make_env,)) as pool:
            folds = pool.map(_run_fold, tasks, chunksize=1)
    for f in folds:
        seconds = f.pop("seconds")
        op._dump(out / "folds" / f"{f['held_out']}.json", f)
        hp = f["hyperparams"]
        print(
            f"fold {f['held_out']}: lam={hp['lam']} min_prob={hp['min_prob']} "
            f"min_edge={hp['min_edge']} inner={hp['objective']:+.4f} "
            f"bandit={float(f['bandit']['net']):,.0f} base={float(f['base']['net']):,.0f} "
            f"({seconds:.0f}s)",
            file=sys.stderr,
        )
    oof = {"candidate": [f["bandit"] for f in folds], "baseline": [f["base"] for f in folds]}
    op._dump(out / "oof.json", oof)
    verdict = judge({r["day"]: r for r in oof["candidate"]}, {r["day"]: r for r in oof["baseline"]})
    op._dump(out / "judge.json", verdict)
    final_hp = train(list(train_rows), out / "final_model.json")
    op._dump(out / "final_hyperparams.json", final_hp)
    policy = policy_to_dict(base)
    policy["selection"]["bandit_model"] = "final_model.json"
    policy["selection"]["rank_key"] = "bandit"
    (out / "policy.json").write_text(json.dumps(policy, ensure_ascii=False, indent=2) + "\n")
    return verdict


def main(argv: list[str] | None = None) -> None:
    from bandit_dataset import OUT as DATASET  # noqa: PLC0415
    from replay_common import CACHE, load_day  # noqa: PLC0415
    from replay_parity import DART, EVENTS, _export_quant_airflow_env  # noqa: PLC0415

    from daytrade_it.application.trading.policy import load_policy  # noqa: PLC0415
    from daytrade_it.infrastructure.external.quant_airflow_db_client import (  # noqa: PLC0415
        QuantAirflowDBClient,
    )
    from daytrade_it.shared.runtime_host import require_backtest_host  # noqa: PLC0415

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, default=DATASET)
    ap.add_argument("--base-policy", type=Path, help="default: live defaults")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--dart", type=Path, default=DART)
    ap.add_argument(
        "--events",
        type=Path,
        default=None,
        help="events directory (default data/eval/replay_events)",
    )
    ap.add_argument("--work-root", type=Path, default=WORK)
    ap.add_argument("--cache-dir", type=Path, default=CACHE)
    ap.add_argument("--jobs", type=int, default=7)
    args = ap.parse_args(argv)

    require_backtest_host()
    if args.base_policy is not None and not args.base_policy.exists():
        ap.error(f"--base-policy {args.base_policy} does not exist")
    base = load_policy(args.base_policy) if args.base_policy else NewsPolicy()
    rows = load_rows(args.dataset)
    # Every event day is a fold, even one without a dataset row.
    events_dir = args.events or EVENTS
    days = [dt.date.fromisoformat(p.stem) for p in sorted(events_dir.glob("2026-*.jsonl"))]
    extra = sorted({r["day"] for r in rows} - {str(d) for d in days})
    if extra:
        ap.error(f"dataset days without an events file: {extra}")
    _export_quant_airflow_env()
    stocks = QuantAirflowDBClient().get_stock_master()
    for d in days:
        load_day(d, args.cache_dir, events_dir)
    started = time.monotonic()
    verdict = run(
        rows,
        days,
        base,
        jobs=args.jobs,
        out=args.out,
        make_env=partial(
            op._replay_env, days, stocks, args.dart, args.work_root, args.cache_dir, events_dir
        ),
    )
    print(json.dumps(verdict, indent=1, ensure_ascii=False))
    print(f"wall {time.monotonic() - started:.0f}s -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
