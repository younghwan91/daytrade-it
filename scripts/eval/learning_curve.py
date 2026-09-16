"""Forward-in-time learning curve for the rule optimizer and the entry bandit.

    uv run python scripts/eval/learning_curve.py --events data/eval/n300_t30/events \\
        --sizes 5,10,15,20,25,30 --holdout 8 --origins 5 --trials 200 --jobs 8 \\
        --out data/eval/n300_t30/curve

Answers "does another judged day still buy anything?" without ever training on a
day at or after the day being scored.

For each training-window size K and each origin o, the window is the K
consecutive complete days starting at o and the evaluation set is the next
``--holdout`` days, which are all strictly later. Several origins per size give
the dispersion; a single origin is one lucky draw and is not reported alone.

Both learners answer the same question against the same baseline (the live
defaults, which optuna sees as trial 0):

* ``optimizer`` -- an optuna study over the window, exactly ``optimize_policy``'s
  full-days study, then its best policy replayed on the held-out days.
* ``bandit``    -- ``bandit_lodo.choose`` + ``fit_rows`` on the window's rows,
  then the held-out days replayed with that model and ``rank_key="bandit"``.

Per window: ``bayes_judge`` over the held-out days. Per size: the spread of the
per-origin out-of-sample net across origins. No Anthropic call; the DB is read
only; Toss and Kiwoom are never touched.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import multiprocessing
import statistics
import sys
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import optimize_policy as op
import optuna

from daytrade_it.application.trading.policy import NewsPolicy

OUT = Path("data/eval/curve")
WORK = Path("data/eval/curve_work")

__all__ = ["WindowTask", "aggregate", "plan_windows", "run"]


def plan_windows(
    days: list[dt.date], sizes: list[int], holdout: int, origins: int
) -> list[tuple[int, int]]:
    """(size, origin) pairs: `origins` evenly spaced starts that leave `holdout` later days.

    Origins are spread over the whole usable range rather than bunched at the
    start, so a size's spread reflects the sample period and not one regime.
    A size that cannot fit even one window is dropped.
    """
    n = len(days)
    plan: list[tuple[int, int]] = []
    for k in sorted(set(sizes)):
        last = n - k - holdout
        if last < 0:
            continue
        if origins <= 1 or last == 0:
            starts = [0]  # one window is all that fits, or all that was asked for
        else:
            step = last / (origins - 1)
            starts = sorted({round(i * step) for i in range(origins)})
        plan += [(k, o) for o in starts]
    return plan


@dataclasses.dataclass(frozen=True)
class WindowTask:
    kind: str  # "optimizer" | "bandit"
    stage: str
    size: int
    origin: int
    train_days: tuple[dt.date, ...]
    eval_days: tuple[dt.date, ...]
    base: NewsPolicy
    trials: int
    seed: int
    out: Path

    @property
    def name(self) -> str:
        return f"{self.kind}-k{self.size}-o{self.origin}"


def _train_optimizer(task: WindowTask) -> tuple[NewsPolicy, dict[str, Any]]:
    """optimize_policy's full-days study restricted to the window."""
    env = op._ENV
    assert env is not None  # noqa: S101

    def trial_objective(trial: optuna.Trial) -> float:
        policy = op.SPACES[task.stage](trial, task.base)
        results = [env.evaluate(d, policy) for d in task.train_days]
        trial.set_user_attr("results", [op._jsonable(r) for r in results])
        return op.objective(results)

    study = optuna.create_study(
        study_name=task.name,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=task.seed),
    )
    study.enqueue_trial(op.base_params(task.stage, task.base))  # trial 0 = live defaults
    study.optimize(trial_objective, n_trials=task.trials)
    best = study.best_trial
    policy = op.SPACES[task.stage](optuna.trial.FixedTrial(best.params), task.base)
    return policy, {
        "best_trial": best.number,
        "best_params": best.params,
        "best_objective": best.value,
        "baseline_objective": study.trials[0].value,
    }


def _train_bandit(task: WindowTask) -> tuple[NewsPolicy, dict[str, Any]]:
    """bandit_lodo hyperparameter choice + fit on the window's trainable rows."""
    import bandit_lodo as bl  # noqa: PLC0415

    rows = [r for r in _ROWS if dt.date.fromisoformat(r["day"]) in set(task.train_days)]
    if len({r["day"] for r in rows}) < 2:
        raise ValueError(f"{task.name}: bandit needs >= 2 training days with rows")
    model_path = task.out / "models" / f"{task.name}.model.json"
    hp = bl.train(rows, model_path)
    model = json.loads(model_path.read_text())
    return bl.bandit_policy(task.base, model_path), {
        "hyperparams": hp,
        "n_train_rows": len(rows),
        "feature_names": model["feature_names"],
        "w": model["w"],
    }


_ROWS: list[dict] = []


def _init_rows(rows: list[dict]) -> None:
    global _ROWS  # noqa: PLW0603
    _ROWS = rows


def _run_window(task: WindowTask) -> dict:
    env = op._ENV
    assert env is not None  # noqa: S101
    started = time.monotonic()
    policy, meta = (_train_optimizer if task.kind == "optimizer" else _train_bandit)(task)
    candidate = [op._jsonable(env.evaluate(d, policy)) for d in task.eval_days]
    baseline = [op._jsonable(env.evaluate(d, task.base)) for d in task.eval_days]
    return {
        "name": task.name,
        "kind": task.kind,
        "size": task.size,
        "origin": task.origin,
        "train_days": [str(d) for d in task.train_days],
        "eval_days": [str(d) for d in task.eval_days],
        **meta,
        "candidate": candidate,
        "baseline": baseline,
        "oos_candidate_net": sum(float(r["net"]) for r in candidate),
        "oos_baseline_net": sum(float(r["net"]) for r in baseline),
        "seconds": time.monotonic() - started,
    }


def _spread(values: list[float]) -> dict[str, float]:
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def aggregate(results: list[dict]) -> dict:
    """Per (kind, size): the spread across origins of per-day OOS net and its delta."""
    from bayes_judge import judge  # noqa: PLC0415

    out: dict[str, Any] = {}
    for r in results:
        n_eval = len(r["eval_days"])
        r["oos_candidate_per_day"] = r["oos_candidate_net"] / n_eval
        r["oos_baseline_per_day"] = r["oos_baseline_net"] / n_eval
        r["oos_delta_per_day"] = r["oos_candidate_per_day"] - r["oos_baseline_per_day"]
        r["judge"] = judge(
            {x["day"]: x for x in r["candidate"]}, {x["day"]: x for x in r["baseline"]}
        )
    for kind in sorted({r["kind"] for r in results}):
        rows = [r for r in results if r["kind"] == kind]
        out[kind] = {
            str(size): {
                "candidate_per_day": _spread(
                    [r["oos_candidate_per_day"] for r in rows if r["size"] == size]
                ),
                "baseline_per_day": _spread(
                    [r["oos_baseline_per_day"] for r in rows if r["size"] == size]
                ),
                "delta_per_day": _spread(
                    [r["oos_delta_per_day"] for r in rows if r["size"] == size]
                ),
                "adopt_count": sum(1 for r in rows if r["size"] == size and r["judge"]["adopt"]),
            }
            for size in sorted({r["size"] for r in rows})
        }
    return out


def run(
    days: list[dt.date],
    kinds: list[str],
    sizes: list[int],
    holdout: int,
    origins: int,
    base: NewsPolicy,
    trials: int,
    jobs: int,
    out: Path,
    rows: list[dict],
    make_env: Callable[[], op.Env],
) -> dict:
    days = sorted(days)
    plan = plan_windows(days, sizes, holdout, origins)
    if not plan:
        raise SystemExit(f"no window fits: {len(days)} day(s), sizes {sizes}, holdout {holdout}")
    tasks = [
        WindowTask(
            kind=kind,
            stage="stage2",
            size=k,
            origin=o,
            train_days=tuple(days[o : o + k]),
            eval_days=tuple(days[o + k : o + k + holdout]),
            base=base,
            trials=trials,
            seed=1000 * k + o,
            out=out,
        )
        for kind in kinds
        for k, o in plan
    ]
    (out / "models").mkdir(parents=True, exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def _init(make: Callable[[], op.Env], rws: list[dict]) -> None:
        op._init_worker(make)
        _init_rows(rws)

    if jobs <= 1:
        saved, op._ENV = op._ENV, make_env()
        _init_rows(rows)
        try:
            results = [_run_window(t) for t in tasks]
        finally:
            op._ENV = saved
    else:
        with multiprocessing.Pool(min(jobs, len(tasks)), _init, (make_env, rows)) as pool:
            results = pool.map(_run_window, tasks, chunksize=1)

    for r in results:
        print(
            f"{r['name']}: train {len(r['train_days'])}d eval {len(r['eval_days'])}d "
            f"cand={r['oos_candidate_net']:,.0f} base={r['oos_baseline_net']:,.0f} "
            f"({r.pop('seconds'):.0f}s)",
            file=sys.stderr,
        )
    summary = aggregate(results)
    out.mkdir(parents=True, exist_ok=True)
    op._dump(out / "windows.json", results)
    op._dump(out / "curve.json", summary)
    return summary


def main(argv: list[str] | None = None) -> None:
    from bandit_dataset import OUT as DATASET  # noqa: PLC0415
    from bandit_dataset import load_rows  # noqa: PLC0415
    from replay_common import CACHE, load_day  # noqa: PLC0415
    from replay_parity import DART, EVENTS, _export_quant_airflow_env  # noqa: PLC0415

    from daytrade_it.application.trading.policy import load_policy  # noqa: PLC0415
    from daytrade_it.infrastructure.external.quant_airflow_db_client import (  # noqa: PLC0415
        QuantAirflowDBClient,
    )
    from daytrade_it.shared.runtime_host import require_backtest_host  # noqa: PLC0415

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--events", type=Path, default=None)
    ap.add_argument("--dataset", type=Path, default=DATASET)
    ap.add_argument("--kinds", default="optimizer,bandit")
    ap.add_argument("--sizes", default="5,10,15,20,25,30")
    ap.add_argument("--holdout", type=int, default=8)
    ap.add_argument("--origins", type=int, default=5)
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--base-policy", type=Path, help="default: live defaults")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--dart", type=Path, default=DART)
    ap.add_argument("--work-root", type=Path, default=WORK)
    ap.add_argument("--cache-dir", type=Path, default=CACHE)
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args(argv)

    require_backtest_host()
    if args.base_policy is not None and not args.base_policy.exists():
        ap.error(f"--base-policy {args.base_policy} does not exist")
    base = load_policy(args.base_policy) if args.base_policy else NewsPolicy()
    kinds = [k for k in args.kinds.split(",") if k]
    unknown = [k for k in kinds if k not in ("optimizer", "bandit")]
    if unknown:
        ap.error(f"--kinds must be optimizer and/or bandit, got {unknown}")
    events_dir = args.events or EVENTS
    days = [dt.date.fromisoformat(p.stem) for p in sorted(events_dir.glob("2026-*.jsonl"))]
    rows = load_rows(args.dataset) if "bandit" in kinds else []
    _export_quant_airflow_env()
    stocks = QuantAirflowDBClient().get_stock_master()
    for d in days:
        load_day(d, args.cache_dir, events_dir)
    started = time.monotonic()
    summary = run(
        days,
        kinds,
        [int(s) for s in args.sizes.split(",") if s],
        args.holdout,
        args.origins,
        base,
        trials=args.trials,
        jobs=args.jobs,
        out=args.out,
        rows=rows,
        make_env=partial(
            op._replay_env, days, stocks, args.dart, args.work_root, args.cache_dir, events_dir
        ),
    )
    print(json.dumps(summary, indent=1, ensure_ascii=False, default=str))
    print(f"wall {time.monotonic() - started:.0f}s -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
