"""Leave-one-day-out x Optuna TPE search over NewsPolicy (Bayesian stages 2-3).

uv run python scripts/eval/optimize_policy.py --stage stage2 --trials 300 --jobs 14 \\
    --dart data/eval/dart_trader_20260915.db                          # simnode
uv run python scripts/eval/optimize_policy.py --stage stage3 \\
    --base-policy data/eval/opt/stage2/policy.json --dart ...

- stage2: selection + buy timing + stop-loss; stage3: profit-taking, on top of
  the stage2 policy (`--base-policy`).
- One study per held-out day (trained on the other days, `TPESampler(seed=fold
  index)`) plus a `full` study over every day (seed = number of days). Trial 0
  of every study is the base policy (`--base-policy`, else the live defaults).
- objective = sum of train-day nets - 0.5 x max(0, -worst train-day net); a
  trial whose replay logged any error scores -1e12.
- Output under `--out`: `fold-<day>.json` (best params, train objective,
  held-out results), `oof.json` (held-out results of each fold's best vs the
  base policy), `full.json`, `judge.json` (`bayes_judge.judge` oof vs base),
  `policy.json` (the full study's best policy) and `trials-<study>.csv`
  (params, objective and per-day nets of every trial, for convergence).
- Studies run in `multiprocessing.Pool(jobs)`; each worker reads the cached
  days once and replays under its own `--work-root/<pid>`.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as dt
import json
import multiprocessing
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from functools import partial
from pathlib import Path
from typing import Any

import optuna

from daytrade_it.application.replay.day_replay import DEFAULT_MAX_PRE_ENTRY_MOVE
from daytrade_it.application.trading.policy import (
    NewsPolicy,
    dump_policy,
    load_policy,
    policy_to_dict,
)

ERROR_SCORE = -1e12
WORST_DAY_WEIGHT = 0.5
OPT_OUT = Path("data/eval/opt")
OPT_WORK = Path("data/eval/opt_work")


# -- search spaces -----------------------------------------------------------


def _dec(v: float | None) -> Decimal | None:
    return None if v is None else Decimal(str(v))


def _undec(v: Decimal | None) -> float | None:
    return None if v is None else float(v)


def _minutes(v: int | None) -> dt.timedelta | None:
    return None if v is None else dt.timedelta(minutes=v)


def _unminutes(v: dt.timedelta | None) -> int | None:
    if v is None:
        return None
    minutes = v.total_seconds() / 60
    return int(minutes) if minutes == int(minutes) else minutes  # type: ignore[return-value]


def _delay(v: int) -> dt.timedelta | None:
    # 0 = enter as judged (None), not a zero-length deferred entry.
    return None if v == 0 else dt.timedelta(minutes=v)


def _undelay(v: dt.timedelta | None) -> int | float:
    return 0 if v is None else _unminutes(v)  # type: ignore[return-value]


def _csv(v: str) -> tuple[str, ...]:
    return tuple(v.split(","))


def _uncsv(v: tuple[str, ...]) -> str:
    return ",".join(v)


def _hhmm(v: str) -> dt.time:
    return dt.time.fromisoformat(v)


def _unhhmm(v: dt.time) -> str:
    return v.strftime("%H:%M") if v.second == 0 and v.microsecond == 0 else v.isoformat()


def _max_pre(v: float) -> Decimal:
    return Decimal(str(v))


def _unmax_pre(v: Decimal | None) -> float:
    # None = the replay's/live default no-chase cap.
    return float(DEFAULT_MAX_PRE_ENTRY_MOVE if v is None else v)


def _same(v: Any) -> Any:
    return v


@dataclass(frozen=True)
class Param:
    name: str
    section: str  # "limits" | "selection"
    field: str
    choices: tuple | None = None  # categorical
    low: float | None = None  # float range (when choices is None)
    high: float | None = None
    to_policy: Callable[[Any], Any] = _same
    from_policy: Callable[[Any], Any] = _same

    def suggest(self, trial: optuna.Trial) -> Any:
        if self.choices is None:
            return trial.suggest_float(self.name, self.low, self.high)
        return trial.suggest_categorical(self.name, list(self.choices))


def _cat(name: str, section: str, field: str, choices: tuple, to=_same, back=_same) -> Param:
    return Param(name, section, field, choices=choices, to_policy=to, from_policy=back)


STAGE2: tuple[Param, ...] = (
    Param("min_score", "selection", "min_score", low=0.15, high=0.6),
    _cat("subjects", "selection", "subjects", ("main,partial", "main"), _csv, _uncsv),
    _cat(
        "min_pre_move",
        "selection",
        "min_pre_move",
        (None, -0.05, -0.03, -0.02, -0.01),
        _dec,
        _undec,
    ),
    # Never 0: `no_chase_cap` would read 0 as "unset" and apply the 3% default.
    _cat(
        "max_pre_move", "selection", "max_pre_move", (0.01, 0.02, 0.03, 0.05), _max_pre, _unmax_pre
    ),
    _cat("sessions", "selection", "sessions", ("premarket,intraday", "premarket"), _csv, _uncsv),
    _cat("rank_key", "selection", "rank_key", ("surprise", "persistence", "score")),
    _cat("entry_delay_min", "limits", "entry_delay", (0, 3, 5, 10, 20), _delay, _undelay),
    _cat("entry_pullback", "limits", "entry_pullback", (None, 0.005, 0.01, 0.02), _dec, _undec),
    _cat("entry_vwap_reclaim", "limits", "entry_vwap_reclaim", (False, True)),
    _cat("entry_limit_ticks", "limits", "entry_limit_ticks", (0, 1)),
    _cat("stop_loss_pct", "limits", "stop_loss_pct", (None, 0.01, 0.015, 0.02, 0.03), _dec, _undec),
    _cat("stop_vwap_gap", "limits", "stop_vwap_gap", (None, 0.005, 0.01), _dec, _undec),
    _cat("stop_time_min", "limits", "stop_time", (None, 30, 60, 120), _minutes, _unminutes),
    _cat("stop_orb_min", "limits", "stop_orb", (None, 5, 15, 30), _minutes, _unminutes),
)

STAGE3: tuple[Param, ...] = (
    _cat(
        "take_profit_pct", "limits", "take_profit_pct", (None, 0.02, 0.03, 0.05, 0.08), _dec, _undec
    ),
    _cat("trail_after_gain", "limits", "trail_after_gain", (None, 0.01, 0.02, 0.03), _dec, _undec),
    _cat("trail_gap", "limits", "trail_gap", (0.005, 0.01, 0.02), _dec, _undec),
    _cat("vwap_exit_gap", "limits", "vwap_exit_gap", (None, 0.005, 0.01, 0.02), _dec, _undec),
    _cat("vwap_exit_min_gain", "limits", "vwap_exit_min_gain", (0.01, 0.02, 0.03), _dec, _undec),
    _cat("add_on_min_gain", "limits", "add_on_min_gain", (None, 0.01, 0.02), _dec, _undec),
    _cat("flatten_at", "limits", "flatten_at", ("14:30", "15:00", "15:15"), _hhmm, _unhhmm),
)

PARAMS: dict[str, tuple[Param, ...]] = {"stage2": STAGE2, "stage3": STAGE3}
assert 0 not in {p.name: p for p in STAGE2}["max_pre_move"].choices  # noqa: S101


def _in_space(p: Param, v: Any) -> bool:
    if p.choices is None:
        return isinstance(v, int | float) and p.low <= v <= p.high  # type: ignore[operator]
    return any(v == c and type(v) is type(c) for c in p.choices)


def base_params(stage: str, base: NewsPolicy) -> dict[str, Any]:
    """`base`'s values as `stage` params (trial 0); ValueError if one is outside the space."""
    out = {}
    for p in PARAMS[stage]:
        v = p.from_policy(getattr(getattr(base, p.section), p.field))
        if not _in_space(p, v):
            raise ValueError(
                f"base policy {p.section}.{p.field} -> {p.name}={v!r} not in the space"
            )
        out[p.name] = v
    return out


def _build(stage: str, trial: optuna.Trial, base: NewsPolicy) -> NewsPolicy:
    """The policy `trial` suggests; a param equal to the base's value keeps the base field."""
    changes: dict[str, dict[str, Any]] = {"limits": {}, "selection": {}}
    for p in PARAMS[stage]:
        v = p.suggest(trial)
        current = p.from_policy(getattr(getattr(base, p.section), p.field))
        if current == v and type(current) is type(v):
            continue
        changes[p.section][p.field] = p.to_policy(v)
    return NewsPolicy(
        limits=dataclasses.replace(base.limits, **changes["limits"]),
        selection=dataclasses.replace(base.selection, **changes["selection"]),
    )


SPACES: dict[str, Callable[[optuna.Trial, NewsPolicy], NewsPolicy]] = {
    stage: partial(_build, stage) for stage in PARAMS
}


# -- objective ---------------------------------------------------------------


def objective(results: list[dict]) -> float:
    """Sum of nets - 0.5 x max(0, -worst net); any replay error -> -1e12."""
    if any(r["errors"] for r in results):
        return ERROR_SCORE
    nets = [float(r["net"]) for r in results]
    return sum(nets) - WORST_DAY_WEIGHT * max(0.0, -min(nets))


# -- studies -----------------------------------------------------------------


@dataclass
class Env:
    """Per-worker state: `evaluate(day, policy) -> day_net record`."""

    evaluate: Callable[[dt.date, NewsPolicy], dict]


@dataclass(frozen=True)
class StudyTask:
    name: str  # "fold-<day>" | "full"
    stage: str
    seed: int
    train_days: tuple[dt.date, ...]
    held_out: dt.date | None
    base: NewsPolicy
    trials: int


_ENV: Env | None = None


def _init_worker(make_env: Callable[[], Env]) -> None:
    global _ENV  # noqa: PLW0603
    _ENV = make_env()


def _jsonable(record: dict) -> dict:
    return json.loads(json.dumps(record, default=str))


def _run_study(task: StudyTask) -> dict:
    env = _ENV
    assert env is not None  # noqa: S101
    started = time.monotonic()

    def trial_objective(trial: optuna.Trial) -> float:
        policy = SPACES[task.stage](trial, task.base)
        results = [env.evaluate(d, policy) for d in task.train_days]
        trial.set_user_attr("results", [_jsonable(r) for r in results])
        return objective(results)

    study = optuna.create_study(
        study_name=task.name,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=task.seed),
    )
    study.enqueue_trial(base_params(task.stage, task.base))
    study.optimize(trial_objective, n_trials=task.trials)

    best = study.best_trial
    best_policy = SPACES[task.stage](optuna.trial.FixedTrial(best.params), task.base)
    out: dict[str, Any] = {
        "name": task.name,
        "stage": task.stage,
        "seed": task.seed,
        "trials": task.trials,
        "train_days": [str(d) for d in task.train_days],
        "best_trial": best.number,
        "best_params": best.params,
        "best_objective": best.value,
        "baseline_objective": study.trials[0].value,
        "train_results": best.user_attrs["results"],
        "baseline_train_results": study.trials[0].user_attrs["results"],
        "policy": policy_to_dict(best_policy),
    }
    if task.held_out is not None:
        out["held_out"] = str(task.held_out)
        out["held_out_result"] = _jsonable(env.evaluate(task.held_out, best_policy))
        out["baseline_held_out_result"] = _jsonable(env.evaluate(task.held_out, task.base))
    out["_table"] = [
        {
            "number": t.number,
            "objective": t.value,
            **{f"param_{k}": v for k, v in t.params.items()},
            **{f"net_{r['day']}": r["net"] for r in t.user_attrs.get("results", [])},
        }
        for t in study.trials
    ]
    out["_seconds"] = time.monotonic() - started
    return out


def _write_table(path: Path, rows: list[dict]) -> None:
    fields: list[str] = []
    for r in rows:
        fields += [k for k in r if k not in fields]
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _dump(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=str) + "\n")


def run(
    stage: str,
    days: list[dt.date],
    base: NewsPolicy,
    trials: int,
    jobs: int,
    out: Path,
    make_env: Callable[[], Env],
) -> dict:
    """LODO folds + a full-days study; writes the output files; returns the judge verdict."""
    from bayes_judge import judge  # noqa: PLC0415

    days = sorted(days)
    base_params(stage, base)  # fail before any replay if trial 0 isn't representable
    tasks = [
        StudyTask(
            name=f"fold-{d}",
            stage=stage,
            seed=i,
            train_days=tuple(x for x in days if x != d),
            held_out=d,
            base=base,
            trials=trials,
        )
        for i, d in enumerate(days)
    ]
    tasks.append(
        StudyTask("full", stage, len(days), tuple(days), None, base, trials),
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    if jobs <= 1:
        global _ENV  # noqa: PLW0603
        saved, _ENV = _ENV, make_env()
        try:
            results = [_run_study(t) for t in tasks]
        finally:
            _ENV = saved
    else:
        with multiprocessing.Pool(min(jobs, len(tasks)), _init_worker, (make_env,)) as pool:
            results = pool.map(_run_study, tasks, chunksize=1)

    out.mkdir(parents=True, exist_ok=True)
    by_name = {r["name"]: r for r in results}
    for r in results:
        _write_table(out / f"trials-{r['name']}.csv", r.pop("_table"))
        print(
            f"{r['name']}: best trial {r['best_trial']} objective {r['best_objective']:,.0f} "
            f"(baseline {r['baseline_objective']:,.0f}) in {r.pop('_seconds'):.0f}s",
            file=sys.stderr,
        )
    for d in days:
        _dump(out / f"fold-{d}.json", by_name[f"fold-{d}"])
    oof = {
        "candidate": [by_name[f"fold-{d}"]["held_out_result"] for d in days],
        "baseline": [by_name[f"fold-{d}"]["baseline_held_out_result"] for d in days],
    }
    _dump(out / "oof.json", oof)
    full = by_name["full"]
    _dump(out / "full.json", full)
    verdict = judge(
        {r["day"]: r for r in oof["candidate"]},
        {r["day"]: r for r in oof["baseline"]},
    )
    _dump(out / "judge.json", verdict)
    dump_policy(
        SPACES[stage](optuna.trial.FixedTrial(full["best_params"]), base),
        out / "policy.json",
    )
    return verdict


# -- CLI ---------------------------------------------------------------------


def _replay_env(
    days: list[dt.date],
    stocks: dict,
    dart: Path,
    work_root: Path,
    cache_dir: Path,
    events_dir: Path | None = None,
) -> Env:
    from replay_common import day_net, load_day  # noqa: PLC0415

    data = {d: load_day(d, cache_dir, events_dir) for d in days}
    work = work_root / str(os.getpid())

    def evaluate(day: dt.date, policy: NewsPolicy) -> dict:
        return day_net(data[day], policy, stocks, dart, work)

    return Env(evaluate=evaluate)


def main(argv: list[str] | None = None) -> None:
    from replay_common import CACHE, load_day  # noqa: PLC0415
    from replay_parity import DART, EVENTS, _export_quant_airflow_env  # noqa: PLC0415

    from daytrade_it.infrastructure.external.quant_airflow_db_client import (  # noqa: PLC0415
        QuantAirflowDBClient,
    )
    from daytrade_it.shared.runtime_host import require_backtest_host  # noqa: PLC0415

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=sorted(PARAMS), required=True)
    ap.add_argument("--trials", type=int, default=300)
    ap.add_argument("--jobs", type=int, default=14)
    ap.add_argument("--base-policy", type=Path, help="default: live defaults")
    ap.add_argument("--out", type=Path, help="default: data/eval/opt/<stage>")
    ap.add_argument("--days", help="comma-separated; default: every replay events day")
    ap.add_argument(
        "--events",
        type=Path,
        default=None,
        help="events directory (default data/eval/replay_events; A7 passes "
        "data/eval/replay_events_universe_a7)",
    )
    ap.add_argument("--dart", type=Path, default=DART)
    ap.add_argument("--work-root", type=Path, default=OPT_WORK)
    ap.add_argument("--cache-dir", type=Path, default=CACHE)
    args = ap.parse_args(argv)

    require_backtest_host()
    if args.base_policy is not None and not args.base_policy.exists():
        ap.error(f"--base-policy {args.base_policy} does not exist")
    base = load_policy(args.base_policy) if args.base_policy else NewsPolicy()
    if args.stage == "stage3" and args.base_policy is None:
        print(
            "WARNING: stage3 without --base-policy searches on the live defaults", file=sys.stderr
        )
    events_dir = args.events or EVENTS
    days = (
        [dt.date.fromisoformat(d) for d in args.days.split(",")]
        if args.days
        else [dt.date.fromisoformat(p.stem) for p in sorted(events_dir.glob("2026-*.jsonl"))]
    )
    out = args.out or OPT_OUT / args.stage
    _export_quant_airflow_env()
    stocks = QuantAirflowDBClient().get_stock_master()
    for d in days:  # build any missing cache once, before the workers read it
        load_day(d, args.cache_dir, events_dir)
    started = time.monotonic()
    verdict = run(
        args.stage,
        days,
        base,
        trials=args.trials,
        jobs=args.jobs,
        out=out,
        make_env=partial(
            _replay_env, days, stocks, args.dart, args.work_root, args.cache_dir, events_dir
        ),
    )
    print(json.dumps(verdict, indent=1, ensure_ascii=False))
    print(f"wall {time.monotonic() - started:.0f}s -> {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
