"""Per-entry fill + minute-bar paths, so exit rules can be searched offline.

uv run python scripts/eval/exit_paths.py --events <dir> --dart <db>   # simnode

Why this exists: searching exit rules through the replay engine costs one full
day-replay per configuration, which is hours for a few hundred configurations.
But `auto_trader.exit_rule_hit` is a *pure function* of (bars, entry price,
entered_at, now, TradingLimits) -- it is the live production predicate. So if we
record, once, each entry's fill and its ticker's minute bars, any exit rule can
be scored offline by calling that same live function. No exit logic is
reimplemented here.

The one thing this file does model is the sell fill: when the rule fires while
evaluating minute `now` (on bars strictly before `now`), the order is placed at
`now` and fills at that minute's open. That is the replay contract's sell row
("the order minute's open is the bid"), minus the retry/discount ladder, which
only matters when the first attempt misses. Any rule that wins here must still
be verified through the real replay engine before it can be judged.

Each day is replayed once with an uncapped copy of the base policy -- caps
lifted, cash 10^12, bandit off, and **add-ons off** so every entry is exactly one
share and its path is a single clean position. Add-ons do not change *which*
pairs are entered (cash and exposure are both uncapped), so the entry set still
matches `bandit_dataset`'s.

Output rows (`--out`, default data/eval/rl_minimal/paths.jsonl):

    day, key, ticker, decide_at, features (bandit.FEATURES), pre_entry_move,
    buy_ts, buy_price, bars: [[minute, o, h, l, c, v], ...] from 09:00 to 15:20
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import multiprocessing
import os
import sys
from decimal import Decimal
from functools import partial
from pathlib import Path

from bandit_dataset import UNCAPPED_CASH, decision_features, uncapped

from daytrade_it.application.replay.day_replay import SESSION_CLOSE
from daytrade_it.application.trading.policy import NewsPolicy

OUT = Path("data/eval/rl_minimal/paths.jsonl")
WORK = Path("data/eval/rl_minimal/paths_work")
_OPEN = dt.time(9, 0)


def no_add_ons(
    base: NewsPolicy, min_score: float | None = None, max_pre_move: Decimal | None = None
) -> NewsPolicy:
    """`uncapped(base)` with pyramiding off, so one entry == one share.

    `min_score` / `max_pre_move` additionally open the two hand-set entry gates
    (the 0.15 score threshold and the no-chase cap), so the recorded set includes
    the pairs the live rule refuses. A contextual threshold cannot be learned from
    a sample that only contains what the current threshold already admits.
    """
    pol = uncapped(base)
    limits = dataclasses.replace(pol.limits, add_on_min_gain=None)
    sel = pol.selection
    if min_score is not None:
        sel = dataclasses.replace(sel, min_score=min_score)
    if max_pre_move is not None:
        sel = dataclasses.replace(sel, max_pre_move=max_pre_move)
    return dataclasses.replace(pol, limits=limits, selection=sel)


def _minute(ts: dt.datetime) -> int:
    return ts.hour * 60 + ts.minute


def day_rows(  # noqa: ANN001
    data,
    base: NewsPolicy,
    stocks: dict,
    dart: Path,
    work_root: Path,
    min_score: float | None = None,
    max_pre_move: Decimal | None = None,
) -> list[dict]:
    """One row per entry that actually filled in the uncapped, add-on-free replay of `data`."""
    from replay_common import run_day  # noqa: PLC0415

    policy = no_add_ons(base, min_score, max_pre_move)
    res, pnl = run_day(data, policy, stocks, dart, work_root, cash=UNCAPPED_CASH)
    if res.errors:
        raise RuntimeError(f"{data.day}: {len(res.errors)} error(s) logged in the uncapped replay")

    first_buy: dict[str, dict] = {}
    for f in res.fills:
        if f["side"] != "buy":
            continue
        prev = first_buy.get(f["ticker"])
        if prev is None or f["ts"] < prev["ts"]:
            first_buy[f["ticker"]] = f

    # The engine's own reward for this exact policy, so the offline lab can be
    # checked against it apples-to-apples (same entries, same add-on setting).
    by_ticker = {r["ticker"]: r for r in pnl}
    events = {e.key: e for e in data.events}
    close_at = dt.datetime.combine(data.day, SESSION_CLOSE)
    rows = []
    for d in res.decisions:
        buy = first_buy.get(d["ticker"])
        if d["outcome"] not in {"entered", "deferred"} or buy is None:
            continue
        e = events[d["key"]]
        bars = [
            [
                _minute(b.ts),
                float(b.open),
                float(b.high),
                float(b.low),
                float(b.close),
                int(b.volume),
            ]
            for b in data.bars.get(d["ticker"], [])
            if b.ts.time() >= _OPEN and b.ts <= close_at
        ]
        if not bars:
            continue
        pre = float(Decimal(d["pre_entry_move"]))
        row = by_ticker.get(d["ticker"])
        engine_reward = float(row["net"] / row["buy_value"]) if row and row["buy_value"] else None
        rows.append(
            {
                "day": str(data.day),
                "engine_reward": engine_reward,
                "key": d["key"],
                "ticker": d["ticker"],
                "decide_at": e.decide_at.isoformat(),
                "pre_entry_move": pre,
                "buy_minute": _minute(buy["ts"]),
                "buy_price": float(buy["price"]),
                "features": decision_features(data, e, pre),
                "bars": bars,
            }
        )
        first_buy.pop(d["ticker"], None)  # one entry per ticker per day
    return rows


_WORKER: dict = {}


def _init(  # noqa: ANN001
    stocks: dict, dart: Path, work_root: Path, cache_dir: Path, base, events_dir, gates
) -> None:
    _WORKER.update(
        stocks=stocks,
        dart=dart,
        work=work_root / str(os.getpid()),
        cache=cache_dir,
        base=base,
        events=events_dir,
        gates=gates,
    )


def _day_job(day: dt.date) -> list[dict]:
    from replay_common import load_day  # noqa: PLC0415

    w = _WORKER
    try:
        data = load_day(day, w["cache"], w["events"])
        return day_rows(data, w["base"], w["stocks"], w["dart"], w["work"], *w["gates"])
    except Exception as exc:  # noqa: BLE001 -- one bad day must not kill the pass
        print(f"{day}: SKIPPED -- {exc}", file=sys.stderr)
        return []


def main(argv: list[str] | None = None) -> None:
    from replay_common import CACHE, load_day  # noqa: PLC0415
    from replay_parity import DART, EVENTS, _export_quant_airflow_env  # noqa: PLC0415

    from daytrade_it.application.trading.policy import load_policy  # noqa: PLC0415
    from daytrade_it.infrastructure.external.quant_airflow_db_client import (  # noqa: PLC0415
        QuantAirflowDBClient,
    )
    from daytrade_it.shared.runtime_host import require_backtest_host  # noqa: PLC0415

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-policy", type=Path, help="default: live defaults")
    ap.add_argument("--days", help="comma-separated; default: every events day")
    ap.add_argument("--days-file", type=Path, help="one ISO day per line")
    ap.add_argument("--events", type=Path, default=None)
    ap.add_argument("--dart", type=Path, default=DART)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--work-root", type=Path, default=WORK)
    ap.add_argument("--cache-dir", type=Path, default=CACHE)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument(
        "--min-score",
        type=float,
        help="override SelectionConfig.min_score (0.0 records every positive-score pair)",
    )
    ap.add_argument(
        "--max-pre-move", type=Decimal, help="override the no-chase cap (e.g. 10 to disable it)"
    )
    args = ap.parse_args(argv)

    require_backtest_host()
    base = load_policy(args.base_policy) if args.base_policy else NewsPolicy()
    events_dir = args.events or EVENTS
    if args.days:
        days = [dt.date.fromisoformat(d) for d in args.days.split(",")]
    elif args.days_file:
        days = [
            dt.date.fromisoformat(line.strip())
            for line in args.days_file.read_text().splitlines()
            if line.strip()
        ]
    else:
        days = [dt.date.fromisoformat(p.stem) for p in sorted(events_dir.glob("2026-*.jsonl"))]
    _export_quant_airflow_env()
    stocks = QuantAirflowDBClient().get_stock_master()
    for d in days:
        load_day(d, args.cache_dir, events_dir)
    gates = (args.min_score, args.max_pre_move)
    init = partial(
        _init, stocks, args.dart, args.work_root, args.cache_dir, base, events_dir, gates
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with (
        multiprocessing.Pool(min(args.jobs, len(days)), init) as pool,
        args.out.open("w", encoding="utf-8") as fh,
    ):
        for day, rows in zip(days, pool.imap(_day_job, days, chunksize=1), strict=True):
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            total += len(rows)
            print(f"{day}: {len(rows)} entries", file=sys.stderr)
    print(f"{total} entries -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
