"""Live policy replayed over every event day -- the baseline stages 1-4 must beat.

uv run python scripts/eval/replay_baseline.py --dart data/eval/dart_trader_20260915.db  # simnode

Net per ticker = sells x (1 - sell cost) - buys x (1 + buy cost). Shares still
held when the replay ends (leftovers) are valued at the day's last regular
session close (bar <= 15:32) as if sold there, with the sell cost; a ticker
whose bought quantity != sold + leftover raises.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from decimal import Decimal
from pathlib import Path

#: Closing auction prints at 15:30, or 15:32 when extended by a VI; NXT from 15:35.
REGULAR_LAST_BAR = dt.time(15, 32)
SELL_COST = Decimal("0.00215")  # 0.20% tax + 0.015% commission
BUY_COST = Decimal("0.00015")
OUT = Path("data/eval/replay_baseline")
#: kr_quant carries a few non-equity codes (rights/warrants, e.g. "0013V0") that
#: are not in the stock master and that the live daemon -- which only ever
#: matches articles against 6-digit KRX equity codes -- would never judge.
#: `to_article` raises on these before the stocks-master check runs, so they
#: are dropped here instead of crashing the whole day's replay.
VALID_TICKER = re.compile(r"^\d{6}$")


def resolve_out(out: Path | None) -> Path:
    """Per-day-file output directory for this run.

    Defaults to the shared `OUT` (unchanged behaviour) so an explicit `--out`
    is the only way to opt into a run-private directory -- parallel runs that
    do not pass it keep merging into the same shared pool they always have,
    but a run that does pass it can no longer clobber another run's files.
    """
    return out if out is not None else OUT


def last_regular_close(bars: dict) -> dict[str, Decimal]:
    """Each ticker's close at its last regular-session bar (ts <= 15:32)."""
    out = {}
    for t, bs in bars.items():
        regular = [b for b in bs if b.ts.time() <= REGULAR_LAST_BAR]
        if regular:
            out[t] = max(regular, key=lambda b: b.ts).close
    return out


def trade_pnl(
    fills: list[dict], positions: dict[str, int], last_close: dict[str, Decimal]
) -> list[dict]:
    rows: dict[str, dict] = {}
    for f in fills:
        r = rows.setdefault(
            f["ticker"],
            {
                "ticker": f["ticker"],
                "buy_qty": 0,
                "sell_qty": 0,
                "leftover_qty": 0,
                "buy_value": Decimal(0),
                "sell_value": Decimal(0),
                "leftover_value": Decimal(0),
            },
        )
        value = Decimal(str(f["price"])) * int(f["qty"])
        if f["side"] == "buy":
            r["buy_qty"] += int(f["qty"])
            r["buy_value"] += value
        else:
            r["sell_qty"] += int(f["qty"])
            r["sell_value"] += value
    for t, r in rows.items():
        left = int(positions.get(t, 0))
        if r["buy_qty"] != r["sell_qty"] + left:
            raise ValueError(
                f"{t}: bought {r['buy_qty']} != sold {r['sell_qty']} + leftover {left}"
            )
        if left:
            if t not in last_close:
                raise ValueError(f"{t}: {left} leftover share(s) but no regular close")
            r["leftover_qty"] = left
            r["leftover_value"] = last_close[t] * left
        proceeds = r["sell_value"] + r["leftover_value"]
        r["net"] = proceeds * (1 - SELL_COST) - r["buy_value"] * (1 + BUY_COST)
    return list(rows.values())


def main() -> None:
    from replay_parity import (  # noqa: PLC0415
        DART,
        EVENTS,
        WORK,
        _export_quant_airflow_env,
        prepare_day,
    )

    from daytrade_it.application.replay.day_replay import (  # noqa: PLC0415
        DayConfig,
        ReplayDataError,
        replay_day,
    )
    from daytrade_it.application.replay.events import load_events  # noqa: PLC0415
    from daytrade_it.application.replay.prev_close import FrozenPrevCloseDB  # noqa: PLC0415
    from daytrade_it.infrastructure.external.quant_airflow_db_client import (  # noqa: PLC0415
        QuantAirflowDBClient,
    )
    from daytrade_it.shared.runtime_host import require_backtest_host  # noqa: PLC0415

    ap = argparse.ArgumentParser()
    ap.add_argument("--dart", type=Path, default=DART)
    ap.add_argument("--work-root", type=Path, default=WORK)
    ap.add_argument(
        "--events",
        type=Path,
        default=EVENTS,
        help="Events directory (default data/eval/replay_events; task-8 passes "
        "data/eval/replay_events_v4 to replay the v4-shadow-judged days).",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Per-day output directory (default data/eval/replay_baseline, the "
        "shared pool every run merges into). Pass a run-private directory so "
        "parallel runs cannot overwrite each other's day files.",
    )
    args = ap.parse_args()

    require_backtest_host()
    _export_quant_airflow_env()
    db = QuantAirflowDBClient()
    stocks = db.get_stock_master()
    cfg = DayConfig()  # NewsPolicy() = the live policy
    out_dir = resolve_out(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    total_errors = 0
    failed_days: list[str] = []
    grand = Decimal(0)
    for path in sorted(args.events.glob("2026-*.jsonl")):
        day = dt.date.fromisoformat(path.stem)
        events = load_events(path)
        bad = [e.ticker for e in events if not VALID_TICKER.match(e.ticker)]
        if bad:
            print(f"{day}: dropping non-equity ticker(s) {sorted(set(bad))}", file=sys.stderr)
            events = [e for e in events if VALID_TICKER.match(e.ticker)]
        try:
            bars, frozen, broker, stale_before = prepare_day(day, events)
            res = replay_day(
                day,
                events,
                broker,
                FrozenPrevCloseDB(frozen),
                stocks,
                stale_before,
                cfg,
                args.work_root,
                args.dart,
            )
        except ReplayDataError as exc:
            print(f"{day} FAILED (data): {exc}", file=sys.stderr)
            failed_days.append(str(day))
            continue
        pnl = trade_pnl(res.fills, res.positions, last_regular_close(bars))
        total = sum((r["net"] for r in pnl), Decimal(0))
        grand += total
        losers = sum(r["net"] < 0 for r in pnl)
        leftovers = {r["ticker"]: r["leftover_qty"] for r in pnl if r["leftover_qty"]}
        src = events[0].source if events else "-"
        n_errors = len(res.errors)
        total_errors += n_errors
        print(
            f"{day} [{src}] trades={len(pnl)} net={total:,.0f} losers={losers} "
            f"leftover_shares={sum(leftovers.values())} {leftovers} "
            f"refusals={len(res.refusals)} cash_rejections={len(res.cash_rejections)} "
            f"missing_bars={len(res.missing_bars)} errors={n_errors}"
        )
        for r in res.refusals:
            print(f"    REFUSED {r['ts'][11:19]} {r['ticker']} {r['reason']}")
        (out_dir / f"{day}.json").write_text(
            json.dumps(
                {
                    "day": str(day),
                    "source": src,
                    "trades": pnl,
                    "net": total,
                    "leftovers": leftovers,
                    "refusals": res.refusals,
                    "cash_rejections": res.cash_rejections,
                    "prev_close_flags": frozen.flags,
                    "missing_bars": res.missing_bars,
                    "skips": res.skips,
                    "run_dir": res.run_dir,
                },
                default=str,
                ensure_ascii=False,
                indent=1,
            )
        )
    print(
        f"total net={grand:,.0f} over "
        f"{len(sorted(args.events.glob('2026-*.jsonl'))) - len(failed_days)} days"
    )
    if failed_days:
        print(f"ERROR: data failures on {failed_days}", file=sys.stderr)
    if total_errors:
        print(f"ERROR: {total_errors} error(s) logged during replay", file=sys.stderr)
    if failed_days or total_errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
