"""Live policy replayed over every event day -- the baseline stages 1-4 must beat.

uv run python scripts/eval/replay_baseline.py   # simnode
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from decimal import Decimal
from pathlib import Path

SELL_COST = Decimal("0.00215")  # 0.20% tax + 0.015% commission
BUY_COST = Decimal("0.00015")
OUT = Path("data/eval/replay_baseline")


def trade_pnl(fills: list[dict]) -> list[dict]:
    rows: dict[str, dict] = {}
    for f in fills:
        r = rows.setdefault(
            f["ticker"],
            {
                "ticker": f["ticker"],
                "buy_qty": 0,
                "buy_value": Decimal(0),
                "sell_value": Decimal(0),
            },
        )
        value = Decimal(str(f["price"])) * int(f["qty"])
        if f["side"] == "buy":
            r["buy_qty"] += int(f["qty"])
            r["buy_value"] += value
        else:
            r["sell_value"] += value
    for r in rows.values():
        r["net"] = r["sell_value"] * (1 - SELL_COST) - r["buy_value"] * (1 + BUY_COST)
    return list(rows.values())


def main() -> None:
    from replay_parity import DART, EVENTS, WORK, load_session_bars, prev_closes  # noqa: PLC0415

    from daytrade_it.application.replay.day_replay import DayConfig, replay_day  # noqa: PLC0415
    from daytrade_it.application.replay.events import load_events  # noqa: PLC0415
    from daytrade_it.application.trading.auto_trader import TradingLimits  # noqa: PLC0415
    from daytrade_it.domain.value_objects import Ticker  # noqa: PLC0415
    from daytrade_it.infrastructure.brokers.base import KST  # noqa: PLC0415
    from daytrade_it.infrastructure.brokers.sim_broker import SimBroker  # noqa: PLC0415
    from daytrade_it.infrastructure.external.quant_airflow_db_client import (  # noqa: PLC0415
        QuantAirflowDBClient,
    )
    from daytrade_it.shared.runtime_host import require_backtest_host  # noqa: PLC0415

    ap = argparse.ArgumentParser()
    ap.add_argument("--dart", type=Path, default=DART)
    args = ap.parse_args()

    require_backtest_host()
    db = QuantAirflowDBClient()
    stocks = db.get_stock_master()
    cfg = DayConfig(limits=TradingLimits(), max_pre_entry_move=Decimal("0.03"))
    OUT.mkdir(parents=True, exist_ok=True)
    total_errors = 0
    for path in sorted(EVENTS.glob("2026-*.jsonl")):
        day = dt.date.fromisoformat(path.stem)
        events = load_events(path)
        codes = sorted({e.ticker for e in events})
        broker = SimBroker(
            load_session_bars(codes, day), prev_closes(codes, day), lambda: dt.datetime.now(KST)
        )
        prior = db.get_adjusted_daily_bars(
            Ticker(value="005930"),
            start=day - dt.timedelta(days=14),
            end=day - dt.timedelta(days=1),
        )[-1].date
        stale_before = dt.datetime.combine(prior, dt.time(15, 20), tzinfo=KST)
        res = replay_day(
            day, events, broker, db, stocks, stale_before, cfg, WORK / f"baseline-{day}", args.dart
        )
        pnl = trade_pnl(res.fills)
        total = sum((r["net"] for r in pnl), Decimal(0))
        losers = sum(r["net"] < 0 for r in pnl)
        src = events[0].source if events else "-"
        n_errors = len(res.errors)
        total_errors += n_errors
        print(f"{day} [{src}] trades={len(pnl)} net={total:,.0f} losers={losers} errors={n_errors}")
        (OUT / f"{day}.json").write_text(
            json.dumps(
                {"day": str(day), "source": src, "trades": pnl, "net": total, "skips": res.skips},
                default=str,
                ensure_ascii=False,
                indent=1,
            )
        )
    if total_errors:
        print(f"ERROR: {total_errors} error(s) logged during replay", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
