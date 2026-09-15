"""Parity gate: replay live days and diff against the live ledger.

    uv run python scripts/eval/replay_parity.py --days 2026-09-14,2026-09-15   # simnode

Pass = same tickers and quantities for entries, add-ons and exits; buy fill
price within ±0.5% per ticker (live fills from Kiwoom ka10076 matched the
ledger limits); no errors logged during the replay.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from decimal import Decimal
from pathlib import Path
from urllib.parse import unquote, urlparse

import psycopg2

from daytrade_it.application.replay.day_replay import DayConfig, replay_day
from daytrade_it.application.replay.events import load_events
from daytrade_it.application.trading.auto_trader import TradingLimits
from daytrade_it.domain.value_objects import Ticker
from daytrade_it.infrastructure.brokers.base import KST, MinuteBar
from daytrade_it.infrastructure.brokers.sim_broker import SimBroker
from daytrade_it.infrastructure.external.quant_airflow_db_client import QuantAirflowDBClient
from daytrade_it.shared.runtime_host import require_backtest_host

EVENTS = Path("data/eval/replay_events")
LIVE_LEDGER = Path("data/auto_trader")
#: scalp-it's dart.db; simnode's copy may lag trader's (pass --dart).
DART = Path("../scalp-it/data/dart.db")
WORK = Path("data/eval/replay_work")
MAX_FILL_DEV = 0.005

LIVE_CONFIGS = {
    # Code before 4ebd39d (cap 2), c273b9b (+2% no-chase), d357b88 (surprise-ranked
    # batch, VWAP exit) and b57e6d8 (add-on): each BUY entered as judged. Live
    # ledger: entries followed their judgment by about two seconds.
    dt.date(2026, 9, 14): DayConfig(
        limits=TradingLimits(max_open_positions=2, vwap_exit_gap=None, add_on_min_gain=None),
        max_pre_entry_move=Decimal("0.02"),
        enter_on_decide=True,
    ),
    # 48d6cb1 defaults.
    dt.date(2026, 9, 15): DayConfig(limits=TradingLimits(), max_pre_entry_move=Decimal("0.03")),
}

#: Orders placed by the second daemon started by the duplicated 08:55 cron on
#: 09-15 (ledger note at 09:08:40). The spec compares against the first daemon.
DUPLICATE_DAEMON_ORDERS = {
    # Entries removed from the public copy: they held live-account order numbers.
}


def _export_quant_airflow_env() -> None:
    """QuantAirflowDBClient reads GPTQ_QUANT_AIRFLOW_DB_*; derive them from KR_QUANT_DB."""
    u = urlparse(os.environ["KR_QUANT_DB"])
    defaults = {
        "GPTQ_QUANT_AIRFLOW_DB_USER": unquote(u.username or ""),
        "GPTQ_QUANT_AIRFLOW_DB_PASSWORD": unquote(u.password or ""),
        "GPTQ_QUANT_AIRFLOW_DB_HOST": u.hostname or "",
        "GPTQ_QUANT_AIRFLOW_DB_PORT": str(u.port or 5432),
        "GPTQ_QUANT_AIRFLOW_DB_DBNAME": u.path.lstrip("/"),
    }
    for k, v in defaults.items():
        os.environ.setdefault(k, v)


def _conn():  # noqa: ANN202
    return psycopg2.connect(os.environ["KR_QUANT_DB"], connect_timeout=5)


def load_session_bars(codes: list[str], day: dt.date) -> dict[str, list[MinuteBar]]:
    out: dict[str, list[MinuteBar]] = {c: [] for c in codes}
    with _conn() as c, c.cursor() as cur:
        cur.execute("SET max_parallel_workers_per_gather = 0")
        cur.execute(
            "SELECT code, ts, open, high, low, close, volume FROM minute_bars "
            "WHERE code = ANY(%s) AND ts >= %s AND ts < %s ORDER BY code, ts",
            (codes, day, day + dt.timedelta(days=1)),
        )
        for code, ts, o, h, lo, cl, v in cur.fetchall():
            out[code].append(
                MinuteBar(
                    ts=ts.replace(tzinfo=None),
                    open=Decimal(str(o)),
                    high=Decimal(str(h)),
                    low=Decimal(str(lo)),
                    close=Decimal(str(cl)),
                    volume=int(v),
                )
            )
    return out


def prev_closes(codes: list[str], day: dt.date) -> dict[str, Decimal]:
    with _conn() as c, c.cursor() as cur:
        cur.execute(
            "SELECT DISTINCT ON (code) code, close FROM daily_bars_adjusted "
            "WHERE code = ANY(%s) AND date < %s ORDER BY code, date DESC",
            (codes, day),
        )
        return {code: Decimal(str(close)) for code, close in cur.fetchall()}


def first_daemon(ledger: list[dict], day: dt.date) -> list[dict]:
    """Drop the second daemon's orders and their exit retries (attempt > 0 handled in summarize)."""
    dup = DUPLICATE_DAEMON_ORDERS.get(day, set())
    return [r for r in ledger if r.get("order_id") not in dup]


def summarize(ledger: list[dict]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {"entry": {}, "add_on": {}, "exit": {}}
    for r in ledger:
        kind = {
            "entry_submitted": "entry",
            "add_on_submitted": "add_on",
            "exit_submitted": "exit",
        }.get(r["event"])
        if kind is None:
            continue
        qty = int(r.get("quantity", 1))
        if kind == "exit" and int(r.get("attempt", 0)) > 0:
            continue  # re-priced retries of the same exit
        out[kind][r["ticker"]] = out[kind].get(r["ticker"], 0) + qty
    return out


def fill_deviations(live: list[dict], fills: list[dict]) -> list[tuple[str, str, str, str, float]]:
    """(kind, ticker, live limit, sim fill, dev) for each live entry/add-on with a sim buy fill."""
    buys: dict[str, list[dict]] = {}
    for f in fills:
        if f["side"] == "buy":
            buys.setdefault(f["ticker"], []).append(f)
    out = []
    for r in live:
        kind = {"entry_submitted": "entry", "add_on_submitted": "add_on"}.get(r["event"])
        if kind is None or not buys.get(r["ticker"]):
            continue
        f = buys[r["ticker"]].pop(0)  # entry precedes add-on per ticker in both ledgers
        dev = float(f["price"]) / float(r["limit"]) - 1
        out.append((kind, r["ticker"], str(r["limit"]), str(f["price"]), dev))
    return out


def main() -> None:
    require_backtest_host()
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", required=True)
    ap.add_argument("--dart", type=Path, default=DART)
    args = ap.parse_args()
    _export_quant_airflow_env()
    db = QuantAirflowDBClient()
    stocks = db.get_stock_master()
    ok_all = True
    for d in args.days.split(","):
        day = dt.date.fromisoformat(d)
        events = load_events(EVENTS / f"{day}.jsonl")
        codes = sorted({e.ticker for e in events})
        bars = load_session_bars(codes, day)
        broker = SimBroker(bars, prev_closes(codes, day), lambda: dt.datetime.now(KST))
        stale_before = dt.datetime.combine(
            db.get_adjusted_daily_bars(
                Ticker(value="005930"),
                start=day - dt.timedelta(days=14),
                end=day - dt.timedelta(days=1),
            )[-1].date,
            dt.time(15, 20),
            tzinfo=KST,
        )
        res = replay_day(
            day,
            events,
            broker,
            db,
            stocks,
            stale_before,
            LIVE_CONFIGS[day],
            WORK / str(day),
            args.dart,
        )
        raw_live = [
            json.loads(line) for line in (LIVE_LEDGER / f"{day}.jsonl").open() if line.strip()
        ]
        live = first_daemon(raw_live, day)
        s_raw, s_live, s_sim = summarize(raw_live), summarize(live), summarize(res.ledger)
        devs = fill_deviations(live, res.fills)
        bad_devs = [x for x in devs if abs(x[4]) > MAX_FILL_DEV]
        same = s_live == s_sim
        ok = same and not bad_devs and not res.errors
        ok_all &= ok
        print(
            f"== {day} parity {'PASS' if ok else 'FAIL'}  ({len(events)} events, {sum(map(len, bars.values()))} bars)"
        )
        print(f"  event tickers without bars: {sorted(c for c, b in bars.items() if not b)}")
        for kind in ("entry", "add_on", "exit"):
            mark = "ok " if s_live[kind] == s_sim[kind] else "DIFF"
            print(f"  {mark} {kind:6} live={s_live[kind]} sim={s_sim[kind]}")
            if s_raw[kind] != s_live[kind]:
                print(f"       {kind:6} raw live (incl. duplicate daemon)={s_raw[kind]}")
        for kind, ticker, limit, price, dev in devs:
            flag = "" if abs(dev) <= MAX_FILL_DEV else "  <-- over ±0.5%"
            print(f"  fill {kind:6} {ticker} live_limit={limit} sim={price} dev={dev:+.3%}{flag}")
        if devs:
            print(f"  max |fill dev| = {max(abs(x[4]) for x in devs):.3%}")
        sells = [f for f in res.fills if f["side"] == "sell"]
        for f in sells:
            print(f"  sim sell {f['ticker']} qty={f['qty']} price={f['price']} ts={f['ts']}")
        print(
            f"  sim ledger events: {[(r['ts'][11:19], r['event'], r.get('ticker')) for r in res.ledger]}"
        )
        print("  skips:", sorted({(s["ticker"], s["reason"]) for s in res.skips}))
        print(f"  errors ({len(res.errors)}):")
        for e in res.errors:
            print("   ", {k: v for k, v in e.items() if k != "exception"})
            if e.get("exception"):
                print("   ", str(e["exception"])[-1500:])
        (WORK / str(day) / "result.json").write_text(
            json.dumps(
                {
                    "ledger": res.ledger,
                    "skips": res.skips,
                    "fills": res.fills,
                    "errors": res.errors,
                },
                default=str,
                ensure_ascii=False,
                indent=1,
            )
        )
    raise SystemExit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
