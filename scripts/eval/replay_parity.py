"""Parity gate: replay live days and diff against the live ledger.

    uv run python scripts/eval/replay_parity.py --days 2026-09-14,2026-09-15   # simnode

Exact comparison (printed): same tickers and quantities for entries, add-ons
and exits; buy fill price within ±0.5% per ticker.

Verdict (controller ruling R9): every judged pair whose outcome differs between
the live daemon log and the replay, and every ticker-level difference (add-on,
exit quantity, entry, fill > ±0.5%), is listed and classified:
``intra_minute_price`` when the live price behind it (ledger limit, or
pre_entry_move x prev close) is not the open of that minute -- the replay
quotes the open and never looks inside the current bar -- else ``other``.
PASS = no ``other`` difference and no errors logged during the replay.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from urllib.parse import unquote, urlparse

import psycopg2
from krx_quant_core.market.ticks import tick_size

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
#: Copy of trader's data/news_signal_daemon.log (the live daemon's JSON log).
LIVE_LOG = Path("data/eval/news_signal_daemon_trader_0915.log")
INTRA = "intra_minute_price"
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


def _kst(ts: str) -> dt.datetime:
    return dt.datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(KST)


def live_decisions(rows: list[dict], day: dt.date, exclude_orders: set[str]) -> list[dict]:
    """One row per live `judged` log line on `day`, with its outcome from later rows.

    extended / persisted rows attach to the latest undecided judgment of that
    ticker; ENTRY SUBMITTED / entry skipped rows to the earliest judgment of
    that ticker still holding an unentered signal. Orders in
    `exclude_orders` (the duplicate daemon's) are ignored.
    """
    out: list[dict] = []

    def latest(t: str, outcome: str) -> dict | None:
        return next(
            (d for d in reversed(out) if d["ticker"] == t and d["outcome"] == outcome), None
        )

    def earliest(t: str, outcome: str) -> dict | None:
        return next((d for d in out if d["ticker"] == t and d["outcome"] == outcome), None)

    for r in rows:
        if "timestamp" not in r or "event" not in r:
            continue
        ts = _kst(r["timestamp"])
        if ts.date() != day:
            continue
        ev, t = r["event"], r.get("ticker")
        if ev == "news_signal_poller: judged":
            out.append({"ticker": t, "at": ts, "outcome": "no_signal"})
        elif ev == "entry_filter: extended, no entry":
            if (d := latest(t, "no_signal")) is not None:
                d.update(outcome="extended", pre_entry_move=r.get("pre_entry_move"))
        elif ev == "news_signal_poller: ACTIONABLE SIGNAL PERSISTED":
            if (d := latest(t, "no_signal")) is not None:
                d.update(outcome="signal", pre_entry_move=r.get("pre_entry_move"))
        elif ev == "auto_trader: ENTRY SUBMITTED":
            if r.get("order_id") in exclude_orders:
                continue
            if (d := earliest(t, "signal")) is not None:
                d.update(outcome="entered", limit=r.get("limit"), entered_at=ts)
        elif ev == "auto_trader: entry skipped" and (d := earliest(t, "signal")) is not None:
            d.update(outcome="skipped", reason=r.get("reason"))
    return out


def match_decisions(
    sim: list[dict], live: list[dict], within: dt.timedelta
) -> list[tuple[dict, dict | None]]:
    """Each replay decision with the nearest unused live judgment of the same ticker."""
    used: set[int] = set()
    out: list[tuple[dict, dict | None]] = []
    for s in sim:
        best = None
        for i, d in enumerate(live):
            if i in used or d["ticker"] != s["ticker"]:
                continue
            gap = abs(d["at"] - s["decide_at"])
            if gap <= within and (best is None or gap < best[0]):
                best = (gap, i)
        if best is None:
            out.append((s, None))
        else:
            used.add(best[1])
            out.append((s, live[best[1]]))
    return out


def _round_tick(price: Decimal) -> Decimal:
    tick = tick_size(price)
    return (price / tick).quantize(Decimal(1), rounding=ROUND_HALF_UP) * tick


def reproducible_from_open(live_price: Decimal, bar_open: Decimal) -> bool:
    return _round_tick(live_price) == bar_open


def _outcome_key(d: dict) -> str:
    return f"skipped:{d.get('reason')}" if d["outcome"] == "skipped" else d["outcome"]


def _open_at(bars: dict[str, list[MinuteBar]], t: str, at: dt.datetime) -> Decimal | None:
    minute = at.astimezone(KST).replace(tzinfo=None, second=0, microsecond=0)
    return next((b.open for b in bars.get(t, []) if b.ts == minute), None)


def _price_class(
    bars: dict[str, list[MinuteBar]], t: str, price: Decimal, at: dt.datetime, src: str
) -> tuple[str, str]:
    o = _open_at(bars, t, at)
    if o is None:
        return "other", f"live {src} {price}: no bar at {at.time()}"
    note = f"live {src} {price:.1f} @{at.strftime('%H:%M:%S')} vs open {o}"
    if _round_tick(price) - price > Decimal("0.01") or price - _round_tick(price) > Decimal("0.01"):
        note += " (implied price off tick grid: prev-close vintage?)"
    return (INTRA if not reproducible_from_open(price, o) else "other"), note


def divergences(
    pairs: list[tuple[dict, dict | None]],
    live_ledger: list[dict],
    sim_ledger: list[dict],
    sim_fills: list[dict],
    bars: dict[str, list[MinuteBar]],
    prev_close: dict[str, Decimal],
) -> list[dict]:
    """Decision- and ticker-level live/replay differences, each classified."""
    out: list[dict] = []
    decision_cls: dict[str, str] = {}
    for sim, live in pairs:
        t = sim["ticker"]
        if live is None:
            out.append(
                {
                    "kind": "decision",
                    "ticker": t,
                    "at": sim["decide_at"],
                    "live": "no live judgment within window",
                    "sim": _outcome_key(sim),
                    "cls": "other",
                    "note": "",
                }
            )
            continue
        if _outcome_key(sim) == _outcome_key(live):
            continue
        cls, note = "other", "no live price for this pair"
        if live["outcome"] == "entered" and live.get("limit"):
            cls, note = _price_class(bars, t, Decimal(live["limit"]), live["entered_at"], "limit")
        elif live.get("pre_entry_move") and t in prev_close:
            price = prev_close[t] * (1 + Decimal(live["pre_entry_move"]))
            cls, note = _price_class(bars, t, price, live["at"], "pre_entry_move x prev_close")
        out.append(
            {
                "kind": "decision",
                "ticker": t,
                "at": sim["decide_at"],
                "live": _outcome_key(live),
                "sim": _outcome_key(sim),
                "cls": cls,
                "note": note,
            }
        )
        if decision_cls.get(t) != INTRA:
            decision_cls[t] = cls

    live_entry = {}
    for r in live_ledger:
        if r["event"] == "entry_submitted" and r["ticker"] not in live_entry:
            live_entry[r["ticker"]] = r

    def ticker_cls(t: str) -> tuple[str, str]:
        if decision_cls.get(t) == INTRA:
            return INTRA, "follows the intra-minute decision difference on this ticker"
        r = live_entry.get(t)
        if r is None or not r.get("ts"):
            return "other", "no live entry price"
        return _price_class(bars, t, Decimal(r["limit"]), _kst(r["ts"]), "entry limit")

    s_live, s_sim = summarize(live_ledger), summarize(sim_ledger)
    for kind, label in (("entry", "entry"), ("add_on", "add_on"), ("exit", "exit_qty")):
        for t in sorted(set(s_live[kind]) | set(s_sim[kind])):
            lq, sq = s_live[kind].get(t, 0), s_sim[kind].get(t, 0)
            if lq == sq or (kind == "entry" and t in decision_cls):
                continue
            cls, note = ticker_cls(t)
            out.append(
                {
                    "kind": label,
                    "ticker": t,
                    "at": None,
                    "live": lq,
                    "sim": sq,
                    "cls": cls,
                    "note": note,
                }
            )
    for kind, t, limit, price, dev in fill_deviations(live_ledger, sim_fills):
        if abs(dev) > MAX_FILL_DEV:
            r = next(x for x in live_ledger if x["ticker"] == t and x.get("limit") == limit)
            cls, note = (
                _price_class(bars, t, Decimal(limit), _kst(r["ts"]), f"{kind} limit")
                if r.get("ts")
                else ("other", "no live ts")
            )
            out.append(
                {
                    "kind": "fill",
                    "ticker": t,
                    "at": None,
                    "live": limit,
                    "sim": f"{price} ({dev:+.3%})",
                    "cls": cls,
                    "note": note,
                }
            )
    return out


def main() -> None:
    require_backtest_host()
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", required=True)
    ap.add_argument("--dart", type=Path, default=DART)
    ap.add_argument("--live-log", type=Path, default=LIVE_LOG)
    args = ap.parse_args()
    _export_quant_airflow_env()
    db = QuantAirflowDBClient()
    stocks = db.get_stock_master()
    ok_all = True
    log_rows = []
    for line in args.live_log.open():
        try:
            log_rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    for d in args.days.split(","):
        day = dt.date.fromisoformat(d)
        events = load_events(EVENTS / f"{day}.jsonl")
        codes = sorted({e.ticker for e in events})
        bars = load_session_bars(codes, day)
        prev = prev_closes(codes, day)
        broker = SimBroker(bars, prev, lambda: dt.datetime.now(KST))
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
        exact = same and not bad_devs
        live_dec = live_decisions(log_rows, day, DUPLICATE_DAEMON_ORDERS.get(day, set()))
        pairs = match_decisions(res.decisions, live_dec, within=dt.timedelta(seconds=5))
        divs = divergences(pairs, live, res.ledger, res.fills, bars, prev)
        n_dec = sum(1 for x in divs if x["kind"] == "decision")
        n_other = sum(1 for x in divs if x["cls"] != INTRA)
        ok = n_other == 0 and not res.errors
        ok_all &= ok
        print(
            f"== {day} parity {'PASS' if ok else 'FAIL'}  ({len(events)} events, "
            f"{sum(map(len, bars.values()))} bars)"
        )
        print(
            f"  decision divergences {n_dec}/{len(res.decisions)} judged pairs; "
            f"ticker-level {len(divs) - n_dec}; other={n_other}; errors={len(res.errors)}; "
            f"exact comparison {'MATCH' if exact else 'DIFF'}"
        )
        for x in divs:
            at = x["at"].strftime("%H:%M:%S") if x["at"] else "--------"
            print(
                f"  DIV {x['kind']:8} {x['ticker']} {at} live={x['live']} sim={x['sim']} "
                f"[{x['cls']}] {x['note']}"
            )
        print("  -- exact comparison --")
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
                    "decisions": res.decisions,
                    "divergences": divs,
                },
                default=str,
                ensure_ascii=False,
                indent=1,
            )
        )
    raise SystemExit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
