"""Parity gate: replay live days and diff against the live ledger.

    uv run python scripts/eval/replay_parity.py --days 2026-09-14,2026-09-15   # simnode

Exact comparison (printed): same tickers and quantities for entries, add-ons
and exits; buy fill price within ±0.5% per ticker.

Verdict (controller rulings R9-R11 and task review I1-I3): every judged pair
whose outcome differs between the live daemon log and the replay, every live
judgment no replay decision matched, and every ticker-level difference
(add-on, exit quantity, entry, fill > ±0.5%) is listed with evidence and a
causal class (see `divergences`): intra_minute_price, data_vintage, knock_on,
or other. PASS = no `other` and no errors logged during the replay.
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
from daytrade_it.application.trading.auto_trader import TradingLimits, add_on_hit
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
#: Ruling R11: live's implied prev close differs from today's kr_quant close by
#: more than this, so the no-chase filter saw a different close. Not an engine
#: difference; the replay keeps today's closes.
VINTAGE = "data_vintage"
VINTAGE_TOL = Decimal("0.001")
#: An entry/skip difference caused by an earlier passing divergence taking or
#: freeing a concurrent-position slot.
KNOCK_ON = "knock_on"
PASSING_CLASSES = {INTRA, VINTAGE, KNOCK_ON}
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

    extended / persisted rows attach to that ticker's
    latest judgment of the last 10 s still awaiting a filter result; ENTRY SUBMITTED /
    entry skipped rows to the earliest judgment of that ticker still holding an
    unentered signal. A second judgment of the same (ticker, title) within 60 s
    is the duplicate daemon's: it takes a filter row but makes no decision row.
    Orders in `exclude_orders` (the duplicate daemon's) are ignored.
    """
    out: list[dict] = []
    awaiting: dict[str, list[tuple[dt.datetime, dict | None]]] = {}

    def take(t: str, ts: dt.datetime) -> dict | None:
        q = [x for x in awaiting.get(t, []) if ts - x[0] <= dt.timedelta(seconds=10)]
        awaiting[t] = q[:-1]
        return q[-1][1] if q else None

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
            dup = any(
                d["ticker"] == t
                and d.get("title") == r.get("title")
                and ts - d["at"] <= dt.timedelta(seconds=60)
                for d in out
            )
            d = (
                None
                if dup
                else {"ticker": t, "at": ts, "outcome": "no_signal", "title": r.get("title")}
            )
            if d is not None:
                out.append(d)
            awaiting.setdefault(t, []).append((ts, d))
        elif ev == "entry_filter: extended, no entry":
            if (d := take(t, ts)) is not None:
                d.update(outcome="extended", pre_entry_move=r.get("pre_entry_move"))
        elif ev == "news_signal_poller: ACTIONABLE SIGNAL PERSISTED":
            if (d := take(t, ts)) is not None:
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


def match_summary(
    pairs: list[tuple[dict, dict | None]], live: list[dict]
) -> tuple[list[dict], dict[str, int]]:
    """Live judgments no replay decision matched, and live/replay/matched/union counts."""
    used = {id(lv) for _, lv in pairs if lv is not None}
    unused = [d for d in live if id(d) not in used]
    matched = len(used)
    counts = {
        "live": len(live),
        "replay": len(pairs),
        "matched": matched,
        "union": len(pairs) + len(unused),
    }
    return unused, counts


def _round_tick(price: Decimal) -> Decimal:
    tick = tick_size(price)
    return (price / tick).quantize(Decimal(1), rounding=ROUND_HALF_UP) * tick


def reproducible_from_open(live_price: Decimal, bar_open: Decimal) -> bool:
    """Within half a tick: live may quote the bid/ask mid around the same ask."""
    return abs(live_price - bar_open) <= tick_size(bar_open) / 2


def _outcome_key(d: dict) -> str:
    return f"skipped:{d.get('reason')}" if d["outcome"] == "skipped" else d["outcome"]


def _sim_quote(bars: dict[str, list[MinuteBar]], t: str, at: dt.datetime) -> Decimal | None:
    """The price SimBroker quotes at `at`: this minute's open, else the last close."""
    minute = at.astimezone(KST).replace(tzinfo=None, second=0, microsecond=0)
    done = None
    for b in bars.get(t, []):
        if b.ts == minute:
            return b.open
        if b.ts < minute:
            done = b.close
    return done


def _on_grid(price: Decimal) -> bool:
    return price == _round_tick(price)


def live_price_and_prev(
    move: Decimal, near: Decimal, kr_prev: Decimal
) -> tuple[Decimal, Decimal] | None:
    """(live price, live prev close) consistent with a logged pre_entry_move.

    Live used last price or bid/ask mid (half ticks allowed) against the prev
    close kr_quant held that morning, which must be a valid tick price. Scans
    half ticks within ±15% of `near`; prefers kr_quant's prev close if it fits.
    """
    hits = []
    p = _round_tick(near * Decimal("0.85"))
    hi = near * Decimal("1.15")
    while p <= hi:
        prev = p / (1 + move)
        r = prev.quantize(Decimal(1), rounding=ROUND_HALF_UP)
        if abs(prev - r) < Decimal("0.001") and _on_grid(r):
            hits.append((p, r))
        p += tick_size(p) / 2
    if not hits:
        return None
    return min(hits, key=lambda h: (h[1] != kr_prev, abs(h[0] - near)))


def _price_class(
    bars: dict[str, list[MinuteBar]], t: str, price: Decimal, at: dt.datetime, src: str
) -> tuple[str, str]:
    o = _sim_quote(bars, t, at)
    if o is None:
        return (
            INTRA,
            f"live {src} {price:.1f} @{at.strftime('%H:%M:%S')}: no bar yet (call auction/VI)",
        )
    note = f"live {src} {price:.1f} @{at.strftime('%H:%M:%S')} vs sim quote {o}"
    return (INTRA if not reproducible_from_open(price, o) else "other"), note


def _filter_side(d: dict) -> str:
    """What the no-chase filter decided for a pair: extended, pass or the outcome."""
    o = d["outcome"]
    if o in ("entered", "skipped", "signal", "pending", "enter_failed"):
        return "pass"
    return o


def _slot_changed(sim: dict, live: dict) -> bool:
    return (sim["outcome"] == "entered") != (live["outcome"] == "entered")


CAP_REASONS = {"concurrent position cap reached", "daily entry cap reached"}


def _cap_skipped_side(sim: dict, live: dict) -> str | None:
    """ "sim" / "live" when one side entered and the other was skipped by a cap."""
    for skipped, entered, name in ((sim, live, "sim"), (live, sim, "live")):
        if (
            entered["outcome"] == "entered"
            and skipped["outcome"] == "skipped"
            and skipped.get("reason") in CAP_REASONS
        ):
            return name
    return None


def _exited_by(ledger: list[dict], ticker: str, at: dt.datetime) -> bool:
    return any(
        r["event"] == "exit_submitted"
        and r.get("ticker") == ticker
        and r.get("ts")
        and _kst(r["ts"]) <= at
        for r in ledger
    )


def _filter_class(
    bars: dict[str, list[MinuteBar]],
    t: str,
    sim: dict,
    live: dict,
    prev_close: dict[str, Decimal],
    max_move: Decimal,
) -> tuple[str, str]:
    """extended-vs-pass pair: which single change (price or prev close) crosses the cap."""
    if not live.get("pre_entry_move") or t not in prev_close:
        return "other", "no live pre_entry_move or kr_quant prev close"
    o = _sim_quote(bars, t, sim["decide_at"])
    if o is None:
        return "other", "replay filtered without a quote"
    kr = Decimal(prev_close[t])
    found = live_price_and_prev(Decimal(live["pre_entry_move"]), o, kr)
    if found is None:
        return "other", "pre_entry_move has no tick-grid solution"
    price, lp = found

    def ext(p: Decimal, c: Decimal) -> bool:
        return p / c - 1 > max_move

    note = (
        f"sim quote {o} {o / kr - 1:+.2%} vs live price {price} {price / kr - 1:+.2%} "
        f"over kr_quant prev {kr.normalize():f}; live prev {lp} gives {price / lp - 1:+.2%}; "
        f"cap {max_move:.2%}"
    )
    if ext(o, kr) != (_filter_side(sim) == "extended") or ext(price, lp) != (
        _filter_side(live) == "extended"
    ):
        return "other", note + " (recomputed filter disagrees with a logged outcome)"
    if ext(o, kr) != ext(price, kr):
        return INTRA, note
    if ext(price, kr) != ext(price, lp) and abs(lp / kr - 1) > VINTAGE_TOL:
        return VINTAGE, note
    return "other", note + " (no single price or prev-close change crosses the cap)"


def _decision_row(t: str, at: dt.datetime, live: str, sim: str, cls: str, note: str) -> dict:
    return {
        "kind": "decision",
        "ticker": t,
        "at": at,
        "live": live,
        "sim": sim,
        "cls": cls,
        "note": note,
    }


def divergences(
    pairs: list[tuple[dict, dict | None]],
    live_ledger: list[dict],
    sim_ledger: list[dict],
    sim_fills: list[dict],
    bars: dict[str, list[MinuteBar]],
    prev_close: dict[str, Decimal],
    *,
    max_move: Decimal = Decimal("0.03"),
    limits: TradingLimits | None = None,
    unmatched_live: list[dict] | tuple = (),
) -> list[dict]:
    """Decision- and ticker-level live/replay differences, each classified causally.

    - no replay quote yet (call auction / VI) while live filtered: intra_minute_price
    - extended vs pass: intra_minute_price if the sim quote and the live price
      fall on opposite sides of `max_move` over the same prev close;
      data_vintage if only swapping kr_quant's prev close for live's crosses it
    - entered vs cap-skipped (both passed the filter): knock_on only if a
      strictly earlier intra_minute_price / data_vintage divergence entered a
      ticker on the cap-skipped side alone and that position was still open
    - add_on: intra_minute_price if add_on_hit at the add-on time flips between
      the live and replay entry limits; exit_qty inherits the entry / add-on
      cause only if the exit gap equals the bought (entry + add-on) gap
    - anything else, including live judgments with no replay decision: other
    """
    limits = limits or TradingLimits()
    out: list[dict] = []
    items = [(s["decide_at"], s, lv) for s, lv in pairs]
    items += [(lv["at"], None, lv) for lv in unmatched_live]
    items.sort(key=lambda x: x[0])
    #: (time, ticker, class, side that entered it) of passing slot-changing roots
    roots: list[tuple[dt.datetime, str, str, str]] = []
    decision_cls: dict[str, str] = {}
    for at, sim, live in items:
        if sim is None:
            out.append(
                _decision_row(
                    live["ticker"],
                    at,
                    _outcome_key(live),
                    "no replay decision",
                    "other",
                    "live judgment with no replay decision within 5 s",
                )
            )
            continue
        t = sim["ticker"]
        if live is None:
            out.append(
                _decision_row(
                    t, at, "no live judgment within window", _outcome_key(sim), "other", ""
                )
            )
            continue
        if _outcome_key(sim) == _outcome_key(live):
            continue
        fl, fs = _filter_side(live), _filter_side(sim)
        if fs == "no_signal" and fl in ("extended", "pass"):
            if _sim_quote(bars, t, sim["decide_at"]) is None:
                cls, note = (
                    INTRA,
                    f"no bar yet at {sim['decide_at'].strftime('%H:%M:%S')} (call auction/VI)",
                )
            else:
                cls, note = "other", "replay had a quote but no signal"
        elif {fl, fs} == {"extended", "pass"}:
            cls, note = _filter_class(bars, t, sim, live, prev_close, max_move)
        elif fl == fs == "pass":
            side = _cap_skipped_side(sim, live)
            ledger = sim_ledger if side == "sim" else live_ledger
            root = (
                None
                if side is None
                else next(
                    (
                        r
                        for r in reversed(roots)
                        if r[0] < at
                        and r[1] != t
                        and r[3] == side
                        and not _exited_by(ledger, r[1], at)
                    ),
                    None,
                )
            )
            if side is None:
                cls, note = "other", "pass-vs-pass mismatch that is not entered vs cap-skipped"
            elif root is None:
                cls, note = (
                    "other",
                    f"{side} cap-skipped but no earlier intra/vintage root entered on {side} "
                    "and still open",
                )
            else:
                cls = KNOCK_ON
                note = (
                    f"{side} cap-skipped while holding {root[1]} ({root[2]}, entered only on "
                    f"{side} at {root[0].strftime('%H:%M:%S')}, not exited)"
                )
        else:
            cls, note = "other", f"filter outcome live {fl} vs replay {fs}"
        out.append(_decision_row(t, at, _outcome_key(live), _outcome_key(sim), cls, note))
        if cls in PASSING_CLASSES:
            decision_cls.setdefault(t, cls)
            if cls in (INTRA, VINTAGE) and _slot_changed(sim, live):
                roots.append((at, t, cls, "sim" if sim["outcome"] == "entered" else "live"))
        else:
            decision_cls[t] = "other"

    def first(ledger: list[dict], event: str, t: str) -> dict | None:
        return next((r for r in ledger if r["event"] == event and r.get("ticker") == t), None)

    s_live, s_sim = summarize(live_ledger), summarize(sim_ledger)
    diff = {
        k: {t for t in set(s_live[k]) | set(s_sim[k]) if s_live[k].get(t, 0) != s_sim[k].get(t, 0)}
        for k in ("entry", "add_on", "exit")
    }

    def row(kind: str, t: str, key: str, cls: str, note: str) -> None:
        out.append(
            {
                "kind": kind,
                "ticker": t,
                "at": None,
                "live": s_live[key].get(t, 0),
                "sim": s_sim[key].get(t, 0),
                "cls": cls,
                "note": note,
            }
        )

    add_on_cls: dict[str, str] = {}
    for t in sorted(diff["entry"]):
        if t not in decision_cls:
            row("entry", t, "entry", "other", "entry differs without a decision divergence")
    for t in sorted(diff["add_on"]):
        if t in diff["entry"]:
            cls = decision_cls.get(t, "other")
            add_on_cls[t] = cls
            row("add_on", t, "add_on", cls, "follows the entry decision divergence on this ticker")
            continue
        le, se = first(live_ledger, "entry_submitted", t), first(sim_ledger, "entry_submitted", t)
        ev = first(sim_ledger, "add_on_submitted", t) or first(live_ledger, "add_on_submitted", t)
        if limits.add_on_min_gain is None or not (le and se and ev and ev.get("ts")):
            add_on_cls[t] = "other"
            row("add_on", t, "add_on", "other", "add-on disabled or entry/add-on rows missing")
            continue
        now = _kst(ev["ts"])
        hit_live = add_on_hit(bars.get(t, []), Decimal(le["limit"]), now, limits.add_on_min_gain)
        hit_sim = add_on_hit(bars.get(t, []), Decimal(se["limit"]), now, limits.add_on_min_gain)
        cls = INTRA if hit_live != hit_sim else "other"
        add_on_cls[t] = cls
        row(
            "add_on",
            t,
            "add_on",
            cls,
            f"add_on_hit at {now.strftime('%H:%M:%S')}: live entry {le['limit']} -> {hit_live}, "
            f"replay entry {se['limit']} -> {hit_sim} (min gain {limits.add_on_min_gain:.0%})",
        )
    for t in sorted(diff["exit"]):
        held = {
            side: summ["entry"].get(t, 0) + summ["add_on"].get(t, 0)
            for side, summ in (("live", s_live), ("sim", s_sim))
        }
        exit_gap = s_sim["exit"].get(t, 0) - s_live["exit"].get(t, 0)
        if exit_gap != held["sim"] - held["live"]:
            row(
                "exit_qty",
                t,
                "exit",
                "other",
                f"exit gap {exit_gap:+d} != bought gap {held['sim'] - held['live']:+d}",
            )
            continue
        causes = []
        if t in diff["entry"]:
            causes.append(("entry decision", decision_cls.get(t, "other")))
        if t in diff["add_on"]:
            causes.append(("add-on", add_on_cls.get(t, "other")))
        if not causes:
            row(
                "exit_qty",
                t,
                "exit",
                "other",
                "exit quantity differs without entry/add-on difference",
            )
            continue
        ok = all(c in PASSING_CLASSES for _, c in causes)
        cls = causes[0][1] if ok else "other"
        row("exit_qty", t, "exit", cls, "follows " + ", ".join(f"{n} ({c})" for n, c in causes))
    for kind, t, limit, price, dev in fill_deviations(live_ledger, sim_fills):
        if abs(dev) > MAX_FILL_DEV:
            r = next(x for x in live_ledger if x.get("ticker") == t and x.get("limit") == limit)
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
        unused, counts = match_summary(pairs, live_dec)
        cfg = LIVE_CONFIGS[day]
        divs = divergences(
            pairs,
            live,
            res.ledger,
            res.fills,
            bars,
            prev,
            max_move=cfg.max_pre_entry_move,
            limits=cfg.limits,
            unmatched_live=unused,
        )
        n_dec = sum(1 for x in divs if x["kind"] == "decision")
        n_other = sum(1 for x in divs if x["cls"] not in PASSING_CLASSES)
        by_cls = {
            c: sum(1 for x in divs if x["cls"] == c) for c in (INTRA, VINTAGE, KNOCK_ON, "other")
        }
        ok = n_other == 0 and not res.errors
        ok_all &= ok
        print(
            f"== {day} parity {'PASS' if ok else 'FAIL'}  ({len(events)} events, "
            f"{sum(map(len, bars.values()))} bars)"
        )
        print(
            f"  live judged {counts['live']} / replay decisions {counts['replay']} / "
            f"matched {counts['matched']} (union {counts['union']})"
        )
        print(
            f"  decision divergences {n_dec}/{counts['union']} pairs "
            f"({n_dec / max(counts['union'], 1):.1%}); "
            f"ticker-level {len(divs) - n_dec}; by class {by_cls}; errors={len(res.errors)}; "
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
