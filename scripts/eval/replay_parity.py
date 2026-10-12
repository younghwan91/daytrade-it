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

Final-review checks (I5): every sim buy fill must equal SimBroker's own rule
price at its timestamp (`fill_rule`), and for tickers exited on both sides
the first exit's submission time must agree within one manage tick (30 s,
`exit_time`) and the sim sell fill within ±0.5% of the live exit limit
(`sell_fill`).

Previous closes (I2) come from `data/eval/prev_close/YYYY-MM-DD.json`: the
prior session's last minute bar with ts <= 20:00 (`close`, incl. KRX after-market
16:00-20:00 trades, 2026-09-14~) and the KRX regular-session close < 15:33
(`krx_close`), plus each code's last daily_bars_adjusted row (`daily`), frozen
once. SimBroker's price-limit base is `krx_close` (R18). The no-chase filter (R17)
gets the inputs live read that day: `close` before `entry_filter.REGULAR_CLOSE_FROM`
(2026-10-13); from then on `krx_close` and `daily`, decided by the same
`entry_filter.choose_prev_close` live runs (daily fallback where the regular bar is
missing or older than the daily row). Tickers whose frozen basis is > 0.1% off an
independent reference are printed: daily_bars_adjusted before the switch,
supply_demand's implied price-limit base from it (skipped while that is empty).
`data_vintage` (R11) is judged against the frozen basis (`no_chase_prices`). The live
log must agree on the basis (`check_no_chase_basis`: startup `no_chase_basis`, and
`prev_close_source` on extended / persisted rows).

The day's live configuration comes from the live ledger's `baseline` row when
it carries one (`limits`, `cash`, `git_sha`; AutoTrader writes them from
2026-10-05): the replay runs `TradingLimits` exactly as live did and SimBroker
starts with the deposit live saw. Older days fall back to the hand-kept
`LIVE_CONFIGS` table; a day with neither fails loudly.

Runs on simnode only (`require_backtest_host`, user directive 2026-09-13).
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

from daytrade_it.application.replay.day_replay import (
    MAX_MISSING_BARS,
    DayConfig,
    ReplayDataError,
    no_chase_cap,
    replay_day,
)
from daytrade_it.application.replay.events import load_events
from daytrade_it.application.replay.prev_close import (
    FrozenPrevCloseDB,
    FrozenPrevCloses,
    dump_prev_closes,
    load_prev_closes,
)
from daytrade_it.application.signals.entry_filter import LOOKBACK_DAYS, REGULAR_CLOSE_FROM
from daytrade_it.application.trading.auto_trader import (
    TradingLimits,
    add_on_hit,
    limits_from_strings,
    vwap_exit_hit,
)
from daytrade_it.domain.services.krx_session import REGULAR_END, REGULAR_OPEN
from daytrade_it.domain.services.krx_session import last_regular_closes as _last_regular_closes
from daytrade_it.infrastructure.brokers.base import KST, MinuteBar
from daytrade_it.infrastructure.brokers.sim_broker import SimBroker
from daytrade_it.infrastructure.external.quant_airflow_db_client import QuantAirflowDBClient
from daytrade_it.shared.runtime_host import require_backtest_host

EVENTS = Path("data/eval/replay_events")
LIVE_LEDGER = Path("data/auto_trader")
#: scalp-it's dart.db; simnode's copy may lag trader's (pass --dart).
DART = Path("../scalp-it/data/dart.db")
WORK = Path("data/eval/replay_work")
PREV_CLOSE = Path("data/eval/prev_close")
#: Regular session incl. the closing auction print (REGULAR_OPEN <= ts < REGULAR_END,
#: 15:30 or 15:32 when a VI extends the auction) lives in `domain.services.krx_session`,
#: shared with the live no-chase filter and the research gate `scripts/eval/krx_session`.
#: Before REGULAR_CLOSE_FROM live no-chase read kr_quant's daily close, which from
#: 2026-09-14 includes KRX after-market trades (16:00-20:00) = last bar by 20:00.
AFTERMARKET_END = dt.time(20, 0)
#: No trading before 08:00 KST: an earlier bar means a timezone mix-up.
EARLIEST_BAR = dt.time(8, 0)
PREV_CLOSE_LOOKBACK = dt.timedelta(days=LOOKBACK_DAYS)
MANAGE_TICK = dt.timedelta(seconds=30)
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

#: Fallback for ledgers written before the baseline row carried `limits`
#: (2026-10-05). Days absent here and without baseline limits fail the day.
LIVE_CONFIGS = {
    # Code before 4ebd39d (cap 2), c273b9b (+2% no-chase), d357b88 (surprise-ranked
    # batch, VWAP exit) and b57e6d8 (add-on): each BUY entered as judged. Live
    # ledger: entries followed their judgment by about two seconds.
    dt.date(2026, 9, 14): DayConfig(
        limits=TradingLimits(max_open_positions=2, vwap_exit_gap=None, add_on_min_gain=None),
        max_pre_move=Decimal("0.02"),
        enter_on_decide=True,
    ),
    # 48d6cb1 / eb1c8b3 defaults: cap 5, +3% no-chase, VWAP exit, add-on at +1%
    # (pyramiding was on until 2026-09-23; main's default is now None).
    dt.date(2026, 9, 15): DayConfig(limits=TradingLimits(add_on_min_gain=Decimal("0.01"))),
    dt.date(2026, 9, 16): DayConfig(limits=TradingLimits(add_on_min_gain=Decimal("0.01"))),
}


def baseline_row(ledger: list[dict]) -> dict | None:
    """The day's `baseline` row (AutoTrader writes one before its first order)."""
    return next((r for r in ledger if r.get("event") == "baseline"), None)


def live_day_config(day: dt.date, ledger: list[dict]) -> tuple[DayConfig, Decimal | None]:
    """(config the live trader ran with, deposit cash it saw) for `day`.

    Limits and cash come from the ledger baseline when it carries them; the
    poller-version switches (`enter_on_decide`, the no-chase cap) are not in
    the ledger and keep coming from `LIVE_CONFIGS` when the day has an entry.
    Without baseline limits the table alone decides; with neither the day
    cannot be replayed faithfully and fails.
    """
    table = LIVE_CONFIGS.get(day)
    base = baseline_row(ledger)
    if base is not None and base.get("limits"):
        limits = limits_from_strings(base["limits"])
        cash = Decimal(str(base["cash"])) if base.get("cash") is not None else None
        cfg = DayConfig(
            limits=limits,
            max_pre_move=table.max_pre_move if table else None,
            enter_on_decide=table.enter_on_decide if table else False,
        )
        return cfg, cash
    if table is not None:
        return table, None
    raise ReplayDataError(
        f"{day}: live ledger baseline carries no limits and LIVE_CONFIGS has no entry"
    )


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


def to_kst_naive(ts: dt.datetime) -> dt.datetime:
    """KST wall time. kr_quant `minute_bars.ts` is `timestamp without time zone`
    holding KST wall time (first bar 09:00); an aware value is converted."""
    return ts if ts.tzinfo is None else ts.astimezone(KST).replace(tzinfo=None)


def check_session_hours(bars: dict[str, list[MinuteBar]]) -> None:
    """Fail loudly if any bar starts before 08:00 KST (timezone mix-up)."""
    bad = sorted(c for c, bs in bars.items() if any(b.ts.time() < EARLIEST_BAR for b in bs))
    if bad:
        raise ReplayDataError(f"minute bars before {EARLIEST_BAR} KST (timezone?): {bad}")


def load_session_bars(codes: list[str], day: dt.date) -> dict[str, list[MinuteBar]]:
    out: dict[str, list[MinuteBar]] = {c: [] for c in codes}
    with _conn() as c, c.cursor() as cur:
        cur.execute("SET max_parallel_workers_per_gather = 0")
        # Day bounds are KST wall time; an aware column would be compared in UTC.
        cur.execute(
            "SELECT data_type FROM information_schema.columns "
            "WHERE table_name = 'minute_bars' AND column_name = 'ts'"
        )
        (kind,) = cur.fetchone()
        if kind != "timestamp without time zone":
            raise ReplayDataError(f"minute_bars.ts is {kind}; day bounds assume KST wall time")
        cur.execute(
            "SELECT code, ts, open, high, low, close, volume FROM minute_bars "
            "WHERE code = ANY(%s) AND ts >= %s AND ts < %s ORDER BY code, ts",
            (codes, day, day + dt.timedelta(days=1)),
        )
        for code, ts, o, h, lo, cl, v in cur.fetchall():
            out[code].append(
                MinuteBar(
                    ts=to_kst_naive(ts),
                    open=Decimal(str(o)),
                    high=Decimal(str(h)),
                    low=Decimal(str(lo)),
                    close=Decimal(str(cl)),
                    volume=int(v),
                )
            )
    check_session_hours(out)
    return out


def daily_adjusted_last(codes: list[str], day: dt.date) -> dict[str, tuple[Decimal, dt.date]]:
    """Each code's last daily_bars_adjusted (close, date) before `day`, as kr_quant holds it now.

    Frozen as `FrozenPrevCloses.daily` (live's no-chase daily input from REGULAR_CLOSE_FROM)
    and, before the switch, the vintage_flags reference. Rows can be revised after the day.
    """
    with _conn() as c, c.cursor() as cur:
        cur.execute(
            "SELECT DISTINCT ON (code) code, close, date FROM daily_bars_adjusted "
            "WHERE code = ANY(%s) AND date < %s AND close IS NOT NULL ORDER BY code, date DESC",
            (codes, day),
        )
        return {code: (Decimal(str(close)), d) for code, close, d in cur.fetchall()}


def regular_base_prices(codes: list[str], day: dt.date) -> dict[str, Decimal]:
    """`day`'s price-limit base implied by supply_demand: close / (1 + flu_rt / 1e4).

    flu_rt is % x 100 and both close and flu_rt include the after-market, so the ratio is the
    base = the prior regular-session close (005930 09-18: 260,000 / 1.0297 = 252,500), up to
    flu_rt's 0.01% rounding. A corporate action that resets the base shows up as a gap.
    Reference for vintage_flags only.
    """
    with _conn() as c, c.cursor() as cur:
        cur.execute(
            "SELECT code, close, flu_rt FROM supply_demand WHERE code = ANY(%s) AND date = %s",
            (codes, day),
        )
        return {
            code: (Decimal(close) / (1 + Decimal(str(flu)) / Decimal(10_000))).quantize(
                Decimal(1), rounding=ROUND_HALF_UP
            )
            for code, close, flu in cur.fetchall()
            if close and flu is not None
        }


def prior_regular_bars(codes: list[str], day: dt.date) -> list[tuple[str, dt.datetime, float]]:
    """(code, ts, close) of each code's last regular bar and last bar by 20:00 in the
    lookback before `day`."""
    out = []
    with _conn() as c, c.cursor() as cur:
        cur.execute("SET max_parallel_workers_per_gather = 0")
        for lo, hi, op in (
            (REGULAR_OPEN, REGULAR_END, "<"),
            (dt.time(0, 0), AFTERMARKET_END, "<="),
        ):
            cur.execute(
                "SELECT DISTINCT ON (code) code, ts, close FROM minute_bars "
                "WHERE code = ANY(%s) AND ts >= %s AND ts < %s "
                f"AND ts::time >= %s AND ts::time {op} %s ORDER BY code, ts DESC",
                (codes, day - PREV_CLOSE_LOOKBACK, day, lo, hi),
            )
            out += [(code, to_kst_naive(ts), close) for code, ts, close in cur.fetchall()]
    return out


def last_regular_closes(
    rows: list[tuple[str, dt.datetime, float]], day: dt.date
) -> dict[str, tuple[Decimal, dt.date]]:
    """Each code's close at its latest regular-session bar strictly before `day`.

    R18 price-limit base, and R17 no-chase basis from REGULAR_CLOSE_FROM.
    `domain.services.krx_session.last_regular_closes` is the one implementation; the live
    filter's SQL and research scripts use the same bounds.
    """
    return {c: (Decimal(str(cl)), d) for c, (cl, d) in _last_regular_closes(rows, day).items()}


def last_aftermarket_closes(
    rows: list[tuple[str, dt.datetime, float]], day: dt.date
) -> dict[str, tuple[Decimal, dt.date]]:
    """Each code's close at its latest bar with ts <= 20:00 strictly before `day` (KRX
    after-market incl.) -- what kr_quant's daily close holds from 2026-09-14."""
    best: dict[str, tuple[dt.datetime, float]] = {}
    for code, ts, close in rows:
        if ts.date() >= day or ts.time() > AFTERMARKET_END:
            continue
        if code not in best or ts > best[code][0]:
            best[code] = (ts, close)
    return {c: (Decimal(str(cl)), ts.date()) for c, (ts, cl) in best.items()}


def vintage_flags(
    closes: dict[str, tuple[Decimal, dt.date]],
    reference: dict[str, Decimal],
    ref_name: str = "daily_adjusted",
) -> list[dict]:
    """Frozen closes more than VINTAGE_TOL off `reference` (or absent there).

    Reference: today's daily_bars_adjusted for the daily-close basis (before
    REGULAR_CLOSE_FROM), `regular_base_prices` ("sd_base") for the regular-close basis --
    daily_bars_adjusted carries the after-market close and would flag most tickers.
    """
    out = []
    for code in sorted(closes):
        frozen, date = closes[code]
        d = reference.get(code)
        dev = None if d is None else frozen / d - 1
        if dev is None or abs(dev) > VINTAGE_TOL:
            out.append(
                {
                    "code": code,
                    "frozen": str(frozen),
                    "date": date.isoformat(),
                    ref_name: str(d),
                    "dev": None if dev is None else f"{dev:+.4%}",
                }
            )
    return out


def day_codes(events: list) -> list[str]:
    """A day's replay tickers: the events' tickers, sorted and unique."""
    return sorted({e.ticker for e in events})


def frozen_prev_closes(
    codes: list[str],
    day: dt.date,
    out_dir: Path = PREV_CLOSE,
    fetch_bars=prior_regular_bars,  # noqa: ANN001
    fetch_daily=daily_adjusted_last,  # noqa: ANN001
    fetch_base=regular_base_prices,  # noqa: ANN001
) -> FrozenPrevCloses:
    """Load `out_dir/<day>.json`, or build it once from minute bars and freeze it.

    A file frozen with extra codes (the research branch added 069500 for its
    bandit) is fine; one missing an event code is refused, never patched. So is a
    regular-close day (>= REGULAR_CLOSE_FROM) frozen without `daily`: the replay would
    feed the no-chase filter a different daily input than live read.

    Flags need a reference; an empty one (supply_demand not loaded yet when frozen)
    skips them rather than freezing every code as flagged for good.
    """
    path = out_dir / f"{day.isoformat()}.json"
    if path.exists():
        frozen = load_prev_closes(path)
        uncovered = sorted(set(codes) - set(frozen.closes) - set(frozen.missing))
        if uncovered:
            raise ValueError(f"{path} was frozen without event codes {uncovered}")
        if frozen.regular_no_chase and frozen.daily is None:
            raise ReplayDataError(
                f"{path} was frozen without `daily` (needed from {REGULAR_CLOSE_FROM})"
            )
        return frozen
    rows = fetch_bars(codes, day)
    closes = last_aftermarket_closes(rows, day)
    krx = last_regular_closes(rows, day)
    daily = fetch_daily(codes, day)
    if day >= REGULAR_CLOSE_FROM:
        ref, name, basis = fetch_base(codes, day), "sd_base", krx
    else:
        ref, name, basis = {c: p for c, (p, _) in daily.items()}, "daily_adjusted", closes
    if ref:
        flags = vintage_flags(basis, ref, name)
    else:
        flags = []
        print(f"  prev close flags skipped for {day}: no {name} reference rows yet")
    frozen = FrozenPrevCloses(
        day=day,
        closes=closes,
        krx=krx,
        flags=flags,
        missing=sorted(set(codes) - set(closes)),
        daily=daily,
    )
    dump_prev_closes(path, frozen)
    return frozen


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


#: Live log rows that carry the no-chase basis: the filter's `extended` row and, from
#: REGULAR_CLOSE_FROM, the persisted-signal row (`prev_close_source`), and the daemon's
#: startup line (`no_chase_basis`).
EXTENDED_EVENT = "entry_filter: extended, no entry"
PERSISTED_EVENT = "news_signal_poller: ACTIONABLE SIGNAL PERSISTED"
BASIS_EVENT = "news_signal_poller: no-chase basis"


def check_no_chase_basis(rows: list[dict], day: dt.date) -> None:
    """Fail the day if live's no-chase log contradicts the basis the replay serves.

    From REGULAR_CLOSE_FROM the live daemon logs `no_chase_basis=regular` at startup and
    `prev_close_source` on every `extended` and `PERSISTED` row; before it none of these
    existed. A mismatch means the switch was deployed on another day than
    REGULAR_CLOSE_FROM says, and the replay would measure the move against a different
    prior close than live did. Checked even on a day with no `extended` row.
    """
    seen = set()
    for r in rows:
        if "timestamp" not in r or _kst(r["timestamp"]).date() != day:
            continue
        ev = r.get("event")
        if ev in (EXTENDED_EVENT, PERSISTED_EVENT):
            seen.add("prev_close_source" in r)
        elif ev == BASIS_EVENT:
            seen.add(r.get("no_chase_basis") == "regular")
    want = day >= REGULAR_CLOSE_FROM
    if seen and seen != {want}:
        raise ReplayDataError(
            f"live no-chase rows on {day} {'lack' if want else 'carry'} the regular-close "
            f"basis (prev_close_source / no_chase_basis=regular): the switch did not deploy "
            f"on REGULAR_CLOSE_FROM={REGULAR_CLOSE_FROM}"
        )


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
        elif ev == EXTENDED_EVENT:
            if (d := take(t, ts)) is not None:
                d.update(outcome="extended", pre_entry_move=r.get("pre_entry_move"))
        elif ev == PERSISTED_EVENT:
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
    if o in ("entered", "skipped", "deferred", "signal", "pending", "enter_failed"):
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
        f"over frozen prev {kr.normalize():f}; live prev {lp} gives {price / lp - 1:+.2%}; "
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


def _rule_price(bars: dict[str, list[MinuteBar]], f: dict) -> Decimal | None:
    """SimBroker's buy fill price for `f`: quote at placement, or min(limit, open)
    of the completed bar that ended at the fill ts."""
    ts = f["ts"]
    if not f["resting"]:
        return _sim_quote(bars, f["ticker"], ts)
    bar_start = ts.astimezone(KST).replace(tzinfo=None) - dt.timedelta(minutes=1)
    b = next((x for x in bars.get(f["ticker"], []) if x.ts == bar_start), None)
    if b is None or b.low > Decimal(f["limit"]):
        return None
    return min(Decimal(f["limit"]), b.open)


def fill_rule_divergences(fills: list[dict], bars: dict[str, list[MinuteBar]]) -> list[dict]:
    """Sim buy fills (entries, add-ons) not at SimBroker's own rule price: `other`."""
    out = []
    for f in fills:
        if f["side"] != "buy":
            continue
        want = _rule_price(bars, f)
        if want is not None and Decimal(f["price"]) == want:
            continue
        mode = "resting" if f["resting"] else "marketable"
        out.append(
            {
                "kind": "fill_rule",
                "ticker": f["ticker"],
                "at": f["ts"],
                "live": "-",
                "sim": str(f["price"]),
                "cls": "other",
                "note": f"{mode} buy limit {f['limit']} filled {f['price']}, rule price {want}",
            }
        )
    return out


def _first_exit(ledger: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for r in ledger:
        if r["event"] == "exit_submitted" and int(r.get("attempt", 0)) == 0:
            out.setdefault(r["ticker"], r)
    return out


def _entry_limit(ledger: list[dict], t: str) -> Decimal | None:
    r = next((x for x in ledger if x["event"] == "entry_submitted" and x.get("ticker") == t), None)
    return Decimal(r["limit"]) if r and r.get("limit") else None


def exit_divergences(
    live_ledger: list[dict],
    sim_ledger: list[dict],
    sim_fills: list[dict],
    bars: dict[str, list[MinuteBar]],
    limits: TradingLimits,
) -> list[dict]:
    """Exit submission time (one manage tick) and sell fill (±0.5%) vs live, per ticker
    exited on both sides.

    - exit_time: intra_minute_price only if `vwap_exit_hit` at the earlier exit
      flips between the live and replay entry limits; else other
    - sell_fill: inherits the ticker's exit_time class; else intra_minute_price
      if the live limit is not reproducible from the minute open (live bid off
      the bar), other if it is (the sim fill itself is off)
    """
    out = []
    live_x, sim_x = _first_exit(live_ledger), _first_exit(sim_ledger)
    for t in sorted(set(live_x) & set(sim_x)):
        lx, sx = live_x[t], sim_x[t]
        lt, st = _kst(lx["ts"]), _kst(sx["ts"])
        time_cls = None
        if abs(lt - st) > MANAGE_TICK:
            le, se = _entry_limit(live_ledger, t), _entry_limit(sim_ledger, t)
            at = min(lt, st)
            if limits.vwap_exit_gap is not None and le is not None and se is not None:
                hits = [
                    vwap_exit_hit(
                        bars.get(t, []), e, at, limits.vwap_exit_gap, limits.vwap_exit_min_gain
                    )[0]
                    for e in (le, se)
                ]
                flip = hits[0] != hits[1]
                note = f"vwap_exit_hit at {at:%H:%M:%S}: live entry {le} -> {hits[0]}, replay entry {se} -> {hits[1]}"
            else:
                flip, note = False, "no VWAP exit rule or entry limit missing"
            time_cls = INTRA if flip else "other"
            out.append(
                {
                    "kind": "exit_time",
                    "ticker": t,
                    "at": at,
                    "live": lt.strftime("%H:%M:%S"),
                    "sim": st.strftime("%H:%M:%S"),
                    "cls": time_cls,
                    "note": note,
                }
            )
        sim_rows = {
            r.get("order_id"): r
            for r in sim_ledger
            if r["event"] == "exit_submitted" and r.get("ticker") == t
        }
        live_rows = [
            r for r in live_ledger if r["event"] == "exit_submitted" and r.get("ticker") == t
        ]
        for f in sim_fills:
            if f["side"] != "sell" or f["ticker"] != t:
                continue
            attempt = int(sim_rows.get(f["order_id"], {}).get("attempt", 0))
            lr = next((r for r in live_rows if int(r.get("attempt", 0)) == attempt), live_rows[-1])
            limit = Decimal(lr["limit"])
            dev = float(Decimal(f["price"]) / limit - 1)
            if abs(dev) <= MAX_FILL_DEV:
                continue
            if time_cls is not None:
                cls, note = time_cls, "follows the exit_time divergence"
            else:
                disc = min(limits.exit_discount_step * attempt, limits.exit_discount_cap)
                q = _sim_quote(bars, t, _kst(lr["ts"]))
                if q is None:
                    cls, note = INTRA, f"no bar at live exit {_kst(lr['ts']):%H:%M:%S}"
                else:
                    sim_limit = (q - tick_size(q)) * (1 - disc)
                    live_bid = limit / (1 - disc)
                    ok = abs(limit - sim_limit) <= tick_size(q) / 2
                    cls = "other" if ok else INTRA
                    note = f"live exit limit {limit} (bid {live_bid:.0f}) vs sim bid {q - tick_size(q)} at {_kst(lr['ts']):%H:%M:%S}"
            out.append(
                {
                    "kind": "sell_fill",
                    "ticker": t,
                    "at": None,
                    "live": str(limit),
                    "sim": f"{f['price']} ({dev:+.3%})",
                    "cls": cls,
                    "note": note,
                }
            )
    return out


def check_prev_close_coverage(frozen: FrozenPrevCloses) -> None:
    """Fail the day if > 5% of event tickers have no no-chase basis at all.

    Without one the no-chase filter finds no prior close and silently drops the
    pair, so the replay would trade a different universe than live did. A ticker
    with a bar by 20:00 but no regular bar still has a basis (the daily-close
    fallback, `FrozenPrevCloses.no_chase_fallbacks`), and from REGULAR_CLOSE_FROM so
    does one with no minute bar but a frozen `daily` row -- live falls back to it too.
    """
    n = len(frozen.closes) + len(frozen.missing)
    uncovered = sorted(set(frozen.missing) - set(frozen.no_chase_basis))
    if n and len(uncovered) / n > MAX_MISSING_BARS:
        raise ReplayDataError(
            f"{len(uncovered)}/{n} event tickers have no prior-session bar or daily row "
            f"for a frozen prev close (> {MAX_MISSING_BARS:.0%}): {uncovered}"
        )


def prepare_day(
    day: dt.date, events: list, cash: Decimal | None = None
) -> tuple[dict[str, list[MinuteBar]], FrozenPrevCloses, SimBroker, dt.datetime]:
    """Bars, frozen prev closes (flags printed), a fresh SimBroker and stale_before.

    `cash` is the deposit the live account reported that morning (ledger
    baseline); None keeps SimBroker's default.
    """
    codes = day_codes(events)
    bars = load_session_bars(codes, day)
    frozen = frozen_prev_closes(codes, day)
    print(
        f"  prev close frozen: {len(frozen.closes)} codes (prior session "
        f"{frozen.prev_session}), no close: {frozen.missing}; no-chase basis "
        f"{'regular-session close' if frozen.regular_no_chase else 'daily close'}"
        f"{f', daily fallback: {frozen.no_chase_fallbacks}' if frozen.regular_no_chase else ''}; "
        f"> {VINTAGE_TOL:.1%} off reference: {len(frozen.flags)}"
    )
    for f in frozen.flags:
        print(f"    PREV_CLOSE_FLAG {f}")
    check_prev_close_coverage(frozen)
    broker = SimBroker(
        bars,
        frozen.prices,
        lambda: dt.datetime.now(KST),
        limit_base=frozen.krx_prices,
        **({} if cash is None else {"cash": cash}),
    )
    stale_before = dt.datetime.combine(frozen.prev_session, dt.time(15, 20), tzinfo=KST)
    return bars, frozen, broker, stale_before


def load_live_ledger(day: dt.date, ledger_dir: Path = LIVE_LEDGER) -> list[dict]:
    path = ledger_dir / f"{day}.jsonl"
    return [json.loads(line) for line in path.open() if line.strip()]


def main() -> None:
    require_backtest_host()
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", required=True)
    ap.add_argument("--dart", type=Path, default=DART)
    ap.add_argument("--live-log", type=Path, default=LIVE_LOG)
    ap.add_argument("--work-root", type=Path, default=WORK)
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
        raw_live = load_live_ledger(day)
        print(f"== {day}")
        try:
            cfg, cash = live_day_config(day, raw_live)
            check_no_chase_basis(log_rows, day)
            base = baseline_row(raw_live) or {}
            print(
                f"  live config: {'ledger baseline' if base.get('limits') else 'LIVE_CONFIGS'}; "
                f"git sha {base.get('git_sha')}; cash {cash if cash is not None else 'default'}; "
                f"no-chase cap {no_chase_cap(cfg):.2%}; add_on {cfg.limits.add_on_min_gain}"
            )
            bars, frozen, broker, stale_before = prepare_day(day, events, cash)
            prev = frozen.no_chase_prices
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
            print(f"== {day} parity FAIL (data): {exc}")
            ok_all = False
            continue
        live = first_daemon(raw_live, day)
        s_raw, s_live, s_sim = summarize(raw_live), summarize(live), summarize(res.ledger)
        devs = fill_deviations(live, res.fills)
        bad_devs = [x for x in devs if abs(x[4]) > MAX_FILL_DEV]
        same = s_live == s_sim
        exact = same and not bad_devs
        live_dec = live_decisions(log_rows, day, DUPLICATE_DAEMON_ORDERS.get(day, set()))
        pairs = match_decisions(res.decisions, live_dec, within=dt.timedelta(seconds=5))
        unused, counts = match_summary(pairs, live_dec)
        divs = divergences(
            pairs,
            live,
            res.ledger,
            res.fills,
            bars,
            prev,
            max_move=no_chase_cap(cfg),
            limits=cfg.limits,
            unmatched_live=unused,
        )
        divs += fill_rule_divergences(res.fills, bars)
        divs += exit_divergences(live, res.ledger, res.fills, bars, cfg.limits)
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
        print(
            f"  risk refusals {len(res.refusals)}; cash rejections {len(res.cash_rejections)}; "
            f"end positions {res.positions}; run dir {res.run_dir}"
        )
        for r in res.refusals:
            print(f"    REFUSED {r['ts'][11:19]} {r['ticker']} {r['reason']}")
        for x in divs:
            at = x["at"].strftime("%H:%M:%S") if x["at"] else "--------"
            print(
                f"  DIV {x['kind']:8} {x['ticker']} {at} live={x['live']} sim={x['sim']} "
                f"[{x['cls']}] {x['note']}"
            )
        print("  -- exact comparison --")
        print(f"  event tickers without bars: {res.missing_bars}")
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
        assert res.run_dir is not None
        (res.run_dir / "result.json").write_text(
            json.dumps(
                {
                    "ledger": res.ledger,
                    "skips": res.skips,
                    "fills": res.fills,
                    "errors": res.errors,
                    "decisions": res.decisions,
                    "divergences": divs,
                    "refusals": res.refusals,
                    "cash_rejections": res.cash_rejections,
                    "positions": res.positions,
                    "prev_close_flags": frozen.flags,
                },
                default=str,
                ensure_ascii=False,
                indent=1,
            )
        )
    raise SystemExit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
