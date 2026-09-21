"""Per-pair outcome labels for every judged pair, at its live decision minute.

READ-ONLY on kr_quant (one SELECT per trading day). No API, no Toss.

Live parity: the entry price is SimBroker's own quote rule at `decide_at` --
the decision minute's bar open, else the last completed bar's close. Nothing
after `decide_at` enters any input. Everything after it is outcome only.

Labels written per pair:
  ret_close   exit at the day's last regular close (<= 15:33)
  ret_1520    exit at the last bar <= 15:20 (the live signal expiry)
  mfe         max favourable excursion: max high after the decision minute / entry - 1
  mae         max adverse excursion: min low after the decision minute / entry - 1
  t_mfe_min   minutes from the decision minute to the bar holding the max high
  mkt_close   069500 over the same window
  exc_close   ret_close - mkt_close   (the 2026-09-13 doc's `excess`)
  pre_move    entry / frozen prev close - 1   (decision-time input, no look-ahead)
  act_min     minutes from 09:00 to the acting minute (0 for every premarket pair)
  turnover    the code's trade_value that day (liquidity)
Costs are NOT deducted; `COST` below is applied by the consumer so both the
gross and the net view stay available.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import psycopg2

MASTER = Path("data/eval/prompt_v3/master.jsonl")
PREV_CLOSE = Path("data/eval/prev_close")
OUT = Path("data/eval/prompt_v3/labels.jsonl")
MARKET = "069500"
REGULAR_END = dt.time(15, 33)
EXPIRY = dt.time(15, 20)
OPEN = dt.time(9, 0)


def _dsn() -> str:
    """KR_QUANT_DB when set, else the shared read-only dsn from edge_db."""
    if os.environ.get("KR_QUANT_DB"):
        return os.environ["KR_QUANT_DB"]
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import edge_db

    return edge_db.dsn()


def load_master(master: Path | None = None) -> dict[dt.date, list[dict]]:
    by_day: dict[dt.date, list[dict]] = defaultdict(list)
    with (master or MASTER).open() as f:
        for line in f:
            r = json.loads(line)
            by_day[dt.date.fromisoformat(r["day"])].append(r)
    return dict(by_day)


def fetch_bars(cur, day: dt.date, codes: list[str]) -> dict[str, list[tuple]]:
    cur.execute(
        "SELECT code, ts, open, high, low, close, trade_value FROM minute_bars "
        "WHERE code = ANY(%s) AND ts >= %s AND ts < %s ORDER BY code, ts",
        (codes, day, day + dt.timedelta(days=1)),
    )
    out: dict[str, list[tuple]] = defaultdict(list)
    for code, ts, o, h, lo, c, tv in cur.fetchall():
        out[code].append((ts, o, h, lo, c, tv))
    return out


def prev_closes(day: dt.date, prev_close_dir: Path | None = None) -> dict[str, float]:
    """Frozen prev closes for `day`, or {} when no frozen file exists.

    A missing file is not an error: `pre_move` is then None for every pair of
    that day and the pair is kept (see `label_day`).
    """
    path = (prev_close_dir or PREV_CLOSE) / f"{day}.json"
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())["closes"]
    return {k: float(v["close"]) for k, v in raw.items() if v.get("close")}


def window(bars: list[tuple], start: dt.datetime) -> list[tuple]:
    """Regular-session bars whose minute starts at or after `start`."""
    return [b for b in bars if start <= b[0] and OPEN <= b[0].time() <= REGULAR_END]


def entry_price(bars: list[tuple], minute: dt.datetime) -> float | None:
    """SimBroker's quote rule: this minute's open, else the last completed close."""
    for ts, o, _h, _l, _c, _tv in bars:
        if ts == minute:
            return float(o)
    done = [b for b in bars if b[0] < minute and b[0].time() >= OPEN]
    return float(done[-1][4]) if done else None


def market_ret(mbars: list[tuple], minute: dt.datetime, exit_ts: dt.datetime) -> float | None:
    e = entry_price(mbars, minute)
    if not e:
        return None
    done = [b for b in mbars if b[0] <= exit_ts and b[0].time() >= OPEN]
    if not done:
        return None
    return float(done[-1][4]) / e - 1


def label_day(
    cur, day: dt.date, rows: list[dict], prev_close_dir: Path | None = None
) -> list[dict]:
    codes = sorted({r["code"] for r in rows} | {MARKET})
    bars = fetch_bars(cur, day, codes)
    mbars = bars.get(MARKET, [])
    pc = prev_closes(day, prev_close_dir)
    turnover = {c: sum(b[5] or 0 for b in bs) for c, bs in bars.items()}

    out = []
    for r in rows:
        code = r["code"]
        bs = bars.get(code)
        if not bs:
            continue
        decide = dt.datetime.fromisoformat(r["decide_at"]).replace(tzinfo=None)
        minute = decide.replace(second=0, microsecond=0)
        if minute.date() != day:
            continue
        # Live parity: a judgment finished before the open cannot act until the
        # first poll (09:00:30), and SimBroker's quote inside the 09:00 minute is
        # that bar's open -- the day's opening price, as the 2026-09-13 eval used.
        if minute.time() < OPEN:
            minute = dt.datetime.combine(day, OPEN)
        e = entry_price(bs, minute)
        if not e or e <= 0:
            continue
        fwd = window(bs, minute)
        if not fwd:
            continue
        last = fwd[-1]
        upto = [b for b in fwd if b[0].time() <= EXPIRY]
        rec = {
            "key": r["key"],
            "day": str(day),
            "entry": e,
            "ret_close": float(last[4]) / e - 1,
            "ret_1520": (float(upto[-1][4]) / e - 1) if upto else None,
            "mfe": max(float(b[2]) for b in fwd) / e - 1,
            "mae": min(float(b[3]) for b in fwd) / e - 1,
            "t_mfe_min": int(
                (max(fwd, key=lambda b: float(b[2]))[0] - minute).total_seconds() // 60
            ),
            "n_bars_fwd": len(fwd),
            "act_min": int((minute - dt.datetime.combine(day, OPEN)).total_seconds() // 60),
            "premarket": minute.time() == OPEN and decide.time() < OPEN,
            "turnover": turnover.get(code),
        }
        mk = market_ret(mbars, minute, last[0])
        rec["mkt_close"] = mk
        rec["exc_close"] = None if mk is None else rec["ret_close"] - mk
        prev = pc.get(code)
        rec["pre_move"] = (e / prev - 1) if prev else None
        out.append(rec)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", default=str(MASTER), help="master.jsonl-shaped input")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument(
        "--prev-close",
        default=str(PREV_CLOSE),
        help="dir of frozen <day>.json prev closes; days without a file get pre_move=None",
    )
    args = ap.parse_args()

    prev_close_dir = Path(args.prev_close)
    by_day = load_master(Path(args.master))
    conn = psycopg2.connect(_dsn(), connect_timeout=5)
    conn.set_session(readonly=True, autocommit=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_in = n_out = 0
    with conn.cursor() as cur, out_path.open("w") as f:
        for day in sorted(by_day):
            rows = by_day[day]
            n_in += len(rows)
            labelled = label_day(cur, day, rows, prev_close_dir)
            for rec in labelled:
                f.write(json.dumps(rec) + "\n")
            n_out += len(labelled)
            print(f"{day}  pairs={len(rows):5d}  labelled={len(labelled):5d}", flush=True)
    conn.close()
    print(f"\ntotal {n_out} / {n_in} labelled -> {out_path}")


if __name__ == "__main__":
    main()
