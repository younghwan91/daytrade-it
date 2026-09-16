"""A matched null for scalp-it's dip fills.

Their question: of the gross return on a dip fill, how much is real dip-buying
and how much is just being present during a volatile minute?

The answer needs a null that is wrong in the SAME way their number is. Their
`ret_bp` is an oracle exit -- the maximum bid inside a 300-second window
divided by the fill price. Comparing that to a pre-committed exit would measure
the oracle premium, which is large and has nothing to do with their hypothesis.
So the null here takes its own 300-second maximum too: a universe name, entered
at the same minute at the ordinary quote, sold at its own best moment. The
difference is then dip-buying, with the oracle and the minute differenced out.

Matching, as everywhere in this investigation: the same day, the same entry
minute, and the same prior-day volatility x prior-day turnover quartile, both
known before the open.

One asymmetry, flagged rather than hidden: their maximum is of `bid1` and mine
is of the bar `high`. A high is a trade print and sits at or above the bid, so
the null is if anything slightly generous to itself, which biases the reported
alpha DOWNWARD. Minute resolution is theirs by choice -- a 300-second hold does
not need finer.

READ-ONLY on kr_quant.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import datetime as dt
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from edge_db import connect  # noqa: E402
from edge_strata import quantile_edges, sigma_at  # noqa: E402

BP = 10_000
OPEN = dt.time(9, 0)
REGULAR_END = dt.time(15, 33)
SLOTS = 394
HOLD_MIN = 5  # 300 seconds
NQ = 4
MIN_CELL = 5


class Bars:
    """Per-minute entry quote and forward 5-minute maximum."""

    __slots__ = ("entry", "high")

    def __init__(self, rows: list[tuple]) -> None:
        opens: list[float | None] = [None] * SLOTS
        closes: list[float | None] = [None] * SLOTS
        highs: list[float | None] = [None] * SLOTS
        for ts, o, h, c in rows:
            t = ts.time()
            if t < OPEN or t > REGULAR_END:
                continue
            i = (ts.hour - 9) * 60 + ts.minute
            if 0 <= i < SLOTS:
                opens[i], highs[i], closes[i] = float(o), float(h), float(c)
        entry: list[float | None] = [None] * SLOTS
        prev: float | None = None
        for i in range(SLOTS):
            entry[i] = opens[i] if opens[i] is not None else prev
            if closes[i] is not None:
                prev = closes[i]
        self.entry = entry
        self.high = highs

    def oracle(self, i: int) -> float | None:
        """(max high over [i, i+5] / entry(i) - 1) in bp -- their exit rule."""
        e = self.entry[i]
        if not e or e <= 0:
            return None
        window = [h for h in self.high[i : i + HOLD_MIN + 1] if h is not None]
        if not window:
            return None
        return (max(window) / e - 1) * BP


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input", default=str(Path.home() / "git/scalp-it/data/null_input_dipfill.csv")
    )
    ap.add_argument("--size", type=int, default=300)
    ap.add_argument("--out", default="data/eval/edge/scalp_null.csv")
    args = ap.parse_args()

    rows: dict[dt.date, list[dict]] = defaultdict(list)
    with Path(args.input).open() as f:
        for r in csv.DictReader(f):
            d = dt.date.fromisoformat(r["date"])
            hh, mm, _ = r["entry_ts"].split(":")
            slot = (int(hh) - 9) * 60 + int(mm)
            if not (0 <= slot < SLOTS):
                continue
            rows[d].append(
                {"code": r["code"], "slot": slot, "ret": float(r["ret_bp"]), "tag": r["tag"]}
            )
    days = sorted(rows)
    print(f"{sum(len(v) for v in rows.values()):,} fills over {len(days)} days")

    conn = connect()
    cur = conn.cursor()
    cur.execute(
        "SELECT DISTINCT date FROM daily_bars_adjusted WHERE date >= %s AND date <= %s ORDER BY date",
        (days[0] - dt.timedelta(days=90), days[-1]),
    )
    cal = [r[0] for r in cur.fetchall()]
    prev_of = {d: cal[i - 1] for i, d in enumerate(cal) if i}

    cur.execute(
        "SELECT code, date, close, trade_value FROM daily_bars_adjusted "
        "WHERE date >= %s AND date <= %s ORDER BY code, date",
        (days[0] - dt.timedelta(days=90), days[-1]),
    )
    hist: dict[str, list[tuple]] = defaultdict(list)
    value_on: dict[dt.date, dict[str, float]] = defaultdict(dict)
    for code, date, close, value in cur.fetchall():
        if close and float(close) > 0:
            hist[code].append((date, float(close), float(value or 0.0)))
            if value and float(value) > 0 and len(code) == 6 and code.isdigit():
                value_on[date][code] = float(value)

    out_rows = []
    for day in days:
        prev = prev_of.get(day)
        if prev is None:
            continue
        vals = value_on[prev]
        univ = sorted(vals, key=lambda c: (-vals[c], c))[: args.size]
        theirs = {r["code"] for r in rows[day]}
        sig = {c: sigma_at(hist.get(c, []), day) for c in set(univ) | theirs}
        u_sig = [sig[c] for c in univ if sig.get(c) is not None]
        if len(u_sig) < NQ * 4:
            continue
        sig_edges = quantile_edges(u_sig, NQ)
        val_edges = quantile_edges([vals[c] for c in univ], NQ)

        def cell(
            code: str,
            sig: dict = sig,
            vals: dict = vals,
            sig_edges: list = sig_edges,
            val_edges: list = val_edges,
        ) -> tuple[int, int]:
            # bound as defaults: this closure is used only inside the current
            # day's iteration, and leaving it to capture the loop variables
            # would silently follow the last day if that ever changed
            s = sig.get(code)
            v = vals.get(code)
            return (
                bisect.bisect_right(sig_edges, s) if s is not None else -1,
                bisect.bisect_right(val_edges, v) if v is not None else -1,
            )

        cur.execute(
            "SELECT code, ts, open, high, close FROM minute_bars WHERE code = ANY(%s) "
            "AND ts >= %s AND ts < %s ORDER BY code, ts",
            (sorted(set(univ) | theirs), day, day + dt.timedelta(days=1)),
        )
        raw: dict[str, list[tuple]] = defaultdict(list)
        for code, ts, o, h, c in cur.fetchall():
            raw[code].append((ts, o, h, c))
        bars = {c: Bars(v) for c, v in raw.items()}

        members: dict[tuple[int, int], list[Bars]] = defaultdict(list)
        for c in univ:
            if c in bars:
                members[cell(c)].append(bars[c])

        slots = sorted({r["slot"] for r in rows[day]})
        null: dict[int, dict[tuple[int, int], tuple[float, int]]] = {}
        for s in slots:
            per: dict[tuple[int, int], tuple[float, int]] = {}
            for cl, bs in members.items():
                v = [x for b in bs if (x := b.oracle(s)) is not None]
                if len(v) >= MIN_CELL:
                    per[cl] = (statistics.fmean(v), len(v))
            null[s] = per

        for r in rows[day]:
            cl = cell(r["code"])
            got = null.get(r["slot"], {}).get(cl)
            if got is None:
                continue
            out_rows.append(
                {
                    "day": str(day),
                    "tag": r["tag"],
                    "ret_bp": r["ret"],
                    "null_bp": got[0],
                    "alpha_bp": r["ret"] - got[0],
                    "cell_n": got[1],
                }
            )
        print(f"{day}  fills={len(rows[day])}  matched={len(out_rows)}", flush=True)
    conn.close()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["day", "tag", "ret_bp", "null_bp", "alpha_bp", "cell_n"])
        w.writeheader()
        w.writerows(out_rows)

    print(f"\n{len(out_rows):,} of {sum(len(v) for v in rows.values()):,} fills matched\n")
    print(f"{'tag':<22}{'n':>9}{'gross':>9}{'null':>9}{'alpha':>9}{'se':>7}{'days':>6}")
    for tag in ("dipfill_tick_narrow", "dipfill_tick_mid", "dipfill_tick_wide", "ALL"):
        sel = out_rows if tag == "ALL" else [r for r in out_rows if r["tag"] == tag]
        if not sel:
            continue
        by: dict[str, list[float]] = defaultdict(list)
        for r in sel:
            by[r["day"]].append(r["alpha_bp"])
        per = [statistics.fmean(v) for v in by.values()]
        se = statistics.stdev(per) / (len(per) ** 0.5) if len(per) > 1 else float("nan")
        print(
            f"{tag:<22}{len(sel):>9,}{statistics.fmean(r['ret_bp'] for r in sel):>9.1f}"
            f"{statistics.fmean(r['null_bp'] for r in sel):>9.1f}"
            f"{statistics.fmean(per):>+9.1f}{se:>7.1f}{len(per):>6}"
        )
    print(f"\n-> {out}\n")


if __name__ == "__main__":
    main()
