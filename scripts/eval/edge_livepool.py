"""Does the universe LIVE actually trades carry the momentum drag?

`select_universe`'s top-300-by-prior-day-turnover is the A-plan sweep pool. It
exists to bound the 1-req/s company-news sweep's lap time and it has never
gated a live entry: live trades whatever tickers Toss tags on a dashboard
article, filtered only by the stock master. So "we chose a bad universe" is
true of our research pool and says nothing, by itself, about live.

This settles which. Two measurements, because the dashboard-tagged ticker set
exists for seven days in 2026-09 and two of those are genuine archive:

  DIRECT   the dashboard set's own open-to-close drift on those days, beside
           the market, our top-300, and the movers/calm split. Honest but
           nearly powerless, and provenance is kept separate throughout.

  IMPLIED  characterise the dashboard-tagged names by two things observable at
           decision time -- their prior-day turnover rank and their prior-day
           absolute move, both as percentiles within that day's traded names --
           then measure what names with THAT profile drifted over ten years.
           This buys a long history for the kind of name the dashboard picks
           without needing ten years of dashboards.

If the implied drift lands near the market's -6.4 bp, live's universe is fine
and the 22 bp problem is confined to research. If it lands near the top-300's
-28.6 or the movers third's -42.0, the dashboard is a momentum screen by
editorial accident and live inherits the drag without anyone having chosen it.

READ-ONLY on kr_quant.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from edge_db import connect  # noqa: E402

BP = 10_000
#: percentile bands for the two decision-time characterisations
VAL_BANDS = (0.5, 0.8, 0.9, 0.95, 0.99)
MOVE_BANDS = (0.5, 0.8, 0.9, 0.95, 0.99)


def load_bars(
    cur, lo: dt.date, hi: dt.date
) -> dict[dt.date, dict[str, tuple[float, float, float]]]:
    cur.execute(
        "SELECT date, code, open, close, trade_value FROM daily_bars_adjusted "
        "WHERE date >= %s AND date <= %s",
        (lo, hi),
    )
    out: dict[dt.date, dict[str, tuple[float, float, float]]] = defaultdict(dict)
    for date, code, o, c, v in cur.fetchall():
        if o and c and float(o) > 0 and len(code) == 6 and code.isdigit():
            out[date][code] = (float(o), float(c), float(v or 0.0))
    return dict(out)


def pct_ranks(
    prev: dict[str, tuple[float, float, float]],
) -> tuple[dict[str, float], dict[str, float]]:
    """Percentile of each code by prior-day traded value and by prior-day |move|.

    1.0 = the most traded / the biggest mover that day. Both are known before
    the open of the day being scored.
    """
    traded = [c for c, (_o, _c, v) in prev.items() if v > 0]
    by_val = sorted(traded, key=lambda c: prev[c][2])
    by_mov = sorted(traded, key=lambda c: abs(prev[c][1] / prev[c][0] - 1))
    n = len(traded)
    return (
        {c: (i + 1) / n for i, c in enumerate(by_val)},
        {c: (i + 1) / n for i, c in enumerate(by_mov)},
    )


def band(p: float, edges: tuple[float, ...]) -> str:
    lo = 0.0
    for e in edges:
        if p < e:
            return f"{lo:.2f}-{e:.2f}"
        lo = e
    return f"{lo:.2f}-1.00"


def ew(day: dict[str, tuple[float, float, float]], codes) -> float | None:
    r = [day[c][1] / day[c][0] - 1 for c in codes if c in day]
    return statistics.fmean(r) if r else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dash",
        default="data/eval/edge/dash_labels.jsonl",
        help="labelled pairs -- used ONLY for the direct drift, which needs bars",
    )
    ap.add_argument(
        "--events",
        default="data/eval/replay_events",
        help="raw dashboard events -- used for the profile, which needs only "
        "prior-day data and so must not inherit the bar-coverage filter",
    )
    ap.add_argument("--labels", default="data/eval/prompt_v3/labels.jsonl")
    ap.add_argument("--start", default="2016-09-12")
    ap.add_argument("--end", default="2026-09-15")
    args = ap.parse_args()

    # Two ticker sets, deliberately. `dash` needs minute bars because it is
    # priced; `tagged` does not, because a percentile rank only needs the
    # PREVIOUS day's daily bar. Using the priced set for the profile would
    # inherit its bar-coverage filter, which drops thin names non-randomly and
    # would bias the turnover percentile upward -- the same silent, outcome-
    # correlated filtering that has bitten this project three times.
    dash: dict[dt.date, set[str]] = defaultdict(set)
    archive: dict[dt.date, bool] = {}
    with Path(args.dash).open() as f:
        for line in f:
            r = json.loads(line)
            d = dt.date.fromisoformat(r["day"])
            dash[d].add(r["code"])
            archive[d] = bool(r.get("archive"))

    tagged: dict[dt.date, set[str]] = defaultdict(set)
    ev_dir = Path(args.events)
    for path in sorted(ev_dir.glob("*.jsonl")):
        d = dt.date.fromisoformat(path.stem)
        with path.open() as f:
            for line in f:
                tagged[d].add(json.loads(line)["ticker"])
        archive.setdefault(d, False)
    meta = ev_dir / "meta.json"
    if meta.exists():
        for day in json.loads(meta.read_text()).get("archive_days", []):
            archive[dt.date.fromisoformat(day)] = True
    n_drop = sum(len(tagged[d]) for d in tagged) - sum(len(dash[d]) for d in dash)
    print(
        f"\nprofile uses {sum(len(v) for v in tagged.values())} tagged (day, ticker) pairs; "
        f"the priced set is {n_drop} smaller"
    )

    sweep: dict[dt.date, set[str]] = defaultdict(set)
    with Path(args.labels).open() as f:
        for line in f:
            r = json.loads(line)
            sweep[dt.date.fromisoformat(r["day"])].add(r["key"].split("|")[1])

    conn = connect()
    cur = conn.cursor()
    bars = load_bars(
        cur,
        dt.date.fromisoformat(args.start) - dt.timedelta(days=15),
        dt.date.fromisoformat(args.end),
    )
    conn.close()
    cal = sorted(bars)
    prev_of = {d: cal[i - 1] for i, d in enumerate(cal) if i}

    # ---- 1. what kind of names does each pool pick?
    print("\n=== 1. decision-time profile of the tagged names (percentile within that day)")
    print(f"{'pool':<26}{'n':>7}{'prior-day turnover pct':>26}{'prior-day |move| pct':>24}")
    print(f"{'':<26}{'':>7}{'p25':>8}{'median':>9}{'p75':>9}{'p25':>8}{'median':>8}{'p75':>8}")
    for label, pool, days in (
        ("dashboard (all 7 days)", tagged, sorted(tagged)),
        ("dashboard archive only", tagged, [d for d in sorted(tagged) if archive.get(d)]),
        ("dashboard reconstructed", tagged, [d for d in sorted(tagged) if not archive.get(d)]),
        ("sweep pool (same 7 days)", sweep, sorted(tagged)),
        ("sweep pool (all 125 days)", sweep, sorted(sweep)),
    ):
        vals, movs = [], []
        for d in days:
            prev = prev_of.get(d)
            if prev is None or d not in bars:
                continue
            vr, mr = pct_ranks(bars[prev])
            for c in pool.get(d, ()):
                if c in vr:
                    vals.append(vr[c])
                    movs.append(mr[c])
        if len(vals) < 20:
            print(f"{label:<26}{len(vals):>7}  (too few)")
            continue
        vals.sort()
        movs.sort()
        q = lambda s, p: s[int(len(s) * p)]  # noqa: E731
        print(
            f"{label:<26}{len(vals):>7}{q(vals, 0.25):>8.2f}{q(vals, 0.5):>9.2f}{q(vals, 0.75):>9.2f}"
            f"{q(movs, 0.25):>8.2f}{q(movs, 0.5):>8.2f}{q(movs, 0.75):>8.2f}"
        )

    # ---- 2. direct drift on the seven dashboard days
    print("\n=== 2. direct: open-to-close on the dashboard days (7 days, 2 of them archive)")
    print(f"{'set':<26}{'days':>6}{'drift bp':>10}")
    groups: dict[str, list[float]] = defaultdict(list)
    for d in sorted(dash):
        prev = prev_of.get(d)
        if prev is None or d not in bars:
            continue
        today, pv = bars[d], bars[prev]
        traded = [c for c, (_o, _c, v) in pv.items() if v > 0]
        ranked = sorted(traded, key=lambda c: (-pv[c][2], c))
        top = ranked[:300]
        moved = sorted(top, key=lambda c: -abs(pv[c][1] / pv[c][0] - 1))
        k = len(top) // 3
        for name, codes in (
            ("dashboard tagged", dash[d]),
            ("all traded", traded),
            ("top300", top),
            ("movers third", moved[:k]),
            ("calm two-thirds", moved[k:]),
        ):
            r = ew(today, codes)
            if r is not None:
                groups[name].append(r)
        if archive.get(d):
            r = ew(today, dash[d])
            if r is not None:
                groups["dashboard (archive days)"].append(r)
    for name in (
        "dashboard tagged",
        "dashboard (archive days)",
        "all traded",
        "top300",
        "movers third",
        "calm two-thirds",
    ):
        v = groups.get(name)
        if v:
            print(f"{name:<26}{len(v):>6}{statistics.fmean(v) * BP:>+10.1f}")

    # ---- 3. ten years of the profile the dashboard picks
    print("\n=== 3. implied: ten years of drift by decision-time profile")
    print("    the dashboard's own weight in each cell is in the last column\n")
    dash_w: dict[tuple[str, str], int] = defaultdict(int)
    total_w = 0
    for d in sorted(tagged):
        prev = prev_of.get(d)
        if prev is None:
            continue
        vr, mr = pct_ranks(bars[prev])
        for c in tagged[d]:
            if c in vr:
                dash_w[(band(vr[c], VAL_BANDS), band(mr[c], MOVE_BANDS))] += 1
                total_w += 1

    cells: dict[tuple[str, str], list[float]] = defaultdict(list)
    lo_d, hi_d = dt.date.fromisoformat(args.start), dt.date.fromisoformat(args.end)
    for i, d in enumerate(cal):
        if i == 0 or d < lo_d or d > hi_d:
            continue
        prev, today = bars[cal[i - 1]], bars[d]
        vr, mr = pct_ranks(prev)
        members: dict[tuple[str, str], list[float]] = defaultdict(list)
        for c, p in vr.items():
            if c in today:
                members[(band(p, VAL_BANDS), band(mr[c], MOVE_BANDS))].append(
                    today[c][1] / today[c][0] - 1
                )
        for cell, rets in members.items():
            if rets:
                cells[cell].append(statistics.fmean(rets))

    print(f"{'turnover pct':<14}{'|move| pct':<14}{'days':>6}{'drift bp':>10}{'dash weight':>13}")
    implied = w_used = 0.0
    for cell in sorted(cells, key=lambda c: -dash_w.get(c, 0)):
        v = cells[cell]
        w = dash_w.get(cell, 0)
        if w == 0 or len(v) < 200:
            continue
        drift = statistics.fmean(v)
        implied += drift * w
        w_used += w
        print(f"{cell[0]:<14}{cell[1]:<14}{len(v):>6}{drift * BP:>+10.1f}{w / total_w:>13.1%}")
    if w_used:
        print(
            f"\n  dashboard-weighted implied drift over ten years: "
            f"{implied / w_used * BP:+.1f} bp  ({w_used / total_w:.0%} of its weight covered)"
        )
    print(
        "\n  reference: market -6.8, our top300 -28.6, movers third -42.0, "
        "calm two-thirds -21.8 (same-day, ten years)\n"
    )


if __name__ == "__main__":
    main()
