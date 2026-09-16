"""Multi-day outcomes for every judged pair, with a null matched at each horizon.

Two of this project's binding constraints weaken only on a longer horizon. We
are ~21.8 minutes late to every article and most of that is upstream of us, so
an impulse is unreachable -- but if the news has lasting impact, arriving late
costs little. And the 20 bp sell tax is charged per disposal, so it amortises
over days: 36.7 bp is the whole toll at T+0 and 7.3 bp/day at T+5. Neither
constraint weakens intraday.

This writes the raw material for that test and nothing else, so the
conditioning can be done offline without touching the DB again.

Entry is unchanged: the same acting minute, the same SimBroker quote rule. The
exit moves to the close of day d+h. The null moves with it -- same universe,
same minute, same horizon -- because a longer horizon carries more drift and
more variance, and a null that did not move with it would turn both into
"alpha".

Two nulls are emitted per pair per horizon:
  null_all   the whole top-300 cross-section at that minute
  null_cell  only the names in the pair's own volatility x liquidity cell
             (`edge_strata.py`), which is what took the intraday result from
             +8.1 bp to +1.6 bp -- the difference was composition, not skill

Returns are composed from two legs, not read off one price, because
`minute_bars` is raw and `daily_bars_adjusted` is adjusted; dividing one by the
other breaks on any split or dividend in between:

    1 + ret_h = (1 + same-day raw return from minute bars)
              * (adjusted close at d+h / adjusted close at d)

Each leg is internally consistent, so the product is split-safe.

Output, one JSON line per pair:
    key, day, code, act_min, sig_q, val_q,
    r   {h: return},  na {h: null_all},  nc {h: null_cell}

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
from edge_null import fetch, trading_days, universe_for  # noqa: E402

HORIZONS = (0, 1, 2, 5, 10, 20, 40)
MIN_CELL = 5


def daily_closes(cur, lo: dt.date, hi: dt.date) -> dict[dt.date, dict[str, float]]:
    cur.execute(
        "SELECT date, code, close FROM daily_bars_adjusted WHERE date >= %s AND date <= %s",
        (lo, hi),
    )
    out: dict[dt.date, dict[str, float]] = defaultdict(dict)
    for date, code, c in cur.fetchall():
        if c and float(c) > 0:
            out[date][code] = float(c)
    return dict(out)


def load_strata(path: Path) -> dict[tuple[str, str], tuple[int, int]]:
    out: dict[tuple[str, str], tuple[int, int]] = {}
    with path.open() as f:
        for line in f:
            s = json.loads(line)
            out[(s["day"], s["code"])] = (s["sig_q"], s["val_q"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/eval/prompt_v3/labels.jsonl")
    ap.add_argument("--strata", default="data/eval/edge/strata_sweep.jsonl")
    ap.add_argument("--out", default="data/eval/edge/horizon_pairs.jsonl")
    ap.add_argument("--size", type=int, default=300)
    args = ap.parse_args()

    strata = load_strata(Path(args.strata))
    pairs: dict[dt.date, list[dict]] = defaultdict(list)
    with Path(args.labels).open() as f:
        for line in f:
            r = json.loads(line)
            r["code"] = r.get("code") or r["key"].split("|")[1]
            pairs[dt.date.fromisoformat(r["day"])].append(r)
    days = sorted(pairs)

    conn = connect()
    cur = conn.cursor()
    cal = trading_days(cur, days[0] - dt.timedelta(days=30), days[-1] + dt.timedelta(days=40))
    idx = {d: i for i, d in enumerate(cal)}
    prev_of = {d: cal[i - 1] for i, d in enumerate(cal) if i}
    closes = daily_closes(cur, days[0], cal[-1])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_out = 0
    with out_path.open("w") as out:
        for day in days:
            prev = prev_of.get(day)
            if prev is None:
                continue
            ds = str(day)
            univ = universe_for(cur, prev, args.size)
            rows = pairs[day]
            bars = fetch(cur, day, sorted({r["code"] for r in rows} | set(univ)))
            i = idx[day]
            today_adj = closes.get(day, {})

            # carry[h][code] = adjusted close at d+h / adjusted close at d
            carry: dict[int, dict[str, float]] = {}
            for h in HORIZONS:
                ex = cal[i + h] if i + h < len(cal) else None
                px = closes.get(ex, {}) if ex else {}
                carry[h] = {
                    c: px[c] / today_adj[c] for c in px if c in today_adj and today_adj[c] > 0
                }

            members = [c for c in univ if c in bars]
            cell_of = {c: strata.get((ds, c)) for c in set(members) | {r["code"] for r in rows}}
            minutes = sorted({r["act_min"] for r in rows})

            # null[h][minute] -> (all, {cell: mean})
            null: dict[int, dict[int, tuple[float | None, dict[tuple[int, int], float]]]] = {}
            for h in HORIZONS:
                k = carry[h]
                per_min: dict[int, tuple[float | None, dict[tuple[int, int], float]]] = {}
                for m in minutes:
                    allv: list[float] = []
                    by_cell: dict[tuple[int, int], list[float]] = defaultdict(list)
                    for c in members:
                        if c not in k:
                            continue
                        r0 = bars[c].ret(m)
                        if r0 is None:
                            continue
                        v = (1 + r0) * k[c] - 1
                        allv.append(v)
                        cell = cell_of.get(c)
                        if cell:
                            by_cell[cell].append(v)
                    per_min[m] = (
                        statistics.fmean(allv) if allv else None,
                        {
                            cl: statistics.fmean(vs)
                            for cl, vs in by_cell.items()
                            if len(vs) >= MIN_CELL
                        },
                    )
                null[h] = per_min

            for r in rows:
                d = bars.get(r["code"])
                if d is None:
                    continue
                r0 = d.ret(r["act_min"])
                if r0 is None:
                    continue
                cell = cell_of.get(r["code"])
                rec: dict[str, object] = {
                    "key": r["key"],
                    "day": ds,
                    "code": r["code"],
                    "act_min": r["act_min"],
                    "sig_q": cell[0] if cell else None,
                    "val_q": cell[1] if cell else None,
                    "pre_move": r.get("pre_move"),
                }
                rr, na, nc = {}, {}, {}
                for h in HORIZONS:
                    k = carry[h].get(r["code"])
                    if k is None:
                        continue
                    rr[str(h)] = (1 + r0) * k - 1
                    a, cells = null[h].get(r["act_min"], (None, {}))
                    if a is not None:
                        na[str(h)] = a
                    if cell and cell in cells:
                        nc[str(h)] = cells[cell]
                if not rr:
                    continue
                rec["r"], rec["na"], rec["nc"] = rr, na, nc
                out.write(json.dumps(rec) + "\n")
                n_out += 1
            print(f"{day}  pairs={len(rows)}  written={n_out}", flush=True)
    conn.close()
    print(f"\n{n_out} pairs -> {out_path}")


if __name__ == "__main__":
    main()
