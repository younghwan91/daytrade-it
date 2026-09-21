"""Does `persistence` ORDER the horizon response? A paired, time-ordered test.

Three separate group means each carry a huge standard error at h=40, which says
nothing about whether the ORDER structural > multi_quarter > one_off is real:
the three groups share the same days, so the right statistic is the per-day
DIFFERENCE, where the common market factor cancels.

For each judged day and each horizon we form the mean stratum-matched alpha of
each persistence group, then the paired differences. Reported: the mean
difference, its standard error inflated by sqrt(h) for the h-fold window
overlap, and the fraction of days on which the difference has the expected
sign -- an ordering statistic, not a magnitude.

Time-ordered, LODO-free. `data/eval/rlnews/CONFIRM_WINDOWS.json` fixes the
split: the ordering is read on `search` (2026-03-17..2026-08-18) and the held
out `confirm` days (2026-08-19..2026-09-15) are touched exactly once, at the
end.

READ-ONLY on kr_quant.
"""

from __future__ import annotations

import argparse
import bisect
import datetime as dt
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path.cwd()
sys.path.insert(0, str(REPO / "data" / "eval" / "rlnews"))
sys.path.insert(0, str(REPO / "scripts" / "eval"))

from daily_horizon import HORIZONS, MARKET, MIN_CELL, load_panel, rows_by_day  # noqa: E402
from edge_db import connect  # noqa: E402
from edge_overnight import sigma_index, sigma_of  # noqa: E402
from edge_strata import NQ, quantile_edges  # noqa: E402
from universe_coverage import select_universe  # noqa: E402

BP = 10_000
GROUPS = ("structural", "multi_quarter", "one_off")
PAIRS = (("structural", "one_off"), ("structural", "multi_quarter"), ("multi_quarter", "one_off"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judged", default="data/eval/prompt_v3/master.jsonl")
    ap.add_argument("--windows", default="data/eval/rlnews/CONFIRM_WINDOWS.json")
    ap.add_argument("--size", type=int, default=300)
    ap.add_argument("--out", default="data/eval/rlnews/persist_order.json")
    args = ap.parse_args()

    win = json.loads(Path(args.windows).read_text())["track_a_judged"]
    confirm = set(win["confirm"])
    s_lo, s_hi = win["search"]

    members: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    with Path(args.judged).open() as f:
        for line in f:
            r = json.loads(line)
            p = (r.get("v2") or {}).get("persistence")
            if p in GROUPS:
                members[r["day"]][p].add(r["code"])
    days = sorted(members)
    print(
        f"judged days {days[0]}..{days[-1]}  n={len(days)}  "
        f"search={sum(1 for d in days if d not in confirm)} confirm={len(confirm)}"
    )

    conn = connect()
    cur = conn.cursor()
    lo = dt.date.fromisoformat(days[0]) - dt.timedelta(days=120)
    hi = dt.date.fromisoformat(days[-1]) + dt.timedelta(days=90)
    bars, codes, di, ci, opn, C, V = load_panel(cur, lo, hi)
    conn.close()
    rbd = rows_by_day(bars, codes, V)
    hist = {}
    for i, c in enumerate(codes):
        ok = np.isfinite(C[i])
        if ok.any():
            hist[c] = [(bars[j], float(C[i, j]), float(V[i, j])) for j in np.nonzero(ok)[0]]
    sindex = sigma_index(hist)

    # series[h][group][day] = mean alpha
    series: dict[int, dict[str, dict[str, float]]] = {h: {g: {} for g in GROUPS} for h in HORIZONS}
    ncd: dict[str, int] = defaultdict(int)
    for ds in days:
        d = dt.date.fromisoformat(ds)
        j = di.get(d)
        if j is None or j == 0:
            continue
        univ = [c for c in select_universe(rbd[bars[j - 1]], size=args.size) if c != MARKET]
        univ = [c for c in univ if c in ci]
        sig = {c: sigma_of(sindex, c, d) for c in univ}
        u_sig = [sig[c] for c in univ if sig[c] is not None]
        if len(u_sig) < NQ * 4:
            continue
        se_ = quantile_edges(u_sig, NQ)
        pv = {c: float(V[ci[c], j - 1]) for c in univ}
        ve_ = quantile_edges([pv[c] for c in univ], NQ)
        cell_of = {
            c: (
                bisect.bisect_right(se_, sig[c]) if sig[c] is not None else -1,
                bisect.bisect_right(ve_, pv[c]) if pv[c] else -1,
            )
            for c in univ
        }
        for h in HORIZONS:
            k = j + h
            if k >= len(bars):
                continue
            r_all = C[:, k] / C[:, j] - 1.0
            grp: dict = defaultdict(list)
            for c in univ:
                v = r_all[ci[c]]
                if np.isfinite(v):
                    grp[cell_of[c]].append(float(v))
            cm = {kk: statistics.fmean(vs) for kk, vs in grp.items() if len(vs) >= MIN_CELL}
            for g in GROUPS:
                vals = []
                for c in members[ds][g]:
                    if c not in cell_of:
                        continue
                    v = r_all[ci[c]]
                    base = cm.get(cell_of[c])
                    if np.isfinite(v) and base is not None:
                        vals.append(float(v) - base)
                if vals:
                    series[h][g][ds] = statistics.fmean(vals)
                    ncd[f"{g}|{h}"] += len(vals)

    def report(title, keep):
        print(f"\n=== {title}")
        print(
            f"{'h':>4}" + "".join(f"{g:>15}" for g in GROUPS) + f"{'days':>6}"
            f"   ordering  "
            + "".join(f"{a[:4]}-{b[:4]:<6}{'  se':>6}{'  sign':>7}" for a, b in PAIRS)
        )
        rows = {}
        for h in HORIZONS:
            common = sorted(
                d
                for d in series[h][GROUPS[0]]
                if keep(d) and all(d in series[h][g] for g in GROUPS)
            )
            if len(common) < 5:
                continue
            m = {g: statistics.fmean(series[h][g][d] for d in common) for g in GROUPS}
            ordered = m["structural"] >= m["multi_quarter"] >= m["one_off"]
            line = f"{h:>4}" + "".join(f"{m[g] * BP:>+15.1f}" for g in GROUPS)
            line += f"{len(common):>6}   {'YES ' if ordered else 'no  '}     "
            pr = {}
            for a, b in PAIRS:
                dif = [series[h][a][d] - series[h][b][d] for d in common]
                mu = statistics.fmean(dif)
                se = (statistics.stdev(dif) / len(dif) ** 0.5) * (h**0.5)
                frac = sum(1 for x in dif if x > 0) / len(dif)
                pr[f"{a}-{b}"] = [mu * BP, se * BP, frac, len(dif)]
                line += f"{mu * BP:>+10.1f}{se * BP:>6.1f}{frac:>7.2f}"
            print(line)
            rows[h] = {
                "means": {g: m[g] * BP for g in GROUPS},
                "ordered": ordered,
                "days": len(common),
                "pairs": pr,
            }
        return rows

    out = {
        "search": report(
            f"SEARCH {s_lo}..{s_hi} -- mean group alpha (bp) and paired "
            f"differences (bp, se overlap-inflated, sign = frac of days "
            f"with the expected order)",
            lambda d: d not in confirm,
        ),
        "confirm": report(
            f"CONFIRM {min(confirm)}..{max(confirm)} ({len(confirm)} days) -- touched once",
            lambda d: d in confirm,
        ),
        "code_days": dict(ncd),
    }
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
