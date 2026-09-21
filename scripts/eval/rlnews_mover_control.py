"""The mover split, and an explicit negative control for the mechanism that has
produced four false findings in this project: conditioning a sample on its own
outcome.

Three splits of the same pool, same stratum-matched null (refined to
cell x bucket so the comparison stays like-for-like):

  prev-day    |close(d-1)/close(d-2) - 1|.  Known a full session before entry.
              This is the split the main run uses.
  same-day    |close(d)/close(d-1) - 1|.  ALSO clean here, and worth saying so:
              entry is the close of day d, so the day's own move is in the
              past at the moment of the trade. It would be look-ahead for an
              intraday entry and is not for this one.
  BANNED      |close(d+h)/close(d) - 1|, the absolute size of the very return
              being measured. Nothing about it is available at entry. It is run
              only to calibrate how large the artefact is, so that a real
              effect can be told apart from it.

Each split is taken at the median inside that day's own universe.

READ-ONLY on kr_quant.
"""

from __future__ import annotations

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

from daily_horizon import MARKET, MIN_CELL, load_panel, rows_by_day  # noqa: E402
from edge_db import connect  # noqa: E402
from edge_overnight import sigma_index, sigma_of  # noqa: E402
from edge_strata import NQ, quantile_edges  # noqa: E402
from universe_coverage import select_universe  # noqa: E402

BP = 10_000
H = (1, 5, 20, 40)
LO, HI = dt.date(2023, 3, 16), dt.date(2026, 9, 15)


def main() -> None:
    conn = connect()
    cur = conn.cursor()
    days, codes, di, ci, opn, C, V = load_panel(cur, dt.date(2022, 11, 1), dt.date(2026, 9, 21))
    conn.close()
    rbd = rows_by_day(days, codes, V)
    hist = {}
    for i, c in enumerate(codes):
        ok = np.isfinite(C[i])
        if ok.any():
            hist[c] = [(days[j], float(C[i, j]), float(V[i, j])) for j in np.nonzero(ok)[0]]
    sidx = sigma_index(hist)
    absmove = np.full_like(C, np.nan)
    absmove[:, 1:] = np.abs(C[:, 1:] / C[:, :-1] - 1.0)

    pool: dict[dt.date, set[str]] = defaultdict(set)
    with open("data/eval/rlnews/master_daily_full.jsonl") as f:
        for line in f:
            r = json.loads(line)
            pool[dt.date.fromisoformat(r["day"])].add(r["code"])

    res: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for d in sorted(pool):
        j = di.get(d)
        if not j or d < LO or d > HI:
            continue
        univ = [c for c in select_universe(rbd[days[j - 1]], size=300) if c != MARKET and c in ci]
        sg = {c: sigma_of(sidx, c, d) for c in univ}
        us = [sg[c] for c in univ if sg[c] is not None]
        if len(us) < 16:
            continue
        se_ = quantile_edges(us, NQ)
        pv = {c: float(V[ci[c], j - 1]) for c in univ}
        ve_ = quantile_edges([pv[c] for c in univ], NQ)
        cell = {
            c: (
                bisect.bisect_right(se_, sg[c]) if sg[c] is not None else -1,
                bisect.bisect_right(ve_, pv[c]) if pv[c] else -1,
            )
            for c in univ
        }
        for h in H:
            k = j + h
            if k >= len(days):
                continue
            r_all = C[:, k] / C[:, j] - 1.0
            splits = {
                "prev-day (clean)": {c: float(absmove[ci[c], j - 1]) for c in univ},
                "same-day (clean here)": {c: float(absmove[ci[c], j]) for c in univ},
                "BANNED own outcome": {c: abs(float(r_all[ci[c]])) for c in univ},
            }
            for name, vals in splits.items():
                vals = {c: v for c, v in vals.items() if np.isfinite(v)}
                if len(vals) < 50:
                    continue
                med = statistics.median(vals.values())
                mv = {c: (vals[c] > med) for c in vals}
                g: dict = defaultdict(list)
                for c in univ:
                    v = r_all[ci[c]]
                    if np.isfinite(v) and c in mv:
                        g[(cell[c], mv[c])].append(float(v))
                cm = {kk: statistics.fmean(vs) for kk, vs in g.items() if len(vs) >= MIN_CELL}
                for tag, want in (("mover", True), ("calm", False)):
                    out = []
                    for c in pool[d]:
                        if c not in mv or mv[c] is not want or c not in cell:
                            continue
                        v = r_all[ci[c]]
                        b = cm.get((cell[c], want))
                        if np.isfinite(v) and b is not None:
                            out.append(float(v) - b)
                    if out:
                        res[f"{name}|{tag}"][h].append(statistics.fmean(out))

    print("\nstratum-matched alpha, bp, pool 2023-03-16..2026-09-15")
    print(f"{'split':<26}{'bucket':<8}" + "".join(f"{('h=' + str(h)):>12}" for h in H))
    for name in ("prev-day (clean)", "same-day (clean here)", "BANNED own outcome"):
        for tag in ("mover", "calm"):
            row = res[f"{name}|{tag}"]
            cells = ""
            for h in H:
                v = row.get(h)
                if v:
                    se = statistics.stdev(v) / len(v) ** 0.5 * h**0.5 if len(v) > 1 else 0.0
                    cells += f"{statistics.fmean(v) * BP:>+8.1f}±{se * BP:>3.0f}"
                else:
                    cells += f"{'.':>12}"
            print(f"{name:<26}{tag:<8}{cells}")
    Path("data/eval/rlnews/mover_control.json").write_text(
        json.dumps(
            {
                k: {str(h): [statistics.fmean(v), len(v)] for h, v in hs.items()}
                for k, hs in res.items()
            },
            indent=1,
        )
    )
    print("\n-> data/eval/rlnews/mover_control.json")


if __name__ == "__main__":
    main()
