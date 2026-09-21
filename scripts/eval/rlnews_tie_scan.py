"""How often does the vectorised twin's float barrier disagree with the live
Decimal barrier on the prior holdout?

`exit_rule_hit` compares ``close >= entry * (1 + take_profit_pct)`` in Decimal --
exact. `exit_fast`/`Stack` compare in float64, where e.g.
345000.0 * 1.1 == 379500.00000000006 and a close of exactly 379500 does NOT fire.
Disagreement therefore needs an EXACT tie on the Decimal barrier; this counts them.
"""

from __future__ import annotations

import json
import sys
from decimal import Decimal
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts/eval")
from rl_exits import STOP_LOSS, TAKE_PROFIT, TRAIL  # noqa: E402

with Path("data/eval/rlnews/paths_prior.jsonl").open() as _fh:
    rows = [json.loads(line) for line in _fh]
print(f"paths={len(rows)}")

tp_levels = [p for p in TAKE_PROFIT if p is not None]
sl_levels = [p for p in STOP_LOSS if p is not None]
tr_levels = sorted({a for a, _ in TRAIL if a is not None})

hits = {"take_profit": 0, "stop_loss": 0, "trail_after": 0}
paths_hit = set()
per_level: dict[tuple[str, str], int] = {}

for r in rows:
    e = r["buy_price"]
    ed = Decimal(str(e))
    bm = r["buy_minute"]
    arr = np.array([b for b in r["bars"] if b[0] > bm], dtype=float)
    if arr.size == 0:
        continue
    close = arr[:, 4]
    high = arr[:, 2]
    for name, levels, series, sign in (
        ("take_profit", tp_levels, close, +1),
        ("stop_loss", sl_levels, close, -1),
        ("trail_after", tr_levels, high, +1),
    ):
        for p in levels:
            bar_dec = ed * (1 + p) if sign > 0 else ed * (1 - p)
            bar_f = e * (1 + float(p)) if sign > 0 else e * (1 - float(p))
            barrier = float(bar_dec)
            tie = series == barrier
            if not tie.any():
                continue
            # live (Decimal) fires on the tie; twin (float) fires iff the float
            # barrier lands on the permissive side
            twin_fires = (barrier >= bar_f) if sign > 0 else (barrier <= bar_f)
            if not twin_fires:
                n = int(tie.sum())
                hits[name] += n
                paths_hit.add(r["key"] + "@" + r["day"])
                k = (name, str(p))
                per_level[k] = per_level.get(k, 0) + n

print("bar-level disagreements (live fires, twin does not):", hits)
print("distinct paths affected:", len(paths_hit), f"({len(paths_hit) / len(rows):.4%})")
for k in sorted(per_level, key=lambda k: -per_level[k])[:20]:
    print(f"  {k[0]:12s} {k[1]:>7s} {per_level[k]}")
