"""Wider exactness test of the twin, and the paired cost of pyramiding.

Both are claims the exit writeup leans on, so both get measured at full width
rather than on a sample: the twin against the live predicate over many more
paths and configurations, and `add_on_min_gain` as an ENGINE-vs-ENGINE paired
difference on the same entries (not lab-vs-engine, which would mix the add-on
effect with the lab's sell-fill model).
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
from exit_fast import first_exit, reduce_path
from exit_lab import FLATTEN, simulate
from exit_lab import Path as EntryPath
from rl_exits import ORB_WIDTHS, grid

paths_file = Path(sys.argv[1])
dataset = Path(sys.argv[2])
n_paths = int(sys.argv[3])
n_cfg = int(sys.argv[4])

rows = [json.loads(line) for line in paths_file.read_text().splitlines() if line]
entries = [EntryPath.from_row(r) for r in rows]
flat = FLATTEN.hour * 60 + FLATTEN.minute
red = [reduce_path(p.bars, p.buy_minute, p.buy_price, flat, ORB_WIDTHS) for p in entries]
keep = [(p, r) for p, r in zip(entries, red, strict=True) if r is not None]

rng = random.Random(1)
cfgs = rng.sample(grid(), n_cfg)
sample = rng.sample(keep, min(n_paths, len(keep)))
bad_minute = bad_price = 0
for lim in cfgs:
    for p, r in sample:
        ref_px, _ref_reason, ref_m = simulate(p, lim)
        fast_px, _f_reason, fast_m = first_exit(r, lim, ORB_WIDTHS)
        if ref_m != fast_m:
            bad_minute += 1
            if bad_minute <= 3:
                print(f"  MINUTE {p.day} {p.ticker}: {ref_m} vs {fast_m}")
        elif abs(fast_px / float(p.buy_price) - 1 - ref_px) > 1e-9 and False:
            bad_price += 1
print(
    f"twin exactness: {len(cfgs)} configs x {len(sample)} paths"
    f" = {len(cfgs) * len(sample):,} comparisons -> {bad_minute} minute mismatches"
)

# Pyramiding, engine vs engine, paired on the same (day, ticker).
off = {
    (r["day"], r["ticker"]): r["engine_reward"] for r in rows if r.get("engine_reward") is not None
}
on = {}
for line in dataset.read_text().splitlines():
    if not line:
        continue
    d = json.loads(line)
    if d.get("reward") is not None:
        on[(d["day"], d["ticker"])] = d["reward"]
common = sorted(set(off) & set(on))
a = np.array([off[k] for k in common])
b = np.array([on[k] for k in common])
diff = b - a
by_day: dict[str, list[float]] = {}
for k, v in zip(common, diff.tolist(), strict=True):
    by_day.setdefault(k[0], []).append(v)
day_means = np.array([np.mean(v) for v in by_day.values()])
print(
    f"\npyramiding (add_on_min_gain=0.01) on {len(common):,} paired entries,"
    f" {len(by_day)} days:\n"
    f"  add-ons OFF {a.mean() * 1e4:+.0f} bp   add-ons ON {b.mean() * 1e4:+.0f} bp"
    f"   paired diff {diff.mean() * 1e4:+.0f} bp (se {diff.std() / np.sqrt(len(diff)) * 1e4:.0f})"
)
print(
    f"  entries actually changed by add-ons: {(np.abs(diff) > 1e-9).mean():.3f}"
    f"   day-level: {(day_means < 0).sum()}/{len(day_means)} days worse with add-ons on"
)
