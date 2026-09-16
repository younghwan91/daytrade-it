"""Which of our own judged names were yesterday's movers?

The universe work says the top third of our top-300 by yesterday's absolute
move drifts -42.0 bp against -21.8 for the calm two-thirds, and that the split
holds in 115 of 121 months. That is a screen change we can actually make: it
stays inside the universe we already judge, so no transfer assumption is
needed and no new labelling is required.

Re-running the horizon ladder on "rank 601-900" would need news labels for
names we have never judged, and would silently assume our selection alpha
transfers to a different liquidity stratum. This does not.

Emits one row per (day, code) for every code in the day's top 300:
    day, code, movers (bool), prev_move

READ-ONLY on kr_quant.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from edge_db import connect  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/eval/prompt_v3/labels.jsonl")
    ap.add_argument("--out", default="data/eval/edge/movers.jsonl")
    ap.add_argument("--size", type=int, default=300)
    args = ap.parse_args()

    days = set()
    with Path(args.labels).open() as f:
        for line in f:
            days.add(dt.date.fromisoformat(json.loads(line)["day"]))
    lo, hi = min(days), max(days)

    conn = connect()
    cur = conn.cursor()
    cur.execute(
        "SELECT date, code, open, close, trade_value FROM daily_bars_adjusted "
        "WHERE date >= %s AND date <= %s",
        (lo - dt.timedelta(days=15), hi),
    )
    bars: dict[dt.date, dict[str, tuple[float, float, float]]] = defaultdict(dict)
    for date, code, o, c, v in cur.fetchall():
        if o and c and float(o) > 0 and len(code) == 6 and code.isdigit():
            bars[date][code] = (float(o), float(c), float(v or 0.0))
    conn.close()

    cal = sorted(bars)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w") as out:
        for i, day in enumerate(cal):
            if i == 0 or day not in days:
                continue
            prev = bars[cal[i - 1]]
            ranked = sorted(
                (c for c, (_o, _c, v) in prev.items() if v > 0),
                key=lambda c: (-prev[c][2], c),
            )[: args.size]
            moved = sorted(ranked, key=lambda c: -abs(prev[c][1] / prev[c][0] - 1))
            k = len(moved) // 3
            for rank, code in enumerate(moved):
                out.write(
                    json.dumps(
                        {
                            "day": str(day),
                            "code": code,
                            "movers": rank < k,
                            "prev_move": prev[code][1] / prev[code][0] - 1,
                        }
                    )
                    + "\n"
                )
                n += 1
    print(f"{n} (day, code) rows over {len(days)} days -> {out_path}")


if __name__ == "__main__":
    main()
