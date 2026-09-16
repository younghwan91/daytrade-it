"""Does `edge_null.Day` reproduce `prompt_labels.py`'s entry and ret_close?

The null is only comparable to the pool if both are priced by the same rule.
This re-prices the pool's own pairs through the null's machinery and compares
them to labels.jsonl, which was written by the original code.
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from edge_db import connect  # noqa: E402
from edge_null import fetch  # noqa: E402

LABELS = Path("data/eval/prompt_v3/labels.jsonl")
DAYS = [dt.date(2026, 3, 17), dt.date(2026, 5, 20), dt.date(2026, 7, 15), dt.date(2026, 9, 10)]


def main() -> None:
    want = {str(d) for d in DAYS}
    by_day: dict[str, list[dict]] = defaultdict(list)
    with LABELS.open() as f:
        for line in f:
            r = json.loads(line)
            if r["day"] in want:
                by_day[r["day"]].append(r)

    conn = connect()
    cur = conn.cursor()
    bad_entry = bad_ret = n = 0
    worst_e = worst_r = 0.0
    for day_s, rows in sorted(by_day.items()):
        day = dt.date.fromisoformat(day_s)
        codes = sorted({r["key"].split("|")[1] for r in rows})
        bars = fetch(cur, day, codes)
        for r in rows:
            code = r["key"].split("|")[1]
            d = bars.get(code)
            if d is None:
                continue
            i = int(r["act_min"])
            e, got = d.entry[i], d.ret(i)
            if e is None or got is None:
                continue
            n += 1
            de = abs(e - r["entry"]) / r["entry"]
            dr = abs(got - r["ret_close"])
            worst_e, worst_r = max(worst_e, de), max(worst_r, dr)
            bad_entry += de > 1e-9
            bad_ret += dr > 1e-9
        print(f"{day}  pairs={len(rows)}")
    conn.close()
    print(
        f"\ncompared {n}: entry mismatches {bad_entry} (worst rel {worst_e:.2e}), "
        f"ret_close mismatches {bad_ret} (worst abs {worst_r:.2e})"
    )


if __name__ == "__main__":
    main()
