"""Label the dashboard (live) pool with `prompt_labels.py`'s exact rules.

`data/eval/replay_events/*.jsonl` is the only pool in this system that makes
money, and it has never been priced against a null. This writes it in the
shape of `labels.jsonl` -- same keys, same `act_min` -- so `edge_null.py` and
`edge_ladder.py` can treat both pools identically.

Carried through from the event row so the ladder can condition on them:
`score` (the live BUY gate is score > 0.15), `market`, `sector`, `category`,
`in_universe` (was the ticker in that day's top-300 sweep universe at all).

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
from edge_null import fetch, trading_days, universe_for  # noqa: E402

EVENTS = Path("data/eval/replay_events")
OUT = Path("data/eval/edge/dash_labels.jsonl")
OPEN = dt.time(9, 0)


def act_min(decide: str, day: dt.date) -> int | None:
    d = dt.datetime.fromisoformat(decide).replace(tzinfo=None)
    minute = d.replace(second=0, microsecond=0)
    if minute.date() != day:
        return None
    if minute.time() < OPEN:  # live parity: cannot act before the 09:00 poll
        minute = dt.datetime.combine(day, OPEN)
    return int((minute - dt.datetime.combine(day, OPEN)).total_seconds() // 60)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default=str(EVENTS))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--size", type=int, default=300)
    args = ap.parse_args()

    ev_dir = Path(args.events)
    meta = json.loads((ev_dir / "meta.json").read_text())
    archive = set(meta.get("archive_days", []))

    by_day: dict[dt.date, list[dict]] = defaultdict(list)
    for path in sorted(ev_dir.glob("*.jsonl")):
        day = dt.date.fromisoformat(path.stem)
        with path.open() as f:
            for line in f:
                by_day[day].append(json.loads(line))

    conn = connect()
    cur = conn.cursor()
    days = sorted(by_day)
    cal = trading_days(cur, days[0] - dt.timedelta(days=30), days[-1])
    prev_of = {d: cal[i - 1] for i, d in enumerate(cal) if i}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_in = n_out = 0
    with out_path.open("w") as f:
        for day in days:
            rows = by_day[day]
            n_in += len(rows)
            univ = set(universe_for(cur, prev_of[day], args.size))
            codes = sorted({r["ticker"] for r in rows})
            bars = fetch(cur, day, codes)
            kept = 0
            for r in rows:
                code = r["ticker"]
                d = bars.get(code)
                i = act_min(r["decide_at"], day)
                if d is None or i is None:
                    continue
                ret = d.ret(i)
                if ret is None:
                    continue
                feats = r.get("features") or {}
                f.write(
                    json.dumps(
                        {
                            "key": r["key"],
                            "day": str(day),
                            "code": code,
                            "act_min": i,
                            "entry": d.entry[i],
                            "ret_close": ret,
                            "ret_1520": d.ret_1520(i),
                            "score": feats.get("score"),
                            "market": r.get("market"),
                            "sector": r.get("sector"),
                            "category": r.get("category"),
                            "in_universe": code in univ,
                            "archive": str(day) in archive,
                        }
                    )
                    + "\n"
                )
                kept += 1
            n_out += kept
            print(
                f"{day}  rows={len(rows):4d}  labelled={kept:4d}  "
                f"{'archive' if str(day) in archive else 'reconstructed'}",
                flush=True,
            )
    conn.close()
    print(f"\ntotal {n_out} / {n_in} labelled -> {out_path}")


if __name__ == "__main__":
    main()
