"""Path rows for the strictly-prior holdout, from `labels_prior.jsonl` + kr_quant bars.

Emits the row shape `exit_lab.Path.from_row` consumes:
    day, ticker, key, buy_minute, buy_price, bars=[[minute, o, h, l, c, v], ...], features

Conventions verified against `~/git/daytrade-it-rl/data/eval/rl_minimal/paths.jsonl`:
  * `buy_minute` is minutes from MIDNIGHT (that file: 540..869, bars 540..919),
    while `labels_prior.act_min` is minutes from 09:00 -> buy_minute = 540 + act_min.
  * bars are the regular session [09:00, 15:20), i.e. minute 540..919.

`minute_bars.ts` is `timestamp WITHOUT time zone`, already naive KST: ts is
compared against naive day bounds directly. No AT TIME ZONE, ever.

Sampling: deterministic, outcome-independent. Within each day the pairs are
ordered by md5(key) and the first --cap are kept. Nothing about the return,
MFE, MAE or entry enters the ordering.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
sys.path.insert(0, "scripts/eval")

import edge_db  # noqa: E402

OPEN_MINUTE = 9 * 60
END_MINUTE = 15 * 60 + 20  # exclusive; matches the reference file's 919 maximum


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", default="data/eval/rlnews/labels_prior.jsonl")
    ap.add_argument("--out", default="data/eval/rlnews/paths_prior.jsonl")
    ap.add_argument("--cap", type=int, default=200, help="max pairs kept per day")
    args = ap.parse_args()

    by_day: dict[str, list[dict]] = defaultdict(list)
    n_in = 0
    with open(args.labels) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            n_in += 1
            by_day[r["day"]].append(r)

    kept: dict[str, list[dict]] = {}
    for day, rows in by_day.items():
        rows.sort(key=lambda r: hashlib.md5(r["key"].encode()).hexdigest())
        kept[day] = rows[: args.cap]
    n_kept = sum(len(v) for v in kept.values())
    print(f"labels={n_in} days={len(by_day)} cap={args.cap} sampled={n_kept}")

    conn = edge_db.connect()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = n_nobars = 0
    with conn.cursor() as cur, out_path.open("w") as out:
        for day in sorted(kept):
            rows = kept[day]
            codes = sorted({r["key"].split("|", 1)[1] for r in rows})
            d = dt.date.fromisoformat(day)
            cur.execute(
                "SELECT code, ts, open, high, low, close, volume FROM minute_bars "
                "WHERE code = ANY(%s) AND ts >= %s AND ts < %s ORDER BY code, ts",
                (codes, d, d + dt.timedelta(days=1)),
            )
            bars: dict[str, list[list]] = defaultdict(list)
            for code, ts, o, h, lo, c, v in cur.fetchall():
                m = ts.hour * 60 + ts.minute
                if OPEN_MINUTE <= m < END_MINUTE:
                    bars[code].append([m, float(o), float(h), float(lo), float(c), int(v or 0)])
            for r in rows:
                code = r["key"].split("|", 1)[1]
                bs = bars.get(code)
                if not bs:
                    n_nobars += 1
                    continue
                out.write(
                    json.dumps(
                        {
                            "day": day,
                            "ticker": code,
                            "key": r["key"],
                            "buy_minute": OPEN_MINUTE + int(r["act_min"]),
                            "buy_price": float(r["entry"]),
                            "bars": bs,
                            "features": {"premarket": float(bool(r["premarket"]))},
                        }
                    )
                    + "\n"
                )
                n_written += 1
            print(
                f"{day} codes={len(codes):4d} rows={len(rows):4d} written={n_written}", flush=True
            )
    conn.close()
    print(f"wrote {n_written} paths ({n_nobars} dropped for no bars) -> {out_path}")


if __name__ == "__main__":
    main()
